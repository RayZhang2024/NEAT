"""Worker threads for preprocessing and filtering tasks."""

import gc
import os
import re
import shutil

import numpy as np
import pandas as pd
import psutil
from PIL import Image, TiffImagePlugin
from PyQt5.QtCore import Qt, QEventLoop, QThread, pyqtSignal

from .batch import get_raden_tiff_stack_info, load_image_file, write_fits_image_file
from ..domain import (
    LoadedImageRun,
    PreprocessingOperationResult,
    PreprocessingStatus,
)
from ..services.preprocessing_clean import clean_loaded_image_runs, positive_neighbor_mean
from ..services.preprocessing_filtering import (
    copy_filtering_related_files,
    filter_loaded_image_runs,
)
from ..services.preprocessing_layout import discover_full_process_summation
from ..services.preprocessing_summation import sum_loaded_image_runs

NORMALISATION_WINDOW_HALF_RANGE = (0, 100)
NORMALISATION_ADJACENT_RANGE = (0, 10)


def validate_normalisation_windows(window_half, adjacent_sum):
    """Return validated normalisation window settings."""
    try:
        window_half = int(window_half)
        adjacent_sum = int(adjacent_sum)
    except (TypeError, ValueError) as exc:
        raise ValueError("Normalisation n and m must be integers.") from exc

    n_min, n_max = NORMALISATION_WINDOW_HALF_RANGE
    m_min, m_max = NORMALISATION_ADJACENT_RANGE
    if not n_min <= window_half <= n_max:
        raise ValueError(
            f"Normalisation n must be between {n_min} and {n_max}."
        )
    if not m_min <= adjacent_sum <= m_max:
        raise ValueError(
            f"Normalisation m must be between {m_min} and {m_max}."
        )
    return window_half, adjacent_sum


class OutlierFilteringWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, image_runs, output_folder, base_name):
        super().__init__()
        self.image_runs = image_runs
        self.output_folder = output_folder
        self.base_name = base_name
        self._is_running = True
        self.succeeded = False
        self.failed_frames = []
        self.report_path = os.path.join(output_folder, f"{base_name}_outlier_report.csv")
        self.result: PreprocessingOperationResult | None = None

    def run(self):
        try:
            runs = tuple(
                LoadedImageRun(
                    primary_source=run["folder_path"],
                    frames=run["images"],
                    source_folders=run.get("run_folders") or None,
                    load_errors=run.get("load_errors") or (),
                )
                for run in self.image_runs
            )
            self.result = clean_loaded_image_runs(
                runs,
                self.output_folder,
                self.base_name,
                progress_callback=self.progress_updated.emit,
                message_callback=self.message.emit,
                cancellation_check=lambda: not self._is_running,
                frame_failure_callback=lambda suffix: self.failed_frames.append(suffix),
            )
            self.succeeded = self.result.status is PreprocessingStatus.SUCCEEDED
        except Exception as exc:
            self.succeeded = False
            self.result = PreprocessingOperationResult(
                PreprocessingStatus.FAILED, 0, errors=(str(exc),)
            )
            self.message.emit(f"[FATAL] {exc}")
        finally:
            self.finished.emit()

    _positive_neighbor_mean = staticmethod(positive_neighbor_mean)

    def stop(self):
        self._is_running = False

class SummationWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished         = pyqtSignal()
    message          = pyqtSignal(str)

    def __init__(self, summation_image_runs: list, base_name: str, output_folder: str):
        super().__init__()
        self.summation_image_runs = summation_image_runs
        self.base_name            = base_name
        self.output_folder        = output_folder
        self._is_running          = True     # Cooperative‑cancel flag
        self.succeeded            = False
        self.result: PreprocessingOperationResult | None = None

    # ───────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────
    def stop(self):
        self._is_running = False
        self.message.emit("Stop signal received – cancelling at next safe point.")

    # ───────────────────────────────────────────────────────────────
    # Thread entry
    # ───────────────────────────────────────────────────────────────
    def run(self):
        try:
            runs = tuple(
                _loaded_image_run_from_legacy(run)
                for run in self.summation_image_runs
            )
            self.result = sum_loaded_image_runs(
                runs,
                self.base_name,
                self.output_folder,
                progress_callback=self.progress_updated.emit,
                message_callback=self.message.emit,
                cancellation_check=lambda: not self._is_running,
            )
            self.succeeded = self.result.status is PreprocessingStatus.SUCCEEDED
            for error in self.result.errors:
                self.message.emit(f"[FATAL] Summation aborted: {error}")
        except Exception as exc:
            self.succeeded = False
            self.result = PreprocessingOperationResult(
                PreprocessingStatus.FAILED, 0, errors=(str(exc),)
            )
            self.message.emit(f"[FATAL] Summation aborted: {exc}")
        finally:
            gc.collect()
            self.finished.emit()


def _loaded_image_run_from_legacy(run: dict) -> LoadedImageRun:
    """Translate a historical GUI-worker dictionary at the adapter boundary."""
    primary_source = run.get("folder_path")
    if not isinstance(primary_source, str) or not primary_source:
        raise ValueError("Summation run has no valid source folder.")
    if "run_folders" in run and run["run_folders"] is None:
        raise ValueError("Summation run_folders must contain valid source folders.")
    source_folders = run.get("run_folders", [primary_source])
    return LoadedImageRun(
        primary_source=primary_source,
        frames=run.get("images", {}),
        source_folders=source_folders,
        load_errors=run.get("load_errors") or (),
    )

class OverlapCorrectionWorker(QThread):
    progress_updated = pyqtSignal(int)  # Emits progress percentage
    finished = pyqtSignal()             # Emits when processing is finished
    message = pyqtSignal(str)           # Emits messages for user feedback

    def __init__(self, run, base_name, output_folder):
        """
        Initialize the OverlapCorrectionWorker.

        Args:
            run (dict): Dictionary containing 'folder_path', 'images', 'spectra', 'shutter_count'.
            base_name (str): Base name for the output files.
            output_folder (str): Folder where corrected images will be saved.
        """
        super().__init__()
        self.run_data = run
        self.base_name = base_name
        self.output_folder = output_folder
        self._is_running = True
        self.succeeded = False
        self.failed_frames = []

    def run(self):
        """
        Execute the Overlap Correction process.
        """
        try:
            folder_path = self.run_data['folder_path']
            images_dict = self.run_data['images']
            spectra_data = self.run_data['spectra']
            shutter_count_data = self.run_data['shutter_count']
            
            normalized_path = os.path.normpath(folder_path)
            path_parts = normalized_path.split(os.sep)
            if len(path_parts) >= 2:
                short_path = os.path.join(path_parts[-2], path_parts[-1])
            else:
                short_path = normalized_path

            # 1) Extract the first column (ToF values) from spectra
            try:
                # Only cast to float32 if needed
                if spectra_data.dtype != np.float32:
                    spectra_data = spectra_data.astype(np.float32)

                tof_values = spectra_data[:, 0]  # Now guaranteed float32 if it wasn't
                self.message.emit("Extracted ToF values from Spectra data.")
            except Exception as e:
                self.message.emit(f"Error extracting ToF values: {e}. Aborting run.")
                self.finished.emit()
                return

            # 2) Calculate intervals and identify segmentation points
            try:
                tof_intervals = np.diff(tof_values)
                segmentation_indices = np.where(tof_intervals > 0.0001)[0] + 1
                segments = np.split(np.arange(len(tof_values)), segmentation_indices)
                n_segments = len(segments)
                self.message.emit(f"Identified {n_segments} segments based on ToF intervals.")
            except Exception as e:
                self.message.emit(f"Error during ToF segmentation: {e}. Aborting run.")
                self.finished.emit()
                return

            # 3) Extract shutter counts for these segments
            try:
                if len(shutter_count_data) < n_segments:
                    self.message.emit(f"Insufficient shutter counts for {n_segments} segments. Aborting run.")
                    self.finished.emit()
                    return

                # Filter only counts > 1000
                filtered_shutter_counts = shutter_count_data[shutter_count_data > 1000]
                if len(filtered_shutter_counts) < n_segments:
                    self.message.emit(
                        f"Only {len(filtered_shutter_counts)} shutter counts > 1000 found, "
                        f"but {n_segments} segments identified. Aborting run."
                    )
                    self.finished.emit()
                    return

                # Take the first n_segments shutter counts from the filtered list
                selected_shutter_counts = filtered_shutter_counts[:n_segments]
                self.message.emit(f"Selected {n_segments} shutter counts for {n_segments} segments.")
            except Exception as e:
                self.message.emit(f"Error processing shutter counts: {e}. Aborting run.")
                self.finished.emit()
                return

            # 3.1) Compute each segment’s mean ToF interval
            segment_intervals = []
            for idx, seg in enumerate(segments):
                if len(seg) <= 1:
                    # No interval or only 1 point
                    mean_interval = 0.0 if len(seg) < 1 else 1e-5
                else:
                    seg_tofs = tof_values[seg]
                    seg_diffs = np.diff(seg_tofs)
                    mean_interval = float(np.mean(seg_diffs))
                segment_intervals.append(mean_interval)

            # Reference interval is that of the first segment
            ref_interval = segment_intervals[0]
            if ref_interval == 0:
                self.message.emit("First segment's interval is 0. Cannot normalize to T1=0.")
                self.finished.emit()
                return

            # Debug messages
            self.message.emit(f"Segment intervals: {segment_intervals}")
            self.message.emit(f"Reference interval (segment 1) = {ref_interval:.7f}")

            # 4) Initialize cumulative intensity arrays for each segment.
            #    We'll base the shape/dtype on the first image in images_dict.
            #    If no images, this could raise KeyError.
            try:
                first_img_key = next(iter(images_dict.keys()))
                first_img_data = images_dict[first_img_key]
                if first_img_data.dtype != np.float32:
                    first_img_data = first_img_data.astype(np.float32)
                shape_512 = first_img_data.shape

                # Quick check the shape is what's expected (512, 512)
                if shape_512 != (512, 512):
                    self.message.emit(f"Image dimensions {shape_512} do not match expected (512, 512). Aborting run.")
                    self.finished.emit()
                    return

                # Create a zero array for each segment
                cumulative_intensities = [
                    np.zeros(shape_512, dtype=np.float32) for _ in segments
                ]
            except StopIteration:
                self.message.emit("No images found in the run. Aborting.")
                self.finished.emit()
                return
            except Exception as e:
                self.message.emit(f"Error initializing cumulative arrays: {e}. Aborting.")
                self.finished.emit()
                return

            # 5) Sort images by ToF order based on numeric suffix
            try:
                def extract_numeric_suffix(suf):
                    # safer approach: filter digits out of the string
                    digits = ''.join(filter(str.isdigit, suf))
                    return int(digits) if digits else -1

                sorted_suffixes = sorted(images_dict.keys(), key=extract_numeric_suffix)
                sorted_images = [images_dict[suf] for suf in sorted_suffixes]
            except Exception as e:
                self.message.emit(f"Error sorting images: {e}. Aborting run.")
                self.finished.emit()
                return

            self.message.emit("--- Starting Overlap Correction ---")

            # 6) Process each image individually
            total_imgs = len(sorted_suffixes)
            processed_images = 0
            for img_idx, suf in enumerate(sorted_suffixes):
                if not self._is_running:
                    self.message.emit("Overlap Correction process has been stopped by the user.")
                    break

                try:
                    image_data = sorted_images[img_idx]
                    # Only cast if needed
                    if image_data.dtype != np.float32:
                        image_data = image_data.astype(np.float32)

                    # Identify the segment this image belongs to
                    # (img_idx is the index in sorted order, not necessarily the real "ToF" index,
                    #  so you might want a different logic if needed. We'll keep your approach.)
                    segment_number = None
                    for seg_num, seg_indices in enumerate(segments):
                        if img_idx in seg_indices:
                            segment_number = seg_num
                            break

                    if segment_number is None:
                        self.message.emit(f"Image {img_idx+1}: No matching segment. Skipping.")
                        self.failed_frames.append(str(suf))
                        continue

                    # If it's the first image in that segment, overwrite
                    seg_idx_within = np.where(segments[segment_number] == img_idx)[0][0]
                    if seg_idx_within == 0:
                        cumulative_intensities[segment_number] = image_data.copy()
                    else:
                        cumulative_intensities[segment_number] += image_data

                    shutter_count = np.float32(selected_shutter_counts[segment_number])
                    if shutter_count == 0:
                        self.message.emit(
                            f"Image {img_idx+1}: Shutter count = 0 for segment {segment_number+1}. Skipping normalisation."
                        )
                        self.failed_frames.append(str(suf))
                        continue

                    # Step 7) Calculate p value
                    #   p = (cumulative_intensity in that segment) / shutter_count
                    p = cumulative_intensities[segment_number] / shutter_count

                    # Step 8) Correct intensities = original intensity / (1 - p)
                    #   and scale by (ref_interval / this_interval)
                    # denom = 1.0 - p
                    denom  = np.where(1.0 - p <= 0, np.float32(1e-10), 1.0 - p)
                    epsilon = 1e-10
                    denom = np.where(denom <= 0, epsilon, denom)  # avoid div-by-zero
                    corrected_intensity = image_data / denom

                    # Scale factor for time intervals
                    this_interval = np.float32(segment_intervals[segment_number])
                    ref_interval  = np.float32(segment_intervals[0])
                    scale_factor  = ref_interval / this_interval if this_interval > 0 else np.float32(1.0)

                    # this_interval = segment_intervals[segment_number]
                    # scale_factor = ref_interval / this_interval if this_interval > 0 else 1.0
                    corrected_intensity *= scale_factor

                    # Check NaN/Inf
                    if np.isnan(corrected_intensity).any() or np.isinf(corrected_intensity).any():
                        self.message.emit(f"Image {img_idx+1}: NaN or Inf after correction. Skipping.")
                        self.failed_frames.append(str(suf))
                        continue

                    # Construct output path
                    try:
                        numeric_suffix = ''.join(filter(str.isdigit, suf))
                        original_filename = f"{self.base_name}_{numeric_suffix}.fits"
                        corrected_filename = f"Corrected_{original_filename}"
                        output_path = os.path.join(self.output_folder, corrected_filename)
                    except Exception as e:
                        self.message.emit(f"Error constructing filename: {e}. Skipping.")
                        self.failed_frames.append(str(suf))
                        continue

                    # Save corrected image
                    write_fits_image_file(output_path, corrected_intensity, overwrite=True)
                    processed_images += 1

                    # Update progress
                    overall_progress = int(((img_idx + 1) / total_imgs) * 100)
                    self.progress_updated.emit(overall_progress)

                except Exception as e:
                    self.message.emit(f"Error processing image '{suf}': {e}. Skipping.")
                    self.failed_frames.append(str(suf))
                    continue

            # Final messages
            self.succeeded = (
                self._is_running
                and not self.failed_frames
                and processed_images == total_imgs
            )
            if self.succeeded:
                self.message.emit("Overlap Correction process completed successfully.")
            else:
                self.message.emit(
                    "Overlap Correction failed or stopped before all frames "
                    "were written."
                )

            # 9) Copy spectra and shuttercount files to output folder
            try:
                spectra_suffix = '_Spectra.txt'
                shuttercount_suffix = '_ShutterCount.txt'

                # Copy spectra files
                spectra_files = [f for f in os.listdir(folder_path) if f.endswith(spectra_suffix)]
                if not spectra_files:
                    self.message.emit(f"No files ending with '{spectra_suffix}' found in \\{short_path}.")
                else:
                    for spectra_file in spectra_files:
                        source_path = os.path.join(folder_path, spectra_file)
                        dest_path = os.path.join(self.output_folder, spectra_file)
                        shutil.copyfile(source_path, dest_path)
                        self.message.emit(f"'{spectra_file}' copied to output folder.")

                # Copy shuttercount files
                shuttercount_files = [f for f in os.listdir(folder_path) if f.endswith(shuttercount_suffix)]
                if not shuttercount_files:
                    self.message.emit(f"No files ending with '{shuttercount_suffix}' found in \\{short_path}.")
                else:
                    for shuttercount_file in shuttercount_files:
                        source_path = os.path.join(folder_path, shuttercount_file)
                        dest_path = os.path.join(self.output_folder, shuttercount_file)
                        shutil.copyfile(source_path, dest_path)
                        self.message.emit(f"'{shuttercount_file}' copied to output folder.")

            except Exception as e:
                self.message.emit(f"Error copying spectra or shuttercount files: {e}")

        except Exception as e:
            # If a top-level error happened, log and skip gracefully
            self.succeeded = False
            self.message.emit(f"Error in OverlapCorrectionWorker: {e}")

        # Optional: if memory usage is extremely high, you could call gc.collect() once here
        # gc.collect()

        # Emit finished signal
        if not self.succeeded:
            self.message.emit("Overlap Correction did not complete successfully.")
        self.finished.emit()

    def stop(self):
        """
        Stop the Overlap Correction process.
        """
        self._is_running = False
        self.message.emit("Stop signal received. Terminating Overlap Correction process.")

class NormalisationWorker(QThread):
    progress_updated = pyqtSignal(int)   # Emits progress percentage (0-100)
    finished         = pyqtSignal()      # Emits when processing is finished
    message          = pyqtSignal(str)   # Emits messages for user feedback

    def __init__(
        self,
        normalisation_image_runs,
        normalisation_open_beam_runs,
        output_folder,
        base_name,
        window_half,
        adjacent_sum,
    ):
        super().__init__()
        window_half, adjacent_sum = validate_normalisation_windows(
            window_half, adjacent_sum
        )
        self.normalisation_image_runs      = normalisation_image_runs
        self.normalisation_open_beam_runs  = normalisation_open_beam_runs
        self.output_folder                 = output_folder
        self.base_name                     = base_name
        self.window_half                   = window_half
        self.adjacent_sum                  = adjacent_sum
        self._is_running                   = True
        self.succeeded                     = False
        self.failed_frames                 = []

    @staticmethod
    def _read_shutter_count(folder_path):
        """Read the 2nd column, 1st row from *_ShutterCount.txt."""
        try:
            fname = next(f for f in os.listdir(folder_path)
                         if f.endswith('_ShutterCount.txt'))
        except StopIteration:
            raise FileNotFoundError("no *_ShutterCount.txt found")

        with open(os.path.join(folder_path, fname), 'r') as fh:
            first_line = fh.readline().strip()

        parts = [p for p in re.split(r'[,\s]+', first_line) if p]
        if len(parts) < 2:
            raise ValueError(f"cannot parse shutter count in {fname}")

        return float(parts[1])

    def run(self):
        try:
            # prepare a short display path
            norm_path = os.path.normpath(self.output_folder)
            parts = norm_path.split(os.sep)
            short_path = os.path.join(parts[-2], parts[-1]) if len(parts) >= 2 else norm_path

            # basic validation
            if not self.normalisation_image_runs or not self.normalisation_open_beam_runs:
                self.message.emit("No runs provided. Aborting.")
                return

            if len(self.normalisation_image_runs) != len(self.normalisation_open_beam_runs):
                self.message.emit("Data vs. Open‐beam count mismatch. Aborting.")
                return

            total_images = sum(len(r['images']) for r in self.normalisation_image_runs)
            processed_images = 0

            self.message.emit("<b>--- Starting Normalisation ---</b>")
            window_half = self.window_half
            full_win    = (2*window_half+1)**2
            thresh      = 1e-7
            frame_win   = 2*self.adjacent_sum+1

            self.message.emit(
                f"Using {2*window_half+1}×{2*window_half+1} spatial window "
                f"and {frame_win} frames."
            )

            for run_idx, (data_run, ob_run) in enumerate(
                zip(self.normalisation_image_runs,
                    self.normalisation_open_beam_runs), start=1
            ):
            
                if not self._is_running:
                    self.message.emit("User stopped the process.")
                    break

                # read shutter counts
                try:
                    sc = self._read_shutter_count(data_run['folder_path'])
                    ob = self._read_shutter_count(ob_run['folder_path'])
                    scale = np.float32(ob/sc) if sc>0 else 1.0
                    self.message.emit(
                        f"sample={sc:.0f}, open‐beam={ob:.0f}, scale={scale:.4f}"
                    )
                except Exception as e:
                    self.message.emit(
                        f"shutter‐count error ({e}), scale=1.0"
                    )
                    scale = np.float32(1.0)

                data_imgs = data_run['images']
                ob_imgs   = ob_run['images']
                common    = sorted(set(data_imgs) & set(ob_imgs))
                if not common:
                    self.message.emit(f" no matching suffixes—skipping.")
                    self.failed_frames.append(f"run-{run_idx}:no-matching-suffix")
                    continue

                for i, suffix in enumerate(common):
                    if not self._is_running:
                        break
  

                    try:
                        img  = data_imgs[suffix].astype(np.float32)
                        
                        if self.adjacent_sum == 0:
                            ob0 = ob_imgs[suffix].astype(np.float64).copy()
                            start = 0
                            end = 0

                        else:
                        # build combined open‐beam
                            start = max(0, i-self.adjacent_sum)
                            end   = min(len(common)-1, i+self.adjacent_sum)
                            ob0   = ob_imgs[common[start]].astype(np.float64).copy()
                            for j in range(start+1, end+1):
                                ob0 += ob_imgs[common[j]].astype(np.float64)

                        if img.shape != ob0.shape:
                            self.message.emit(
                                f" {suffix}: shape mismatch—skipping."
                            )
                            self.failed_frames.append(str(suffix))
                            continue

                        h, w = img.shape
                        if h < 2*window_half+1 or w < 2*window_half+1:
                            self.message.emit(
                                f" {suffix}: too small for window—skipping."
                            )
                            self.failed_frames.append(str(suffix))
                            continue

                        # integral images
                        II = ob0.cumsum(0).cumsum(1)
                        II = np.pad(II, ((1,0),(1,0)), 'constant')
                        II1 = np.pad(np.ones_like(img).cumsum(0).cumsum(1), ((1,0),(1,0)), 'constant')

                        # get sums via broadcasted indices
                        I, J = np.ogrid[:h, :w]
                        i0, i1 = I-window_half, I+window_half+1
                        j0, j1 = J-window_half, J+window_half+1
                        i0, i1 = np.clip(i0,0,h), np.clip(i1,0,h)
                        j0, j1 = np.clip(j0,0,w), np.clip(j1,0,w)

                        part_sum = II[i1, j1] - II[i0, j1] - II[i1, j0] + II[i0, j0]
                        part_cnt = II1[i1,j1] - II1[i0,j1] - II1[i1,j0] + II1[i0,j0]
                        scaled   = np.where(part_cnt>0,
                                            part_sum*(full_win/part_cnt),
                                            thresh).astype(np.float32)

                        normed = ((end-start+1)*full_win*img/scaled)*scale
                        normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                        out_fname = f"{self.base_name}_{suffix}.fits"
                        write_fits_image_file(os.path.join(self.output_folder, out_fname),
                                              normed, overwrite=True)

                        del data_imgs[suffix]
                        processed_images += 1
                        self.progress_updated.emit(int(100*processed_images/total_images))

                    except Exception as e:
                        self.failed_frames.append(str(suffix))
                        self.message.emit(
                            f" {suffix}: error ({e})—skipping."
                        )

                        
                    finally:
                        # free big arrays
                        for v in ('img','ob0','II','II1','scaled','normed'):
                            if v in locals():
                                del locals()[v]
                    if suffix == 1500:
                        gc.collect()
                        QThread.sleep(2)
                    if suffix == 2500:
                        gc.collect()
                        QThread.sleep(2)
            
                # per‐run cleanup
                data_imgs.clear()
                gc.collect()
                try:
                    self.copy_related_files(run_idx, data_run)
                except Exception as e:
                    self.message.emit(f"copy_related_files failed: {e}")

                self.message.emit("Run done.")
                if self._is_running:
                    QThread.sleep(5)

            self.succeeded = (
                self._is_running
                and not self.failed_frames
                and processed_images == total_images
            )
            status = "completed" if self.succeeded else "failed or incomplete"
            self.message.emit(
                f"Normalisation {status}: {processed_images} of "
                f"{total_images} images written to {short_path}."
            )

        except Exception as e:
            self.succeeded = False
            self.message.emit(f"Fatal error in normalisation: {e}")

        finally:
            # final cleanup & memory report
            gc.collect()
            proc  = psutil.Process(os.getpid())
            memMB = proc.memory_info().rss / (1024.**2)
            self.message.emit(f"<b>Final memory usage:</b> {memMB:.1f} MB")
            self.finished.emit()

    def stop(self):
        """
        Stop the Normalisation process.
        """
        self._is_running = False
        self.message.emit("Stop signal received. Terminating Normalisation process.")

    def copy_related_files(self, run_idx, data_run):
        """
        Copies related files (e.g., *_Spectra.txt and *_ShutterCount.txt) from the data run's folder
        to the output folder with unique run identifiers.
        """
        try:
            spectra_suffix = '_Spectra.txt'
            shuttercount_suffix = '_ShutterCount.txt'

            def create_unique_filename(run_number, original_filename):
                return f"Run{run_number}_{original_filename}"

            data_folder = data_run['folder_path']
            data_files = os.listdir(data_folder)

            # Spectra
            spectra_files = [f for f in data_files if f.endswith(spectra_suffix)]
            if spectra_files:
                for sf in spectra_files:
                    source_path = os.path.join(data_folder, sf)
                    dest_filename = create_unique_filename(run_idx, sf)
                    dest_path = os.path.join(self.output_folder, dest_filename)
                    shutil.copyfile(source_path, dest_path)
                    self.message.emit(f"Copied '{sf}' to '{dest_filename}'.")
                    # If only one file is expected, you could break here
            else:
                self.message.emit(f"No file ending with '{spectra_suffix}' found in {data_folder}.")

            # ShutterCount
            shuttercount_files = [f for f in data_files if f.endswith(shuttercount_suffix)]
            if shuttercount_files:
                for scf in shuttercount_files:
                    source_path = os.path.join(data_folder, scf)
                    dest_filename = create_unique_filename(run_idx, scf)
                    dest_path = os.path.join(self.output_folder, dest_filename)
                    shutil.copyfile(source_path, dest_path)
                    self.message.emit(f"Copied '{scf}' to '{dest_filename}'.")
                    # If only one file is expected, you could break here
            else:
                self.message.emit(f"No file ending with '{shuttercount_suffix}' found in {data_folder}.")

        except Exception as e:
            self.message.emit(f"Error copying related files: {e}")

class FullProcessWorker(QThread):
    # Signals for messages, progress updates and final completion
    message = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    load_progress_updated = pyqtSignal(int) 
    finished = pyqtSignal()

    def __init__(self, sample_folder: str, open_beam_folder: str, output_folder: str,
                 base_name: str, window_half: int, adjacent_sum: int):
        super().__init__()
        window_half, adjacent_sum = validate_normalisation_windows(
            window_half, adjacent_sum
        )
        self.sample_folder = sample_folder
        self.open_beam_folder = open_beam_folder
        self.output_folder = output_folder
        self.base_name = base_name
        self.window_half = window_half
        self.adjacent_sum = adjacent_sum
        self._is_running = True  # Flag to track whether the user requested a stop
        self.succeeded = False
        self._active_child = None
        
    def get_short_path(self, full_path, levels=2):
        """
        Returns the last `levels` parts of a path.
        
        Args:
            full_path (str): The full file or folder path.
            levels (int): How many trailing parts to keep. Default is 2.
            
        Returns:
            str: The shortened path.
        """
        normalized_path = os.path.normpath(full_path)
        path_parts = normalized_path.split(os.sep)
        if len(path_parts) >= levels:
            short_path = os.path.join(*path_parts[-levels:])
        else:
            short_path = normalized_path
        return short_path


    def run(self):
        try:
            self.message.emit("=== <b>Full Process Pipeline Started</b> ===")

            # 1) Summation ---------------------------------------------------
            sample_after_sum = self.maybe_do_summation(self.sample_folder, "Sample")
            if not self._continue("sample summation"): return
            self.progress_updated.emit(0)       # reset for next stage

            openbeam_after_sum = self.maybe_do_summation(self.open_beam_folder, "OpenBeam")
            if not self._continue("open‑beam summation"): return
            self.progress_updated.emit(0)

            # 2) Clean sample ----------------------------------------------
            sample_after_clean = self.do_outlier_removal(sample_after_sum, "Sample")
            if not self._continue("sample cleaning"): return
            self.progress_updated.emit(0)

            # 3) Clean OB ---------------------------------------------------
            openbeam_after_clean = self.do_outlier_removal(openbeam_after_sum, "OpenBeam")
            if not self._continue("open‑beam cleaning"): return
            self.progress_updated.emit(0)

            # 4) Overlap sample --------------------------------------------
            sample_after_overlap = self.do_overlap_correction(sample_after_clean, "Sample")
            if not self._continue("sample overlap"): return
            self.progress_updated.emit(0)

            # 5) Overlap OB -------------------------------------------------
            openbeam_after_overlap = self.do_overlap_correction(openbeam_after_clean, "OpenBeam")
            if not self._continue("open‑beam overlap"): return
            self.progress_updated.emit(0)

            # 6) Normalisation ---------------------------------------------
            self.do_normalisation(sample_after_overlap, openbeam_after_overlap)
            if not self._continue("normalisation"): return
            self.progress_updated.emit(0)

            self.succeeded = True
            self.message.emit("=== <b>Full Process Completed Successfully</b> ===")

        except Exception as exc:
            self.succeeded = False
            self.message.emit(f"[ERROR] {exc}")
        finally:
            gc.collect()
            self.finished.emit()

    # ------------------------------------------------------------------
    # HELPER: continue or exit early
    # ------------------------------------------------------------------
    def _continue(self, phase):
        if not self._is_running:
            self.message.emit(f"Stopped during {phase}.")
            return False
        return True
    
    def _early_exit(self, reason: str):
        """Emits a final message and returns quickly."""
        self.message.emit(reason)
        # (The final self.finished.emit() is in the 'finally' block of run().) 
        
    def stop(self):
        """
        Called from the main UI when the user clicks 'Stop Full Process'.
        Sets _is_running=False so the while loops in each step can stop the worker.
        """
        self._is_running = False
        child = self._active_child
        if child is not None and hasattr(child, "stop"):
            child.stop()
        self.message.emit("FullProcessWorker: Stop signal received.")

    # ---------------------------------------------------------------
    # Summation Step (conditionally skipped if no subfolders)
    # ---------------------------------------------------------------
    def maybe_do_summation(self, folder: str, label: str) -> str:
        if not self._is_running:
            return folder
    
        # Check for subfolders
        discovery = discover_full_process_summation(folder)
        subfolders = discovery.folders
        
        short_path = self.get_short_path(folder, levels=2)
    
        if not discovery.should_sum:
            self.message.emit(f"0_sumation_{label}: No subfolders found in '\\{short_path}'. Skipping Summation.")
            return folder
    
        # Get the original folder name (e.g., "sample_data")
        original_folder_name = os.path.basename(folder.rstrip(os.sep))
        # Build the output folder: e.g., "0_summed_sample_data"
        summation_output = os.path.join(self.output_folder, f"0_summed_{original_folder_name}")
        os.makedirs(summation_output, exist_ok=True)
    
        self.message.emit(f"0_sumation_{label}: Found {len(subfolders)} subfolder(s) in '\\{short_path}'. Performing Summation...")
        runs = []
        for sf in subfolders:
            r = self.load_run_dict(sf)
            if r.get("images"):
                runs.append(r)
    
        if not runs:
            raise RuntimeError(
                f"0_summation_{label}: no valid FITS/TIFF images were found "
                f"in subfolders of '\\{short_path}'."
            )
    
        # Create a modified base name for image naming (e.g., "summed_sample_data")
        modified_base_name = f"summed_{original_folder_name}"
        worker = SummationWorker(runs, modified_base_name, summation_output)
        # worker.progress_updated.connect(lambda v: self.progress_updated.emit(v // 4))
        worker.progress_updated.connect(self.progress_updated, Qt.QueuedConnection)
               
        worker.message.connect(self.message.emit, Qt.QueuedConnection)

        # block until child finishes, no busy‑wait
        loop = QEventLoop(); 
        worker.finished.connect(loop.quit); 
        self._active_child = worker
        worker.start()
        loop.exec_()
        self._active_child = None
        if not worker.succeeded:
            raise RuntimeError(f"0_summation_{label} failed.")
    
        # Free memory from runs
        del runs
        import gc
        gc.collect()
        
        summation_output_short = self.get_short_path(summation_output, levels = 3)
    
        if self._is_running:
            self.message.emit(f"<b>0_sumation_{label} complete</b>, saved at: <b>\\{summation_output_short}</b>")
            return summation_output
        else:
            return folder


    # ---------------------------------------------------------------
    # Outlier Removal (Clean) Step
    # ---------------------------------------------------------------

    def do_outlier_removal(self, folder: str, label: str) -> str:
        if not self._is_running:
            return folder
        
        short_path = self.get_short_path(folder, levels=2)
    
        self.message.emit(f"1_clean_{label}: Starting Outlier Removal on \\{short_path}...")
        run = self.load_run_dict(folder)
        if not run.get("images"):
            raise RuntimeError(
                f"1_clean_{label}: no images found in \\{short_path}."
            )
        if run.get("load_errors"):
            raise RuntimeError(
                f"1_clean_{label}: one or more input frames could not be loaded."
            )
    
        original_folder_name = os.path.basename(folder.rstrip(os.sep))
        # Output folder: e.g., "1_cleaned_sample_data" or "1_cleaned_openbeam_data"
        outlier_output = os.path.join(self.output_folder, f"1_cleaned_{original_folder_name}")
        os.makedirs(outlier_output, exist_ok=True)
    
        # Create a base name for cleaned image names (e.g., "cleaned_sample_data")
        modified_base_name = f"cleaned_{original_folder_name}"
        worker = OutlierFilteringWorker([run], outlier_output, modified_base_name)
        # worker.progress_updated.connect(lambda v: self.progress_updated.emit(25 + v // 4))
        worker.progress_updated.connect(self.progress_updated, Qt.QueuedConnection)
        worker.message.connect(self.message.emit, Qt.QueuedConnection)

        # block until child finishes, no busy‑wait
        loop = QEventLoop(); 
        worker.finished.connect(loop.quit); 
        self._active_child = worker
        worker.start()
        loop.exec_()
        self._active_child = None
        if not getattr(worker, "succeeded", True):
            raise RuntimeError(f"1_clean_{label} failed or skipped frames.")
    
        del run
        import gc
        gc.collect()
        
        short_path = self.get_short_path(outlier_output, levels=2)
    
        if self._is_running:
            self.message.emit(f"<b>1_clean_{label} complete</b>, saved at: <b>\\{short_path}</b>")
            return outlier_output
        else:
            return folder


    # ---------------------------------------------------------------
    # Overlap Correction Step
    # ---------------------------------------------------------------

    def do_overlap_correction(self, folder: str, label: str) -> str:
        if not self._is_running:
            return folder
        
        short_path = self.get_short_path(folder, levels=2)
    
        self.message.emit(f"2_correction_{label}: Starting Overlap Correction on \\{short_path}...")
        run = self.load_run_dict(folder)
        if not run.get("images"):
            raise RuntimeError(
                f"2_correction_{label}: no images found in \\{short_path}."
            )
        if run.get("load_errors"):
            raise RuntimeError(
                f"2_correction_{label}: one or more input frames could not be loaded."
            )
    
        try:
            all_files = os.listdir(folder)
        except Exception as e:
            raise RuntimeError(
                f"2_correction_{label}: cannot access \\{short_path}: {e}"
            ) from e
    
        # Load additional data (Spectra and ShutterCount)
        spectra_file = next((os.path.join(folder, f) for f in all_files if f.endswith("_Spectra.txt")), None)
        # short_spectra=self.get_short_path(spectra_file, levels=2)
        if spectra_file:
            try:
                run["spectra"] = np.loadtxt(spectra_file)
                # self.message.emit(f"2_correction_{label}: Loaded Spectra from {short_spectra}")
            except Exception as e:
                self.message.emit(f"2_correction_{label}: Error loading Spectra: {e}")
                run["spectra"] = None
        else:
            run["spectra"] = None
    
        shutter_file = next((os.path.join(folder, f) for f in all_files if f.endswith("_ShutterCount.txt")), None)
        # short_shutter=self.get_short_path(shutter_file, levels=2)
        if shutter_file:
            try:
                sc_data = np.loadtxt(shutter_file)
                run["shutter_count"] = sc_data[sc_data != 0]
                # self.message.emit(f"2_correction_{label}: Loaded ShutterCount from {short_shutter}")
            except Exception as e:
                self.message.emit(f"2_correction_{label}: Error loading ShutterCount: {e}")
                run["shutter_count"] = None
        else:
            run["shutter_count"] = None
    
        if run["spectra"] is None or run["shutter_count"] is None:
            raise RuntimeError(
                f"2_correction_{label}: Spectra or ShutterCount is missing; "
                "Full Process requires overlap correction."
            )
    
        original_folder_name = os.path.basename(folder.rstrip(os.sep))
        # Output folder: e.g., "2_corrected_sample_data"
        overlap_output = os.path.join(self.output_folder, f"2_corrected_{original_folder_name}")
        os.makedirs(overlap_output, exist_ok=True)
    
        # Create a base name to be used for image naming (e.g., "corrected_sample_data")
        modified_base_name = f"corrected_{original_folder_name}"
        worker = OverlapCorrectionWorker(run, modified_base_name, overlap_output)
        # worker.progress_updated.connect(lambda v: self.progress_updated.emit(50 + v // 4))
        worker.progress_updated.connect(self.progress_updated, Qt.QueuedConnection)
        worker.message.connect(self.message.emit, Qt.QueuedConnection)

        # block until child finishes, no busy‑wait
        loop = QEventLoop(); 
        worker.finished.connect(loop.quit); 
        self._active_child = worker
        worker.start()
        loop.exec_()
        self._active_child = None
        if not worker.succeeded:
            raise RuntimeError(f"2_correction_{label} failed.")
    
        del run
        import gc
        gc.collect()
        overlap_output_short=self.get_short_path(overlap_output, levels=3)
    
        if self._is_running:
            self.message.emit(f"<b>2_correction_{label} complete</b>, saved at: <b>\\{overlap_output_short}</b>")
            return overlap_output
        else:
            return folder


    # ---------------------------------------------------------------
    # Normalisation Step
    # ---------------------------------------------------------------

    def do_normalisation(self, sample_folder: str, openbeam_folder: str):
        if not self._is_running:
            return
        short_path_sample = self.get_short_path(sample_folder, levels=2)
        short_path_ob = self.get_short_path(openbeam_folder, levels=2)
    
        self.message.emit(
            f"3_normalisation: Sample='{short_path_sample}', OpenBeam='{short_path_ob}'"
        )
    
        sample_run = self.load_run_dict(sample_folder)
        openbeam_run = self.load_run_dict(openbeam_folder)
        if (not sample_run.get("images")) or (not openbeam_run.get("images")):
            raise RuntimeError(
                "Normalisation cannot start because sample or open beam has no images."
            )
        if sample_run.get("load_errors") or openbeam_run.get("load_errors"):
            raise RuntimeError(
                "Normalisation cannot start because one or more input frames "
                "could not be loaded."
            )
    
        # Set output folder to a fixed name for the merged result
        normalised_output = os.path.join(self.output_folder, "3_normalised_original")
        os.makedirs(normalised_output, exist_ok=True)
    
        # Base name for normalised images (here using "normalised" as the prefix)
        modified_base_name = "normalised"
        worker = NormalisationWorker(
            [sample_run],
            [openbeam_run],
            normalised_output,
            modified_base_name,
            self.window_half,
            self.adjacent_sum
        )
        # worker.progress_updated.connect(lambda v: self.progress_updated.emit(75 + v // 4))
        worker.progress_updated.connect(self.progress_updated.emit, Qt.QueuedConnection)
        worker.message.connect(self.message.emit, Qt.QueuedConnection)

        # block until child finishes, no busy‑wait
        loop = QEventLoop(); 
        worker.finished.connect(loop.quit); 
        self._active_child = worker
        worker.start()
        loop.exec_()
        self._active_child = None
        if not worker.succeeded:
            raise RuntimeError("3_normalisation failed or skipped one or more frames.")
    
        del sample_run, openbeam_run
        import gc
        gc.collect()
        
        normalised_output_short=self.get_short_path(normalised_output, levels=3)
    
        if self._is_running:
            self.message.emit(f"<b>3_normalisation complete</b>, saved at: <b>\\{normalised_output_short}</b>")

  
    def load_run_dict(self, folder: str) -> dict:
        """
        Scan a folder for FITS/TIFF files and use the final underscore-delimited
        stem component as the frame suffix. The suffix need not contain five
        digits.

        Returns a dictionary:
            { 'folder_path': folder, 'images': {suffix: data} }.
        Also emits loading progress via load_progress_updated.
        """
        run = {"folder_path": folder, "images": {}, "load_errors": []}
        if not os.path.isdir(folder):
            self.message.emit(f"Folder not found: {folder}")
            return run
        
        short_path = self.get_short_path(folder, levels=2)

        try:
            image_files = [
                f for f in os.listdir(folder)
                if f.lower().endswith((".fits", ".fit", ".tiff", ".tif"))
            ]
            total_files = len(image_files)
            processed_files = 0

            for f in image_files:
                processed_files += 1
                stem = os.path.splitext(f)[0]
                suffix = stem.rsplit("_", 1)[-1].strip()
                if not suffix:
                    error = f"File '{f}' has an empty frame suffix."
                    run["load_errors"].append(error)
                    self.message.emit(error)
                    self.load_progress_updated.emit(
                        int((processed_files / total_files) * 100)
                    )
                    continue
                if suffix in run["images"]:
                    error = (
                        f"File '{f}' duplicates frame suffix '{suffix}'. "
                        "Frame names must be unique."
                    )
                    run["load_errors"].append(error)
                    self.message.emit(error)
                    self.load_progress_updated.emit(
                        int((processed_files / total_files) * 100)
                    )
                    continue
                try:
                    path = os.path.join(folder, f)
                    data = load_image_file(path)
                    run["images"][suffix] = data.astype(np.float32)
                except Exception as e:
                    error = f"Error loading file {f} in \\{short_path}: {e}"
                    run["load_errors"].append(error)
                    self.message.emit(error)
                # Update loading progress after processing each file
                self.load_progress_updated.emit(int((processed_files / total_files) * 100))
        except Exception as e:
            error = f"Error reading folder \\{short_path}: {e}"
            run["load_errors"].append(error)
            self.message.emit(error)
        return run

class FilteringWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, filtering_image_runs, filtering_mask, output_folder, base_name):
        super().__init__()
        self.filtering_image_runs = filtering_image_runs
        self.filtering_mask = filtering_mask
        self.output_folder = output_folder
        self.base_name = base_name
        self._is_running = True
        self.succeeded = False
        self.failed_frames = []
        self.result: PreprocessingOperationResult | None = None

    def run(self):
        try:
            runs = tuple(
                LoadedImageRun(
                    primary_source=run["folder_path"],
                    frames=run["images"],
                    load_errors=run.get("load_errors") or (),
                )
                for run in self.filtering_image_runs
            )
            self.result = filter_loaded_image_runs(
                runs,
                self.filtering_mask,
                self.output_folder,
                self.base_name,
                progress_callback=self.progress_updated.emit,
                message_callback=self.message.emit,
                cancellation_check=lambda: not self._is_running,
                frame_failure_callback=lambda suffix: self.failed_frames.append(suffix),
                validated_mask_callback=self._set_validated_mask,
                summary_path_callback=self._set_summary_path,
            )
            self.succeeded = self.result.status is PreprocessingStatus.SUCCEEDED
            if not self.filtering_image_runs or self.filtering_mask is None:
                self.finished.emit()  # Preserve the legacy early-return signal.
                return
        except Exception as exc:
            self.succeeded = False
            self.result = PreprocessingOperationResult(
                PreprocessingStatus.FAILED, 0, errors=(str(exc),)
            )
            self.message.emit(f"Error during filtering: {exc}")
        finally:
            gc.collect()
            self.finished.emit()

    def _set_validated_mask(self, mask: np.ndarray) -> None:
        self.filtering_mask = mask

    def _set_summary_path(self) -> str:
        self.output_folder_short = self.get_short_path(self.output_folder, levels=2)
        return self.output_folder_short

    def stop(self):
        self._is_running = False
        self.message.emit("Stop signal received. Terminating Filtering process.")

    @staticmethod
    def get_short_path(full_path, levels=2):
        """Return the final path components for concise progress messages."""
        normalized_path = os.path.normpath(full_path)
        path_parts = normalized_path.split(os.sep)
        if len(path_parts) >= levels:
            return os.path.join(*path_parts[-levels:])
        return normalized_path

    def copy_related_files(self, run_idx, data_run):
        """Keep the public sidecar-copy entry point as a service delegate."""
        copy_filtering_related_files(
            run_idx,
            data_run.get("folder_path", None),
            self.output_folder,
            message_callback=self.message.emit,
        )

class RadenNormalisationWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(
        self,
        sample_run,
        open_beam_run,
        output_folder,
        base_name,
        window_half,
        adjacent_sum,
    ):
        super().__init__()
        window_half, adjacent_sum = validate_normalisation_windows(
            window_half, adjacent_sum
        )
        self.sample_run = sample_run
        self.open_beam_run = open_beam_run
        self.output_folder = output_folder
        self.base_name = base_name
        self.window_half = window_half
        self.adjacent_sum = adjacent_sum
        self._is_running = True
        self.succeeded = False

    @staticmethod
    def _pulse_count(info):
        meta = info.get("axes", {}).get("meta", {})
        for key in ("pulses_with_data", "pulses"):
            value = meta.get(key)
            if value is not None and np.isfinite(value) and value > 0:
                return float(value)
        return None

    @staticmethod
    def _read_frame(tiff, index):
        tiff.seek(index)
        frame = np.array(tiff, dtype=np.float32, copy=True)
        np.nan_to_num(frame, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return frame

    @staticmethod
    def _same_tof_axis(sample_info, open_beam_info):
        sample_tof = sample_info.get("axes", {}).get("tof", {})
        open_beam_tof = open_beam_info.get("axes", {}).get("tof", {})
        for key in ("bins", "min", "max"):
            try:
                if not np.isclose(float(sample_tof[key]), float(open_beam_tof[key])):
                    return False
            except (KeyError, TypeError, ValueError):
                return False
        return str(sample_tof.get("units", "")).lower() == str(open_beam_tof.get("units", "")).lower()

    def _validate(self, sample_info, open_beam_info):
        if int(sample_info["n_frames"]) != int(open_beam_info["n_frames"]):
            raise ValueError(
                f"Frame count mismatch: sample={sample_info['n_frames']}, open beam={open_beam_info['n_frames']}."
            )
        if tuple(sample_info["image_shape"]) != tuple(open_beam_info["image_shape"]):
            raise ValueError(
                f"Image shape mismatch: sample={sample_info['image_shape']}, open beam={open_beam_info['image_shape']}."
            )
        if not self._same_tof_axis(sample_info, open_beam_info):
            raise ValueError("Sample and open-beam RADEN TOF axes do not match.")

    def _normalise_frame(self, sample_frame, open_beam_sum, frame_count, scale):
        window_half = self.window_half
        full_win = (2 * window_half + 1) ** 2
        thresh = 1e-7

        if sample_frame.shape != open_beam_sum.shape:
            raise ValueError("sample/open-beam frame shape mismatch")
        height, width = sample_frame.shape
        if height < 2 * window_half + 1 or width < 2 * window_half + 1:
            raise ValueError("frame is too small for the selected spatial window")

        integral = open_beam_sum.cumsum(0).cumsum(1)
        integral = np.pad(integral, ((1, 0), (1, 0)), "constant")
        counts = np.pad(np.ones_like(sample_frame).cumsum(0).cumsum(1), ((1, 0), (1, 0)), "constant")

        rows, cols = np.ogrid[:height, :width]
        r0, r1 = rows - window_half, rows + window_half + 1
        c0, c1 = cols - window_half, cols + window_half + 1
        r0, r1 = np.clip(r0, 0, height), np.clip(r1, 0, height)
        c0, c1 = np.clip(c0, 0, width), np.clip(c1, 0, width)

        part_sum = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
        part_count = counts[r1, c1] - counts[r0, c1] - counts[r1, c0] + counts[r0, c0]
        scaled = np.where(part_count > 0, part_sum * (full_win / part_count), thresh).astype(np.float32)

        normalised = (frame_count * full_win * sample_frame / scaled) * np.float32(scale)
        return np.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _copy_sidecars(self, sample_info, output_tiff):
        output_stem = os.path.splitext(os.path.basename(output_tiff))[0]
        source_folder = os.path.dirname(sample_info["file_path"])
        copied_exts = set()
        metadata_path = sample_info.get("metadata_path")
        if metadata_path and os.path.exists(metadata_path):
            ext = os.path.splitext(metadata_path)[1].lower()
            shutil.copyfile(metadata_path, os.path.join(self.output_folder, output_stem + ext))
            copied_exts.add(ext)

        for ext in (".stat", ".json", ".log"):
            if ext in copied_exts:
                continue
            for name in os.listdir(source_folder):
                source_path = os.path.join(source_folder, name)
                if os.path.isfile(source_path) and os.path.splitext(name)[1].lower() == ext:
                    shutil.copyfile(source_path, os.path.join(self.output_folder, output_stem + ext))
                    copied_exts.add(ext)
                    break

    def run(self):
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            sample_info = self.sample_run.get("info") or get_raden_tiff_stack_info(self.sample_run["folder_path"])
            open_beam_info = self.open_beam_run.get("info") or get_raden_tiff_stack_info(self.open_beam_run["folder_path"])
            self._validate(sample_info, open_beam_info)

            sample_pulses = self._pulse_count(sample_info)
            open_beam_pulses = self._pulse_count(open_beam_info)
            if sample_pulses and open_beam_pulses:
                scale = np.float32(open_beam_pulses / sample_pulses)
                self.message.emit(
                    f"RADEN pulse scale: sample={sample_pulses:.0f}, open-beam={open_beam_pulses:.0f}, scale={scale:.4f}"
                )
            else:
                scale = np.float32(1.0)
                self.message.emit("RADEN pulse metadata unavailable; using scale=1.0")

            output_stem = self.base_name.strip() or "normalised"
            sample_stem = os.path.splitext(os.path.basename(sample_info["file_path"]))[0]
            if output_stem.lower().endswith((".tif", ".tiff")):
                output_stem = os.path.splitext(output_stem)[0]
            else:
                output_stem = f"{output_stem}_{sample_stem}"
            output_tiff = os.path.join(self.output_folder, output_stem + ".tiff")

            total = int(sample_info["n_frames"])
            processed_frames = 0
            self.message.emit(
                f"<b>--- Starting RADEN stack normalisation ---</b> "
                f"{total} frames, output='{os.path.basename(output_tiff)}'"
            )

            with Image.open(sample_info["file_path"]) as sample_tiff, Image.open(open_beam_info["file_path"]) as open_beam_tiff:
                with TiffImagePlugin.AppendingTiffWriter(output_tiff, new=True) as writer:
                    for index in range(total):
                        if not self._is_running:
                            self.message.emit("User stopped RADEN normalisation.")
                            break

                        sample_frame = self._read_frame(sample_tiff, index)
                        start = max(0, index - self.adjacent_sum)
                        end = min(total - 1, index + self.adjacent_sum)
                        open_beam_sum = None
                        for open_beam_index in range(start, end + 1):
                            open_beam_frame = self._read_frame(open_beam_tiff, open_beam_index)
                            if open_beam_sum is None:
                                open_beam_sum = open_beam_frame.astype(np.float64)
                            else:
                                open_beam_sum += open_beam_frame

                        normalised = self._normalise_frame(sample_frame, open_beam_sum, end - start + 1, scale)
                        Image.fromarray(normalised).save(writer, format="TIFF")
                        processed_frames += 1
                        if index != total - 1:
                            writer.newFrame()

                        self.progress_updated.emit(int(100 * (index + 1) / total))
                        if index % 100 == 0:
                            gc.collect()

            if self._is_running:
                self._copy_sidecars(sample_info, output_tiff)
            self.succeeded = self._is_running and processed_frames == total
            if self.succeeded:
                self.message.emit(f"RADEN stack normalisation complete: {output_tiff}")
            else:
                self.message.emit(
                    f"RADEN normalisation incomplete: {processed_frames} of "
                    f"{total} frames written."
                )

        except Exception as exc:
            self.succeeded = False
            self.message.emit(f"Fatal error in RADEN normalisation: {exc}")
        finally:
            gc.collect()
            self.finished.emit()

    def stop(self):
        self._is_running = False
        self.message.emit("Stop signal received. Terminating RADEN normalisation process.")


__all__ = [
    "OutlierFilteringWorker",
    "SummationWorker",
    "OverlapCorrectionWorker",
    "NormalisationWorker",
    "RadenNormalisationWorker",
    "FullProcessWorker",
    "FilteringWorker",
    "validate_normalisation_windows",
]

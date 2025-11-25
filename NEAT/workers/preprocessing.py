"""Worker threads for preprocessing and filtering tasks."""

import gc
import os
import re
import shutil

import numpy as np
import pandas as pd
import psutil
from astropy.io import fits
from PyQt5.QtCore import Qt, QEventLoop, QThread, pyqtSignal

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
        self.report_path = os.path.join(output_folder, f"{base_name}_outlier_report.csv")

    # ────────────────────────────────────────────────────────────────────────────
    # Main thread entry
    # ────────────────────────────────────────────────────────────────────────────
    def run(self):
        try:
            self._prepare_output()

            total_images   = sum(len(run["images"]) for run in self.image_runs)
            processed      = 0
            total_cleaned  = 0

            for run_idx, data_run in enumerate(self.image_runs, start=1):
                if not self._is_running:
                    self.message.emit("Process stopped by user")
                    break

                self._copy_related_files(run_idx, data_run)

                for suffix, img_data in data_run["images"].items():
                    if not self._is_running:
                        break

                    try:
                        cleaned_pixels = self._clean_one_frame(suffix, img_data)
                        total_cleaned += cleaned_pixels
                        processed     += 1
                        self._update_progress(processed, total_images)

                    except Exception as exc:
                        self.message.emit(f"[ERROR] Frame {suffix}: {exc}")

            self.message.emit(
                f"[SUCCESS] Processed {processed} frame(s) • "
                f"Total cleaned pixels: {total_cleaned} • "
                f"Report: {os.path.basename(self.report_path)}"
            )
        except Exception as exc:
            self.message.emit(f"[FATAL] {exc}")
        finally:
            self.finished.emit()

    # ────────────────────────────────────────────────────────────────────────────
    # Helper methods
    # ────────────────────────────────────────────────────────────────────────────
    def _prepare_output(self):
        """Ensure output folder exists and create a fresh report file."""
        os.makedirs(self.output_folder, exist_ok=True)
        try:
            with open(self.report_path, "w", encoding="utf-8") as fh:
                fh.write("frame_idx,pixel_x,pixel_y,outlier_value,replace_value\n")
        except OSError as exc:
            raise RuntimeError(f"Cannot create report file: {exc}") from exc

    def _clean_one_frame(self, suffix: str, img_data: np.ndarray) -> int:
        """Return number of pixels cleaned for this frame."""
        img = img_data.astype(np.float32, copy=True)
        invalid_mask = (img <= 0) | np.isnan(img) | np.isinf(img)
        bad_pixels   = np.argwhere(invalid_mask)

        if bad_pixels.size == 0:
            self.message.emit(f"Frame {suffix}: no outliers detected")
            return 0

        records = []
        cleaned = 0

        for y, x in bad_pixels:
            original = img[y, x]

            y0, y1 = max(0, y - 2), min(img.shape[0], y + 3)
            x0, x1 = max(0, x - 2), min(img.shape[1], x + 3)

            nbhood      = img[y0:y1, x0:x1]
            valid_mask  = (nbhood > 0) & np.isfinite(nbhood)
            valid_values = nbhood[valid_mask]

            if valid_values.size:          # We have something to average
                replacement = float(np.mean(valid_values))
                img[y, x]   = replacement
            else:
                replacement = np.nan       # Do not inject a hard zero
                # Leave img[y, x] unchanged to flag it downstream

            records.append(
                f"{suffix},{x},{y},{original:.4f},{replacement:.4f}"
            )
            cleaned += 1

        # Append to CSV
        try:
            with open(self.report_path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(records) + "\n")
        except OSError as exc:
            self.message.emit(f"[WARNING] Could not append to report: {exc}")

        # Write FITS
        out_fits = os.path.join(self.output_folder, f"{self.base_name}_{suffix}.fits")
        try:
            fits.writeto(out_fits, img, overwrite=True)
        except Exception as exc:  # astropy throws its own subclass of OSError
            raise RuntimeError(f"Cannot write FITS {out_fits}: {exc}") from exc

        return cleaned

    def _update_progress(self, processed: int, total: int) -> None:
        progress = int((processed / total) * 100) if total else 0
        self.progress_updated.emit(progress)

    def _copy_related_files(self, run_idx: int, data_run: dict) -> None:
        """Copy *_Spectra.txt and *_ShutterCount.txt (if present)"""
        spectra_suffix      = "_Spectra.txt"
        shuttercount_suffix = "_ShutterCount.txt"

        def _copy_if_exists(suffix: str):
            for f in os.listdir(data_run["folder_path"]):
                if f.endswith(suffix):
                    src = os.path.join(data_run["folder_path"], f)
                    dst = os.path.join(self.output_folder, f"Run{run_idx}_{f}")
                    try:
                        shutil.copy2(src, dst)
                        self.message.emit(f"Copied {f} → {os.path.basename(dst)}")
                    except OSError as exc:
                        self.message.emit(f"[WARNING] Could not copy {f}: {exc}")
                    break          # only the first match

        _copy_if_exists(spectra_suffix)
        _copy_if_exists(shuttercount_suffix)

    # --------------------------------------------------------------------
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
            self._prepare_output()
            self._sum_images()
            if self._is_running:
                self._sum_and_save_shuttercounts()
            if self._is_running:
                self._copy_spectra_files()
        except Exception as exc:
            self.message.emit(f"[FATAL] Summation aborted: {exc}")
        finally:
            gc.collect()
            self.finished.emit()

    # ───────────────────────────────────────────────────────────────
    # Phase 0  –  Ensure output folder & log shortcut
    # ───────────────────────────────────────────────────────────────
    def _prepare_output(self):
        try:
            os.makedirs(self.output_folder, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Cannot create output folder '{self.output_folder}': {exc}") from exc

        parts = os.path.normpath(self.output_folder).split(os.sep)
        self._short_path = os.path.join(*parts[-2:]) if len(parts) >= 2 else self.output_folder

    # ───────────────────────────────────────────────────────────────
    # Phase 1  –  Image summation
    # ───────────────────────────────────────────────────────────────
    def _sum_images(self):
        if not self.summation_image_runs:
            self.message.emit("No summation runs supplied – nothing to do.")
            return
        
        first_keys  = set(self.summation_image_runs[0]["images"].keys())
        first_count = len(first_keys)
        bad_runs    = []
    
        for idx, run in enumerate(self.summation_image_runs[1:], start=2):
            keys = set(run["images"].keys())
            if len(keys) != first_count:
                bad_runs.append(
                    f"run {idx} has {len(keys)} image(s) vs {first_count}"
                )
            elif keys != first_keys:
                extra   = keys - first_keys
                missing = first_keys - keys
                detail  = []
                if extra:   detail.append(f"extra: {sorted(extra)[:3]}…")
                if missing: detail.append(f"missing: {sorted(missing)[:3]}…")
                bad_runs.append(f"run {idx} suffix mismatch ({'; '.join(detail)})")
    
        if bad_runs:
            self.message.emit(
                "❌  Image‑count / suffix mismatch detected – "
                "aborting summation:\n    " + "\n    ".join(bad_runs)
            )
            return  # hard abort – do not touch FITS files
        
        suffixes = sorted(first_keys)
        total    = len(suffixes)

        # Find common suffixes
        common_suffixes = set(self.summation_image_runs[0]["images"])
        for run in self.summation_image_runs[1:]:
            common_suffixes.intersection_update(run["images"].keys())

        if not common_suffixes:
            self.message.emit("No common suffixes across runs – skipping image summation.")
            return

        suffixes = sorted(common_suffixes)
        total    = len(suffixes)

        self.message.emit("--- Starting image summation ---")
        self.progress_updated.emit(0)

        for idx, suffix in enumerate(suffixes, start=1):
            if not self._is_running:
                self.message.emit("Image summation cancelled by user.")
                break

            summed   = None
            for run in self.summation_image_runs:
                if not self._is_running:
                    break
                try:
                    img = run["images"][suffix]
                    img = img.astype(np.float32, copy=False)

                    if summed is None:
                        summed = img.copy()
                    else:
                        if summed.shape != img.shape:
                            self.message.emit(
                                f"[WARNING] Shape mismatch for suffix '{suffix}' in "
                                f"run '{run['folder_path']}'. Skipping this suffix."
                            )
                            summed = None
                            break
                        summed += img
                except KeyError:
                    self.message.emit(
                        f"[WARNING] Run '{run['folder_path']}' missing suffix '{suffix}' – skipped."
                    )
                    summed = None
                    break
                except Exception as exc:
                    self.message.emit(
                        f"[ERROR] Problem reading suffix '{suffix}' in run "
                        f"'{run['folder_path']}': {exc}"
                    )
                    summed = None
                    break

            # Write FITS
            if self._is_running and summed is not None:
                out_name = f"{self.base_name}_Summed_{suffix}.fits"
                out_path = os.path.join(self.output_folder, out_name)
                try:
                    fits.writeto(out_path, summed, overwrite=True)
                    # self.message.emit(f"Saved summed image '{out_name}'.")
                except Exception as exc:
                    self.message.emit(f"[ERROR] Could not save '{out_name}': {exc}")

            self.progress_updated.emit(int((idx / total) * 100))

    # ───────────────────────────────────────────────────────────────
    # Phase 2  –  ShutterCount summation
    # ───────────────────────────────────────────────────────────────
    def _sum_and_save_shuttercounts(self):
        if not self._is_running:
            return

        shutter_lists = []
        for run_idx, run in enumerate(self.summation_image_runs, start=1):
            if not self._is_running:
                break

            combined = None
            for folder in run.get("run_folders", [run["folder_path"]]):
                if not self._is_running:
                    break

                sc_file = self._find_file(folder, "_ShutterCount.txt")
                if sc_file:
                    try:
                        data = np.loadtxt(sc_file, dtype=np.float32)
                        if data.ndim != 2 or data.shape[1] != 2:
                            self.message.emit(
                                f"[WARNING] Run {run_idx}: Bad ShutterCount format in '{sc_file}'."
                            )
                            continue
                        counts = data[:, 1]
                    except Exception as exc:
                        self.message.emit(
                            f"[WARNING] Run {run_idx}: Cannot read '{sc_file}': {exc}"
                        )
                        counts = np.zeros(256, np.float32)
                else:
                    self.message.emit(
                        f"Run {run_idx}: No ShutterCount in '{folder}'. Using zeros."
                    )
                    counts = np.zeros(256, np.float32)

                combined = counts if combined is None else (
                    combined + counts if len(combined) == len(counts) else combined
                )

            shutter_lists.append(combined if combined is not None
                                 else np.zeros(256, np.float32))

        if not shutter_lists:
            self.message.emit("No ShutterCount data found in any run.")
            return

        base_len = len(shutter_lists[0])
        if any(len(arr) != base_len for arr in shutter_lists):
            self.message.emit("[WARNING] ShutterCount length mismatch – skipping save.")
            return

        summed   = np.sum(shutter_lists, axis=0)
        out_data = np.column_stack((np.arange(base_len), summed))
        out_name = f"{self.base_name}_summed_ShutterCount.txt"
        out_path = os.path.join(self.output_folder, out_name)

        if not self._is_running:
            return
        try:
            np.savetxt(out_path, out_data, fmt="%d\t%d")
            self.message.emit(f"Saved summed ShutterCount '{out_name}'.")
        except OSError as exc:
            self.message.emit(f"[ERROR] Could not write '{out_name}': {exc}")

    # ───────────────────────────────────────────────────────────────
    # Phase 3  –  Copy Spectra files
    # ───────────────────────────────────────────────────────────────
    def _copy_spectra_files(self):
        if not self._is_running:
            return

        for run_idx, run in enumerate(self.summation_image_runs, start=1):
            if not self._is_running:
                break

            copied = False
            for folder in run.get("run_folders", [run["folder_path"]]):
                spectra = self._find_file(folder, "_Spectra.txt")
                if spectra:
                    dest_name = f"{self.base_name}_{run_idx}_Spectra.txt"
                    dest_path = os.path.join(self.output_folder, dest_name)
                    try:
                        shutil.copyfile(spectra, dest_path)
                        self.message.emit(f"Copied Spectra → '{dest_name}'.")
                        copied = True
                    except OSError as exc:
                        self.message.emit(
                            f"[WARNING] Could not copy Spectra from '{folder}': {exc}"
                        )
                    break           # only first spectra per run
            if not copied:
                self.message.emit(f"Run {run_idx}: No Spectra file found.")

        self.message.emit(
            f"Summation complete – files written to <b>'\\{self._short_path}\\'</b> "
            f"with base name '{self.base_name}'."
        )
        self.progress_updated.emit(100)

    # ───────────────────────────────────────────────────────────────
    # Utility helpers
    # ───────────────────────────────────────────────────────────────
    @staticmethod
    def _find_file(folder: str, suffix: str) -> str | None:
        try:
            for f in os.listdir(folder):
                if f.endswith(suffix):
                    return os.path.join(folder, f)
        except OSError:
            return None
        return None

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
        self.run = run
        self.base_name = base_name
        self.output_folder = output_folder
        self._is_running = True

    def run(self):
        """
        Execute the Overlap Correction process.
        """
        try:
            folder_path = self.run['folder_path']
            images_dict = self.run['images']
            spectra_data = self.run['spectra']
            shutter_count_data = self.run['shutter_count']
            
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
                        continue

                    # Construct output path
                    try:
                        numeric_suffix = ''.join(filter(str.isdigit, suf))
                        original_filename = f"{self.base_name}_{numeric_suffix}.fits"
                        corrected_filename = f"Corrected_{original_filename}"
                        output_path = os.path.join(self.output_folder, corrected_filename)
                    except Exception as e:
                        self.message.emit(f"Error constructing filename: {e}. Skipping.")
                        continue

                    # Save corrected image
                    fits.writeto(output_path, corrected_intensity, overwrite=True)

                    # Update progress
                    overall_progress = int(((img_idx + 1) / total_imgs) * 100)
                    self.progress_updated.emit(overall_progress)

                except Exception as e:
                    self.message.emit(f"Error processing image '{suf}': {e}. Skipping.")
                    continue

            # Final messages
            if self._is_running:
                self.message.emit("Overlap Correction process completed successfully.")
            else:
                self.message.emit("Overlap Correction process was stopped before completion.")

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
            self.message.emit(f"Error in OverlapCorrectionWorker: {e}")

        # Optional: if memory usage is extremely high, you could call gc.collect() once here
        # gc.collect()

        # Emit finished signal
        self.message.emit("Overlap Correction completed successfully.")
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
        self.normalisation_image_runs      = normalisation_image_runs
        self.normalisation_open_beam_runs  = normalisation_open_beam_runs
        self.output_folder                 = output_folder
        self.base_name                     = base_name
        self.window_half                   = window_half
        self.adjacent_sum                  = adjacent_sum
        self._is_running                   = True

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
                            continue

                        h, w = img.shape
                        if h < 2*window_half+1 or w < 2*window_half+1:
                            self.message.emit(
                                f" {suffix}: too small for window—skipping."
                            )
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
                        fits.writeto(os.path.join(self.output_folder, out_fname),
                                     normed, overwrite=True)

                        del data_imgs[suffix]
                        processed_images += 1
                        self.progress_updated.emit(int(100*processed_images/total_images))

                    except Exception as e:
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

            self.message.emit(f"All done—{processed_images} images → {short_path}")

        except Exception as e:
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
        self.sample_folder = sample_folder
        self.open_beam_folder = open_beam_folder
        self.output_folder = output_folder
        self.base_name = base_name
        self.window_half = window_half
        self.adjacent_sum = adjacent_sum
        self._is_running = True  # Flag to track whether the user requested a stop
        
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

            self.message.emit("=== <b>Full Process Completed Successfully</b> ===")

        except Exception as exc:
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
        self.message.emit("FullProcessWorker: Stop signal received.")

    # ---------------------------------------------------------------
    # Summation Step (conditionally skipped if no subfolders)
    # ---------------------------------------------------------------
    def maybe_do_summation(self, folder: str, label: str) -> str:
        if not self._is_running:
            return folder
    
        # Check for subfolders
        subfolders = [
            os.path.join(folder, d)
            for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        ]
        
        short_path = self.get_short_path(folder, levels=2)
    
        if not subfolders:
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
            self.message.emit(f"0_sumation_{label}: No valid FITS images in subfolders of '\\{short_path}'. Summation skipped.")
            return folder
    
        # Create a modified base name for image naming (e.g., "summed_sample_data")
        modified_base_name = f"summed_{original_folder_name}"
        worker = SummationWorker(runs, modified_base_name, summation_output)
        # worker.progress_updated.connect(lambda v: self.progress_updated.emit(v // 4))
        worker.progress_updated.connect(self.progress_updated, Qt.QueuedConnection)
               
        worker.message.connect(self.message.emit, Qt.QueuedConnection)

        # block until child finishes, no busy‑wait
        loop = QEventLoop(); 
        worker.finished.connect(loop.quit); 
        worker.start(); 
        loop.exec_()     
    
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
            self.message.emit(f"1_clean_{label}: No images found in \\{short_path}. Skipping Clean.")
            return folder
    
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
        worker.start(); 
        loop.exec_()
    
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
            self.message.emit(f"2_correction_{label}: No images found in \\{short_path}. Skipping Overlap Correction.")
            return folder
    
        try:
            all_files = os.listdir(folder)
        except Exception as e:
            self.message.emit(f"Error accessing \\{short_path}: {e}")
            return folder
    
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
            self.message.emit(f"2_correction_{label}: Spectra or ShutterCount missing => Skipping Overlap Correction.")
            return folder
    
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
        worker.start(); 
        loop.exec_()
    
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
            self.message.emit("Normalisation skipped because sample or openbeam has no images.")
            return
    
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
        worker.start(); 
        loop.exec_()
    
        del sample_run, openbeam_run
        import gc
        gc.collect()
        
        normalised_output_short=self.get_short_path(normalised_output, levels=3)
    
        if self._is_running:
            self.message.emit(f"<b>3_normalisation complete</b>, saved at: <b>\\{normalised_output_short}</b>")

  
    def load_run_dict(self, folder: str) -> dict:
        """
        Scans the folder for FITS files with a 5-digit suffix and returns a dictionary:
            { 'folder_path': folder, 'images': {suffix: data} }.
        Also emits loading progress via load_progress_updated.
        """
        # import re
        run = {"folder_path": folder, "images": {}}
        if not os.path.isdir(folder):
            self.message.emit(f"Folder not found: {folder}")
            return run
        
        short_path = self.get_short_path(folder, levels=2)

        try:
            fits_files = [f for f in os.listdir(folder) if f.lower().endswith((".fits", ".fit"))]
            # Filter only files with the proper 5-digit suffix format:
            pattern = re.compile(r'^.+_(\d{5})\.(fits|fit)$', re.IGNORECASE)
            total_files = len(fits_files)
            processed_files = 0

            for f in fits_files:
                processed_files += 1
                match = pattern.match(f)
                if not match:
                    # self.message.emit(f"Skipping file '{f}' because it does not have a 5-digit suffix.")
                    # Emit progress update even if skipped
                    self.load_progress_updated.emit(int((processed_files / total_files) * 100))
                    continue

                # Use the captured 5-digit group as the suffix
                suffix = match.group(1)
                try:
                    path = os.path.join(folder, f)
                    data = fits.getdata(path)
                    run["images"][suffix] = data.astype(np.float32)
                except Exception as e:
                    self.message.emit(f"Error loading file {f} in \\{short_path}: {e}")
                # Update loading progress after processing each file
                self.load_progress_updated.emit(int((processed_files / total_files) * 100))
        except Exception as e:
            self.message.emit(f"Error reading folder \\{short_path}: {e}")
        return run

class FilteringWorker(QThread):
    progress_updated = pyqtSignal(int)  # Emits progress percentage
    finished = pyqtSignal()             # Emits when processing is finished
    message = pyqtSignal(str)           # Emits messages for user feedback

    def __init__(self, filtering_image_runs, filtering_mask, output_folder, base_name):
        super().__init__()
        self.filtering_image_runs = filtering_image_runs  # List of data run dictionaries
        self.filtering_mask = filtering_mask              # Single mask image array
        self.output_folder = output_folder
        self.base_name = base_name
        self._is_running = True

    def run(self):
        """
        Main filtering process.
        """
        try:
            # 1) Basic validations
            if not self.filtering_image_runs:
                self.message.emit("No filtering data runs to process.")
                self.finished.emit()
                return

            if self.filtering_mask is None:
                self.message.emit("No mask image provided.")
                self.finished.emit()
                return

            # 2) Convert the mask to float32 or bool (if not already), only once
            if self.filtering_mask.dtype != np.float32 and self.filtering_mask.dtype != bool:
                # If it's an integer or float32, consider converting
                # Example: let's keep it float32 for consistency:
                self.filtering_mask = self.filtering_mask.astype(np.float32)

            mask_shape = self.filtering_mask.shape

            # 3) Counters for progress
            total_runs = len(self.filtering_image_runs)
            total_images = sum(len(run['images']) for run in self.filtering_image_runs)
            processed_images = 0

            self.message.emit("Filtering started...")

            # 4) Process each run
            for run_idx, data_run in enumerate(self.filtering_image_runs, start=1):
                if not self._is_running:
                    self.message.emit("Filtering process has been stopped by the user.")
                    break

                data_folder = data_run.get('folder_path', None)
                data_images = data_run.get('images', {})

                # Copy .txt files for this run (if desired)
                self.copy_related_files(run_idx, data_run)

                # Sort image keys for a consistent order
                suffixes = sorted(data_images.keys())

                # 5) Process each image in the run
                for suffix in suffixes:
                    if not self._is_running:
                        self.message.emit("Filtering process has been stopped by the user.")
                        break

                    try:
                        image_data = data_images[suffix]
                        # Convert to float32 if needed
                        if image_data.dtype != np.float32:
                            image_data = image_data.astype(np.float32)

                        if image_data.shape != mask_shape:
                            self.message.emit(
                                f"Image {suffix}: Mask shape {mask_shape} "
                                f"does not match image shape {image_data.shape}. Skipping."
                            )
                            continue

                        # Apply the mask
                        # If your mask is float32, we can do: (mask != 0)
                        # Or if you kept it as bool, simply: np.where(mask, image_data, 0)
                        # Example assuming mask != 0 means “keep pixel”:
                        filtered_image = np.where(self.filtering_mask != 0, image_data, 0)

                        # Save the filtered image
                        filtered_filename = f"{self.base_name}_{suffix}.fits"
                        filtered_path = os.path.join(self.output_folder, filtered_filename)
                        try:
                            fits.writeto(filtered_path, filtered_image, overwrite=True)
                        except Exception as e:
                            self.message.emit(f"Image {suffix}: Failed to save '{filtered_filename}': {e}. Skipping.")
                            continue

                        # Update progress
                        processed_images += 1
                        overall_progress = int((processed_images / total_images) * 100)
                        self.progress_updated.emit(overall_progress)

                    except Exception as e:
                        self.message.emit(f"Error filtering image {suffix}: {e}")
                        continue

                # Optional: update overall run progress
                run_progress = int((run_idx / total_runs) * 100)
                self.progress_updated.emit(run_progress)
                
            self.output_folder_short = self.get_short_path(self.output_folder, levels=2)

            self.message.emit(
                f"Filtering completed. {processed_images} images filtered and saved to {self.output_folder_short}."
            )

        except Exception as e:
            self.message.emit(f"Error during filtering: {e}")

        finally:
            # Call gc.collect() once at the end if you need to enforce cleanup
            gc.collect()
            self.finished.emit()

    def stop(self):
        """
        Stop the Filtering process.
        """
        self._is_running = False
        self.message.emit("Stop signal received. Terminating Filtering process.")

    def copy_related_files(self, run_idx, data_run):
        """
        Copies related files (e.g., *_Spectra.txt and *_ShutterCount.txt) from the data run's folder
        to the output folder, renaming them with the run_idx to avoid collisions.
        """
        try:
            folder_path = data_run.get('folder_path', None)
            if not folder_path or not os.path.isdir(folder_path):
                self.message.emit(f"Data run folder not found or invalid: {folder_path}")
                return

            data_files = os.listdir(folder_path)
            spectra_suffix = '_Spectra.txt'
            shuttercount_suffix = '_ShutterCount.txt'

            def create_unique_filename(run_number, original_filename):
                return f"Run{run_number}_{original_filename}"

            # Copy one spectra file if it exists
            spectra_files = [f for f in data_files if f.endswith(spectra_suffix)]
            if spectra_files:
                file = spectra_files[0]
                src = os.path.join(folder_path, file)
                dest_filename = create_unique_filename(run_idx, file)
                dst = os.path.join(self.output_folder, dest_filename)
                shutil.copyfile(src, dst)
                self.message.emit(f"Copied '{file}' to '{dest_filename}'.")
            else:
                self.message.emit(f"No spectra file (*{spectra_suffix}) found in {folder_path}.")

            # Copy one shuttercount file if it exists
            shuttercount_files = [f for f in data_files if f.endswith(shuttercount_suffix)]
            if shuttercount_files:
                file = shuttercount_files[0]
                src = os.path.join(folder_path, file)
                dest_filename = create_unique_filename(run_idx, file)
                dst = os.path.join(self.output_folder, dest_filename)
                shutil.copyfile(src, dst)
                self.message.emit(f"Copied '{file}' to '{dest_filename}'.")
            else:
                self.message.emit(f"No shuttercount file (*{shuttercount_suffix}) found in {folder_path}.")

        except Exception as e:
            self.message.emit(f"Error copying related files: {e}")

__all__ = [
    "OutlierFilteringWorker",
    "SummationWorker",
    "OverlapCorrectionWorker",
    "NormalisationWorker",
    "FullProcessWorker",
    "FilteringWorker",
]

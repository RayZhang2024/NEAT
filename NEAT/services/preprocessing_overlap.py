"""Headless, legacy-compatible overlap correction for one loaded image run."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable

import numpy as np

from ..domain import (
    LoadedImageRun,
    PreprocessingOperationResult,
    PreprocessingStatus,
    ProducedOutput,
)
from .image_io import write_fits_image_file

ProgressCallback = Callable[[int], None]
MessageCallback = Callable[[str], None]
CancellationCheck = Callable[[], bool]
FrameFailureCallback = Callable[[str], None]
FatalErrorCallback = Callable[[Exception], None]


def segment_mean_intervals(tof_values: np.ndarray, segments: list[np.ndarray]) -> list[float]:
    """Calculate legacy mean intervals, including empty and singleton segments."""
    intervals = []
    for seg in segments:
        if len(seg) <= 1:
            mean_interval = 0.0 if len(seg) < 1 else 1e-5
        else:
            seg_tofs = tof_values[seg]
            seg_diffs = np.diff(seg_tofs)
            mean_interval = float(np.mean(seg_diffs))
        intervals.append(mean_interval)
    return intervals


def correct_overlap_frame(
    image_data: np.ndarray,
    cumulative_intensity: np.ndarray,
    shutter_count: np.float32,
    this_interval: float,
    reference_interval: float,
) -> np.ndarray:
    """Apply the original double-clamped overlap equation without changing inputs."""
    p = cumulative_intensity / shutter_count
    denom = np.where(1.0 - p <= 0, np.float32(1e-10), 1.0 - p)
    epsilon = 1e-10
    denom = np.where(denom <= 0, epsilon, denom)
    corrected_intensity = image_data / denom
    this_interval_32 = np.float32(this_interval)
    reference_interval_32 = np.float32(reference_interval)
    scale_factor = (
        reference_interval_32 / this_interval_32
        if this_interval_32 > 0
        else np.float32(1.0)
    )
    corrected_intensity *= scale_factor
    return corrected_intensity


def _numeric_key(suffix: str) -> int:
    digits = "".join(filter(str.isdigit, suffix))
    return int(digits) if digits else -1


def correct_loaded_image_run(
    run: LoadedImageRun,
    spectra_data: np.ndarray,
    shutter_count_data: np.ndarray,
    base_name: str,
    output_folder: str,
    *,
    progress_callback: ProgressCallback | None = None,
    message_callback: MessageCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
    frame_failure_callback: FrameFailureCallback | None = None,
    fatal_error_callback: FatalErrorCallback | None = None,
) -> PreprocessingOperationResult:
    """Correct frames sequentially, retaining legacy messages and partial outputs."""
    expected_count = len(run.frames)
    processed = 0
    outputs: list[ProducedOutput] = []
    errors: list[str] = []
    warnings: list[str] = []
    failed_frames: list[str] = []

    def emit(message: str) -> None:
        if message_callback is not None:
            message_callback(message)

    def cancelled() -> bool:
        return cancellation_check is not None and cancellation_check()

    def result(status: PreprocessingStatus) -> PreprocessingOperationResult:
        return PreprocessingOperationResult(
            status=status,
            processed_count=processed,
            outputs=tuple(outputs),
            expected_count=expected_count,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def abort(message: str) -> PreprocessingOperationResult:
        errors.append(message)
        emit(message)
        return result(PreprocessingStatus.FAILED)

    def fail_frame(suffix: str, message: str) -> None:
        failed_frames.append(str(suffix))
        errors.append(message)
        emit(message)
        if frame_failure_callback is not None:
            frame_failure_callback(str(suffix))

    try:
        folder_path = run.primary_source
        images_dict = run.frames
        normalized_path = os.path.normpath(folder_path)
        path_parts = normalized_path.split(os.sep)
        short_path = (
            os.path.join(path_parts[-2], path_parts[-1])
            if len(path_parts) >= 2
            else normalized_path
        )

        try:
            if spectra_data.dtype != np.float32:
                spectra_data = spectra_data.astype(np.float32)
            tof_values = spectra_data[:, 0]
            emit("Extracted ToF values from Spectra data.")
        except Exception as exc:
            return abort(f"Error extracting ToF values: {exc}. Aborting run.")

        try:
            tof_intervals = np.diff(tof_values)
            segmentation_indices = np.where(tof_intervals > 0.0001)[0] + 1
            segments = np.split(np.arange(len(tof_values)), segmentation_indices)
            n_segments = len(segments)
            emit(f"Identified {n_segments} segments based on ToF intervals.")
        except Exception as exc:
            return abort(f"Error during ToF segmentation: {exc}. Aborting run.")

        try:
            if len(shutter_count_data) < n_segments:
                return abort(
                    f"Insufficient shutter counts for {n_segments} segments. Aborting run."
                )
            filtered_shutter_counts = shutter_count_data[shutter_count_data > 1000]
            if len(filtered_shutter_counts) < n_segments:
                return abort(
                    f"Only {len(filtered_shutter_counts)} shutter counts > 1000 found, "
                    f"but {n_segments} segments identified. Aborting run."
                )
            selected_shutter_counts = filtered_shutter_counts[:n_segments]
            emit(f"Selected {n_segments} shutter counts for {n_segments} segments.")
        except Exception as exc:
            return abort(f"Error processing shutter counts: {exc}. Aborting run.")

        segment_intervals = segment_mean_intervals(tof_values, segments)
        ref_interval = segment_intervals[0]
        if ref_interval == 0:
            return abort("First segment's interval is 0. Cannot normalize to T1=0.")
        emit(f"Segment intervals: {segment_intervals}")
        emit(f"Reference interval (segment 1) = {ref_interval:.7f}")

        try:
            first_img_key = next(iter(images_dict.keys()))
            first_img_data = images_dict[first_img_key]
            if first_img_data.dtype != np.float32:
                first_img_data = first_img_data.astype(np.float32)
            shape_512 = first_img_data.shape
            if shape_512 != (512, 512):
                return abort(
                    f"Image dimensions {shape_512} do not match expected "
                    "(512, 512). Aborting run."
                )
            cumulative_intensities = [
                np.zeros(shape_512, dtype=np.float32) for _ in segments
            ]
        except StopIteration:
            return abort("No images found in the run. Aborting.")
        except Exception as exc:
            return abort(f"Error initializing cumulative arrays: {exc}. Aborting.")

        try:
            sorted_suffixes = sorted(images_dict.keys(), key=_numeric_key)
            sorted_images = [images_dict[suf] for suf in sorted_suffixes]
        except Exception as exc:
            return abort(f"Error sorting images: {exc}. Aborting run.")

        emit("--- Starting Overlap Correction ---")
        total_imgs = len(sorted_suffixes)
        for img_idx, suf in enumerate(sorted_suffixes):
            if cancelled():
                emit("Overlap Correction process has been stopped by the user.")
                break
            try:
                image_data = sorted_images[img_idx]
                if image_data.dtype != np.float32:
                    image_data = image_data.astype(np.float32)

                segment_number = None
                for seg_num, seg_indices in enumerate(segments):
                    if img_idx in seg_indices:
                        segment_number = seg_num
                        break
                if segment_number is None:
                    fail_frame(suf, f"Image {img_idx+1}: No matching segment. Skipping.")
                    continue

                seg_idx_within = np.where(segments[segment_number] == img_idx)[0][0]
                if seg_idx_within == 0:
                    cumulative_intensities[segment_number] = image_data.copy()
                else:
                    cumulative_intensities[segment_number] += image_data

                shutter_count = np.float32(selected_shutter_counts[segment_number])
                if shutter_count == 0:
                    fail_frame(
                        suf,
                        f"Image {img_idx+1}: Shutter count = 0 for segment "
                        f"{segment_number+1}. Skipping normalisation.",
                    )
                    continue
                corrected_intensity = correct_overlap_frame(
                    image_data,
                    cumulative_intensities[segment_number],
                    shutter_count,
                    segment_intervals[segment_number],
                    segment_intervals[0],
                )
                if np.isnan(corrected_intensity).any() or np.isinf(corrected_intensity).any():
                    fail_frame(
                        suf, f"Image {img_idx+1}: NaN or Inf after correction. Skipping."
                    )
                    continue

                try:
                    numeric_suffix = "".join(filter(str.isdigit, suf))
                    original_filename = f"{base_name}_{numeric_suffix}.fits"
                    corrected_filename = f"Corrected_{original_filename}"
                    output_path = os.path.join(output_folder, corrected_filename)
                except Exception as exc:
                    fail_frame(suf, f"Error constructing filename: {exc}. Skipping.")
                    continue

                write_fits_image_file(output_path, corrected_intensity, overwrite=True)
                outputs.append(ProducedOutput(output_path, "corrected_image"))
                processed += 1
                if progress_callback is not None:
                    progress_callback(int(((img_idx + 1) / total_imgs) * 100))
            except Exception as exc:
                fail_frame(suf, f"Error processing image '{suf}': {exc}. Skipping.")
                continue

        # Snapshot completion before any sidecar copy can trigger cancellation.
        was_cancelled = cancelled()
        succeeded = not was_cancelled and not failed_frames and processed == total_imgs
        if succeeded:
            emit("Overlap Correction process completed successfully.")
        else:
            emit("Overlap Correction failed or stopped before all frames were written.")
            if not was_cancelled and not errors:
                errors.append("Overlap Correction failed or stopped before all frames were written.")
        status = (
            PreprocessingStatus.CANCELLED
            if was_cancelled
            else PreprocessingStatus.SUCCEEDED
            if succeeded
            else PreprocessingStatus.FAILED
        )

        try:
            spectra_suffix = "_Spectra.txt"
            shuttercount_suffix = "_ShutterCount.txt"
            spectra_files = [f for f in os.listdir(folder_path) if f.endswith(spectra_suffix)]
            if not spectra_files:
                emit(f"No files ending with '{spectra_suffix}' found in \\{short_path}.")
            else:
                for spectra_file in spectra_files:
                    source_path = os.path.join(folder_path, spectra_file)
                    dest_path = os.path.join(output_folder, spectra_file)
                    shutil.copyfile(source_path, dest_path)
                    outputs.append(ProducedOutput(dest_path, "related_file_copy"))
                    emit(f"'{spectra_file}' copied to output folder.")

            shuttercount_files = [
                f for f in os.listdir(folder_path) if f.endswith(shuttercount_suffix)
            ]
            if not shuttercount_files:
                emit(f"No files ending with '{shuttercount_suffix}' found in \\{short_path}.")
            else:
                for shuttercount_file in shuttercount_files:
                    source_path = os.path.join(folder_path, shuttercount_file)
                    dest_path = os.path.join(output_folder, shuttercount_file)
                    shutil.copyfile(source_path, dest_path)
                    outputs.append(ProducedOutput(dest_path, "related_file_copy"))
                    emit(f"'{shuttercount_file}' copied to output folder.")
        except Exception as exc:
            warning = f"Error copying spectra or shuttercount files: {exc}"
            warnings.append(warning)
            emit(warning)

        if not succeeded:
            emit("Overlap Correction did not complete successfully.")
        return result(status)
    except Exception as exc:
        errors.append(str(exc))
        if fatal_error_callback is not None:
            fatal_error_callback(exc)
        emit("Overlap Correction did not complete successfully.")
        return result(PreprocessingStatus.FAILED)

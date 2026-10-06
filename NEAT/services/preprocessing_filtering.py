"""Headless binary-mask Filtering for loaded classic image runs."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Sequence

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
ValidatedMaskCallback = Callable[[np.ndarray], None]
SummaryPathCallback = Callable[[], str]


def validate_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Apply the legacy finite/binary checks, then convert to float32."""
    mask_values = np.asarray(mask)
    if not np.isfinite(mask_values).all():
        raise ValueError("Filtering mask contains NaN or infinite values.")
    unique_values = np.unique(mask_values)
    if not np.isin(unique_values, (0, 1)).all():
        raise ValueError("Filtering mask must be binary and contain only 0 and 1.")
    return mask_values.astype(np.float32, copy=False)


def apply_binary_mask(image_data: np.ndarray, validated_mask: np.ndarray) -> np.ndarray:
    """Keep mask-one pixels and zero mask-zero pixels without input mutation."""
    if image_data.dtype != np.float32:
        image_data = image_data.astype(np.float32)
    return np.where(validated_mask == 1, image_data, 0)


def short_filtering_path(full_path: str, levels: int = 2) -> str:
    """Use the final path components in the legacy completion message."""
    normalized_path = os.path.normpath(full_path)
    path_parts = normalized_path.split(os.sep)
    if len(path_parts) >= levels:
        return os.path.join(*path_parts[-levels:])
    return normalized_path


def copy_filtering_related_files(
    run_idx: int,
    primary_source: str | None,
    output_folder: str,
    *,
    message_callback: MessageCallback | None = None,
) -> tuple[tuple[ProducedOutput, ...], tuple[str, ...]]:
    """Copy first raw-order sidecars with one shared legacy error boundary."""
    outputs: list[ProducedOutput] = []
    warnings: list[str] = []

    def emit(message: str) -> None:
        if message_callback is not None:
            message_callback(message)

    try:
        if not primary_source or not os.path.isdir(primary_source):
            emit(f"Data run folder not found or invalid: {primary_source}")
            return (), ()

        data_files = os.listdir(primary_source)
        spectra_suffix = "_Spectra.txt"
        shuttercount_suffix = "_ShutterCount.txt"

        spectra_files = [f for f in data_files if f.endswith(spectra_suffix)]
        if spectra_files:
            filename = spectra_files[0]
            src = os.path.join(primary_source, filename)
            dst_name = f"Run{run_idx}_{filename}"
            dst = os.path.join(output_folder, dst_name)
            shutil.copyfile(src, dst)
            outputs.append(ProducedOutput(dst, "related_file_copy"))
            emit(f"Copied '{filename}' to '{dst_name}'.")
        else:
            emit(f"No spectra file (*{spectra_suffix}) found in {primary_source}.")

        shuttercount_files = [f for f in data_files if f.endswith(shuttercount_suffix)]
        if shuttercount_files:
            filename = shuttercount_files[0]
            src = os.path.join(primary_source, filename)
            dst_name = f"Run{run_idx}_{filename}"
            dst = os.path.join(output_folder, dst_name)
            shutil.copyfile(src, dst)
            outputs.append(ProducedOutput(dst, "related_file_copy"))
            emit(f"Copied '{filename}' to '{dst_name}'.")
        else:
            emit(
                f"No shuttercount file (*{shuttercount_suffix}) found in {primary_source}."
            )
    except Exception as exc:  # noqa: BLE001 - one legacy sidecar error boundary
        warning = f"Error copying related files: {exc}"
        warnings.append(warning)
        emit(warning)

    return tuple(outputs), tuple(warnings)


def filter_loaded_image_runs(
    runs: Sequence[LoadedImageRun],
    mask: np.ndarray | None,
    output_folder: str,
    base_name: str,
    *,
    progress_callback: ProgressCallback | None = None,
    message_callback: MessageCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
    frame_failure_callback: FrameFailureCallback | None = None,
    validated_mask_callback: ValidatedMaskCallback | None = None,
    summary_path_callback: SummaryPathCallback | None = None,
) -> PreprocessingOperationResult:
    """Run Filtering without Qt, preserving partial outputs and dual progress."""
    expected_count = sum(len(run.frames) for run in runs)
    processed = 0
    outputs: list[ProducedOutput] = []
    errors: list[str] = []
    warnings: list[str] = []

    def emit(message: str) -> None:
        if message_callback is not None:
            message_callback(message)

    def emit_progress(value: int) -> None:
        if progress_callback is not None:
            progress_callback(value)

    def cancelled() -> bool:
        return cancellation_check is not None and cancellation_check()

    def fail_frame(suffix: str, message: str) -> None:
        errors.append(message)
        if frame_failure_callback is not None:
            frame_failure_callback(str(suffix))
        emit(message)

    def result(status: PreprocessingStatus) -> PreprocessingOperationResult:
        return PreprocessingOperationResult(
            status=status,
            processed_count=processed,
            outputs=tuple(outputs),
            expected_count=expected_count,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    if not runs:
        message = "No filtering data runs to process."
        emit(message)
        errors.append(message)
        return result(PreprocessingStatus.FAILED)
    if mask is None:
        message = "No mask image provided."
        emit(message)
        errors.append(message)
        return result(PreprocessingStatus.FAILED)

    try:
        validated_mask = validate_binary_mask(mask)
        if validated_mask_callback is not None:
            validated_mask_callback(validated_mask)
        mask_shape = validated_mask.shape
    except Exception as exc:  # noqa: BLE001 - legacy setup failure
        message = f"Error during filtering: {exc}"
        errors.append(message)
        emit(message)
        return result(PreprocessingStatus.FAILED)

    try:
        total_runs = len(runs)
        emit("Filtering started...")

        for run_idx, run in enumerate(runs, start=1):
            if cancelled():
                emit("Filtering process has been stopped by the user.")
                break

            sidecar_outputs, sidecar_warnings = copy_filtering_related_files(
                run_idx, run.primary_source, output_folder, message_callback=emit
            )
            outputs.extend(sidecar_outputs)
            warnings.extend(sidecar_warnings)

            suffixes = sorted(run.frames.keys())
            for suffix in suffixes:
                if cancelled():
                    emit("Filtering process has been stopped by the user.")
                    break

                try:
                    image_data = run.frames[suffix]
                    if image_data.dtype != np.float32:
                        image_data = image_data.astype(np.float32)

                    if image_data.shape != mask_shape:
                        fail_frame(
                            suffix,
                            f"Image {suffix}: Mask shape {mask_shape} "
                            f"does not match image shape {image_data.shape}. Skipping.",
                        )
                        continue

                    filtered_image = apply_binary_mask(image_data, validated_mask)
                    filtered_filename = f"{base_name}_{suffix}.fits"
                    filtered_path = os.path.join(output_folder, filtered_filename)
                    try:
                        write_fits_image_file(
                            filtered_path, filtered_image, overwrite=True
                        )
                    except Exception as exc:  # noqa: BLE001 - FITS library failures
                        fail_frame(
                            suffix,
                            f"Image {suffix}: Failed to save '{filtered_filename}': "
                            f"{exc}. Skipping.",
                        )
                        continue

                    outputs.append(ProducedOutput(filtered_path, "filtered_image"))
                    processed += 1
                    emit_progress(int((processed / expected_count) * 100))
                except Exception as exc:  # noqa: BLE001 - legacy frame continuation
                    fail_frame(suffix, f"Error filtering image {suffix}: {exc}")
                    continue

            emit_progress(int((run_idx / total_runs) * 100))

        short_path = (
            summary_path_callback()
            if summary_path_callback is not None
            else short_filtering_path(output_folder, levels=2)
        )
        status = (
            PreprocessingStatus.CANCELLED
            if cancelled()
            else PreprocessingStatus.FAILED
            if errors or processed != expected_count
            else PreprocessingStatus.SUCCEEDED
        )
        label = (
            "completed"
            if status is PreprocessingStatus.SUCCEEDED
            else "failed or incomplete"
        )
        emit(
            f"Filtering {label}. {processed} of {expected_count} "
            f"images saved to {short_path}."
        )
        return result(status)
    except Exception as exc:  # noqa: BLE001 - legacy outer service boundary
        message = f"Error during filtering: {exc}"
        errors.append(message)
        emit(message)
        return result(PreprocessingStatus.FAILED)

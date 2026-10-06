"""Headless classic-image summation and sidecar persistence."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class _ValidatedInputs:
    suffixes: tuple[str, ...]
    shutter_counts_by_run: tuple[np.ndarray, ...]


def sum_corresponding_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Sum frames in input order using the historical float32 accumulation."""
    if not frames:
        raise ValueError("At least one image frame is required.")

    total = frames[0].astype(np.float32, copy=False).copy()
    for frame in frames[1:]:
        total += frame.astype(np.float32, copy=False)
    return total


def sum_loaded_image_runs(
    runs: Sequence[LoadedImageRun],
    base_name: str,
    output_folder: str,
    *,
    progress_callback: ProgressCallback | None = None,
    message_callback: MessageCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
) -> PreprocessingOperationResult:
    """Sum loaded runs and persist images, shutter counts, and spectra.

    Validation is completed before the output folder is created. Once
    validation succeeds, existing outputs are retained on later failure or
    cancellation and ``expected_count`` is the number of frame suffixes.
    """
    outputs: list[ProducedOutput] = []
    warnings: list[str] = []
    processed_count = 0
    expected_count: int | None = None

    def emit_message(message: str) -> None:
        if message_callback is not None:
            message_callback(message)

    def emit_progress(value: int) -> None:
        if progress_callback is not None:
            progress_callback(value)

    def is_cancelled() -> bool:
        return cancellation_check is not None and cancellation_check()

    def result(
        status: PreprocessingStatus, *, errors: tuple[str, ...] = ()
    ) -> PreprocessingOperationResult:
        return PreprocessingOperationResult(
            status,
            processed_count,
            tuple(outputs),
            expected_count,
            errors=errors,
            warnings=tuple(warnings),
        )

    try:
        validated = _validate_inputs(runs)
        expected_count = len(validated.suffixes)
        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create output folder '{output_folder}': {exc}"
            ) from exc

        emit_message("--- Starting image summation ---")
        emit_progress(0)

        for idx, suffix in enumerate(validated.suffixes, start=1):
            if is_cancelled():
                emit_message("Image summation cancelled by user.")
                return result(PreprocessingStatus.CANCELLED)

            frame_list: list[np.ndarray] = []
            for run in runs:
                if is_cancelled():
                    break
                frame_list.append(run.frames[suffix])
            if is_cancelled():
                emit_message("Image summation cancelled by user.")
                emit_progress(int((idx / expected_count) * 100))
                return result(PreprocessingStatus.CANCELLED)

            summed = sum_corresponding_frames(frame_list)
            out_name = f"{base_name}_Summed_{suffix}.fits"
            out_path = os.path.join(output_folder, out_name)
            try:
                write_fits_image_file(out_path, summed, overwrite=True)
            except Exception as exc:
                raise RuntimeError(f"Could not save '{out_name}': {exc}") from exc
            outputs.append(ProducedOutput(out_path, "summed_image"))
            processed_count += 1
            emit_progress(int((idx / expected_count) * 100))

        if is_cancelled():
            return result(PreprocessingStatus.CANCELLED)

        shutter_counts = validated.shutter_counts_by_run
        if not shutter_counts:
            raise RuntimeError("Validated ShutterCount data are unavailable.")
        base_len = len(shutter_counts[0])
        summed_counts = np.sum(shutter_counts, axis=0)
        shutter_data = np.column_stack((np.arange(base_len), summed_counts))
        shutter_name = f"{base_name}_summed_ShutterCount.txt"
        shutter_path = os.path.join(output_folder, shutter_name)
        if not is_cancelled():
            try:
                np.savetxt(shutter_path, shutter_data, fmt="%d\t%d")
            except OSError as exc:
                raise RuntimeError(
                    f"Could not write '{shutter_name}': {exc}"
                ) from exc
            outputs.append(ProducedOutput(shutter_path, "summed_shutter_count"))
            emit_message(f"Saved summed ShutterCount '{shutter_name}'.")
        else:
            return result(PreprocessingStatus.CANCELLED)

        if is_cancelled():
            return result(PreprocessingStatus.CANCELLED)

        for run_idx, run in enumerate(runs, start=1):
            if is_cancelled():
                break
            copied = False
            for folder in run.source_folders:
                spectra_path = _find_file(folder, "_Spectra.txt")
                if spectra_path:
                    dest_name = f"{base_name}_{run_idx}_Spectra.txt"
                    dest_path = os.path.join(output_folder, dest_name)
                    try:
                        shutil.copyfile(spectra_path, dest_path)
                        outputs.append(ProducedOutput(dest_path, "spectrum_copy"))
                        emit_message(f"Copied Spectra → '{dest_name}'.")
                        copied = True
                    except OSError as exc:
                        warning = (
                            f"Could not copy Spectra from '{folder}': {exc}"
                        )
                        warnings.append(warning)
                        emit_message(f"[WARNING] {warning}")
                    break
            if not copied:
                warning = f"Run {run_idx}: No Spectra file found."
                warnings.append(warning)
                emit_message(warning)

        emit_message(
            f"Summation complete – files written to "
            f"<b>'\\{_short_path(output_folder)}\\'</b> "
            f"with base name '{base_name}'."
        )
        emit_progress(100)
        if is_cancelled():
            return result(PreprocessingStatus.CANCELLED)
        return result(PreprocessingStatus.SUCCEEDED)
    except Exception as exc:
        return result(PreprocessingStatus.FAILED, errors=(str(exc),))


def _validate_inputs(runs: Sequence[LoadedImageRun]) -> _ValidatedInputs:
    if not runs:
        raise ValueError("No summation runs were supplied.")
    if not all(isinstance(run, LoadedImageRun) for run in runs):
        raise TypeError("Summation inputs must be LoadedImageRun values.")

    first_keys = set(runs[0].frames)
    if not first_keys:
        raise ValueError("The first summation run contains no images.")
    reference_shapes = {
        suffix: np.asarray(image).shape
        for suffix, image in runs[0].frames.items()
    }

    for run_idx, run in enumerate(runs, start=1):
        if run.load_errors:
            raise ValueError(
                f"Run {run_idx} contains load errors: {run.load_errors[0]}"
            )
        keys = set(run.frames)
        if keys != first_keys:
            raise ValueError(
                f"Run {run_idx} image suffix mismatch; "
                f"missing={sorted(first_keys - keys)[:3]}, "
                f"extra={sorted(keys - first_keys)[:3]}."
            )
        for suffix, image in run.frames.items():
            shape = np.asarray(image).shape
            if len(shape) != 2:
                raise ValueError(f"Run {run_idx} frame {suffix} is not a 2D image.")
            if shape != reference_shapes[suffix]:
                raise ValueError(
                    f"Run {run_idx} frame {suffix} has shape {shape}; "
                    f"expected {reference_shapes[suffix]}."
                )

    shutter_counts_by_run: list[np.ndarray] = []
    for run_idx, run in enumerate(runs, start=1):
        combined: np.ndarray | None = None
        if not run.source_folders:
            raise ValueError(f"Run {run_idx} has no valid source folder.")
        for folder in run.source_folders:
            shutter_path = _find_file(folder, "_ShutterCount.txt")
            if not shutter_path:
                raise ValueError(
                    f"Run {run_idx} has no ShutterCount file in '{folder}'."
                )
            try:
                data = np.loadtxt(shutter_path, dtype=np.float32)
            except Exception as exc:
                raise ValueError(
                    f"Run {run_idx} cannot read '{shutter_path}': {exc}"
                ) from exc
            if data.ndim == 1 and data.size == 2:
                data = data.reshape(1, 2)
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError(
                    f"Run {run_idx} has malformed ShutterCount data in "
                    f"'{shutter_path}'; exactly two columns are required."
                )
            counts = data[:, 1]
            if combined is None:
                combined = counts.copy()
            elif len(combined) != len(counts):
                raise ValueError(f"Run {run_idx} has unequal ShutterCount lengths.")
            else:
                combined += counts
        assert combined is not None
        shutter_counts_by_run.append(combined)

    base_len = len(shutter_counts_by_run[0])
    if any(len(values) != base_len for values in shutter_counts_by_run):
        raise ValueError("ShutterCount lengths differ between summation runs.")

    return _ValidatedInputs(
        tuple(sorted(first_keys)),
        tuple(shutter_counts_by_run),
    )


def _find_file(folder: str, suffix: str) -> str | None:
    """Return the first unsorted directory entry matching a suffix."""
    try:
        for filename in os.listdir(folder):
            if filename.endswith(suffix):
                return os.path.join(folder, filename)
    except OSError:
        return None
    return None


def _short_path(output_folder: str) -> str:
    parts = os.path.normpath(output_folder).split(os.sep)
    return os.path.join(*parts[-2:]) if len(parts) >= 2 else output_folder

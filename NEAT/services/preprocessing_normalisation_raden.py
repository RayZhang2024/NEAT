"""Headless RADEN multi-page TIFF normalisation operation."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from PIL import Image, TiffImagePlugin

from ..domain import PreprocessingOperationResult, PreprocessingStatus, ProducedOutput
from .preprocessing_normalisation_kernel import normalise_local_open_beam_frame

MessageCallback = Callable[[str], None]
ProgressCallback = Callable[[int], None]
CancellationCheck = Callable[[], bool]
PageCompleteCallback = Callable[[int], None]
OutputCallback = Callable[[ProducedOutput], None]


def raden_pulse_count(info: Mapping[str, Any]) -> float | None:
    """Return the first valid RADEN pulse count using the legacy preference."""
    meta = info.get("axes", {}).get("meta", {})
    for key in ("pulses_with_data", "pulses"):
        value = meta.get(key)
        if value is not None and np.isfinite(value) and value > 0:
            return float(value)
    return None


def read_raden_frame(tiff: Any, index: int) -> np.ndarray:
    """Read one unflipped float32 TIFF page and replace nonfinite values."""
    tiff.seek(index)
    frame = np.array(tiff, dtype=np.float32, copy=True)
    np.nan_to_num(frame, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return frame


def same_raden_tof_axis(
    sample_info: Mapping[str, Any], open_beam_info: Mapping[str, Any]
) -> bool:
    """Apply the legacy bins/min/max and lowercased-units comparison."""
    sample_tof = sample_info.get("axes", {}).get("tof", {})
    open_beam_tof = open_beam_info.get("axes", {}).get("tof", {})
    for key in ("bins", "min", "max"):
        try:
            if not np.isclose(float(sample_tof[key]), float(open_beam_tof[key])):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return str(sample_tof.get("units", "")).lower() == str(
        open_beam_tof.get("units", "")
    ).lower()


def validate_raden_stacks(
    sample_info: Mapping[str, Any], open_beam_info: Mapping[str, Any]
) -> None:
    """Validate frame count, shape, then ToF in the established order."""
    if int(sample_info["n_frames"]) != int(open_beam_info["n_frames"]):
        raise ValueError(
            f"Frame count mismatch: sample={sample_info['n_frames']}, "
            f"open beam={open_beam_info['n_frames']}."
        )
    if tuple(sample_info["image_shape"]) != tuple(open_beam_info["image_shape"]):
        raise ValueError(
            f"Image shape mismatch: sample={sample_info['image_shape']}, "
            f"open beam={open_beam_info['image_shape']}."
        )
    if not same_raden_tof_axis(sample_info, open_beam_info):
        raise ValueError("Sample and open-beam RADEN TOF axes do not match.")


def normalise_raden_frame(
    sample_frame: np.ndarray,
    open_beam_sum: np.ndarray,
    frame_count: int,
    window_half: int,
    scale: float | np.float32,
) -> np.ndarray:
    """Delegate prepared RADEN arrays to the shared local spatial kernel."""
    return normalise_local_open_beam_frame(
        sample_frame,
        open_beam_sum,
        frame_count,
        window_half,
        np.float32(scale),
    )


def copy_raden_sidecars(
    sample_info: Mapping[str, Any],
    output_tiff: str,
    output_folder: str,
    *,
    output_callback: OutputCallback | None = None,
) -> tuple[ProducedOutput, ...]:
    """Copy RADEN metadata then the first raw-order sidecar per extension.

    Errors intentionally propagate: unlike classic normalisation, a RADEN
    sidecar copy failure fails the operation while retaining prior artifacts.
    """
    output_stem = os.path.splitext(os.path.basename(output_tiff))[0]
    source_folder = os.path.dirname(sample_info["file_path"])
    copied_exts: set[str] = set()
    outputs: list[ProducedOutput] = []

    def copied(path: str) -> None:
        output = ProducedOutput(path, "related_file_copy")
        outputs.append(output)
        if output_callback is not None:
            output_callback(output)

    metadata_path = sample_info.get("metadata_path")
    if metadata_path and os.path.exists(metadata_path):
        ext = os.path.splitext(metadata_path)[1].lower()
        destination = os.path.join(output_folder, output_stem + ext)
        shutil.copyfile(metadata_path, destination)
        copied_exts.add(ext)
        copied(destination)

    for ext in (".stat", ".json", ".log"):
        if ext in copied_exts:
            continue
        for name in os.listdir(source_folder):
            source_path = os.path.join(source_folder, name)
            if os.path.isfile(source_path) and os.path.splitext(name)[1].lower() == ext:
                destination = os.path.join(output_folder, output_stem + ext)
                shutil.copyfile(source_path, destination)
                copied_exts.add(ext)
                copied(destination)
                break
    return tuple(outputs)


def normalise_raden_tiff_stack(
    sample_info: Mapping[str, Any],
    open_beam_info: Mapping[str, Any],
    output_folder: str,
    base_name: str,
    window_half: int,
    adjacent_sum: int,
    *,
    cancellation_check: CancellationCheck | None = None,
    progress_callback: ProgressCallback | None = None,
    message_callback: MessageCallback | None = None,
    page_complete_callback: PageCompleteCallback | None = None,
) -> PreprocessingOperationResult:
    """Normalise a RADEN stack, retaining persisted pages on failure/stop."""
    processed = 0
    expected_count: int | None = None
    outputs: list[ProducedOutput] = []
    errors: list[str] = []

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
        )

    try:
        # Preserve the legacy ordering: create the folder before validation.
        os.makedirs(output_folder, exist_ok=True)
        validate_raden_stacks(sample_info, open_beam_info)

        sample_pulses = raden_pulse_count(sample_info)
        open_beam_pulses = raden_pulse_count(open_beam_info)
        if sample_pulses and open_beam_pulses:
            scale = np.float32(open_beam_pulses / sample_pulses)
            emit(
                "RADEN pulse scale: "
                f"sample={sample_pulses:.0f}, open-beam={open_beam_pulses:.0f}, "
                f"scale={scale:.4f}"
            )
        else:
            scale = np.float32(1.0)
            emit("RADEN pulse metadata unavailable; using scale=1.0")

        output_stem = base_name.strip() or "normalised"
        sample_stem = os.path.splitext(os.path.basename(sample_info["file_path"]))[0]
        if output_stem.lower().endswith((".tif", ".tiff")):
            output_stem = os.path.splitext(output_stem)[0]
        else:
            output_stem = f"{output_stem}_{sample_stem}"
        output_tiff = os.path.join(output_folder, output_stem + ".tiff")

        total = int(sample_info["n_frames"])
        expected_count = total
        emit(
            "<b>--- Starting RADEN stack normalisation ---</b> "
            f"{total} frames, output='{os.path.basename(output_tiff)}'"
        )

        with Image.open(sample_info["file_path"]) as sample_tiff, Image.open(
            open_beam_info["file_path"]
        ) as open_beam_tiff:
            with TiffImagePlugin.AppendingTiffWriter(output_tiff, new=True) as writer:
                for index in range(total):
                    if cancelled():
                        emit("User stopped RADEN normalisation.")
                        break

                    sample_frame = read_raden_frame(sample_tiff, index)
                    start = max(0, index - adjacent_sum)
                    end = min(total - 1, index + adjacent_sum)
                    open_beam_sum = read_raden_frame(
                        open_beam_tiff, start
                    ).astype(np.float64)
                    for open_beam_index in range(start + 1, end + 1):
                        open_beam_frame = read_raden_frame(open_beam_tiff, open_beam_index)
                        open_beam_sum += open_beam_frame

                    normalised = normalise_raden_frame(
                        sample_frame,
                        open_beam_sum,
                        end - start + 1,
                        window_half,
                        scale,
                    )
                    Image.fromarray(normalised).save(writer, format="TIFF")
                    if not outputs:
                        outputs.append(ProducedOutput(output_tiff, "normalised_stack"))
                    processed += 1
                    if index != total - 1:
                        writer.newFrame()

                    if progress_callback is not None:
                        progress_callback(int(100 * (index + 1) / total))
                    if page_complete_callback is not None:
                        page_complete_callback(index)

        # Match the legacy two cancellation boundaries: before sidecars, then
        # once more for the final status after an already-started copy finishes.
        if not cancelled():
            copy_raden_sidecars(
                sample_info,
                output_tiff,
                output_folder,
                output_callback=outputs.append,
            )
        still_running = not cancelled()
        succeeded = still_running and processed == total
        if succeeded:
            emit(f"RADEN stack normalisation complete: {output_tiff}")
            return result(PreprocessingStatus.SUCCEEDED)

        incomplete = f"RADEN normalisation incomplete: {processed} of {total} frames written."
        emit(incomplete)
        if not still_running:
            return result(PreprocessingStatus.CANCELLED)
        errors.append(incomplete)
        return result(PreprocessingStatus.FAILED)
    except Exception as exc:
        errors.append(str(exc))
        emit(f"Fatal error in RADEN normalisation: {exc}")
        return result(PreprocessingStatus.FAILED)

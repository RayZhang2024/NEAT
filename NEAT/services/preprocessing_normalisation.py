"""Headless classic FITS/TIFF normalisation with legacy-compatible ordering."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from ..domain import (
    LoadedImageRun,
    PreprocessingOperationResult,
    PreprocessingStatus,
    ProducedOutput,
)
from .image_io import write_fits_image_file
from .preprocessing_normalisation_kernel import (
    LocalNormalisationShapeMismatch,
    LocalNormalisationWindowTooLarge,
    normalise_local_open_beam_frame,
)

NORMALISATION_WINDOW_HALF_RANGE = (0, 100)
NORMALISATION_ADJACENT_RANGE = (0, 10)

ProgressCallback = Callable[[int], None]
MessageCallback = Callable[[str], None]
CancellationCheck = Callable[[], bool]
FailureCallback = Callable[[str], None]
WrittenFrameCallback = Callable[[int, str], None]
RunCallback = Callable[[int], None]
FatalCallback = Callable[[Exception], None]


def validate_normalisation_windows(window_half, adjacent_sum) -> tuple[int, int]:
    """Retain the existing public n/m conversion, limits, and error messages."""
    try:
        window_half = int(window_half)
        adjacent_sum = int(adjacent_sum)
    except (TypeError, ValueError) as exc:
        raise ValueError("Normalisation n and m must be integers.") from exc

    n_min, n_max = NORMALISATION_WINDOW_HALF_RANGE
    m_min, m_max = NORMALISATION_ADJACENT_RANGE
    if not n_min <= window_half <= n_max:
        raise ValueError(f"Normalisation n must be between {n_min} and {n_max}.")
    if not m_min <= adjacent_sum <= m_max:
        raise ValueError(f"Normalisation m must be between {m_min} and {m_max}.")
    return window_half, adjacent_sum


def read_normalisation_shutter_count(
    folder_path: str,
    *,
    sidecar_paths: Sequence[str] | None = None,
) -> float:
    """Read the second token of line one in the selected shutter sidecar.

    Standalone callers retain the legacy first raw-order match. Full Process
    supplies a current-run manifest so stale files in a reused output folder
    cannot choose the normalisation scale.
    """
    if sidecar_paths is None:
        try:
            fname = next(
                f for f in os.listdir(folder_path) if f.endswith("_ShutterCount.txt")
            )
        except StopIteration:
            raise FileNotFoundError("no *_ShutterCount.txt found") from None
        source_path = os.path.join(folder_path, fname)
    else:
        matches = []
        for sidecar_path in sidecar_paths:
            candidate_path = os.fspath(sidecar_path)
            if not os.path.isabs(candidate_path):
                candidate_path = os.path.join(folder_path, candidate_path)
            if os.path.basename(candidate_path).endswith("_ShutterCount.txt"):
                matches.append(candidate_path)
        if not matches:
            raise FileNotFoundError("no current-run *_ShutterCount.txt found")
        if len(matches) != 1:
            names = ", ".join(os.path.basename(path) for path in matches)
            raise ValueError(
                f"ambiguous current-run ShutterCount sidecars: {names}"
            )
        source_path = matches[0]
        fname = os.path.basename(source_path)

    with open(source_path, "r") as fh:
        first_line = fh.readline().strip()
    parts = [p for p in re.split(r"[,\s]+", first_line) if p]
    if len(parts) < 2:
        raise ValueError(f"cannot parse shutter count in {fname}")
    return float(parts[1])


class NormalisationShapeMismatch(ValueError):
    """The sample and temporally combined open beam differ in shape."""


class NormalisationWindowTooLarge(ValueError):
    """The nominal spatial window exceeds at least one image dimension."""


def normalise_classic_frame(
    sample_image: np.ndarray,
    open_beam_images: Mapping[str, np.ndarray],
    common_suffixes: Sequence[str],
    index: int,
    window_half: int,
    adjacent_sum: int,
    scale: float | np.float32,
) -> np.ndarray:
    """Pure legacy temporal/spatial integral-image calculation for one frame."""
    img = sample_image.astype(np.float32)
    if adjacent_sum == 0:
        ob0 = open_beam_images[common_suffixes[index]].astype(np.float64).copy()
        start = 0
        end = 0
    else:
        start = max(0, index - adjacent_sum)
        end = min(len(common_suffixes) - 1, index + adjacent_sum)
        ob0 = open_beam_images[common_suffixes[start]].astype(np.float64).copy()
        for j in range(start + 1, end + 1):
            ob0 += open_beam_images[common_suffixes[j]].astype(np.float64)

    try:
        return normalise_local_open_beam_frame(
            img,
            ob0,
            end - start + 1,
            window_half,
            scale,
        )
    except LocalNormalisationShapeMismatch as exc:
        raise NormalisationShapeMismatch from exc
    except LocalNormalisationWindowTooLarge as exc:
        raise NormalisationWindowTooLarge from exc


def copy_normalisation_related_files(
    run_idx: int,
    data_folder: str,
    output_folder: str,
    *,
    sidecar_paths: Sequence[str] | None = None,
    message_callback: MessageCallback | None = None,
) -> tuple[tuple[ProducedOutput, ...], tuple[str, ...]]:
    """Copy sample sidecars in type-grouped order.

    With an explicit manifest, only current-run sidecars are copied and prior
    sidecars for this output run are pruned. Without one, standalone callers
    retain the legacy full-folder copy behavior.
    """
    outputs: list[ProducedOutput] = []
    warnings: list[str] = []

    def emit(message: str) -> None:
        if message_callback is not None:
            message_callback(message)

    try:
        spectra_suffix = "_Spectra.txt"
        shuttercount_suffix = "_ShutterCount.txt"
        if sidecar_paths is None:
            data_files = os.listdir(data_folder)
            spectra_files = [
                (filename, os.path.join(data_folder, filename))
                for filename in data_files if filename.endswith(spectra_suffix)
            ]
            shuttercount_files = [
                (filename, os.path.join(data_folder, filename))
                for filename in data_files if filename.endswith(shuttercount_suffix)
            ]
        else:
            spectra_files = []
            shuttercount_files = []
            for sidecar_path in sidecar_paths:
                source_path = os.fspath(sidecar_path)
                if not os.path.isabs(source_path):
                    source_path = os.path.join(data_folder, source_path)
                filename = os.path.basename(source_path)
                if filename.endswith(spectra_suffix):
                    spectra_files.append((filename, source_path))
                elif filename.endswith(shuttercount_suffix):
                    shuttercount_files.append((filename, source_path))

            current_names = {
                f"Run{run_idx}_{filename}"
                for filename, source in (*spectra_files, *shuttercount_files)
                if os.path.isfile(source)
            }
            for filename in os.listdir(output_folder):
                if not filename.startswith(f"Run{run_idx}_"):
                    continue
                if not filename.endswith((spectra_suffix, shuttercount_suffix)):
                    continue
                if filename not in current_names:
                    stale_path = os.path.join(output_folder, filename)
                    if os.path.isfile(stale_path):
                        os.remove(stale_path)
                        emit(f"Removed stale sidecar '{filename}'.")

        if spectra_files:
            for filename, source_path in spectra_files:
                dest_filename = f"Run{run_idx}_{filename}"
                dest_path = os.path.join(output_folder, dest_filename)
                shutil.copyfile(source_path, dest_path)
                outputs.append(ProducedOutput(dest_path, "related_file_copy"))
                emit(f"Copied '{filename}' to '{dest_filename}'.")
        else:
            emit(f"No file ending with '{spectra_suffix}' found in {data_folder}.")

        if shuttercount_files:
            for filename, source_path in shuttercount_files:
                dest_filename = f"Run{run_idx}_{filename}"
                dest_path = os.path.join(output_folder, dest_filename)
                shutil.copyfile(source_path, dest_path)
                outputs.append(ProducedOutput(dest_path, "related_file_copy"))
                emit(f"Copied '{filename}' to '{dest_filename}'.")
        else:
            emit(f"No file ending with '{shuttercount_suffix}' found in {data_folder}.")
    except Exception as exc:
        warning = f"Error copying related files: {exc}"
        warnings.append(warning)
        emit(warning)
    return tuple(outputs), tuple(warnings)


def normalise_loaded_image_runs(
    sample_runs: Sequence[LoadedImageRun],
    open_beam_runs: Sequence[LoadedImageRun],
    output_folder: str,
    base_name: str,
    window_half: int,
    adjacent_sum: int,
    *,
    progress_callback: ProgressCallback | None = None,
    message_callback: MessageCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
    failed_frame_callback: FailureCallback | None = None,
    written_frame_callback: WrittenFrameCallback | None = None,
    run_cleanup_callback: RunCallback | None = None,
    run_collect_callback: RunCallback | None = None,
    run_complete_callback: RunCallback | None = None,
    fatal_error_callback: FatalCallback | None = None,
    sample_sidecars_by_run: Sequence[Sequence[str]] | None = None,
    open_beam_sidecars_by_run: Sequence[Sequence[str]] | None = None,
) -> PreprocessingOperationResult:
    """Normalise paired runs while preserving partial outputs and adapter events."""
    expected_count = sum(len(run.frames) for run in sample_runs)
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

    def fail_frame(token: str, message: str) -> None:
        failed_frames.append(token)
        errors.append(message)
        emit(message)
        if failed_frame_callback is not None:
            failed_frame_callback(token)

    def result(status: PreprocessingStatus) -> PreprocessingOperationResult:
        return PreprocessingOperationResult(
            status=status,
            processed_count=processed,
            outputs=tuple(outputs),
            expected_count=expected_count,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    try:
        norm_path = os.path.normpath(output_folder)
        parts = norm_path.split(os.sep)
        short_path = (
            os.path.join(parts[-2], parts[-1])
            if len(parts) >= 2
            else norm_path
        )
        if not sample_runs or not open_beam_runs:
            message = "No runs provided. Aborting."
            errors.append(message)
            emit(message)
            return result(PreprocessingStatus.FAILED)
        if len(sample_runs) != len(open_beam_runs):
            message = "Data vs. Open‐beam count mismatch. Aborting."
            errors.append(message)
            emit(message)
            return result(PreprocessingStatus.FAILED)
        for name, manifests in (
            ("sample_sidecars_by_run", sample_sidecars_by_run),
            ("open_beam_sidecars_by_run", open_beam_sidecars_by_run),
        ):
            if manifests is not None and len(manifests) != len(sample_runs):
                message = f"{name} must contain one entry per run."
                errors.append(message)
                emit(message)
                return result(PreprocessingStatus.FAILED)

        emit("<b>--- Starting Normalisation ---</b>")
        emit(
            f"Using {2 * window_half + 1}×{2 * window_half + 1} spatial window "
            f"and {2 * adjacent_sum + 1} frames."
        )
        # Zip ordered positions, not LoadedImageRun objects: zip retains its
        # last yielded pair, which would keep cleared sample arrays alive.
        for run_idx, (sample_pos, beam_pos) in enumerate(
            zip(range(len(sample_runs)), range(len(open_beam_runs))), start=1
        ):
            if cancelled():
                emit("User stopped the process.")
                break
            sample_run = sample_runs[sample_pos]
            open_beam_run = open_beam_runs[beam_pos]
            sample_sidecars = (
                None if sample_sidecars_by_run is None
                else sample_sidecars_by_run[sample_pos]
            )
            open_beam_sidecars = (
                None if open_beam_sidecars_by_run is None
                else open_beam_sidecars_by_run[beam_pos]
            )

            try:
                sc = read_normalisation_shutter_count(
                    sample_run.primary_source, sidecar_paths=sample_sidecars
                )
                ob = read_normalisation_shutter_count(
                    open_beam_run.primary_source, sidecar_paths=open_beam_sidecars
                )
                scale = np.float32(ob / sc) if sc > 0 else 1.0
                emit(f"sample={sc:.0f}, open‐beam={ob:.0f}, scale={scale:.4f}")
            except Exception as exc:
                if (
                    sample_sidecars_by_run is not None
                    or open_beam_sidecars_by_run is not None
                ):
                    raise ValueError(
                        f"current-run shutter-count metadata is unavailable: {exc}"
                    ) from exc
                emit(f"shutter‐count error ({exc}), scale=1.0")
                scale = np.float32(1.0)

            data_imgs = sample_run.frames
            ob_imgs = open_beam_run.frames
            common = sorted(set(data_imgs) & set(ob_imgs))
            if not common:
                fail_frame(
                    f"run-{run_idx}:no-matching-suffix",
                    " no matching suffixes—skipping.",
                )
                continue

            for i, suffix in enumerate(common):
                if cancelled():
                    break
                try:
                    normed = normalise_classic_frame(
                        data_imgs[suffix], ob_imgs, common, i,
                        window_half, adjacent_sum, scale,
                    )
                    out_fname = f"{base_name}_{suffix}.fits"
                    output_path = os.path.join(output_folder, out_fname)
                    write_fits_image_file(output_path, normed, overwrite=True)
                    outputs.append(ProducedOutput(output_path, "normalised_image"))
                    if written_frame_callback is not None:
                        written_frame_callback(run_idx, suffix)
                    processed += 1
                    if progress_callback is not None:
                        progress_callback(int(100 * processed / expected_count))
                except NormalisationShapeMismatch:
                    fail_frame(str(suffix), f" {suffix}: shape mismatch—skipping.")
                except NormalisationWindowTooLarge:
                    fail_frame(str(suffix), f" {suffix}: too small for window—skipping.")
                except Exception as exc:
                    fail_frame(str(suffix), f" {suffix}: error ({exc})—skipping.")

            if run_cleanup_callback is not None:
                run_cleanup_callback(run_idx)
            sample_source = sample_run.primary_source
            del data_imgs, sample_run
            if run_collect_callback is not None:
                run_collect_callback(run_idx)
            sidecar_outputs, sidecar_warnings = copy_normalisation_related_files(
                run_idx,
                sample_source,
                output_folder,
                sidecar_paths=sample_sidecars,
                message_callback=emit,
            )
            outputs.extend(sidecar_outputs)
            warnings.extend(sidecar_warnings)
            emit("Run done.")
            if not cancelled() and run_complete_callback is not None:
                run_complete_callback(run_idx)

        # This snapshot follows cleanup, sidecars, and the final run pause.
        was_cancelled = cancelled()
        succeeded = not was_cancelled and not failed_frames and processed == expected_count
        status = (
            PreprocessingStatus.CANCELLED
            if was_cancelled
            else PreprocessingStatus.SUCCEEDED
            if succeeded
            else PreprocessingStatus.FAILED
        )
        label = "completed" if succeeded else "failed or incomplete"
        summary = (
            f"Normalisation {label}: {processed} of {expected_count} "
            f"images written to {short_path}."
        )
        if status is PreprocessingStatus.FAILED and not errors:
            errors.append(summary)
        emit(summary)
        return result(status)
    except Exception as exc:
        errors.append(str(exc))
        if fatal_error_callback is not None:
            fatal_error_callback(exc)
        return result(PreprocessingStatus.FAILED)

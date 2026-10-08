"""Deterministic, headless input preparation for overlap correction."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..domain import LoadedImageRun
from .image_io import load_image_file

ProgressCallback = Callable[[int], None]
MessageCallback = Callable[[str], None]
CancellationCheck = Callable[[], bool]

IMAGE_EXTENSIONS = (".fits", ".fit", ".tiff", ".tif")
SPECTRA_SUFFIX = "_Spectra.txt"
SHUTTER_SUFFIX = "_ShutterCount.txt"
_FRAME_SUFFIX = re.compile(r"[0-9]{1,10}\Z")


@dataclass(frozen=True, slots=True)
class PreparedOverlapInputs:
    """One selected frame sequence and its unambiguous overlap sidecars."""

    run: LoadedImageRun
    spectra_data: np.ndarray | None
    shutter_count_data: np.ndarray | None
    expected_count: int
    spectra_filename: str | None
    related_files: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    image_paths: tuple[str, ...]


def validate_frame_suffixes(suffixes: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return numeric frame order and diagnostics for ambiguous identity/order."""
    errors: list[str] = []
    indexed: list[tuple[int, str]] = []
    first_by_id: dict[int, str] = {}

    for suffix in suffixes:
        if not _FRAME_SUFFIX.fullmatch(suffix):
            errors.append(
                f"Frame suffix '{suffix}' is invalid; expected 1–10 ASCII digits."
            )
            continue
        frame_id = int(suffix)
        previous = first_by_id.get(frame_id)
        if previous is not None:
            errors.append(
                f"Frame suffix collision: '{previous}' and '{suffix}' both identify "
                f"frame {frame_id}."
            )
            continue
        first_by_id[frame_id] = suffix
        indexed.append((frame_id, suffix))

    indexed.sort(key=lambda item: item[0])
    for (previous_id, _), (frame_id, suffix) in zip(indexed, indexed[1:]):
        if frame_id != previous_id + 1:
            errors.append(
                f"Frame suffix gap: expected frame {previous_id + 1}, found '{suffix}'."
            )
            break

    return tuple(suffix for _frame_id, suffix in indexed), tuple(errors)


def prepare_overlap_inputs(
    folder: str,
    *,
    image_paths: Sequence[str] | None = None,
    stage: str = "Overlap Correction",
    progress_callback: ProgressCallback | None = None,
    message_callback: MessageCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
) -> PreparedOverlapInputs:
    """Discover and validate images and sidecars before loading image payloads.

    An explicit ``image_paths`` sequence is a producer manifest. Files present
    in the folder but absent from that manifest are ignored and reported as
    stale/unmanifested; they are never added to the selected frame sequence.
    """
    errors: list[str] = []
    warnings: list[str] = []
    spectra_data: np.ndarray | None = None
    shutter_data: np.ndarray | None = None
    spectra_path: str | None = None
    shutter_path: str | None = None
    selected_paths: list[str] = []
    selected_suffixes: list[str] = []
    all_image_names: set[str] = set()
    folder_names: list[str] = []

    def emit(message: str) -> None:
        if message_callback is not None:
            message_callback(message)

    try:
        folder_names = os.listdir(folder)
        all_image_names = {
            name
            for name in folder_names
            if name.lower().endswith(IMAGE_EXTENSIONS)
            and os.path.isfile(os.path.join(folder, name))
        }
    except Exception as exc:
        errors.append(_context(stage, folder, 0, None, None, f"cannot list folder: {exc}"))

    preliminary_image_count = len(all_image_names) if image_paths is None else len(image_paths)
    spectra_matches = sorted(name for name in folder_names if name.endswith(SPECTRA_SUFFIX))
    shutter_matches = sorted(name for name in folder_names if name.endswith(SHUTTER_SUFFIX))
    if len(spectra_matches) != 1:
        detail = (
            "missing Spectra sidecar"
            if not spectra_matches
            else f"ambiguous Spectra sidecars: {', '.join(spectra_matches)}"
        )
        errors.append(
            _context(stage, folder, preliminary_image_count, None, None, detail)
        )
    else:
        spectra_path = os.path.join(folder, spectra_matches[0])
        try:
            spectra_data = np.loadtxt(spectra_path, ndmin=2)
            if spectra_data.ndim != 2 or spectra_data.shape[0] == 0 or spectra_data.shape[1] == 0:
                raise ValueError("expected at least one row and one column")
            if not np.isfinite(spectra_data).all():
                raise ValueError("contains non-finite values")
        except Exception as exc:
            errors.append(
                _context(
                    stage,
                    folder,
                    preliminary_image_count,
                    None,
                    os.path.basename(spectra_path),
                    f"invalid Spectra sidecar: {exc}",
                )
            )
            spectra_data = None

    if len(shutter_matches) != 1:
        detail = (
            "missing ShutterCount sidecar"
            if not shutter_matches
            else f"ambiguous ShutterCount sidecars: {', '.join(shutter_matches)}"
        )
        errors.append(
            _context(
                stage,
                folder,
                preliminary_image_count,
                _row_count(spectra_data),
                os.path.basename(spectra_path) if spectra_path else None,
                detail,
            )
        )
    else:
        shutter_path = os.path.join(folder, shutter_matches[0])
        try:
            raw_shutter = np.loadtxt(shutter_path, ndmin=2)
            if raw_shutter.ndim != 2 or raw_shutter.size == 0:
                raise ValueError("expected at least one numeric value")
            if not np.isfinite(raw_shutter).all():
                raise ValueError("contains non-finite values")
            # Keep the legacy nonzero flattening and the correction service's
            # existing >1000 shutter selection exactly as before.
            shutter_data = raw_shutter[raw_shutter != 0]
            if shutter_data.size == 0:
                raise ValueError("contains no nonzero shutter counts")
        except Exception as exc:
            errors.append(
                _context(
                    stage,
                    folder,
                    preliminary_image_count,
                    _row_count(spectra_data),
                    os.path.basename(spectra_path) if spectra_path else None,
                    f"invalid ShutterCount sidecar '{os.path.basename(shutter_path)}': {exc}",
                )
            )
            shutter_data = None

    if image_paths is None:
        candidate_names = sorted(all_image_names)
    else:
        candidate_names = []
        for path in image_paths:
            name = os.path.basename(os.fspath(path))
            candidate_path = os.fspath(path)
            if not os.path.isabs(candidate_path):
                candidate_path = os.path.join(folder, candidate_path)
            if name.lower().endswith(IMAGE_EXTENSIONS):
                candidate_names.append(name)
                if not os.path.isfile(candidate_path):
                    errors.append(
                        _context(
                            stage,
                            folder,
                            len(image_paths),
                            _row_count(spectra_data),
                            os.path.basename(spectra_path) if spectra_path else None,
                            f"manifest image '{name}' is missing",
                        )
                    )
            else:
                errors.append(
                    _context(
                        stage,
                        folder,
                        len(image_paths),
                        _row_count(spectra_data),
                        os.path.basename(spectra_path) if spectra_path else None,
                        f"manifest entry '{name}' is not a supported FITS/TIFF image",
                    )
                )

        manifest_names = set(candidate_names)
        unmanifested = sorted(all_image_names - manifest_names)
        if unmanifested:
            warning = (
                f"{stage}: ignored {len(unmanifested)} unmanifested image(s) in "
                f"'{folder}'; first: {unmanifested[0]}"
            )
            warnings.append(warning)
            emit(warning)

    image_count = len(candidate_names)
    for filename in candidate_names:
        stem = os.path.splitext(filename)[0]
        if "_" not in stem:
            errors.append(
                _context(
                    stage,
                    folder,
                    image_count,
                    _row_count(spectra_data),
                    os.path.basename(spectra_path) if spectra_path else None,
                    f"problem image '{filename}': no underscore-separated numeric suffix",
                )
            )
            continue
        suffix = stem.rsplit("_", 1)[-1]
        if not _FRAME_SUFFIX.fullmatch(suffix):
            errors.append(
                _context(
                    stage,
                    folder,
                    image_count,
                    _row_count(spectra_data),
                    os.path.basename(spectra_path) if spectra_path else None,
                    f"problem image '{filename}' has invalid suffix '{suffix}'; "
                    "expected 1–10 ASCII digits",
                )
            )
            continue
        selected_paths.append(os.path.join(folder, filename))
        selected_suffixes.append(suffix)

    ordered_suffixes, identity_errors = validate_frame_suffixes(selected_suffixes)
    for detail in identity_errors:
        errors.append(
            _context(
                stage,
                folder,
                image_count,
                _row_count(spectra_data),
                os.path.basename(spectra_path) if spectra_path else None,
                detail,
            )
        )

    if spectra_data is not None and image_count != len(spectra_data):
        if image_count > len(spectra_data) and ordered_suffixes:
            problem = (
                f"extra/unmatched image suffix '{ordered_suffixes[len(spectra_data)]}' "
                f"at sorted frame position {len(spectra_data) + 1}"
            )
        else:
            problem = (
                f"missing image for ToF row {image_count + 1}"
                if image_count < len(spectra_data)
                else "image/ToF count mismatch"
            )
        errors.append(
            _context(
                stage,
                folder,
                image_count,
                len(spectra_data),
                os.path.basename(spectra_path) if spectra_path else None,
                problem,
            )
        )

    frames: dict[str, np.ndarray] = {}
    selected_path_by_suffix = dict(zip(selected_suffixes, selected_paths))
    # Fail cheaply: do not read multi-thousand-frame image stacks if names,
    # sidecars, identity, or frame/ToF counts have already failed preflight.
    if not errors:
        total = len(ordered_suffixes)
        for index, suffix in enumerate(ordered_suffixes, start=1):
            if cancellation_check is not None and cancellation_check():
                errors.append(f"{stage}: input loading cancelled by user.")
                frames.clear()
                break
            image_path = selected_path_by_suffix[suffix]
            try:
                data = load_image_file(image_path)
                if data is None:
                    raise ValueError("no image data found")
                frames[suffix] = np.asarray(data).astype(np.float32, copy=False)
            except Exception as exc:
                errors.append(
                    _context(
                        stage,
                        folder,
                        image_count,
                        _row_count(spectra_data),
                        os.path.basename(spectra_path) if spectra_path else None,
                        f"problem image '{os.path.basename(image_path)}' "
                        f"(suffix '{suffix}') could not be loaded: {exc}",
                    )
                )
                frames.clear()
                break
            if progress_callback is not None:
                progress_callback(int(index / total * 100))

    for error in errors:
        emit(error)
    related_files: tuple[str, ...] = ()
    if spectra_path is not None and shutter_path is not None:
        related_files = (spectra_path, shutter_path)
    run = LoadedImageRun(folder, frames, load_errors=errors)
    return PreparedOverlapInputs(
        run,
        spectra_data,
        shutter_data,
        image_count,
        os.path.basename(spectra_path) if spectra_path else None,
        related_files,
        tuple(errors),
        tuple(warnings),
        tuple(selected_path_by_suffix[suffix] for suffix in ordered_suffixes),
    )


def _row_count(spectra: np.ndarray | None) -> int | None:
    return int(spectra.shape[0]) if spectra is not None and spectra.ndim == 2 else None


def _context(
    stage: str,
    folder: str,
    image_count: int,
    tof_count: int | None,
    spectra_filename: str | None,
    problem: str,
) -> str:
    tof_text = "unknown" if tof_count is None else str(tof_count)
    spectra_text = spectra_filename or "not selected"
    return (
        f"{stage}: folder='{folder}', images={image_count}, ToF rows={tof_text}, "
        f"Spectra='{spectra_text}': {problem}."
    )

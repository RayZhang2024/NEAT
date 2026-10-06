"""Headless Clean/outlier filtering for loaded classic image runs."""

from __future__ import annotations

import heapq
import os
import shutil
from collections.abc import Callable, Sequence

import numpy as np
from scipy import ndimage

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


def positive_neighbor_mean(
    img: np.ndarray, y: int, x: int, *, radius: int
) -> float | None:
    """Return the positive finite neighbor mean, excluding the center."""
    y0, y1 = max(0, y - radius), min(img.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(img.shape[1], x + radius + 1)
    neighborhood = img[y0:y1, x0:x1]
    valid = (neighborhood > 0) & np.isfinite(neighborhood)
    valid[y - y0, x - x0] = False
    values = neighborhood[valid]
    if values.size == 0:
        return None
    return float(np.mean(values))


def clean_outlier_frame(
    img_data: np.ndarray, suffix: str
) -> tuple[np.ndarray, tuple[str, ...], int]:
    """Clean one image without I/O or mutation of the caller's array."""
    img = img_data.astype(np.float32, copy=True)
    invalid_mask = (img <= 0) | np.isnan(img) | np.isinf(img)
    bad_pixels = np.argwhere(invalid_mask)

    records: list[str] = []
    cleaned = 0

    # Preserve sequential invalid-pixel replacement and the 5x5/7x7 search.
    for y, x in bad_pixels:
        original = img[y, x]
        replacement = positive_neighbor_mean(img, y, x, radius=2)
        if replacement is None:
            replacement = positive_neighbor_mean(img, y, x, radius=3)

        if replacement is not None:
            img[y, x] = replacement
        else:
            replacement = np.nan  # Leave the invalid image value unchanged.

        records.append(f"{suffix},{x},{y},{original:.4f},{replacement:.4f}")
        cleaned += 1

    # Keep the original float64 5x5 cache and sequential raster/heap updates.
    valid = np.isfinite(img) & (img > 0)
    positive_values = np.where(valid, img, 0.0).astype(np.float64)
    neighbourhood_sum = (
        ndimage.uniform_filter(positive_values, size=5, mode="constant", cval=0.0)
        * 25.0
    )
    neighbourhood_count = (
        ndimage.uniform_filter(
            valid.astype(np.float64), size=5, mode="constant", cval=0.0
        )
        * 25.0
    )

    neighbor_count = neighbourhood_count - valid
    with np.errstate(divide="ignore", invalid="ignore"):
        neighbor_mean = (neighbourhood_sum - positive_values) / neighbor_count
    tolerance = np.finfo(np.float64).eps * 32.0 * np.maximum(1.0, np.abs(img))
    candidate_mask = (
        valid & (neighbor_count > 0.0) & (img >= 10.0 * neighbor_mean - tolerance)
    )
    candidates = [tuple(position) for position in np.argwhere(candidate_mask)]
    heapq.heapify(candidates)
    queued = candidate_mask.copy()
    height, width = img.shape

    while candidates:
        y, x = heapq.heappop(candidates)
        queued[y, x] = False
        original = float(img[y, x])
        replacement = positive_neighbor_mean(img, y, x, radius=2)
        if replacement is None or original < 10.0 * replacement:
            continue

        img[y, x] = replacement
        records.append(f"{suffix},{x},{y},{original:.4f},{replacement:.4f}")
        cleaned += 1

        y0, y1 = max(0, y - 2), min(height, y + 3)
        x0, x1 = max(0, x - 2), min(width, x + 3)
        neighbourhood_sum[y0:y1, x0:x1] += replacement - original

        local_y, local_x = np.mgrid[y0:y1, x0:x1]
        later = (local_y > y) | ((local_y == y) & (local_x > x))
        local_counts = neighbor_count[y0:y1, x0:x1]
        local_values = img[y0:y1, x0:x1]
        with np.errstate(divide="ignore", invalid="ignore"):
            local_means = (
                neighbourhood_sum[y0:y1, x0:x1] - local_values
            ) / local_counts
        local_tolerance = (
            np.finfo(np.float64).eps * 32.0 * np.maximum(1.0, np.abs(local_values))
        )
        newly_eligible = (
            later
            & valid[y0:y1, x0:x1]
            & ~queued[y0:y1, x0:x1]
            & (local_counts > 0.0)
            & (local_values >= 10.0 * local_means - local_tolerance)
        )
        for local_row, local_col in np.argwhere(newly_eligible):
            candidate_y = y0 + int(local_row)
            candidate_x = x0 + int(local_col)
            queued[candidate_y, candidate_x] = True
            heapq.heappush(candidates, (candidate_y, candidate_x))

    return img, tuple(records), cleaned


def clean_loaded_image_runs(
    runs: Sequence[LoadedImageRun],
    output_folder: str,
    base_name: str,
    *,
    progress_callback: ProgressCallback | None = None,
    message_callback: MessageCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
    frame_failure_callback: FrameFailureCallback | None = None,
) -> PreprocessingOperationResult:
    """Persist Clean artifacts, retaining partial outputs on failure or stop."""
    expected_count = sum(len(run.frames) for run in runs)
    report_path = os.path.join(output_folder, f"{base_name}_outlier_report.csv")
    outputs: list[ProducedOutput] = []
    errors: list[str] = []
    warnings: list[str] = []
    processed = 0
    total_cleaned = 0

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

    try:
        os.makedirs(output_folder, exist_ok=True)
        try:
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write("frame_idx,pixel_x,pixel_y,outlier_value,replace_value\n")
        except OSError as exc:
            raise RuntimeError(f"Cannot create report file: {exc}") from exc
        outputs.append(ProducedOutput(report_path, "outlier_report"))

        for run_idx, run in enumerate(runs, start=1):
            if cancelled():
                emit("Process stopped by user")
                break

            for sidecar_suffix in ("_Spectra.txt", "_ShutterCount.txt"):
                # Re-enumerate for each suffix, matching the legacy first-match rule.
                for filename in os.listdir(run.primary_source):
                    if filename.endswith(sidecar_suffix):
                        src = os.path.join(run.primary_source, filename)
                        dst = os.path.join(output_folder, f"Run{run_idx}_{filename}")
                        try:
                            shutil.copy2(src, dst)
                            outputs.append(ProducedOutput(dst, "related_file_copy"))
                            emit(f"Copied {filename} → {os.path.basename(dst)}")
                        except OSError as exc:
                            warning = f"Could not copy {filename}: {exc}"
                            warnings.append(warning)
                            emit(f"[WARNING] {warning}")
                        break

            for suffix, img_data in run.frames.items():
                if cancelled():
                    break
                try:
                    img, records, cleaned = clean_outlier_frame(img_data, suffix)
                    if cleaned == 0:
                        emit(f"Frame {suffix}: no outliers detected")

                    try:
                        with open(report_path, "a", encoding="utf-8") as fh:
                            fh.write("\n".join(records) + "\n")
                    except OSError as exc:
                        warning = f"Could not append to report: {exc}"
                        warnings.append(warning)
                        emit(f"[WARNING] {warning}")

                    out_fits = os.path.join(output_folder, f"{base_name}_{suffix}.fits")
                    try:
                        write_fits_image_file(out_fits, img, overwrite=True)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Cannot write FITS {out_fits}: {exc}"
                        ) from exc
                    outputs.append(ProducedOutput(out_fits, "cleaned_image"))
                    total_cleaned += cleaned
                    processed += 1
                    if progress_callback is not None:
                        progress_callback(
                            int((processed / expected_count) * 100)
                            if expected_count
                            else 0
                        )
                except Exception as exc:  # noqa: BLE001 - legacy frame failures continue
                    error = f"Frame {suffix}: {exc}"
                    errors.append(error)
                    if frame_failure_callback is not None:
                        frame_failure_callback(str(suffix))
                    emit(f"[ERROR] {error}")

        emit(
            f"[SUCCESS] Processed {processed} frame(s) • "
            f"Total cleaned pixels: {total_cleaned} • "
            f"Report: {os.path.basename(report_path)}"
        )
        if cancelled():
            return result(PreprocessingStatus.CANCELLED)
        return result(
            PreprocessingStatus.FAILED if errors else PreprocessingStatus.SUCCEEDED
        )
    except Exception as exc:  # noqa: BLE001 - return fatal operational result
        errors.append(str(exc))
        emit(f"[FATAL] {exc}")
        return result(PreprocessingStatus.FAILED)

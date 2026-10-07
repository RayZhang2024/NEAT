"""Prepared-array numerical kernel shared by classic and RADEN workflows."""

from __future__ import annotations

import numpy as np


class LocalNormalisationShapeMismatch(ValueError):
    """The sample and prepared open-beam arrays have different shapes."""

    def __init__(self) -> None:
        super().__init__("sample/open-beam frame shape mismatch")


class LocalNormalisationWindowTooLarge(ValueError):
    """The selected spatial window does not fit the prepared frame."""

    def __init__(self) -> None:
        super().__init__("frame is too small for the selected spatial window")


def normalise_local_open_beam_frame(
    sample_frame: np.ndarray,
    open_beam_sum: np.ndarray,
    frame_count: int,
    window_half: int,
    scale: float | np.float32,
) -> np.ndarray:
    """Apply local spatial normalisation to already prepared arrays.

    This helper intentionally does not coerce or sanitise its inputs. Classic
    and RADEN callers retain their own preparation and temporal-accumulation
    policies before delegating here.
    """
    if sample_frame.shape != open_beam_sum.shape:
        raise LocalNormalisationShapeMismatch

    height, width = sample_frame.shape
    if height < 2 * window_half + 1 or width < 2 * window_half + 1:
        raise LocalNormalisationWindowTooLarge

    full_win = (2 * window_half + 1) ** 2
    thresh = 1e-7
    integral = open_beam_sum.cumsum(0).cumsum(1)
    integral = np.pad(integral, ((1, 0), (1, 0)), "constant")
    counts = np.pad(
        np.ones_like(sample_frame).cumsum(0).cumsum(1),
        ((1, 0), (1, 0)),
        "constant",
    )

    rows, cols = np.ogrid[:height, :width]
    r0, r1 = rows - window_half, rows + window_half + 1
    c0, c1 = cols - window_half, cols + window_half + 1
    r0, r1 = np.clip(r0, 0, height), np.clip(r1, 0, height)
    c0, c1 = np.clip(c0, 0, width), np.clip(c1, 0, width)

    part_sum = (
        integral[r1, c1]
        - integral[r0, c1]
        - integral[r1, c0]
        + integral[r0, c0]
    )
    part_count = (
        counts[r1, c1]
        - counts[r0, c1]
        - counts[r1, c0]
        + counts[r0, c0]
    )
    scaled = np.where(
        part_count > 0,
        part_sum * (full_win / part_count),
        thresh,
    ).astype(np.float32)

    normalised = (
        frame_count * full_win * sample_frame / scaled
    ) * scale
    return np.nan_to_num(
        normalised,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

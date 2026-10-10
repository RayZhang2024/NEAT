"""Typed, GUI-independent contracts for scientific observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import floor
from threading import Lock
from uuid import uuid4

import numpy as np


class ObservationStatus(str, Enum):
    OK = "ok"
    NOT_APPLICABLE = "not_applicable"
    INVALID = "invalid"


class DatasetMode(str, Enum):
    IMAGES = "images"
    PROFILE = "profile"
    EMPTY = "empty"


class WavelengthAxisStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    LENGTH_MISMATCH = "length_mismatch"
    NON_FINITE = "non_finite"
    NON_MONOTONIC = "non_monotonic"


class RoiClipPolicy(str, Enum):
    REJECT = "reject"
    CLIP = "clip"


@dataclass(frozen=True)
class ObservationError:
    code: str
    message: str


@dataclass(frozen=True)
class RoiBounds:
    """Half-open image coordinates: x in [x_min, x_max), y in [y_min, y_max)."""

    x_min: int
    x_max: int
    y_min: int
    y_max: int
    clip_policy: RoiClipPolicy = RoiClipPolicy.REJECT

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class DatasetCapabilities:
    images_available: bool
    spectrum_available: bool
    mapping_applicable: bool


@dataclass(frozen=True)
class DatasetProvenance:
    """Safe provenance fields; intentionally contains no local file paths."""

    input_format: str = "unknown"
    wavelength_source: str = "unknown"
    flight_path_m: float | None = None
    flight_path_source: str = "unknown"
    delay_ms: float | None = None
    wavelength_unit: str = "angstrom"
    intensity_unit: str = "not_provided"
    companion_metadata_status: str = "unknown"


@dataclass(frozen=True)
class DatasetObservation:
    status: ObservationStatus
    dataset_id: str
    revision: int
    mode: DatasetMode
    frame_count: int | None
    image_shape: tuple[int, int] | None
    wavelength_count: int
    wavelength_min: float | None
    wavelength_max: float | None
    wavelength_axis_status: WavelengthAxisStatus
    display_window: tuple[float, float] | None
    capabilities: DatasetCapabilities
    provenance: DatasetProvenance
    metadata_flags: tuple[str, ...] = ()
    error: ObservationError | None = None


@dataclass(frozen=True)
class ImageViewSpec:
    max_dimension: int = 1024
    vmin: float | None = None
    vmax: float | None = None


@dataclass(frozen=True)
class PixelCoordinateMapping:
    """Nearest-sample mapping from preview pixels to original pixel centres."""

    original_width: int
    original_height: int
    preview_width: int
    preview_height: int
    origin: str = "upper"
    x_axis: str = "column"
    y_axis: str = "row"
    pixel_centres: str = "zero_based_integer_centres"
    roi_bounds: str = "zero_based_half_open"

    def preview_pixel_to_original(self, x: int, y: int) -> tuple[int, int]:
        if not (0 <= x < self.preview_width and 0 <= y < self.preview_height):
            raise ValueError("Preview pixel coordinate is outside the preview.")
        original_x = min(
            floor((x + 0.5) * self.original_width / self.preview_width),
            self.original_width - 1,
        )
        original_y = min(
            floor((y + 0.5) * self.original_height / self.preview_height),
            self.original_height - 1,
        )
        return original_x, original_y


@dataclass(frozen=True)
class ImageObservation:
    status: ObservationStatus
    dataset_id: str
    revision: int
    frame_index: int | None
    original_shape: tuple[int, int] | None
    preview_shape: tuple[int, int] | None
    preview_pixels: np.ndarray | None
    png_bytes: bytes | None
    coordinate_mapping: PixelCoordinateMapping | None
    contrast_vmin: float | None
    contrast_vmax: float | None
    contrast_method: str | None
    sampled_nonfinite_count: int
    error: ObservationError | None = None


@dataclass(frozen=True)
class SpectrumObservation:
    status: ObservationStatus
    dataset_id: str
    revision: int
    mode: DatasetMode
    roi_requested: RoiBounds | None
    roi_effective: RoiBounds | None
    wavelengths: np.ndarray | None
    intensities: np.ndarray | None
    valid_pair_mask: np.ndarray | None
    display_mask: np.ndarray | None
    display_window: tuple[float, float] | None
    wavelength_axis_status: WavelengthAxisStatus
    wavelength_unit: str
    intensity_unit: str
    sample_count: int
    finite_pair_count: int
    nonfinite_intensity_count: int
    intensity_min: float | None
    intensity_max: float | None
    intensity_mean: float | None
    error: ObservationError | None = None


@dataclass(frozen=True)
class SpectrumPlotSpec:
    width_pixels: int = 800
    height_pixels: int = 450
    dpi: int = 100
    display_window: tuple[float, float] | None = None


@dataclass(frozen=True)
class PlotObservation:
    status: ObservationStatus
    dataset_id: str
    revision: int
    png_bytes: bytes | None
    mime_type: str
    width_pixels: int
    height_pixels: int
    x_label: str
    y_label: str
    plotted_point_count: int
    display_window: tuple[float, float] | None
    error: ObservationError | None = None


class DatasetRevisionClock:
    """Thread-safe identity/revision token shared by a GUI adapter and handles."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._dataset_id = uuid4().hex
        self._revision = 0

    def current(self) -> tuple[str, int]:
        with self._lock:
            return self._dataset_id, self._revision

    def invalidate(self, *, replace_dataset: bool = False) -> tuple[str, int]:
        with self._lock:
            if replace_dataset:
                self._dataset_id = uuid4().hex
            self._revision += 1
            return self._dataset_id, self._revision

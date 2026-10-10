"""Read-only scientific observations over an immutable dataset snapshot.

This module is deliberately independent of Qt and never calls GUI plotting or
fitting methods. Image-stack ROI extraction preserves the legacy GUI's
per-frame sum divided by ROI pixel area.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Integral
from threading import Lock
from types import EllipsisType
from uuid import uuid4

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from ..domain.observation import (
    DatasetCapabilities,
    DatasetMode,
    DatasetObservation,
    DatasetProvenance,
    DatasetRevisionClock,
    ImageObservation,
    ImageViewSpec,
    ObservationError,
    ObservationStatus,
    PixelCoordinateMapping,
    PlotObservation,
    RoiBounds,
    RoiClipPolicy,
    SpectrumObservation,
    SpectrumPlotSpec,
    WavelengthAxisStatus,
)

MAX_PREVIEW_DIMENSION = 1024
MAX_PREVIEW_PIXELS = MAX_PREVIEW_DIMENSION * MAX_PREVIEW_DIMENSION
MAX_PREVIEW_PNG_BYTES = 4 * 1024 * 1024
MAX_SPECTRUM_POINTS = 250_000
MAX_ROI_PIXEL_OPERATIONS = 250_000_000
MAX_METADATA_FLAGS = 64
MAX_PLOT_WIDTH_PIXELS = 1600
MAX_PLOT_HEIGHT_PIXELS = 1200
MAX_PLOT_PIXELS = 1_920_000
MAX_PLOT_DPI = 200
MAX_PLOT_PNG_BYTES = 8 * 1024 * 1024

_PLOT_LOCK = Lock()


class ObservationErrorBase(Exception):
    """Base class for errors raised by the read-only observation API."""


class StaleDatasetError(ObservationErrorBase):
    """The live dataset changed after a handle was captured."""


class DatasetUnavailableError(ObservationErrorBase):
    """The requested observation is unavailable for the captured dataset."""


class ObservationValidationError(ObservationErrorBase, ValueError):
    """An observation request is invalid or exceeds a documented bound."""


class ObservationLimitExceeded(ObservationValidationError):
    """An observation would exceed a declared resource limit."""


def _validated_wavelength_window(window, *, label: str):
    if window is None:
        return None
    try:
        if len(window) != 2:
            raise ValueError
        lower, upper = (float(value) for value in window)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ObservationValidationError(
            f"{label} limits must be two finite numbers."
        ) from exc
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ObservationValidationError(
            f"{label} limits must be finite and increasing."
        )
    return lower, upper


def _safe_provenance(provenance: DatasetProvenance | None) -> DatasetProvenance:
    """Allow only bounded vocabulary in model-facing metadata fields."""
    source = provenance or DatasetProvenance()
    choices = {
        "input_format": {"unknown", "fits", "tiff", "nexus", "csv", "text", "xlsx", "image_folder"},
        "wavelength_source": {
            "unknown", "imported_profile", "manual_anchors", "nexus_tof_axis",
            "nexus_wavelength_axis", "raden_tof_axis", "spectra_tof_axis",
        },
        "flight_path_source": {"unknown", "app_setting", "nexus_metadata"},
        "wavelength_unit": {"angstrom", "nm", "not_provided"},
        "intensity_unit": {"not_provided", "counts", "counts_per_pixel"},
        "companion_metadata_status": {"unknown", "provided", "missing", "not_applicable"},
    }

    def safe_choice(field_name):
        value = getattr(source, field_name)
        return value if isinstance(value, str) and value in choices[field_name] else "unknown"

    def safe_number(value):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if np.isfinite(number) else None

    return DatasetProvenance(
        input_format=safe_choice("input_format"),
        wavelength_source=safe_choice("wavelength_source"),
        flight_path_m=safe_number(source.flight_path_m),
        flight_path_source=safe_choice("flight_path_source"),
        delay_ms=safe_number(source.delay_ms),
        wavelength_unit=safe_choice("wavelength_unit"),
        intensity_unit=safe_choice("intensity_unit"),
        companion_metadata_status=safe_choice("companion_metadata_status"),
    )


_SAFE_METADATA_FLAGS = {
    "display_window_unavailable",
    "display_window_invalid",
    "profile_intensity_missing",
    "wavelength_axis_missing",
    "wavelength_axis_length_mismatch",
    "wavelength_axis_non_finite",
    "wavelength_axis_non_monotonic",
    "no_dataset_loaded",
    "companion_metadata_status_unknown",
    "metadata_flags_truncated",
}


def _safe_metadata_flags(flags: Sequence[str]) -> tuple[str, ...]:
    safe_flags = []
    for index, flag in enumerate(flags):
        if index >= MAX_METADATA_FLAGS:
            safe_flags.append("metadata_flags_truncated")
            break
        candidate = flag if isinstance(flag, str) else ""
        safe_flags.append(
            candidate if candidate in _SAFE_METADATA_FLAGS else "unknown_metadata_flag"
        )
    return tuple(dict.fromkeys(safe_flags))


@dataclass(frozen=True)
class ScientificDatasetHandle:
    """Read-only references and small copied axes for one dataset revision."""

    dataset_id: str
    revision: int
    mode: DatasetMode
    frames: tuple[np.ndarray, ...] | None
    image_shape: tuple[int, int] | None
    wavelengths: np.ndarray | None
    profile_intensities: np.ndarray | None
    display_window: tuple[float, float] | None
    provenance: DatasetProvenance
    metadata_flags: tuple[str, ...] = ()
    _revision_clock: DatasetRevisionClock | None = field(
        default=None, repr=False, compare=False
    )

    def assert_current(self) -> None:
        if self._revision_clock is None:
            return
        if self._revision_clock.current() != (self.dataset_id, self.revision):
            raise StaleDatasetError(
                "The dataset changed after this observation handle was captured."
            )


def _as_readonly_float_vector(values, *, name: str) -> np.ndarray | None:
    if values is None:
        return None
    source = np.asarray(values)
    if source.size > MAX_SPECTRUM_POINTS:
        raise ObservationLimitExceeded(
            f"{name} has {source.size} points; the limit is {MAX_SPECTRUM_POINTS}."
        )
    try:
        array = np.asarray(source, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ObservationValidationError(f"{name} must contain numeric values.") from exc
    result = array.copy()
    result.setflags(write=False)
    return result


def _readonly_frame_view(frame, *, index: int) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 2:
        raise ObservationValidationError(
            f"Image frame {index} must be two-dimensional."
        )
    if array.dtype.kind not in "biuf":
        raise ObservationValidationError(
            f"Image frame {index} must contain real numeric pixel values."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ObservationValidationError(
            f"Image frame {index} must have non-empty dimensions."
        )
    view = array.view()
    view.setflags(write=False)
    return view


def create_dataset_handle(
    *,
    mode: DatasetMode | str,
    images: Sequence[np.ndarray] | None = None,
    wavelengths=None,
    profile_intensities=None,
    display_window: tuple[float, float] | None = None,
    provenance: DatasetProvenance | None = None,
    metadata_flags: Sequence[str] = (),
    revision_clock: DatasetRevisionClock | None = None,
) -> ScientificDatasetHandle:
    """Build a coherent, read-only handle without copying the image stack."""
    mode = DatasetMode(mode)
    state_before = revision_clock.current() if revision_clock is not None else None
    frames = None
    image_shape = None

    if mode is DatasetMode.IMAGES:
        if images is None:
            frames = ()
        else:
            frame_count = len(images)
            if frame_count > MAX_SPECTRUM_POINTS:
                raise ObservationLimitExceeded(
                    f"Image stack has {frame_count} frames; the limit is "
                    f"{MAX_SPECTRUM_POINTS}."
                )
            frame_views = tuple(
                _readonly_frame_view(frame, index=index)
                for index, frame in enumerate(images)
            )
            if frame_views:
                image_shape = tuple(int(v) for v in frame_views[0].shape)
                if any(tuple(frame.shape) != image_shape for frame in frame_views):
                    raise ObservationValidationError(
                        "Image frames do not all have the same dimensions."
                    )
            frames = frame_views
        wavelength_values = _as_readonly_float_vector(
            wavelengths, name="Wavelength axis"
        )
        profile_values = None
    elif mode is DatasetMode.PROFILE:
        frames = None
        wavelength_values = _as_readonly_float_vector(
            wavelengths, name="Wavelength profile"
        )
        profile_values = _as_readonly_float_vector(
            profile_intensities, name="Intensity profile"
        )
    else:
        frames = None
        wavelength_values = None
        profile_values = None

    display_window = _validated_wavelength_window(
        display_window, label="Display wavelength"
    )

    if revision_clock is None:
        dataset_id = uuid4().hex
        revision = 0
    else:
        dataset_id, revision = state_before

    state_after = revision_clock.current() if revision_clock is not None else None
    if state_before != state_after:
        raise StaleDatasetError(
            "The live dataset changed while the observation handle was captured."
        )

    handle = ScientificDatasetHandle(
        dataset_id=dataset_id,
        revision=revision,
        mode=mode,
        frames=frames,
        image_shape=image_shape,
        wavelengths=wavelength_values,
        profile_intensities=profile_values,
        display_window=display_window,
        provenance=_safe_provenance(provenance),
        metadata_flags=_safe_metadata_flags(metadata_flags),
        _revision_clock=revision_clock,
    )
    handle.assert_current()
    return handle


def _axis_status(
    wavelengths: np.ndarray | None, expected_count: int | None
) -> WavelengthAxisStatus:
    if wavelengths is None or wavelengths.size == 0:
        return WavelengthAxisStatus.MISSING
    if expected_count is not None and wavelengths.size != expected_count:
        return WavelengthAxisStatus.LENGTH_MISMATCH
    if not np.isfinite(wavelengths).all():
        return WavelengthAxisStatus.NON_FINITE
    if wavelengths.size < 2:
        return WavelengthAxisStatus.VALID
    differences = np.diff(wavelengths)
    if np.all(differences > 0) or np.all(differences < 0):
        return WavelengthAxisStatus.VALID
    return WavelengthAxisStatus.NON_MONOTONIC


def inspect_dataset(handle: ScientificDatasetHandle) -> DatasetObservation:
    """Return privacy-safe metadata and capability status for a snapshot."""
    handle.assert_current()
    if handle.mode is DatasetMode.IMAGES:
        frames = handle.frames or ()
        frame_count: int | None = len(frames)
        expected = frame_count
        image_shape = handle.image_shape
        images_available = bool(frames)
    elif handle.mode is DatasetMode.PROFILE:
        frame_count = None
        expected = (
            handle.profile_intensities.size
            if handle.profile_intensities is not None
            else None
        )
        image_shape = None
        images_available = False
    else:
        frame_count = None
        expected = None
        image_shape = None
        images_available = False

    wavelengths = handle.wavelengths
    axis_status = _axis_status(wavelengths, expected)
    wavelength_count = int(wavelengths.size) if wavelengths is not None else 0
    finite_wavelengths = (
        wavelengths[np.isfinite(wavelengths)]
        if wavelengths is not None and wavelengths.size
        else np.array([], dtype=np.float64)
    )
    wavelength_min = (
        float(np.min(finite_wavelengths)) if finite_wavelengths.size else None
    )
    wavelength_max = (
        float(np.max(finite_wavelengths)) if finite_wavelengths.size else None
    )

    if handle.mode is DatasetMode.IMAGES:
        spectrum_available = (
            images_available
            and axis_status
            not in (WavelengthAxisStatus.MISSING, WavelengthAxisStatus.LENGTH_MISMATCH)
        )
        mapping_applicable = (
            images_available
            and spectrum_available
            and axis_status is WavelengthAxisStatus.VALID
        )
    elif handle.mode is DatasetMode.PROFILE:
        spectrum_available = (
            handle.profile_intensities is not None
            and handle.profile_intensities.size > 0
            and wavelengths is not None
            and wavelengths.size > 0
            and wavelengths.size == handle.profile_intensities.size
        )
        mapping_applicable = False
    else:
        spectrum_available = False
        mapping_applicable = False

    metadata_flags = list(handle.metadata_flags)
    if axis_status is not WavelengthAxisStatus.VALID:
        metadata_flags.append(f"wavelength_axis_{axis_status.value}")
    if handle.mode is DatasetMode.EMPTY:
        metadata_flags.append("no_dataset_loaded")
    if handle.provenance.companion_metadata_status == "unknown":
        metadata_flags.append("companion_metadata_status_unknown")

    if handle.mode is DatasetMode.EMPTY:
        status = ObservationStatus.NOT_APPLICABLE
        error = ObservationError("no_dataset_loaded", "No dataset is loaded.")
    elif not spectrum_available:
        status = ObservationStatus.INVALID
        error = ObservationError(
            "spectrum_unavailable",
            "The loaded data do not have aligned wavelength and signal arrays.",
        )
    else:
        status = ObservationStatus.OK
        error = None

    handle.assert_current()
    return DatasetObservation(
        status=status,
        dataset_id=handle.dataset_id,
        revision=handle.revision,
        mode=handle.mode,
        frame_count=frame_count,
        image_shape=image_shape,
        wavelength_count=wavelength_count,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_axis_status=axis_status,
        display_window=handle.display_window,
        capabilities=DatasetCapabilities(
            images_available=images_available,
            spectrum_available=spectrum_available,
            mapping_applicable=mapping_applicable,
        ),
        provenance=handle.provenance,
        metadata_flags=tuple(dict.fromkeys(metadata_flags)),
        error=error,
    )


def _not_applicable_image(
    handle: ScientificDatasetHandle, frame_index: int | None, code: str, message: str
) -> ImageObservation:
    return ImageObservation(
        status=ObservationStatus.NOT_APPLICABLE,
        dataset_id=handle.dataset_id,
        revision=handle.revision,
        frame_index=frame_index,
        original_shape=None,
        preview_shape=None,
        preview_pixels=None,
        png_bytes=None,
        coordinate_mapping=None,
        contrast_vmin=None,
        contrast_vmax=None,
        contrast_method=None,
        sampled_nonfinite_count=0,
        error=ObservationError(code, message),
    )


def get_image_preview(
    handle: ScientificDatasetHandle,
    frame_index: int,
    view_spec: ImageViewSpec | None = None,
) -> ImageObservation:
    """Create a bounded grayscale preview without changing source pixels."""
    handle.assert_current()
    if handle.mode is not DatasetMode.IMAGES or not handle.frames:
        result = _not_applicable_image(
            handle,
            None,
            "images_not_available",
            "Image previews are unavailable for spectrum-only datasets.",
        )
        handle.assert_current()
        return result

    spec = view_spec or ImageViewSpec()
    if (
        isinstance(spec.max_dimension, bool)
        or not isinstance(spec.max_dimension, int)
        or not 1 <= spec.max_dimension <= MAX_PREVIEW_DIMENSION
    ):
        raise ObservationValidationError(
            f"Preview max_dimension must be between 1 and {MAX_PREVIEW_DIMENSION}."
        )
    if (spec.vmin is None) != (spec.vmax is None):
        raise ObservationValidationError(
            "Specify both contrast limits or neither."
        )
    if spec.vmin is not None:
        try:
            vmin_request = float(spec.vmin)
            vmax_request = float(spec.vmax)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ObservationValidationError(
                "Contrast limits must be finite numbers."
            ) from exc
        if (
            not np.isfinite([vmin_request, vmax_request]).all()
            or vmin_request >= vmax_request
        ):
            raise ObservationValidationError(
                "Contrast limits must be finite and increasing."
            )
    else:
        vmin_request = vmax_request = None

    if isinstance(frame_index, bool) or not isinstance(frame_index, Integral):
        raise ObservationValidationError("frame_index must be an integer.")
    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= len(handle.frames):
        raise ObservationValidationError("frame_index is outside the loaded stack.")

    frame = handle.frames[frame_index]
    height, width = frame.shape
    scale = min(1.0, spec.max_dimension / max(height, width))
    preview_height = max(1, min(height, round(height * scale)))
    preview_width = max(1, min(width, round(width * scale)))
    if preview_height * preview_width > MAX_PREVIEW_PIXELS:
        raise ObservationLimitExceeded("Preview would exceed the pixel limit.")

    y_indices = np.minimum(
        np.floor((np.arange(preview_height) + 0.5) * height / preview_height).astype(int),
        height - 1,
    )
    x_indices = np.minimum(
        np.floor((np.arange(preview_width) + 0.5) * width / preview_width).astype(int),
        width - 1,
    )
    sampled = frame[np.ix_(y_indices, x_indices)]
    finite_mask = np.isfinite(sampled)
    sampled_nonfinite_count = int(sampled.size - np.count_nonzero(finite_mask))

    if spec.vmin is None:
        finite_values = sampled[finite_mask]
        if finite_values.size == 0:
            vmin, vmax = 0.0, 1.0
            contrast_method = "no_finite_sampled_pixels"
        else:
            vmin, vmax = (float(v) for v in np.percentile(finite_values, [5.0, 95.0]))
            contrast_method = "sampled_percentile_5_95"
            if vmin == vmax:
                value = vmin
                vmin, vmax = value - 0.5, value + 0.5
                contrast_method = "constant_frame_mid_gray"
    else:
        vmin, vmax = vmin_request, vmax_request
        contrast_method = "explicit_linear_limits"

    preview = np.zeros(sampled.shape, dtype=np.uint8)
    if np.any(finite_mask):
        values = sampled[finite_mask].astype(np.float64, copy=False)
        scaled = (values - vmin) * (255.0 / (vmax - vmin))
        preview[finite_mask] = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)

    buffer = BytesIO()
    Image.fromarray(preview).save(
        buffer, format="PNG", optimize=False, compress_level=6
    )
    png_bytes = buffer.getvalue()
    if len(png_bytes) > MAX_PREVIEW_PNG_BYTES:
        raise ObservationLimitExceeded(
            f"Preview PNG exceeds {MAX_PREVIEW_PNG_BYTES} bytes."
        )
    preview.setflags(write=False)

    mapping = PixelCoordinateMapping(
        original_width=width,
        original_height=height,
        preview_width=preview_width,
        preview_height=preview_height,
    )
    handle.assert_current()
    return ImageObservation(
        status=ObservationStatus.OK,
        dataset_id=handle.dataset_id,
        revision=handle.revision,
        frame_index=frame_index,
        original_shape=(height, width),
        preview_shape=(preview_height, preview_width),
        preview_pixels=preview,
        png_bytes=png_bytes,
        coordinate_mapping=mapping,
        contrast_vmin=vmin,
        contrast_vmax=vmax,
        contrast_method=contrast_method,
        sampled_nonfinite_count=sampled_nonfinite_count,
    )


def _validated_roi(
    roi: RoiBounds, image_shape: tuple[int, int]
) -> tuple[RoiBounds, RoiBounds]:
    if not isinstance(roi, RoiBounds):
        raise ObservationValidationError("roi must be RoiBounds.")
    coordinates = (roi.x_min, roi.x_max, roi.y_min, roi.y_max)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in coordinates):
        raise ObservationValidationError("ROI coordinates must be integers.")
    roi = RoiBounds(
        int(roi.x_min),
        int(roi.x_max),
        int(roi.y_min),
        int(roi.y_max),
        roi.clip_policy,
    )
    if roi.width <= 0 or roi.height <= 0:
        raise ObservationValidationError("ROI bounds must be non-empty.")
    try:
        clip_policy = RoiClipPolicy(roi.clip_policy)
    except ValueError as exc:
        raise ObservationValidationError("Unknown ROI clip policy.") from exc
    height, width = image_shape
    inside = (
        0 <= roi.x_min < roi.x_max <= width
        and 0 <= roi.y_min < roi.y_max <= height
    )
    if inside:
        return roi, roi
    if clip_policy is RoiClipPolicy.REJECT:
        raise ObservationValidationError(
            f"ROI must lie inside image bounds [0,{width}) x [0,{height})."
        )
    x_min = max(0, min(width, roi.x_min))
    x_max = max(0, min(width, roi.x_max))
    y_min = max(0, min(height, roi.y_min))
    y_max = max(0, min(height, roi.y_max))
    if x_min >= x_max or y_min >= y_max:
        raise ObservationValidationError(
            "Clipped ROI does not overlap the image."
        )
    effective = RoiBounds(
        x_min, x_max, y_min, y_max, clip_policy=RoiClipPolicy.REJECT
    )
    return roi, effective


def _display_mask(
    wavelengths: np.ndarray, display_window: tuple[float, float] | None
) -> np.ndarray:
    if display_window is None:
        mask = np.ones(wavelengths.shape, dtype=bool)
    else:
        lower, upper = display_window
        mask = (wavelengths >= lower) & (wavelengths <= upper)
    mask.setflags(write=False)
    return mask


def get_roi_spectrum(
    handle: ScientificDatasetHandle,
    roi: RoiBounds | None = None,
    *,
    display_window: tuple[float, float] | None | EllipsisType = ...,
) -> SpectrumObservation:
    """Return full source arrays plus an independent display-window mask."""
    handle.assert_current()
    if display_window is ...:
        effective_display_window = handle.display_window
    else:
        effective_display_window = display_window
    effective_display_window = _validated_wavelength_window(
        effective_display_window, label="Display wavelength"
    )

    roi_requested = None
    roi_effective = None
    if handle.mode is DatasetMode.PROFILE:
        if roi is not None:
            raise ObservationValidationError(
                "A spectrum-only dataset does not accept an image ROI."
            )
        wavelengths = handle.wavelengths
        intensities = handle.profile_intensities
        if wavelengths is None or intensities is None:
            raise DatasetUnavailableError(
                "The imported spectrum is incomplete."
            )
    elif handle.mode is DatasetMode.IMAGES:
        if not handle.frames or handle.image_shape is None:
            raise DatasetUnavailableError("No image frames are available.")
        if handle.wavelengths is None:
            raise DatasetUnavailableError("No wavelength axis is available.")
        if roi is None:
            raise ObservationValidationError(
                "An explicit ROI is required for an image-stack spectrum."
            )
        roi_requested, roi_effective = _validated_roi(roi, handle.image_shape)
        if roi_effective.width <= 0 or roi_effective.height <= 0:
            raise ObservationValidationError("ROI must be non-empty.")
        requested_work = len(handle.frames) * roi_effective.pixel_count
        if requested_work > MAX_ROI_PIXEL_OPERATIONS:
            raise ObservationLimitExceeded(
                "ROI extraction exceeds the per-call pixel-operation limit of "
                f"{MAX_ROI_PIXEL_OPERATIONS}."
            )
        roi_sums = []
        for frame in handle.frames:
            selected_area = frame[
                roi_effective.y_min:roi_effective.y_max,
                roi_effective.x_min:roi_effective.x_max,
            ]
            roi_sums.append(np.sum(selected_area))
        # Preserve the GUI's existing expression and dtype promotion exactly.
        intensities = np.array(roi_sums) / [
            roi_effective.width * roi_effective.height
        ]
        wavelengths = handle.wavelengths
    else:
        raise DatasetUnavailableError("No dataset is loaded.")

    if wavelengths.size != intensities.size:
        raise DatasetUnavailableError(
            "Wavelength and signal lengths do not match."
        )
    if wavelengths.size > MAX_SPECTRUM_POINTS:
        raise ObservationLimitExceeded(
            f"Spectrum has {wavelengths.size} points; the limit is "
            f"{MAX_SPECTRUM_POINTS}."
        )

    # Keep returned numerical arrays read-only, including the newly computed
    # image-stack mean profile.
    if intensities.flags.writeable:
        intensities.setflags(write=False)

    axis_status = _axis_status(wavelengths, wavelengths.size)
    valid_pair_mask = np.isfinite(wavelengths) & np.isfinite(intensities)
    valid_pair_mask.setflags(write=False)
    display_mask = _display_mask(wavelengths, effective_display_window)
    finite_intensities = intensities[np.isfinite(intensities)]
    finite_pair_count = int(np.count_nonzero(valid_pair_mask))
    nonfinite_intensity_count = int(
        intensities.size - np.count_nonzero(np.isfinite(intensities))
    )
    intensity_min = (
        float(np.min(finite_intensities)) if finite_intensities.size else None
    )
    intensity_max = (
        float(np.max(finite_intensities)) if finite_intensities.size else None
    )
    intensity_mean = (
        float(np.mean(finite_intensities)) if finite_intensities.size else None
    )

    handle.assert_current()
    return SpectrumObservation(
        status=ObservationStatus.OK,
        dataset_id=handle.dataset_id,
        revision=handle.revision,
        mode=handle.mode,
        roi_requested=roi_requested,
        roi_effective=roi_effective,
        wavelengths=wavelengths,
        intensities=intensities,
        valid_pair_mask=valid_pair_mask,
        display_mask=display_mask,
        display_window=effective_display_window,
        wavelength_axis_status=axis_status,
        wavelength_unit=handle.provenance.wavelength_unit,
        intensity_unit=handle.provenance.intensity_unit,
        sample_count=int(wavelengths.size),
        finite_pair_count=finite_pair_count,
        nonfinite_intensity_count=nonfinite_intensity_count,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        intensity_mean=intensity_mean,
    )


def render_spectrum_preview(
    spectrum: SpectrumObservation,
    plot_spec: SpectrumPlotSpec | None = None,
) -> PlotObservation:
    """Render a deterministic measured-spectrum PNG with Matplotlib's Agg canvas."""
    spec = plot_spec or SpectrumPlotSpec()
    if (
        isinstance(spec.width_pixels, bool)
        or isinstance(spec.height_pixels, bool)
        or isinstance(spec.dpi, bool)
        or not isinstance(spec.width_pixels, int)
        or not isinstance(spec.height_pixels, int)
        or not isinstance(spec.dpi, int)
        or spec.width_pixels <= 0
        or spec.height_pixels <= 0
        or spec.width_pixels > MAX_PLOT_WIDTH_PIXELS
        or spec.height_pixels > MAX_PLOT_HEIGHT_PIXELS
        or spec.width_pixels * spec.height_pixels > MAX_PLOT_PIXELS
        or spec.dpi <= 0
        or spec.dpi > MAX_PLOT_DPI
    ):
        raise ObservationValidationError("Plot dimensions or DPI exceed the bounds.")

    if (
        spectrum.status is not ObservationStatus.OK
        or spectrum.wavelengths is None
        or spectrum.intensities is None
    ):
        return PlotObservation(
            status=ObservationStatus.INVALID,
            dataset_id=spectrum.dataset_id,
            revision=spectrum.revision,
            png_bytes=None,
            mime_type="image/png",
            width_pixels=spec.width_pixels,
            height_pixels=spec.height_pixels,
            x_label="Wavelength",
            y_label="Intensity",
            plotted_point_count=0,
            display_window=None,
            error=ObservationError(
                "spectrum_unavailable", "No valid spectrum is available to plot."
            ),
        )

    wavelengths = np.asarray(spectrum.wavelengths)
    intensities = np.asarray(spectrum.intensities)
    if wavelengths.shape != intensities.shape:
        raise ObservationValidationError(
            "Wavelength and intensity arrays must have the same shape."
        )
    if wavelengths.size > MAX_SPECTRUM_POINTS:
        raise ObservationLimitExceeded(
            f"Spectrum has {wavelengths.size} points; the limit is "
            f"{MAX_SPECTRUM_POINTS}."
        )

    window = (
        spec.display_window
        if spec.display_window is not None
        else spectrum.display_window
    )
    if window is not None:
        window = _validated_wavelength_window(window, label="Plot wavelength")
        mask = (wavelengths >= window[0]) & (wavelengths <= window[1])
    else:
        mask = np.ones(wavelengths.shape, dtype=bool)
    mask &= np.isfinite(wavelengths) & np.isfinite(intensities)
    x_plot = wavelengths[mask]
    y_plot = intensities[mask]
    if x_plot.size == 0:
        return PlotObservation(
            status=ObservationStatus.INVALID,
            dataset_id=spectrum.dataset_id,
            revision=spectrum.revision,
            png_bytes=None,
            mime_type="image/png",
            width_pixels=spec.width_pixels,
            height_pixels=spec.height_pixels,
            x_label="Wavelength",
            y_label="Intensity",
            plotted_point_count=0,
            display_window=window,
            error=ObservationError(
                "empty_display_window",
                "No wavelength samples fall inside the requested plot window.",
            ),
        )

    wavelength_label = (
        "Wavelength (Å)"
        if spectrum.wavelength_unit == "angstrom"
        else f"Wavelength ({spectrum.wavelength_unit})"
    )
    intensity_label = (
        "Intensity"
        if spectrum.intensity_unit == "not_provided"
        else f"Intensity ({spectrum.intensity_unit})"
    )
    # Matplotlib uses process-wide font/configuration state. Serialize only
    # rendering; image and spectrum observations remain concurrent.
    with _PLOT_LOCK:
        figure = Figure(
            figsize=(spec.width_pixels / spec.dpi, spec.height_pixels / spec.dpi),
            dpi=spec.dpi,
            facecolor="white",
        )
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        axes.plot(x_plot, y_plot, color="#1f77b4", linewidth=1.0)
        axes.set_xlabel(wavelength_label)
        axes.set_ylabel(intensity_label)
        axes.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.7)
        if window is not None:
            axes.set_xlim(window)
        figure.tight_layout()

        buffer = BytesIO()
        canvas.print_png(
            buffer,
            metadata={"Software": "NEAT Scientific Observation API"},
        )
        png_bytes = buffer.getvalue()
        figure.clear()
    if len(png_bytes) > MAX_PLOT_PNG_BYTES:
        raise ObservationLimitExceeded(
            f"Rendered plot exceeds {MAX_PLOT_PNG_BYTES} bytes."
        )
    return PlotObservation(
        status=ObservationStatus.OK,
        dataset_id=spectrum.dataset_id,
        revision=spectrum.revision,
        png_bytes=png_bytes,
        mime_type="image/png",
        width_pixels=spec.width_pixels,
        height_pixels=spec.height_pixels,
        x_label=wavelength_label,
        y_label=intensity_label,
        plotted_point_count=int(x_plot.size),
        display_window=window,
    )

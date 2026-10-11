"""Minimal Qt-side adapter for coherent read-only observation snapshots."""

from __future__ import annotations

import os
import weakref
from types import EllipsisType

import numpy as np
from PyQt5.QtCore import QThread

from ..domain.observation import (
    DatasetMode,
    DatasetObservation,
    DatasetProvenance,
    DatasetRevisionClock,
    ImageObservation,
    ImageViewSpec,
    PlotObservation,
    RoiBounds,
    SpectrumObservation,
    SpectrumPlotSpec,
)
from ..services.scientific_observation import (
    DatasetUnavailableError,
    ScientificDatasetHandle,
    create_dataset_handle,
    get_image_preview,
    get_roi_spectrum,
    inspect_dataset,
    render_spectrum_preview,
)


class ObservationThreadError(RuntimeError):
    """Live GUI state may only be captured or invalidated on its owner thread."""


class ScientificObservationAdapter:
    """Capture current FitsViewer state without mutating its UI or fit state.

    Dataset arrays are referenced through read-only NumPy views; only the small
    wavelength axis and spectrum-only arrays are copied. Call invalidate_dataset
    immediately before app code replaces the loaded scientific data or wavelength
    axis. Contrast and image-selection changes do not alter the scientific revision.
    """

    def __init__(self, window):
        self._window_ref = weakref.ref(window)
        clock = getattr(window, "_scientific_observation_revision_clock", None)
        if not isinstance(clock, DatasetRevisionClock):
            clock = DatasetRevisionClock()
            window._scientific_observation_revision_clock = clock
        self._revision_clock = clock
        self._owner_thread = window.thread()
        self._assert_owner_thread(window)

    def _window(self):
        window = self._window_ref()
        if window is None:
            raise DatasetUnavailableError("The NEAT window is no longer available.")
        return window

    def _assert_owner_thread(self, window) -> None:
        if QThread.currentThread() != self._owner_thread:
            raise ObservationThreadError(
                "Capture and invalidation must run on the NEAT window's Qt owner thread."
            )
        if window.thread() != self._owner_thread:
            raise ObservationThreadError(
                "The NEAT window changed Qt thread ownership."
            )

    @property
    def revision_clock(self) -> DatasetRevisionClock:
        return self._revision_clock

    def invalidate_dataset(self, *, replace_dataset: bool = False) -> tuple[str, int]:
        """Mark the current scientific state stale before a data/axis mutation."""
        window = self._window()
        self._assert_owner_thread(window)
        return self._revision_clock.invalidate(replace_dataset=replace_dataset)

    @staticmethod
    def _loader_is_running(window) -> bool:
        for name in ("fits_image_load_worker", "image_load_worker"):
            worker = getattr(window, name, None)
            is_running = getattr(worker, "isRunning", None)
            if callable(is_running):
                try:
                    if is_running():
                        return True
                except RuntimeError:
                    continue
        return False

    @staticmethod
    def _display_window(window, flags):
        lower_widget = getattr(window, "min_wavelength_input", None)
        upper_widget = getattr(window, "max_wavelength_input", None)
        if lower_widget is None or upper_widget is None:
            return None
        try:
            lower = float(lower_widget.text())
            upper = float(upper_widget.text())
        except (TypeError, ValueError):
            flags.append("display_window_unavailable")
            return None
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            flags.append("display_window_invalid")
            return None
        return float(lower), float(upper)

    @staticmethod
    def _safe_input_format(window, mode: DatasetMode) -> str:
        source = str(getattr(window, "current_fitting_input", "") or "")
        extension = os.path.splitext(source)[1].lower()
        known = {
            ".fits": "fits",
            ".fit": "fits",
            ".fts": "fits",
            ".tif": "tiff",
            ".tiff": "tiff",
            ".nxs": "nexus",
            ".h5": "nexus",
            ".hdf5": "nexus",
            ".csv": "csv",
            ".txt": "text",
            ".xlsx": "xlsx",
        }
        if extension in known:
            return known[extension]
        if mode is DatasetMode.IMAGES and source:
            return "image_folder"
        return "unknown"

    @staticmethod
    def _wavelength_source(window, mode: DatasetMode) -> str:
        if mode is DatasetMode.PROFILE:
            return "imported_profile"
        if getattr(window, "manual_wavelength_mode", False):
            return "manual_anchors"
        if getattr(window, "nexus_axis_uses_flight_path", False):
            return "nexus_tof_axis"
        if getattr(window, "nexus_axis_centers", None) is not None:
            return "nexus_wavelength_axis"
        if getattr(window, "wavelength_depends_on_flight_path", False):
            return "raden_tof_axis"
        if getattr(window, "tof_array", None) is not None:
            return "spectra_tof_axis"
        return "unknown"

    @staticmethod
    def _flight_path_source(window) -> str:
        source = getattr(window, "flight_path_source", "")
        if source == "App setting":
            return "app_setting"
        if source == "NeXus file":
            return "nexus_metadata"
        return "unknown"

    def capture_dataset(self) -> ScientificDatasetHandle:
        """Capture state on the UI thread and reject loading or mixed revisions."""
        window = self._window()
        self._assert_owner_thread(window)
        if self._loader_is_running(window):
            raise DatasetUnavailableError(
                "A dataset loader is running; retry after it has completed."
            )

        revision_before = self._revision_clock.current()
        data_source = str(getattr(window, "fitting_data_source", "images"))
        if data_source == "profile":
            mode = DatasetMode.PROFILE
            images = None
        else:
            images = getattr(window, "images", None)
            if images is not None and len(images) > 0:
                mode = DatasetMode.IMAGES
            else:
                mode = DatasetMode.EMPTY
                images = None

        flags: list[str] = []
        wavelengths = getattr(window, "wavelengths", None)
        intensities = (
            getattr(window, "intensities", None)
            if mode is DatasetMode.PROFILE
            else None
        )
        if wavelengths is None or np.asarray(wavelengths).size == 0:
            flags.append("wavelength_axis_missing")
        if mode is DatasetMode.PROFILE and intensities is None:
            flags.append("profile_intensity_missing")
        display_window = self._display_window(window, flags)

        flight_path = getattr(window, "flight_path", None)
        try:
            flight_path = float(flight_path)
            if not np.isfinite(flight_path):
                flight_path = None
        except (TypeError, ValueError):
            flight_path = None

        delay = getattr(window, "delay", None)
        try:
            delay = float(delay)
            if not np.isfinite(delay):
                delay = None
        except (TypeError, ValueError):
            delay = None

        provenance = DatasetProvenance(
            input_format=self._safe_input_format(window, mode),
            wavelength_source=self._wavelength_source(window, mode),
            flight_path_m=flight_path,
            flight_path_source=self._flight_path_source(window),
            delay_ms=delay,
            wavelength_unit="angstrom",
            intensity_unit="not_provided",
            companion_metadata_status="unknown",
        )
        handle = create_dataset_handle(
            mode=mode,
            images=images,
            wavelengths=wavelengths,
            profile_intensities=intensities,
            display_window=display_window,
            provenance=provenance,
            metadata_flags=flags,
            revision_clock=self._revision_clock,
        )
        if self._revision_clock.current() != revision_before:
            raise DatasetUnavailableError(
                "The scientific dataset changed during snapshot capture."
            )
        return handle

    def inspect_dataset(
        self, handle: ScientificDatasetHandle | None = None
    ) -> DatasetObservation:
        return inspect_dataset(handle or self.capture_dataset())

    def get_image_preview(
        self,
        frame_index: int,
        view_spec: ImageViewSpec | None = None,
        *,
        handle: ScientificDatasetHandle | None = None,
    ) -> ImageObservation:
        return get_image_preview(handle or self.capture_dataset(), frame_index, view_spec)

    def get_roi_spectrum(
        self,
        roi: RoiBounds | None = None,
        *,
        display_window: tuple[float, float] | None | EllipsisType = ...,
        handle: ScientificDatasetHandle | None = None,
    ) -> SpectrumObservation:
        return get_roi_spectrum(
            handle or self.capture_dataset(),
            roi,
            display_window=display_window,
        )

    def render_spectrum_preview(
        self,
        spectrum: SpectrumObservation,
        plot_spec: SpectrumPlotSpec | None = None,
    ) -> PlotObservation:
        return render_spectrum_preview(spectrum, plot_spec)

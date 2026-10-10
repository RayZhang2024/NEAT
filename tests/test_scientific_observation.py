"""Coverage for the read-only scientific observation surface."""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from unittest.mock import patch

import numpy as np
import psutil
from PIL import Image
from PyQt5.QtCore import QObject

from NEAT.domain.observation import (
    DatasetMode,
    DatasetProvenance,
    DatasetRevisionClock,
    ImageViewSpec,
    ObservationStatus,
    RoiBounds,
    RoiClipPolicy,
    SpectrumPlotSpec,
    WavelengthAxisStatus,
)
from NEAT.services import scientific_observation as observation
from NEAT.services.scientific_observation import (
    DatasetUnavailableError,
    ObservationLimitExceeded,
    ObservationValidationError,
    StaleDatasetError,
    create_dataset_handle,
    get_image_preview,
    get_roi_spectrum,
    inspect_dataset,
    render_spectrum_preview,
)
from NEAT.ui.scientific_observation_adapter import (
    ObservationThreadError,
    ScientificObservationAdapter,
)


class TestScientificObservation(unittest.TestCase):
    def test_image_metadata_preview_orientation_and_png_consistency(self):
        frame = np.arange(30, dtype=np.uint8).reshape(5, 6)
        handle = create_dataset_handle(
            mode=DatasetMode.IMAGES,
            images=[frame],
            wavelengths=np.linspace(1.0, 3.0, 1),
            display_window=(1.2, 2.8),
            provenance=DatasetProvenance(input_format="fits"),
        )

        dataset = inspect_dataset(handle)
        preview = get_image_preview(
            handle, 0, ImageViewSpec(max_dimension=32, vmin=0, vmax=255)
        )

        self.assertEqual(dataset.mode, DatasetMode.IMAGES)
        self.assertEqual(dataset.image_shape, (5, 6))
        self.assertEqual(dataset.frame_count, 1)
        self.assertEqual(dataset.provenance.input_format, "fits")
        self.assertEqual(preview.preview_shape, (5, 6))
        self.assertEqual(preview.coordinate_mapping.origin, "upper")
        self.assertEqual(preview.coordinate_mapping.preview_pixel_to_original(4, 3), (4, 3))
        np.testing.assert_array_equal(preview.preview_pixels, frame)
        decoded = np.asarray(Image.open(BytesIO(preview.png_bytes)).convert("L"))
        np.testing.assert_array_equal(decoded, preview.preview_pixels)
        self.assertFalse(handle.frames[0].flags.writeable)
        with self.assertRaises(ValueError):
            handle.frames[0][0, 0] = 100
        self.assertEqual(frame[0, 0], 0)

    def test_downsampled_preview_mapping_and_maximum_resolution(self):
        frame = np.zeros((2048, 2048), dtype=np.uint8)
        frame[1, 1] = 200
        handle = create_dataset_handle(
            mode="images", images=[frame], wavelengths=[1.0]
        )
        preview = get_image_preview(
            handle, 0, ImageViewSpec(max_dimension=1024, vmin=0, vmax=255)
        )
        self.assertEqual(preview.preview_shape, (1024, 1024))
        self.assertEqual(
            preview.coordinate_mapping.preview_pixel_to_original(0, 0), (1, 1)
        )
        self.assertEqual(int(preview.preview_pixels[0, 0]), 200)
        with self.assertRaises(ObservationValidationError):
            get_image_preview(handle, 0, ImageViewSpec(max_dimension=1025))

    def test_roi_spectrum_matches_legacy_per_frame_sum_and_area_division(self):
        images = [
            np.arange(36, dtype=np.uint16).reshape(6, 6),
            np.arange(36, dtype=np.float32).reshape(6, 6) * 0.25,
            np.full((6, 6), np.nan, dtype=np.float64),
            np.full((6, 6), np.inf, dtype=np.float64),
        ]
        images[2][2, 1:4] = 3.0
        roi = RoiBounds(1, 4, 2, 4)
        handle = create_dataset_handle(
            mode="images",
            images=images,
            wavelengths=[1.0, 2.0, 3.0, 4.0],
            display_window=(1.5, 3.5),
        )

        actual = get_roi_spectrum(handle, roi)
        legacy_sums = []
        for image in images:
            selected_area = image[roi.y_min:roi.y_max, roi.x_min:roi.x_max]
            legacy_sums.append(np.sum(selected_area))
        legacy = np.array(legacy_sums) / [roi.width * roi.height]

        np.testing.assert_equal(actual.intensities, legacy)
        np.testing.assert_array_equal(actual.wavelengths, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(actual.display_mask, [False, True, True, False])
        self.assertEqual(actual.sample_count, 4)
        self.assertEqual(actual.finite_pair_count, 2)
        self.assertEqual(actual.nonfinite_intensity_count, 2)
        self.assertFalse(actual.intensities.flags.writeable)
        self.assertFalse(actual.wavelengths.flags.writeable)
        self.assertEqual(actual.roi_effective, roi)

    def test_roi_rejects_or_explicitly_clips_out_of_bounds(self):
        handle = create_dataset_handle(
            mode="images", images=[np.ones((4, 5))], wavelengths=[1.0]
        )
        with self.assertRaises(ObservationValidationError):
            get_roi_spectrum(handle, RoiBounds(-1, 3, 0, 2))
        clipped = get_roi_spectrum(
            handle,
            RoiBounds(-1, 3, 0, 2, clip_policy=RoiClipPolicy.CLIP),
        )
        self.assertEqual(clipped.roi_requested.x_min, -1)
        self.assertEqual(clipped.roi_effective, RoiBounds(0, 3, 0, 2))
        np.testing.assert_array_equal(clipped.intensities, [1.0])

    def test_spectrum_plot_is_bounded_and_rendered(self):
        handle = create_dataset_handle(
            mode="profile",
            wavelengths=np.linspace(1.0, 5.0, 20),
            profile_intensities=np.sin(np.linspace(0.0, 3.0, 20)),
        )
        spectrum = get_roi_spectrum(handle)
        plot = render_spectrum_preview(
            spectrum,
            SpectrumPlotSpec(width_pixels=640, height_pixels=360, dpi=100),
        )
        self.assertEqual(plot.status, ObservationStatus.OK)
        self.assertEqual(plot.plotted_point_count, 20)
        self.assertEqual(plot.mime_type, "image/png")
        self.assertEqual(Image.open(BytesIO(plot.png_bytes)).size, (640, 360))
        with self.assertRaises(ObservationValidationError):
            render_spectrum_preview(
                spectrum,
                SpectrumPlotSpec(width_pixels=1601, height_pixels=450),
            )

    def test_profile_only_dataset_has_no_roi_or_image_preview(self):
        handle = create_dataset_handle(
            mode="profile",
            wavelengths=[1.0, 2.0, 3.0],
            profile_intensities=[10.0, 20.0, 15.0],
        )
        dataset = inspect_dataset(handle)
        spectrum = get_roi_spectrum(handle)
        preview = get_image_preview(handle, 0)

        self.assertTrue(dataset.capabilities.spectrum_available)
        self.assertFalse(dataset.capabilities.images_available)
        self.assertFalse(dataset.capabilities.mapping_applicable)
        self.assertEqual(preview.status, ObservationStatus.NOT_APPLICABLE)
        np.testing.assert_array_equal(spectrum.intensities, [10.0, 20.0, 15.0])
        with self.assertRaises(ObservationValidationError):
            get_roi_spectrum(handle, RoiBounds(0, 1, 0, 1))

    def test_invalid_axis_is_reported_without_silently_reordering_samples(self):
        handle = create_dataset_handle(
            mode="profile",
            wavelengths=[1.0, 3.0, 2.0],
            profile_intensities=[10.0, 30.0, 20.0],
        )
        dataset = inspect_dataset(handle)
        spectrum = get_roi_spectrum(handle)
        self.assertEqual(
            dataset.wavelength_axis_status, WavelengthAxisStatus.NON_MONOTONIC
        )
        self.assertEqual(spectrum.wavelength_axis_status, WavelengthAxisStatus.NON_MONOTONIC)
        np.testing.assert_array_equal(spectrum.wavelengths, [1.0, 3.0, 2.0])

        image_handle = create_dataset_handle(
            mode="images",
            images=[np.ones((2, 2)) for _ in range(3)],
            wavelengths=[1.0, np.nan, 2.0],
        )
        image_dataset = inspect_dataset(image_handle)
        self.assertTrue(image_dataset.capabilities.spectrum_available)
        self.assertFalse(image_dataset.capabilities.mapping_applicable)

    def test_revision_clock_invalidates_old_handles_and_replaces_identity(self):
        clock = DatasetRevisionClock()
        handle = create_dataset_handle(
            mode="images",
            images=[np.ones((2, 3))],
            wavelengths=[1.0],
            revision_clock=clock,
        )
        previous = clock.current()
        clock.invalidate(replace_dataset=True)
        current = clock.current()
        self.assertNotEqual(previous[0], current[0])
        self.assertEqual(current[1], previous[1] + 1)
        with self.assertRaises(StaleDatasetError):
            inspect_dataset(handle)

    def test_provenance_allowlist_prevents_path_and_untrusted_text_leaks(self):
        handle = create_dataset_handle(
            mode="profile",
            wavelengths=[1.0],
            profile_intensities=[2.0],
            provenance=DatasetProvenance(
                input_format=r"C:\private\secret.fits",
                wavelength_source=r"C:\private\calibration.csv",
                wavelength_unit="private user label",
            ),
            metadata_flags=[r"C:\private\custom flag"],
        )
        dataset = inspect_dataset(handle)
        self.assertEqual(dataset.provenance.input_format, "unknown")
        self.assertEqual(dataset.provenance.wavelength_source, "unknown")
        self.assertEqual(dataset.provenance.wavelength_unit, "unknown")
        self.assertIn("unknown_metadata_flag", dataset.metadata_flags)
        self.assertNotIn("private", repr(dataset))

    def test_roi_pixel_operation_limit_rejects_expensive_request(self):
        base = np.zeros((1, 1, 1), dtype=np.uint8)
        logical_stack = np.broadcast_to(base, (2, 11200, 11200))
        handle = create_dataset_handle(
            mode="images", images=logical_stack, wavelengths=[1.0, 2.0]
        )
        with self.assertRaises(ObservationLimitExceeded):
            get_roi_spectrum(handle, RoiBounds(0, 11200, 0, 11200))

    def test_invalidation_during_roi_read_rejects_mixed_revision_result(self):
        clock = DatasetRevisionClock()
        handle = create_dataset_handle(
            mode="images",
            images=[np.ones((10, 10)) for _ in range(4)],
            wavelengths=np.arange(4.0),
            revision_clock=clock,
        )
        started = threading.Event()
        resume = threading.Event()
        original_sum = np.sum
        first_call = True

        def pause_sum(*args, **kwargs):
            nonlocal first_call
            if first_call:
                first_call = False
                started.set()
                self.assertTrue(resume.wait(5))
            return original_sum(*args, **kwargs)

        with (
            patch.object(observation.np, "sum", side_effect=pause_sum),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            future = pool.submit(
                get_roi_spectrum, handle, RoiBounds(0, 3, 0, 3)
            )
            self.assertTrue(started.wait(5))
            clock.invalidate(replace_dataset=True)
            resume.set()
            with self.assertRaises(StaleDatasetError):
                future.result(timeout=5)

    def test_large_stack_uses_views_and_bounded_preview_memory(self):
        base = np.zeros((2048, 2048), dtype=np.uint8)
        image_stack = np.broadcast_to(base, (512, 2048, 2048))
        wavelengths = np.arange(512, dtype=float)
        process = psutil.Process()
        before = process.memory_info().rss

        handle = create_dataset_handle(
            mode="images", images=image_stack, wavelengths=wavelengths
        )
        self.assertTrue(np.shares_memory(handle.frames[0], image_stack))
        preview = get_image_preview(handle, 400)
        spectrum = get_roi_spectrum(handle, RoiBounds(100, 116, 200, 216))
        after = process.memory_info().rss

        self.assertLessEqual(max(preview.preview_shape), 1024)
        self.assertLessEqual(preview.preview_pixels.size, 1024 * 1024)
        self.assertEqual(spectrum.sample_count, 512)
        # The logical input stack is 2 GiB; the API retains frame views and
        # allocates only one bounded preview plus the requested ROI profile.
        self.assertLess(after - before, 64 * 1024 * 1024)


class TestScientificObservationAdapter(unittest.TestCase):
    class Window(QObject):
        def __init__(self):
            super().__init__()
            self.images = [np.arange(12, dtype=np.uint8).reshape(3, 4)]
            self.wavelengths = np.array([1.0])
            self.intensities = []
            self.fitting_data_source = "images"
            self.current_fitting_input = r"C:\private\sample.fits"
            self.flight_path = 56.4
            self.flight_path_source = "App setting"
            self.delay = 0.0
            self.min_wavelength_input = self.Value("0.5")
            self.max_wavelength_input = self.Value("2.5")
            self.fits_image_load_worker = None

        class Value:
            def __init__(self, value):
                self._value = value

            def text(self):
                return self._value

    def test_capture_is_privacy_safe_and_uses_read_only_views(self):
        window = self.Window()
        adapter = ScientificObservationAdapter(window)
        handle = adapter.capture_dataset()
        details = adapter.inspect_dataset(handle)

        self.assertEqual(handle.mode, DatasetMode.IMAGES)
        self.assertEqual(details.provenance.input_format, "fits")
        self.assertEqual(details.display_window, (0.5, 2.5))
        self.assertNotIn("private", repr(details))
        self.assertNotIn("sample.fits", repr(details))
        self.assertFalse(handle.frames[0].flags.writeable)

    def test_capture_rejects_loading_and_non_owner_threads(self):
        window = self.Window()
        adapter = ScientificObservationAdapter(window)

        class RunningWorker:
            @staticmethod
            def isRunning():
                return True

        window.fits_image_load_worker = RunningWorker()
        with self.assertRaises(DatasetUnavailableError):
            adapter.capture_dataset()
        window.fits_image_load_worker = None

        errors = []

        def call_from_worker():
            try:
                adapter.capture_dataset()
            except ObservationThreadError as exc:
                errors.append(exc)

        worker = threading.Thread(target=call_from_worker)
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ObservationThreadError)

    def test_adapter_invalidation_makes_previous_snapshot_stale(self):
        window = self.Window()
        adapter = ScientificObservationAdapter(window)
        handle = adapter.capture_dataset()
        adapter.invalidate_dataset(replace_dataset=True)
        with self.assertRaises(StaleDatasetError):
            adapter.inspect_dataset(handle)

    def test_profile_mode_captures_imported_arrays_without_paths(self):
        window = self.Window()
        window.fitting_data_source = "profile"
        window.images = []
        window.wavelengths = np.array([1.0, 2.0, 3.0])
        window.intensities = np.array([3.0, 2.0, 1.0])
        window.current_fitting_input = r"C:\private\profile.csv"
        adapter = ScientificObservationAdapter(window)

        handle = adapter.capture_dataset()
        dataset = adapter.inspect_dataset(handle)
        self.assertEqual(handle.mode, DatasetMode.PROFILE)
        self.assertTrue(dataset.capabilities.spectrum_available)
        self.assertFalse(dataset.capabilities.mapping_applicable)
        self.assertNotIn("private", repr(dataset))
        self.assertFalse(np.shares_memory(handle.wavelengths, window.wavelengths))
        self.assertFalse(
            np.shares_memory(handle.profile_intensities, window.intensities)
        )
        self.assertFalse(handle.profile_intensities.flags.writeable)


if __name__ == "__main__":
    unittest.main()

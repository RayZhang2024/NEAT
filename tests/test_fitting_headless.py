import unittest
import tempfile
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import h5py
from PIL import Image

from NEAT.core import fitting_function_3
from NEAT.core.fitting import (
    DEFAULT_FITTING_PARAMETER_BOUNDS,
    initial_value_within_bounds,
    normalize_fitting_parameter_bounds,
)
from NEAT.domain import FullPatternFitConfig
from NEAT.ui.mixins.fitting import FittingMixin
from NEAT.workers.batch import (
    BatchFitEdgesWorker,
    get_nexus_image_stack_info,
    get_raden_tiff_stack_info,
    load_nexus_image_stack,
    load_raden_tiff_stack,
    parse_raden_stat_file,
)
from NEAT.workers.preprocessing import RadenNormalisationWorker


class _DummyItem:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _DummyTable:
    def __init__(self, rows):
        self._rows = rows

    def rowCount(self):
        return len(self._rows)

    def columnCount(self):
        return max((len(row) for row in self._rows), default=0)

    def item(self, row, col):
        if row < 0 or row >= len(self._rows):
            return None
        current_row = self._rows[row]
        if col < 0 or col >= len(current_row):
            return None
        value = current_row[col]
        if value is None:
            return None
        return _DummyItem(str(value))


class _DummyTextInput:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text

    def setText(self, text):
        self._text = str(text)


class _DummyMessageBox:
    def __init__(self):
        self.messages = []

    def append(self, text):
        self.messages.append(str(text))


class _DummyDropdown:
    def __init__(self, value):
        self._value = value

    def currentText(self):
        return self._value


class _DummyCanvas:
    def __init__(self):
        self.axes = _DummyAxes()
        self.drawn = False

    def draw_idle(self):
        self.drawn = True


class _DummyAxes:
    def __init__(self):
        self._xlim = (0.0, 1.0)
        self._ylim = (1.0, 0.0)
        self.bbox = _DummyBbox()

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def set_xlim(self, *limits):
        self._xlim = tuple(limits)

    def set_ylim(self, *limits):
        self._ylim = tuple(limits)


class _DummyMouseEvent:
    def __init__(self, axes, xdata, ydata, key=None, x=0.0, y=0.0):
        self.button = 1
        self.inaxes = axes
        self.xdata = xdata
        self.ydata = ydata
        self.key = key
        self.x = x
        self.y = y


class _DummyBbox:
    width = 100.0
    height = 100.0


class _HeadlessFitting(FittingMixin):
    pass


class TestFittingHeadless(unittest.TestCase):
    def test_fitting_parameter_bounds_defaults_and_custom_values(self):
        self.assertEqual(
            normalize_fitting_parameter_bounds(),
            DEFAULT_FITTING_PARAMETER_BOUNDS,
        )
        self.assertEqual(
            normalize_fitting_parameter_bounds(
                {
                    "s": {"lower": 0.002, "upper": 0.004},
                    "t": (0.03, 0.06),
                    "eta": (0.2, 0.7),
                }
            ),
            {"s": (0.002, 0.004), "t": (0.03, 0.06), "eta": (0.2, 0.7)},
        )

    def test_fitting_parameter_bounds_reject_invalid_values(self):
        invalid_bounds = (
            {"s": (0.01, 0.01)},
            {"t": (0.0, 0.1)},
            {"eta": (-0.1, 0.5)},
            {"eta": (0.2, 1.1)},
        )
        for bounds in invalid_bounds:
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    normalize_fitting_parameter_bounds(bounds)

    def test_initial_fitting_values_are_clipped_to_configured_bounds(self):
        self.assertEqual(initial_value_within_bounds(-1.0, (0.002, 0.004)), 0.002)
        self.assertEqual(initial_value_within_bounds(1.0, (0.002, 0.004)), 0.004)
        self.assertEqual(initial_value_within_bounds(float("nan"), (0.2, 0.6)), 0.4)

    def test_fit_region_applies_custom_fitting_parameter_bounds(self):
        obj = _HeadlessFitting()
        wavelengths = np.linspace(1.0, 2.2, 1600)
        hkl = (1, 1, 0)
        a_lattice = 1.2
        regions = ((1.2, 1.35), (1.4, 1.55), (1.55, 1.95))
        intensities = fitting_function_3(
            wavelengths,
            0.3,
            0.2,
            0.15,
            0.07,
            0.006,
            0.05,
            0.35,
            [hkl],
            *regions[2],
            "bcc",
            {"a": a_lattice},
        )
        row_data = {
            "hkl": hkl,
            "d": None,
            "regions": [
                {"min_wavelength": regions[1][0], "max_wavelength": regions[1][1]},
                {"min_wavelength": regions[0][0], "max_wavelength": regions[0][1]},
                {"min_wavelength": regions[2][0], "max_wavelength": regions[2][1]},
            ],
            "s": 0.006,
            "t": 0.05,
            "eta": 0.35,
        }
        configured_bounds = {
            "s": (0.002, 0.004),
            "t": (0.03, 0.04),
            "eta": (0.2, 0.3),
        }
        calls = []

        def capture_curve_fit(function, x_values, y_values, p0, bounds=None, **kwargs):
            calls.append((list(p0), bounds))
            p0_array = np.asarray(p0, dtype=float)
            return p0_array, np.eye(len(p0_array))

        with patch("NEAT.ui.mixins.fitting.curve_fit", side_effect=capture_curve_fit):
            result = obj.fit_region(
                0,
                skip_ui_updates=True,
                row_data=row_data,
                wavelengths=wavelengths,
                intensities=intensities,
                fit_flags=(False, False, False),
                selected_phase="Fe_bcc",
                structure_type="bcc",
                lattice_params={"a": a_lattice},
                fitting_parameter_bounds=configured_bounds,
            )

        self.assertIsInstance(result, dict)
        initial_values, (lower, upper) = calls[-1]
        self.assertEqual(lower[-3:], [0.002, 0.03, 0.2])
        self.assertEqual(upper[-3:], [0.004, 0.04, 0.3])
        self.assertEqual(initial_values[-3:], [0.004, 0.04, 0.3])

    def test_manual_wavelength_anchors_interpolate_and_extend_to_stack_edges(self):
        obj = _HeadlessFitting()
        obj.images = [object()] * 5
        obj.message_box = _DummyMessageBox()
        obj.manual_anchor_mode = "wavelength"
        obj.manual_wavelength_anchors = [
            {"index": 2, "value": 1.0},
            {"index": 4, "value": 3.0},
        ]
        obj.flight_path = 10.0
        obj.delay = 0.0

        obj.update_manual_wavelengths()

        np.testing.assert_allclose(obj.wavelengths, [1.0, 1.0, 2.0, 3.0, 3.0])
        self.assertEqual(obj.start_wavelength, 1.0)
        self.assertEqual(obj.end_wavelength, 3.0)

    def test_manual_tof_anchor_uses_milliseconds_flight_path_and_delay(self):
        obj = _HeadlessFitting()
        obj.images = [object()] * 3
        obj.message_box = _DummyMessageBox()
        obj.manual_anchor_mode = "tof"
        obj.manual_wavelength_anchors = [{"index": 1, "value": 1.0}]
        obj.flight_path = 10.0
        obj.delay = 0.5

        obj.update_manual_wavelengths()

        expected = (1.0 + 0.5) * 3.956 / 10.0 * 1000.0
        np.testing.assert_allclose(obj.wavelengths, np.full(3, expected))

    def test_default_bragg_edge_windows_use_current_formula_without_neighbors(self):
        windows = _HeadlessFitting._default_bragg_edge_windows(2.0)

        self.assertAlmostEqual(windows["lower_min"], 1.8)
        self.assertAlmostEqual(windows["lower_max"], 1.96)
        self.assertAlmostEqual(windows["upper_min"], 2.08)
        self.assertAlmostEqual(windows["upper_max"], 2.24)
        self.assertTrue(windows["valid"])

    def test_default_bragg_edge_windows_clamp_to_adjacent_midpoints(self):
        windows = _HeadlessFitting._default_bragg_edge_windows(
            2.0,
            lower_midpoint=1.90,
            upper_midpoint=2.15,
        )

        self.assertAlmostEqual(windows["lower_min"], 1.90)
        self.assertAlmostEqual(windows["lower_max"], 1.96)
        self.assertAlmostEqual(windows["upper_min"], 2.08)
        self.assertAlmostEqual(windows["upper_max"], 2.15)
        self.assertTrue(windows["valid"])

    def test_default_bragg_edge_windows_mark_too_close_neighbors_invalid(self):
        windows = _HeadlessFitting._default_bragg_edge_windows(
            2.0,
            upper_midpoint=2.05,
        )

        self.assertAlmostEqual(windows["upper_min"], 2.08)
        self.assertAlmostEqual(windows["upper_max"], 2.05)
        self.assertFalse(windows["valid"])

    def test_build_batch_fit_context_parses_valid_row(self):
        obj = _HeadlessFitting()
        obj.bragg_table = _DummyTable(
            [
                [
                    "(1, 1, 0)",
                    "1.697",
                    "1.20",
                    "1.35",
                    "1.40",
                    "1.95",
                    "9.99",  # hidden Region 3 min is derived from column 2
                    "9.99",  # hidden Region 3 max is derived from column 5
                    "0.006",
                    "0.05",
                    "0.35",
                ]
            ]
        )
        obj.phase_dropdown = _DummyDropdown("Fe_bcc")
        obj.min_wavelength_input = _DummyTextInput("1.0")
        obj.max_wavelength_input = _DummyTextInput("2.0")
        obj.structure_type = "bcc"
        obj.lattice_params = {"a": 1.2}
        obj.flight_path = 16.0

        ctx = obj._build_batch_fit_context()
        self.assertEqual(ctx["structure_type"], "bcc")
        self.assertEqual(ctx["selected_phase"], "Fe_bcc")
        self.assertEqual(ctx["bragg_rows"][0]["hkl"], (1, 1, 0))
        self.assertTrue(ctx["bragg_rows"][0]["valid"])
        self.assertEqual(ctx["bragg_rows"][0]["regions"][2]["min_wavelength"], 1.20)
        self.assertEqual(ctx["bragg_rows"][0]["regions"][2]["max_wavelength"], 1.95)
        self.assertIn("(1; 1; 0)", ctx["bragg_rows_text"][0])
        with tempfile.TemporaryDirectory() as directory:
            worker = BatchFitEdgesWorker(
                parent=object(), images=[np.ones((1, 1))],
                wavelengths=np.array([1.2]), fit_context=ctx,
                min_x=0, max_x=1, min_y=0, max_y=1,
                box_width=1, box_height=1, step_x=1, step_y=1,
                total_boxes=1, interpolation_enabled=False,
                work_directory=directory,
            )
            self.assertEqual(worker.num_edges, 1)
            self.assertIs(worker.valid_row_configs[0], ctx["bragg_rows"][0])

    def test_build_batch_fit_context_marks_invalid_bounds(self):
        obj = _HeadlessFitting()
        obj.bragg_table = _DummyTable(
            [
                [
                    "(1, 1, 0)",
                    "1.697",
                    "1.40",
                    "1.55",
                    "1.35",
                    "1.20",  # invalid: min >= max
                    "1.55",
                    "1.95",
                    "0.006",
                    "0.05",
                    "0.35",
                ]
            ]
        )
        obj.phase_dropdown = _DummyDropdown("Fe_bcc")
        obj.min_wavelength_input = _DummyTextInput("1.0")
        obj.max_wavelength_input = _DummyTextInput("2.0")
        obj.structure_type = "bcc"
        obj.lattice_params = {"a": 1.2}

        ctx = obj._build_batch_fit_context()
        self.assertFalse(ctx["bragg_rows"][0]["valid"])
        self.assertIsNone(ctx["bragg_rows"][0]["regions"])

    def test_read_intensity_profile_file_accepts_csv_header(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
            handle.write("Wavelength,Summed Intensity\n")
            handle.write("2.0,10\n")
            handle.write("1.5,8\n")
            handle.write("2.5,12\n")
            file_name = handle.name
        try:
            wavelengths, intensities = FittingMixin._read_intensity_profile_file(file_name)
        finally:
            try:
                os.unlink(file_name)
            except OSError:
                pass

        np.testing.assert_allclose(wavelengths, [1.5, 2.0, 2.5])
        np.testing.assert_allclose(intensities, [8.0, 10.0, 12.0])

    def test_read_intensity_profile_file_accepts_whitespace_without_header(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("2.0 10\n")
            handle.write("1.5 8\n")
            handle.write("2.5 12\n")
            file_name = handle.name
        try:
            wavelengths, intensities = FittingMixin._read_intensity_profile_file(file_name)
        finally:
            try:
                os.unlink(file_name)
            except OSError:
                pass

        np.testing.assert_allclose(wavelengths, [1.5, 2.0, 2.5])
        np.testing.assert_allclose(intensities, [8.0, 10.0, 12.0])

    def test_read_intensity_profile_file_accepts_xlsx_header(self):
        fd, file_name = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)
        try:
            pd.DataFrame(
                [["wavelength", "intensity"], [2.0, 10], [1.5, 8], [2.5, 12]]
            ).to_excel(file_name, header=False, index=False)
            wavelengths, intensities = FittingMixin._read_intensity_profile_file(file_name)
        finally:
            try:
                os.unlink(file_name)
            except OSError:
                pass

        np.testing.assert_allclose(wavelengths, [1.5, 2.0, 2.5])
        np.testing.assert_allclose(intensities, [8.0, 10.0, 12.0])

    def test_read_intensity_profile_file_accepts_mantid_nexus_table(self):
        fd, file_name = tempfile.mkstemp(suffix=".nxs")
        os.close(fd)
        try:
            with h5py.File(file_name, "w") as handle:
                entry = handle.create_group("mantid_workspace_1")
                table = entry.create_group("table_workspace")
                x = table.create_dataset("column_1", data=np.array([2.0, 1.5, 2.5]))
                x.attrs["name"] = "XS0"
                y = table.create_dataset("column_2", data=np.array([10.0, 8.0, 12.0]))
                y.attrs["name"] = "YS0"
                e = table.create_dataset("column_3", data=np.array([0.1, 0.1, 0.1]))
                e.attrs["name"] = "ES0"
            wavelengths, intensities = FittingMixin._read_intensity_profile_file(file_name)
        finally:
            try:
                os.unlink(file_name)
            except OSError:
                pass

        np.testing.assert_allclose(wavelengths, [1.5, 2.0, 2.5])
        np.testing.assert_allclose(intensities, [8.0, 10.0, 12.0])

    def test_load_nexus_image_stack_uses_file_geometry_and_tof_edges(self):
        fd, file_name = tempfile.mkstemp(suffix=".nxs")
        os.close(fd)
        try:
            with h5py.File(file_name, "w") as handle:
                entry = handle.create_group("mantid_workspace_1")
                instrument = entry.create_group("instrument")
                parameter_map = instrument.create_group("instrument_parameter_map")
                parameter_map.create_dataset(
                    "data",
                    data=np.array([b"NGEM/source;V3D;pos;[0,0,-10.0];visible:true"]),
                )
                physical_detectors = instrument.create_group("physical_detectors")
                physical_detectors.create_dataset("distance", data=np.full(4, 2.0))
                workspace = entry.create_group("workspace")
                values = workspace.create_dataset(
                    "values",
                    data=np.array(
                        [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0],
                        ]
                    ),
                )
                values.attrs["signal"] = 1
                axis = workspace.create_dataset("axis1", data=np.array([1000.0, 2000.0, 3000.0, 4000.0]))
                axis.attrs["units"] = "TOF"

            info = get_nexus_image_stack_info(file_name)
            self.assertAlmostEqual(info["file_flight_path"], 12.0)
            images, wavelengths, stack_info = load_nexus_image_stack(file_name, info["file_flight_path"])
        finally:
            try:
                os.unlink(file_name)
            except OSError:
                pass

        expected_wavelengths = np.array([1500.0, 2500.0, 3500.0]) * 3.956 / 12.0 / 1000.0
        np.testing.assert_allclose(wavelengths, expected_wavelengths)
        np.testing.assert_allclose(stack_info["axis_centers"], [1500.0, 2500.0, 3500.0])
        self.assertTrue(stack_info["axis_uses_flight_path"])
        self.assertEqual(len(images), 3)
        self.assertEqual(images[0].shape, (2, 2))
        np.testing.assert_allclose(images[0], [[1.0, 4.0], [7.0, 10.0]])

    def test_nexus_wavelengths_recalculate_when_flight_path_changes(self):
        obj = _HeadlessFitting()
        obj.flight_path = 12.0
        obj.flight_path_source = "NeXus file"
        obj.delay = 0.0
        obj.fitting_data_source = "images"
        obj.images = [np.zeros((2, 2)) for _ in range(3)]
        obj.tof_array = None
        obj.tof_axis_centers_us = np.array([1500.0, 2500.0, 3500.0])
        obj.wavelength_depends_on_flight_path = True
        obj.nexus_axis_centers = obj.tof_axis_centers_us
        obj.nexus_axis_uses_flight_path = True
        obj.min_wavelength_input = _DummyTextInput("")
        obj.max_wavelength_input = _DummyTextInput("")
        obj.message_box = _DummyMessageBox()
        obj.update_plots = lambda: None

        self.assertTrue(obj._recalculate_nexus_wavelengths_from_axis(source_label="NeXus file", show_message=False))
        first = obj.wavelengths.copy()

        obj.flight_path = 24.0
        self.assertTrue(obj._apply_instrument_settings_to_loaded_data())

        np.testing.assert_allclose(obj.wavelengths, first / 2.0)
        self.assertEqual(obj.flight_path_source, "App setting")
        self.assertAlmostEqual(obj.wavelength_flight_path, 24.0)
        self.assertEqual(obj.min_wavelength_input.text(), f"{obj.wavelengths[0]:.6g}")

    def test_parse_raden_stat_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".stat", delete=False) as handle:
            handle.write("Version number: 1.4.3\n")
            handle.write("Axes:\n")
            handle.write(" X: Nbins=4, Min=0, Max=8, Bin size=2, Units=mm\n")
            handle.write(" Y: Nbins=3, Min=1, Max=7, Bin size=2, Units=mm\n")
            handle.write(" TOF: Nbins=5, Min=0, Max=10, Bin size=2, Units=ms\n")
            file_name = handle.name
        try:
            axes = parse_raden_stat_file(file_name)
        finally:
            try:
                os.unlink(file_name)
            except OSError:
                pass

        self.assertEqual(axes["x"]["bins"], 4)
        self.assertEqual(axes["y"]["min"], 1.0)
        self.assertEqual(axes["tof"]["units"], "ms")

    def test_load_raden_tiff_stack_uses_stat_tof_axis(self):
        with tempfile.TemporaryDirectory() as folder:
            tiff_path = os.path.join(folder, "NID000001_20210519.tiff")
            stat_path = os.path.join(folder, "NID000001_20210519.stat")
            frames = [
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
                np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
            ]
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(tiff_path, save_all=True, append_images=pil_frames[1:])
            with open(stat_path, "w", encoding="utf-8") as handle:
                handle.write("Version number: 1.4.3\n")
                handle.write("Axes:\n")
                handle.write(" X: Nbins=2, Min=0, Max=2, Bin size=1, Units=mm\n")
                handle.write(" Y: Nbins=2, Min=0, Max=2, Bin size=1, Units=mm\n")
                handle.write(" TOF: Nbins=3, Min=0, Max=3, Bin size=1, Units=ms\n")

            info = get_raden_tiff_stack_info(folder)
            images, wavelengths, stack_info = load_raden_tiff_stack(folder, 10.0)

        self.assertEqual(info["n_frames"], 3)
        self.assertEqual(info["image_shape"], (2, 2))
        np.testing.assert_allclose(info["tof_axis_centers_us"], [500.0, 1500.0, 2500.0])
        np.testing.assert_allclose(wavelengths, np.array([500.0, 1500.0, 2500.0]) * 3.956 / 10.0 / 1000.0)
        self.assertEqual(len(images), 3)
        np.testing.assert_allclose(images[0], np.flipud(frames[0]))
        self.assertTrue(stack_info["axis_uses_flight_path"])

    def test_raden_normalisation_worker_writes_multipage_tiff(self):
        def write_raden_stack(folder, stem, values, pulses):
            os.makedirs(folder, exist_ok=True)
            tiff_path = os.path.join(folder, stem + ".tiff")
            stat_path = os.path.join(folder, stem + ".stat")
            frames = [np.full((5, 5), value, dtype=np.float32) for value in values]
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(tiff_path, save_all=True, append_images=pil_frames[1:])
            with open(stat_path, "w", encoding="utf-8") as handle:
                handle.write("Version number: 1.4.3\n")
                handle.write("Axes:\n")
                handle.write(" X: Nbins=5, Min=0, Max=5, Bin size=1, Units=mm\n")
                handle.write(" Y: Nbins=5, Min=0, Max=5, Bin size=1, Units=mm\n")
                handle.write(f" TOF: Nbins={len(values)}, Min=0, Max={len(values)}, Bin size=1, Units=ms\n")
                handle.write("Entries:100\n")
                handle.write(f"Pulses:{pulses}\n")
                handle.write(f"Pulses with data:{pulses}\n")
            return get_raden_tiff_stack_info(folder)

        with tempfile.TemporaryDirectory() as root:
            sample_folder = os.path.join(root, "sample")
            open_beam_folder = os.path.join(root, "open_beam")
            output_folder = os.path.join(root, "out")
            sample_info = write_raden_stack(sample_folder, "sample_stack", [10.0, 20.0, 30.0], 2)
            open_beam_info = write_raden_stack(open_beam_folder, "open_beam_stack", [20.0, 40.0, 60.0], 4)

            worker = RadenNormalisationWorker(
                {"folder_path": sample_folder, "kind": "raden_tiff_stack", "info": sample_info},
                {"folder_path": open_beam_folder, "kind": "raden_tiff_stack", "info": open_beam_info},
                output_folder,
                "normalised",
                window_half=0,
                adjacent_sum=0,
            )
            worker.run()

            output_tiff = os.path.join(output_folder, "normalised_sample_stack.tiff")
            self.assertTrue(os.path.exists(output_tiff))
            self.assertTrue(os.path.exists(os.path.join(output_folder, "normalised_sample_stack.stat")))
            with Image.open(output_tiff) as image:
                self.assertEqual(image.n_frames, 3)
                for idx in range(3):
                    image.seek(idx)
                    np.testing.assert_allclose(np.array(image), np.ones((5, 5), dtype=np.float32))

    def test_fit_region_headless_known_phase(self):
        obj = _HeadlessFitting()

        wavelengths = np.linspace(1.0, 2.2, 1600)
        hkl = (1, 1, 0)
        a_lattice = 1.2
        a0, b0 = 0.3, 0.2
        a_hkl, b_hkl = 0.15, 0.07
        s_val, t_val, eta_val = 0.006, 0.05, 0.35
        r1 = (1.2, 1.35)
        r2 = (1.4, 1.55)
        r3 = (1.55, 1.95)
        intensities = fitting_function_3(
            wavelengths,
            a0,
            b0,
            a_hkl,
            b_hkl,
            s_val,
            t_val,
            eta_val,
            [hkl],
            r3[0],
            r3[1],
            "bcc",
            {"a": a_lattice},
        )
        row_data = {
            "hkl": hkl,
            "d": None,
            "regions": [
                {"min_wavelength": r2[0], "max_wavelength": r2[1]},
                {"min_wavelength": r1[0], "max_wavelength": r1[1]},
                {"min_wavelength": r3[0], "max_wavelength": r3[1]},
            ],
            "s": s_val,
            "t": t_val,
            "eta": eta_val,
        }

        result = obj.fit_region(
            0,
            skip_ui_updates=True,
            row_data=row_data,
            wavelengths=wavelengths,
            intensities=intensities,
            fit_flags=(False, False, False),
            selected_phase="Fe_bcc",
            structure_type="bcc",
            lattice_params={"a": a_lattice},
        )
        self.assertIsInstance(result, dict)
        self.assertIn("fit_params", result)
        self.assertTrue(np.isfinite(result["d_fit"]))

    def test_fit_region_headless_unknown_phase_has_finite_metrics(self):
        obj = _HeadlessFitting()

        wavelengths = np.linspace(0.6, 1.4, 1400)
        # Unknown-phase model in fitting_function_3 uses x_hkl = 2*a when hkl_list is empty.
        a_unknown = 0.5  # edge position around lambda=1.0
        a0, b0 = 0.2, 0.3
        a_hkl, b_hkl = 0.1, 0.15
        s_val, t_val, eta_val = 0.005, 0.06, 0.4
        r1 = (0.8, 0.95)
        r2 = (0.7, 0.82)
        r3 = (0.9, 1.2)
        intensities = fitting_function_3(
            wavelengths,
            a0,
            b0,
            a_hkl,
            b_hkl,
            s_val,
            t_val,
            eta_val,
            [],
            r3[0],
            r3[1],
            "cubic",
            {"a": a_unknown},
        )
        row_data = {
            "hkl": None,
            "d": 1.0,
            "regions": [
                {"min_wavelength": r2[0], "max_wavelength": r2[1]},
                {"min_wavelength": r1[0], "max_wavelength": r1[1]},
                {"min_wavelength": r3[0], "max_wavelength": r3[1]},
            ],
            "s": s_val,
            "t": t_val,
            "eta": eta_val,
        }

        result = obj.fit_region(
            0,
            skip_ui_updates=True,
            row_data=row_data,
            wavelengths=wavelengths,
            intensities=intensities,
            fit_flags=(False, False, False),
            selected_phase="Unknown_Phase",
            structure_type="cubic",
            lattice_params={"a": 1.0},  # intentionally different from fitted edge location
        )

        self.assertIsInstance(result, dict)
        self.assertTrue(np.isfinite(result["edge_height"]))
        self.assertTrue(np.isfinite(result["edge_width"]))
        self.assertGreater(result["edge_height"], 0.0)
        self.assertGreater(result["edge_width"], 0.0)

    def test_fit_full_pattern_core_with_context(self):
        obj = _HeadlessFitting()
        obj.lattice_params = {"a": 1.19}

        wavelengths = np.linspace(1.0, 2.2, 1600)
        hkl = (1, 1, 0)
        a_lattice = 1.2
        a0, b0 = 0.3, 0.2
        a_hkl, b_hkl = 0.15, 0.07
        s_val, t_val, eta_val = 0.006, 0.05, 0.35
        r1 = (1.2, 1.35)
        r2 = (1.4, 1.55)
        r3 = (1.55, 1.95)
        intensities = fitting_function_3(
            wavelengths,
            a0,
            b0,
            a_hkl,
            b_hkl,
            s_val,
            t_val,
            eta_val,
            [hkl],
            r3[0],
            r3[1],
            "bcc",
            {"a": a_lattice},
        )
        fit_context = {
            "structure_type": "bcc",
            "lattice_params": {"a": a_lattice},
            "bragg_rows": [
                {
                    "row": 0,
                    "hkl": hkl,
                    "d": None,
                    "regions": [
                        {"min_wavelength": r2[0], "max_wavelength": r2[1]},
                        {"min_wavelength": r1[0], "max_wavelength": r1[1]},
                        {"min_wavelength": r3[0], "max_wavelength": r3[1]},
                    ],
                    "s": s_val,
                    "t": t_val,
                    "eta": eta_val,
                    "valid": True,
                }
            ],
        }

        result, error = obj.fit_full_pattern_core(
            fix_s=False,
            fix_t=False,
            fix_eta=False,
            max_nfev=200,
            curve_fit_maxfev=5000,
            fit_context=fit_context,
            wavelengths=wavelengths,
            intensities=intensities,
            apply_lattice_update=False,
        )
        self.assertIsNone(error)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(float(result["lattice_params"]["a"]), a_lattice, places=2)
        self.assertEqual(set(result), {
            "ab_fits", "bragg_edges", "structure_type", "lattice_params",
            "lattice_uncertainties", "fitted_s", "fitted_t", "fitted_eta",
            "s_uncertainties", "t_uncertainties", "eta_uncertainties",
            "x_data", "y_data", "x_exp_sorted", "y_exp_sorted", "residuals",
            "success", "message", "edge_heights", "edge_widths",
        })
        self.assertEqual([edge["hkl"] for edge in result["bragg_edges"]], [hkl])
        self.assertEqual(result["bragg_edges"][0]["regions"], fit_context["bragg_rows"][0]["regions"])
        np.testing.assert_array_equal(
            result["bragg_edges"][0]["x_r3"],
            wavelengths[(wavelengths >= 1.55) & (wavelengths <= 1.95)],
        )
        self.assertIn(hkl, result["edge_heights"])
        self.assertIn(hkl, result["edge_widths"])
        self.assertTrue(np.isfinite(result["edge_heights"][hkl]))
        self.assertTrue(np.isfinite(result["edge_widths"][hkl]))
        self.assertEqual(obj.lattice_params["a"], 1.19)

        fixed_result, fixed_error = obj.fit_full_pattern_core(
            fix_s=True,
            fix_t=True,
            fix_eta=True,
            max_nfev=200,
            curve_fit_maxfev=5000,
            fit_context=fit_context,
            wavelengths=wavelengths,
            intensities=intensities,
            apply_lattice_update=False,
        )
        self.assertIsNone(fixed_error)
        self.assertTrue(np.isnan(fixed_result["s_uncertainties"][hkl]))
        self.assertTrue(np.isnan(fixed_result["t_uncertainties"][hkl]))
        self.assertTrue(np.isnan(fixed_result["eta_uncertainties"][hkl]))

        updated_result, update_error = obj.fit_full_pattern_core(
            fix_s=True,
            fix_t=True,
            fix_eta=True,
            max_nfev=200,
            curve_fit_maxfev=5000,
            fit_context=fit_context,
            wavelengths=wavelengths,
            intensities=intensities,
        )
        self.assertIsNone(update_error)
        self.assertAlmostEqual(obj.lattice_params["a"], updated_result["lattice_params"]["a"])

        invalid_bounds = dict(fit_context, fitting_parameter_bounds={"s": (0.02, 0.01)})
        fallback_result, fallback_error = obj.fit_full_pattern_core(
            fit_context=invalid_bounds, wavelengths=wavelengths,
            intensities=intensities, curve_fit_maxfev=5000,
            apply_lattice_update=False,
        )
        self.assertIsNone(fallback_error)
        self.assertAlmostEqual(fallback_result["fitted_s"][hkl], s_val, places=4)

    def test_interactive_pattern_compatibility_path_still_returns_dict(self):
        obj = _HeadlessFitting()
        obj.bragg_table = _DummyTable([[
            "(1, 1, 0)", "1.697", "1.2", "1.35", "1.4", "1.95",
            "1.55", "1.95", "0.006", "0.05", "0.35",
        ]])
        obj.phase_dropdown = _DummyDropdown("Fe_bcc")
        obj.structure_type = "bcc"
        obj.lattice_params = {"a": 1.2}
        obj.wavelengths = np.linspace(1.0, 2.2, 1600)
        obj.intensities = fitting_function_3(
            obj.wavelengths, 0.3, 0.2, 0.15, 0.07, 0.006, 0.05, 0.35,
            [(1, 1, 0)], 1.55, 1.95, "bcc", {"a": 1.2},
        )
        legacy_context = obj._build_batch_fit_context()
        self.assertIsInstance(legacy_context, dict)
        self.assertIsInstance(legacy_context["bragg_rows"][0], dict)
        self.assertIsInstance(FullPatternFitConfig.from_legacy_dict(legacy_context), FullPatternFitConfig)

        result, error = obj.fit_full_pattern_core(
            curve_fit_maxfev=5000, apply_lattice_update=False
        )
        self.assertIsNone(error)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["bragg_edges"][0]["hkl"], (1, 1, 0))

    def test_canvas_corner_press_without_ctrl_moves_small_roi(self):
        obj = _HeadlessFitting()
        obj.canvas = _DummyCanvas()
        obj.images = [np.zeros((20, 20), dtype=np.float32)]
        obj.selected_area = (10, 12, 10, 12)
        obj._dragging_roi = False
        obj._roi_drag_offset = (0.0, 0.0)
        obj._dragging_roi_mode = "move"
        obj._roi_active_corner = None

        event = _DummyMouseEvent(obj.canvas.axes, 10, 10)
        obj._on_canvas_press(event)

        self.assertTrue(obj._dragging_roi)
        self.assertEqual(obj._dragging_roi_mode, "move")
        self.assertIsNone(obj._roi_active_corner)

    def test_canvas_corner_press_with_ctrl_resizes_small_roi(self):
        obj = _HeadlessFitting()
        obj.canvas = _DummyCanvas()
        obj.images = [np.zeros((20, 20), dtype=np.float32)]
        obj.selected_area = (10, 12, 10, 12)
        obj._dragging_roi = False
        obj._roi_drag_offset = (0.0, 0.0)
        obj._dragging_roi_mode = "move"
        obj._roi_active_corner = None

        event = _DummyMouseEvent(obj.canvas.axes, 10, 10, key="control")
        obj._on_canvas_press(event)

        self.assertTrue(obj._dragging_roi)
        self.assertEqual(obj._dragging_roi_mode, "resize")
        self.assertEqual(obj._roi_active_corner, "tl")

    def test_canvas_drag_outside_rois_pans_image_view(self):
        obj = _HeadlessFitting()
        obj.canvas = _DummyCanvas()
        obj.canvas.axes.set_xlim(20.0, 40.0)
        obj.canvas.axes.set_ylim(50.0, 30.0)
        obj.toolbar = object()
        obj.images = [np.zeros((100, 100), dtype=np.float32)]
        obj.selected_area = (10, 12, 10, 12)
        obj.min_x_input = _DummyTextInput("")
        obj.max_x_input = _DummyTextInput("")
        obj.min_y_input = _DummyTextInput("")
        obj.max_y_input = _DummyTextInput("")
        obj._dragging_roi = False
        obj._dragging_batch_roi = False
        obj._dragging_image_pan = False
        obj._image_pan_start_xpixel = None
        obj._image_pan_start_ypixel = None
        obj._image_pan_start_xlim = None
        obj._image_pan_start_ylim = None

        press = _DummyMouseEvent(obj.canvas.axes, 30.0, 40.0, x=50.0, y=50.0)
        obj._on_canvas_press(press)
        self.assertTrue(obj._dragging_image_pan)

        motion = _DummyMouseEvent(obj.canvas.axes, 35.0, 45.0, x=75.0, y=75.0)
        obj._on_canvas_motion(motion)
        self.assertEqual(obj.canvas.axes.get_xlim(), (15.0, 35.0))
        self.assertEqual(obj.canvas.axes.get_ylim(), (55.0, 35.0))
        self.assertTrue(obj.canvas.drawn)

        obj._on_canvas_motion(motion)
        self.assertEqual(obj.canvas.axes.get_xlim(), (15.0, 35.0))
        self.assertEqual(obj.canvas.axes.get_ylim(), (55.0, 35.0))

        obj._on_canvas_release(motion)
        self.assertFalse(obj._dragging_image_pan)

    def test_shift_image_axis_limits_clamps_to_image_bounds(self):
        shifted = _HeadlessFitting._shift_image_axis_limits((5.0, 25.0), 20.0, (-0.5, 99.5))
        self.assertEqual(shifted, (-0.5, 19.5))

        inverted = _HeadlessFitting._shift_image_axis_limits((25.0, 5.0), -90.0, (-0.5, 99.5))
        self.assertEqual(inverted, (99.5, 79.5))

    def test_slice_box_patch_args_outlines_covered_pixel_edges(self):
        xy, width, height = _HeadlessFitting._slice_box_patch_args(0, 1, 0, 1)
        self.assertEqual(xy, (-0.5, -0.5))
        self.assertEqual(width, 1.0)
        self.assertEqual(height, 1.0)

        xy, width, height = _HeadlessFitting._slice_box_patch_args(10, 12, 20, 23)
        self.assertEqual(xy, (9.5, 19.5))
        self.assertEqual(width, 2.0)
        self.assertEqual(height, 3.0)

    def test_resize_box_uses_pixel_edge_bounds(self):
        obj = _HeadlessFitting()

        expanded = obj._resize_box((0, 1, 0, 1), "br", 1.5, 1.5, 10, 10, min_w=1, min_h=1)
        self.assertEqual(expanded, (0, 2, 0, 2))

        shifted_left = obj._resize_box((2, 5, 2, 5), "tl", 0.5, 0.5, 10, 10, min_w=1, min_h=1)
        self.assertEqual(shifted_left, (1, 5, 1, 5))


if __name__ == "__main__":
    unittest.main()

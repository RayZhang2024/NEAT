"""Compatibility checks for the UI-facing individual-edge adapter."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from NEAT.ui.mixins.fitting import FittingMixin
from tests.test_fitting_headless import _DummyTable
from tests import test_individual_edge_service as golden_fixtures


class _Axes:
    def __init__(self):
        self.plots = []

    def plot(self, *args, **kwargs):
        self.plots.append((args, kwargs))


class _Canvas:
    def __init__(self):
        self.axes = _Axes()
        self.draw_count = 0

    def draw(self):
        self.draw_count += 1


class _Adapter(FittingMixin):
    def __init__(self):
        self.plot_canvas_c = _Canvas()
        self.plot_canvas_b = _Canvas()
        self.plot_canvas_d = _Canvas()
        self.messages = []
        self.cells = []

    def is_single_fit_canvas_mode(self):
        return False

    def _final_fit_canvas(self):
        return self.plot_canvas_d

    def _ensure_fit_canvas_axes(self, canvas):
        return canvas.axes, None

    def _plot_residual_line(self, *_args):
        pass

    def _append_fit_message(self, text):
        self.messages.append(text)

    def _update_cell(self, row, col, text):
        self.cells.append((row, col, text))


class TestIndividualEdgeAdapter(unittest.TestCase):
    def setUp(self):
        self.fixture = golden_fixtures.TestIndividualEdgeService().fixture

    def run_fit(self, *, known=True, skip_ui_updates=False, row=None, row_number=4, **kwargs):
        wavelengths, intensities, fixture_row, _ = self.fixture(known)
        obj = _Adapter()
        result = obj.fit_region(
            row_number,
            skip_ui_updates=skip_ui_updates,
            row_data=fixture_row if row is None else row,
            wavelengths=wavelengths,
            intensities=intensities,
            fit_flags=kwargs.pop("fit_flags", (False, False, False)),
            selected_phase="Fe_bcc" if known else "Unknown_Phase",
            structure_type="bcc" if known else "cubic",
            lattice_params={"a": 1.2 if known else 0.5},
            **kwargs,
        )
        return obj, result

    def test_explicit_row_and_exact_distinct_legacy_shapes(self):
        obj, headless = self.run_fit(skip_ui_updates=True)
        self.assertEqual(set(headless), {
            "hkl", "x", "fit", "d_fit", "d_unc", "s_fit", "s_unc", "t_fit", "t_unc",
            "eta_fit", "eta_unc", "fit_params", "edge_height", "edge_width", "rms", "baseline_params",
        })
        self.assertEqual(len(obj.plot_canvas_d.axes.plots), 0)
        obj, interactive = self.run_fit(skip_ui_updates=False, update_table=False)
        self.assertEqual(set(interactive), {
            "hkl", "x", "y", "fit", "fit_params", "edge_height", "edge_width", "rms", "baseline_params",
        })
        self.assertEqual(len(obj.plot_canvas_c.axes.plots), 2)
        self.assertEqual(len(obj.plot_canvas_b.axes.plots), 2)
        self.assertEqual(len(obj.plot_canvas_d.axes.plots), 2)

    def test_unknown_label_uses_original_source_row(self):
        _, result = self.run_fit(known=False, skip_ui_updates=True, row_number=4)
        self.assertEqual(result["hkl"], "edge5")

    def test_skip_ui_updates_gates_plots_messages_and_table(self):
        obj, result = self.run_fit(skip_ui_updates=True, emit_messages=True, update_table=True)
        self.assertIsNotNone(result)
        self.assertEqual(obj.messages, [])
        self.assertEqual(obj.cells, [])
        self.assertFalse(obj.plot_canvas_c.axes.plots)
        self.assertFalse(obj.plot_canvas_b.axes.plots)
        self.assertFalse(obj.plot_canvas_d.axes.plots)

    def test_only_update_unfixed(self):
        obj, result = self.run_fit(fit_flags=(True, False, True), only_update_unfixed=True)
        self.assertIsNotNone(result)
        self.assertEqual([cell[1] for cell in obj.cells], [9])
        obj, result = self.run_fit(fit_flags=(True, False, True), only_update_unfixed=False)
        self.assertEqual([cell[1] for cell in obj.cells], [8, 9, 10])

    def test_interactive_table_row_parsing(self):
        wavelengths, intensities, _, _ = self.fixture(True)
        obj = _Adapter()
        obj.bragg_table = _DummyTable([[
            "(1, 1, 0)", "1.697", "1.4", "1.55", "1.2", "1.35",
            "1.55", "1.95", "0.006", "0.05", "0.35",
        ]])
        result = obj.fit_region(
            0, wavelengths=wavelengths, intensities=intensities,
            fit_flags=(False, False, False), selected_phase="Fe_bcc",
            structure_type="bcc", lattice_params={"a": 1.2}, update_table=False,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["hkl"], (1, 1, 0))

    def test_malformed_legacy_row_and_invalid_bounds_are_compatible(self):
        _, _, row, _ = self.fixture(True)
        bad_row = dict(row, regions=[{}])
        obj, result = self.run_fit(row=bad_row)
        self.assertIsNone(result)
        self.assertIn("Edge 5: setup failed", obj.messages[-1])
        obj, result = self.run_fit(
            skip_ui_updates=True, fitting_parameter_bounds={"s": (0.02, 0.01)})
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["s_fit"], 0.006, places=4)

    def test_legacy_inverted_region_reports_stage_not_setup(self):
        _, _, row, _ = self.fixture(True)
        row["regions"][1] = {"min_wavelength": 1.35, "max_wavelength": 1.2}
        obj, result = self.run_fit(row=row)
        self.assertIsNone(result)
        self.assertIn("Region 1 has no data in range", obj.messages[-1])
        self.assertNotIn("setup failed", obj.messages[-1])

    def test_legacy_nan_region1_reports_stage_not_setup(self):
        _, _, row, _ = self.fixture(True)
        row["regions"][1] = {"min_wavelength": np.nan, "max_wavelength": 1.35}
        obj, result = self.run_fit(row=row)
        self.assertIsNone(result)
        self.assertIn("Region 1 has no data in range", obj.messages[-1])
        self.assertNotIn("setup failed", obj.messages[-1])

    def test_region3_failure_retains_region1_region2_plots(self):
        from NEAT.services import individual_edge as service
        real_fit = service.curve_fit
        calls = []
        def fail_third(*args, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("forced")
            return real_fit(*args, **kwargs)
        with patch.object(service, "curve_fit", side_effect=fail_third):
            obj, result = self.run_fit()
        self.assertIsNone(result)
        self.assertEqual(len(obj.plot_canvas_c.axes.plots), 2)
        self.assertEqual(len(obj.plot_canvas_b.axes.plots), 2)
        self.assertEqual(len(obj.plot_canvas_d.axes.plots), 1)
        self.assertIn("Region 3 fit failed - forced", obj.messages[-1])

    def test_no_optimizer_staging_remains_in_mixin_method(self):
        source = inspect.getsource(FittingMixin.fit_region)
        self.assertNotIn("curve_fit(", source)
        self.assertNotIn("fitting_function_3(", source)

    def test_batch_ui_builds_ordered_configs_and_keeps_full_metadata(self):
        _, _, row, _ = self.fixture(True)
        invalid = dict(row, row=0, valid=False)
        first = dict(row, row=2, valid=True)
        second = dict(row, row=5, valid=True)
        context = {
            "bragg_rows": [invalid, first, second],
            "bragg_rows_text": ["invalid table row", "first valid", "second valid"],
            "selected_phase": "Fe_bcc", "structure_type": "bcc",
            "lattice_params": {"a": 1.2},
        }
        class Value:
            def __init__(self, value): self.value = value
            def text(self): return str(self.value)
            def setText(self, value): self.value = value
            def setMinimum(self, value): pass
            def setMaximum(self, value): pass
            def setValue(self, value): pass
            def show(self): pass
            def start(self, *args): pass
            def isChecked(self): return False
            def rowCount(self): return 3
            def append(self, value): pass
        class Signal:
            def connect(self, callback): pass
        class Worker:
            captured = None
            def __init__(self, **kwargs):
                Worker.captured = kwargs
                self.progress_updated = Signal()
                self.message = Signal()
                self.finished = Signal()
                self.current_box_changed = Signal()
            def start(self): pass
        obj = _Adapter()
        for name, value in (
            ("box_width_input", 1), ("box_height_input", 1),
            ("step_x_input", 1), ("step_y_input", 1),
            ("min_x_input", 0), ("max_x_input", 1),
            ("min_y_input", 0), ("max_y_input", 1),
        ):
            setattr(obj, name, Value(value))
        for name in (
            "batch_progress_bar", "batch_progress_label", "batch_remaining_time_label",
            "batch_progress_dialog", "remaining_time_timer", "update_timer",
            "interpolation_checkbox", "bragg_table", "message_box",
        ):
            setattr(obj, name, Value(""))
        obj.images = [np.ones((1, 1))]
        obj.wavelengths = np.array([1.0])
        obj.work_directory = "C:/temporary/neat-output"
        obj._confirm_batch_box_size = lambda *_: True
        obj._build_batch_fit_context = lambda: context
        obj.fix_s_enabled = lambda: False
        obj.fix_t_enabled = lambda: False
        obj.fix_eta_enabled = lambda: False
        obj.update_progress_bar = lambda *_: None
        obj.append_message = lambda *_: None
        obj.batch_fit_edges_finished = lambda *_: None
        obj.update_current_box = lambda *_: None
        with patch("NEAT.ui.mixins.fitting.BatchFitEdgesWorker", Worker):
            obj.batch_fit_edges()
        configs = Worker.captured["edge_configs"]
        self.assertEqual([config.source_row for config in configs], [2, 5])
        self.assertEqual([config.hkl for config in configs], [(1, 1, 0), (1, 1, 0)])
        self.assertEqual(Worker.captured["output_metadata"]["bragg_rows_text"],
                         context["bragg_rows_text"])
        self.assertNotIn("parent", Worker.captured)
        self.assertNotIn("fit_context", Worker.captured)

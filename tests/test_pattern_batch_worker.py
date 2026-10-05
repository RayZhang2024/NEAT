"""Worker-level tests for GUI-independent full-pattern mapping."""

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from NEAT.core import fitting_function_3
from NEAT.services.fitting_engine import FittingEngine
from NEAT.workers.batch import BatchFitWorker


class _FailingEngine:
    def fit_full_pattern(self, **_kwargs):
        return None, "controlled fit failure"


class _RecordingEngine(FittingEngine):
    def fit_full_pattern(self, *args, **kwargs):
        self.last_fit_kwargs = kwargs
        return super().fit_full_pattern(*args, **kwargs)


class _CancellingEngine(FittingEngine):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker

    def fit_full_pattern(self, *args, **kwargs):
        result = super().fit_full_pattern(*args, **kwargs)
        self.worker.stop()
        return result


class TestPatternBatchWorker(unittest.TestCase):
    def setUp(self):
        self.wavelengths = np.linspace(1.0, 2.2, 1600)
        self.hkl = (1, 1, 0)
        self.fit_context = {
            "structure_type": "bcc",
            "lattice_params": {"a": 1.2},
            "fitting_parameter_bounds": {
                "s": (0.0001, 0.02),
                "t": (0.001, 0.2),
                "eta": (0.0, 1.0),
            },
            "selected_phase": "Fe_bcc",
            "flight_path": 10.0,
            "bragg_rows_text": ["(1;1;0)|row"],
            "bragg_rows": [
                {
                    "row": 0,
                    "hkl": self.hkl,
                    "regions": [
                        {"min_wavelength": 1.4, "max_wavelength": 1.55},
                        {"min_wavelength": 1.2, "max_wavelength": 1.35},
                        {"min_wavelength": 1.55, "max_wavelength": 1.95},
                    ],
                    "s": 0.006,
                    "t": 0.05,
                    "eta": 0.35,
                    "valid": True,
                }
            ],
        }
        intensities = fitting_function_3(
            self.wavelengths,
            0.3,
            0.2,
            0.15,
            0.07,
            0.006,
            0.05,
            0.35,
            [self.hkl],
            1.55,
            1.95,
            "bcc",
            {"a": 1.2},
        )
        self.images = [np.asarray([intensities[index]], dtype=float).reshape(1, 1)
                       for index in range(len(intensities))]

    def make_worker(self, folder, engine=None, *, fix_s=False, fix_t=False, fix_eta=False):
        return BatchFitWorker(
            fitting_engine=engine or FittingEngine(),
            images=self.images,
            wavelengths=self.wavelengths,
            fit_context=self.fit_context,
            min_x=0,
            max_x=1,
            min_y=0,
            max_y=1,
            box_width=1,
            box_height=1,
            step_x=1,
            step_y=1,
            total_boxes=1,
            interpolation_enabled=False,
            work_directory=str(folder),
            fix_s=fix_s,
            fix_t=fix_t,
            fix_eta=fix_eta,
        )

    def test_engine_fit_saves_outputs_and_emits_expected_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _RecordingEngine()
            worker = self.make_worker(tmp, engine)
            progress, boxes, payloads = [], [], []
            worker.progress_updated.connect(progress.append)
            worker.current_box_changed.connect(lambda *args: boxes.append(args))
            worker.finished.connect(payloads.append)

            worker.run()

            self.assertEqual(progress, [100])
            self.assertEqual(boxes, [(0, 1, 0, 1)])
            self.assertEqual(len(payloads), 1)
            paths = [Path(path) for path in payloads[0].splitlines()]
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.is_absolute() and path.is_file() for path in paths))
            self.assertEqual(
                {path.name.split("_")[1] for path in paths}, {"ungridded", "gridded"}
            )
            self.assertTrue(np.isfinite(worker.param_arrays["a"][0, 0]))
            np.testing.assert_allclose(
                worker.height_array[0, 0, 0], 0.09704296343196434,
                rtol=1e-7, atol=1e-9,
            )
            np.testing.assert_allclose(
                worker.width_array[0, 0, 0], 0.028459175655404012,
                rtol=1e-7, atol=1e-9,
            )
            self.assertEqual(engine.last_fit_kwargs["max_nfev"], 300)
            self.assertEqual(engine.last_fit_kwargs["curve_fit_maxfev"], 300)
            self.assertFalse(engine.last_fit_kwargs["fix_s"])
            self.assertFalse(engine.last_fit_kwargs["fix_t"])
            self.assertFalse(engine.last_fit_kwargs["fix_eta"])
            engine_width = engine.fit_full_pattern(
                wavelengths=self.wavelengths,
                intensities=np.asarray([image[0, 0] for image in self.images]),
                fit_config=worker.fit_context,
                max_nfev=300,
                curve_fit_maxfev=300,
            )[0]["edge_widths"][self.hkl]
            self.assertNotAlmostEqual(worker.width_array[0, 0, 0], engine_width, delta=1e-5)
            csv_text = paths[0].read_text(encoding="utf-8")
            self.assertIn("selected_phase,Fe_bcc", csv_text)
            self.assertIn("bragg_table_row_1,(1;1;0)|row", csv_text)
            self.assertIn("s_110", csv_text)

    def test_fit_failure_emits_message_and_no_result_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = self.make_worker(tmp, _FailingEngine())
            messages, progress, boxes, payloads = [], [], [], []
            worker.message.connect(messages.append)
            worker.progress_updated.connect(progress.append)
            worker.current_box_changed.connect(lambda *args: boxes.append(args))
            worker.finished.connect(payloads.append)

            worker.run()

            self.assertEqual(progress, [100])
            self.assertEqual(boxes, [(0, 1, 0, 1)])
            self.assertEqual(payloads, [""])
            self.assertTrue(any("controlled fit failure" in text for text in messages))
            self.assertEqual(list(Path(tmp).glob("results_*.csv")), [])

    def test_fixed_and_free_shape_flags_are_forwarded_to_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _RecordingEngine()
            worker = self.make_worker(
                tmp, engine, fix_s=True, fix_t=False, fix_eta=True
            )

            worker.run()

            self.assertTrue(worker.fitting_engine.last_fit_kwargs["fix_s"])
            self.assertFalse(worker.fitting_engine.last_fit_kwargs["fix_t"])
            self.assertTrue(worker.fitting_engine.last_fit_kwargs["fix_eta"])
            self.assertTrue(np.isnan(worker.s_unc_array[0, 0, 0]))
            self.assertTrue(np.isfinite(worker.t_unc_array[0, 0, 0]))
            self.assertTrue(np.isnan(worker.eta_unc_array[0, 0, 0]))

    def test_cancellation_during_final_fit_does_not_save_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = self.make_worker(tmp)
            worker.fitting_engine = _CancellingEngine(worker)
            messages, payloads = [], []
            worker.message.connect(messages.append)
            worker.finished.connect(payloads.append)

            worker.run()

            self.assertEqual(payloads, [""])
            self.assertTrue(any("Partial results were not saved" in text for text in messages))
            self.assertEqual(list(Path(tmp).glob("results_*.csv")), [])

    def test_fit_context_is_detached_but_image_stack_remains_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = self.make_worker(tmp)
            self.fit_context["lattice_params"]["a"] = 2.0
            self.fit_context["bragg_rows"][0]["regions"][0]["min_wavelength"] = 9.0
            self.fit_context["fitting_parameter_bounds"]["s"] = (9.0, 10.0)
            self.fit_context["bragg_rows_text"][0] = "changed"

            self.assertEqual(worker.fit_context["lattice_params"]["a"], 1.2)
            self.assertEqual(
                worker.fit_context["bragg_rows"][0]["regions"][0]["min_wavelength"],
                1.4,
            )
            self.assertEqual(worker.fit_context["fitting_parameter_bounds"]["s"], (0.0001, 0.02))
            self.assertEqual(worker.metadata_snapshot["bragg_rows_text"], ["(1;1;0)|row"])
            self.assertIs(worker.images, self.images)
            self.assertIs(worker.images[0], self.images[0])

    def test_worker_has_no_gui_parent_or_compatibility_core_dependency(self):
        source = inspect.getsource(BatchFitWorker)
        self.assertNotIn("self.parent", source)
        self.assertNotIn("fit_full_pattern_core", source)
        self.assertIn("self.fitting_engine.fit_full_pattern", source)


if __name__ == "__main__":
    unittest.main()

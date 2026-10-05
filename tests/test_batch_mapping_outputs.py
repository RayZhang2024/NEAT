import tempfile
import unittest
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from NEAT.domain import FullPatternFitConfig, FittingParameterBounds, IndividualEdgeFitConfig
from NEAT.services.fitting_engine import FittingEngine
from NEAT.workers.batch import (
    BatchFitWorker,
    BatchFitEdgesWorker,
    _representative_pixel_is_valid,
    _sample_valid_pixel_mask,
)


def _make_edges_worker(folder, images=None, edge_configs=None, output_metadata=None):
    fit_context = {
        "bragg_rows": [
            {
                "row": 0,
                "valid": True,
                "hkl": (1, 1, 0),
                "regions": [
                    {"min_wavelength": 1.0, "max_wavelength": 1.1},
                    {"min_wavelength": 1.2, "max_wavelength": 1.3},
                    {"min_wavelength": 1.0, "max_wavelength": 1.3},
                ],
                "s": 0.001,
                "t": 0.02,
                "eta": 0.5,
            }
        ],
        "bragg_rows_text": ["(1;1;0)|test"],
        "selected_phase": "Fe_bcc",
    }
    if edge_configs is None:
        edge_configs = tuple(
            IndividualEdgeFitConfig.from_legacy_row(
                row, source_row=row["row"], is_known_phase=True,
                structure_type="bcc", lattice_params={"a": 1.2},
                fitting_parameter_bounds=FittingParameterBounds(),
            ) for row in fit_context["bragg_rows"] if row["valid"]
        )
    if output_metadata is None:
        output_metadata = {
            "selected_phase": "Fe_bcc",
            "bragg_rows_text": fit_context["bragg_rows_text"],
        }
    return BatchFitEdgesWorker(
        fitting_engine=FittingEngine(),
        edge_configs=edge_configs,
        images=images or [np.ones((5, 5), dtype=np.float32)],
        wavelengths=np.array([1.0]),
        output_metadata=output_metadata,
        min_x=0,
        max_x=5,
        min_y=0,
        max_y=5,
        box_width=1,
        box_height=1,
        step_x=2,
        step_y=2,
        total_boxes=9,
        interpolation_enabled=True,
        work_directory=str(folder),
    )


class _SuccessfulIndividualFit:
    def __init__(self, worker=None):
        self.worker = worker
        self.calls = []

    def fit_individual_edge(self, _wavelengths, _intensities, config, **_kwargs):
        self.calls.append(config)
        return SimpleNamespace(result=SimpleNamespace(
            fit_params=(1.0, 0.001, 0.02, 0.5, 0.01, 0.001, 0.002, 0.01),
            edge_height=0.2, edge_width=0.03,
        ))


class _CancellingIndividualFit(_SuccessfulIndividualFit):
    def fit_individual_edge(self, *args, **kwargs):
        self.worker.stop()
        return super().fit_individual_edge(*args, **kwargs)


class TestBatchMappingOutputs(unittest.TestCase):
    def test_representative_pixel_requires_finite_positive_signal(self):
        valid = [np.array([[0.0, 1.0]], dtype=np.float32)]
        self.assertTrue(_representative_pixel_is_valid(valid, 0, 1))
        self.assertFalse(_representative_pixel_is_valid(valid, 0, 0))

        invalid = [np.array([[np.nan]], dtype=np.float32)]
        self.assertFalse(_representative_pixel_is_valid(invalid, 0, 0))

    def test_sample_mask_blocks_interpolation_outside_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = np.zeros((5, 5), dtype=np.float32)
            for row, col in ((0, 0), (0, 4), (4, 0), (4, 4)):
                image[row, col] = 1.0

            expected_mask = np.zeros((5, 5), dtype=bool)
            expected_mask[0, 0] = expected_mask[0, 4] = True
            expected_mask[4, 0] = expected_mask[4, 4] = True
            np.testing.assert_array_equal(
                _sample_valid_pixel_mask([image]), expected_mask
            )

            worker = _make_edges_worker(Path(tmp), images=[image])
            self.assertFalse(worker.sample_valid_pixel_mask[2, 2])
            worker.a_array[0, 0, 0] = 0.0
            worker.a_array[0, 4, 0] = 4.0
            worker.a_array[4, 0, 0] = 4.0
            worker.a_array[4, 4, 0] = 8.0

            worker.interpolate_results()

            self.assertTrue(np.isnan(worker.a_array[2, 2, 0]))

    def test_invalid_representative_center_skips_individual_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = _make_edges_worker(Path(tmp))
            worker.images = [np.zeros((5, 5), dtype=np.float32)]
            worker.max_x = 1
            worker.max_y = 1
            worker.total_boxes = 1

            worker.run()

            self.assertTrue(worker.invalid_center_mask[0, 0])
            self.assertTrue(np.isnan(worker.a_array[0, 0, 0]))

    def test_invalid_representative_center_skips_pattern_fit(self):
        class UnexpectedPatternFit:
            def fit_full_pattern(self, **_kwargs):
                raise AssertionError("invalid representative center was fitted")

        with tempfile.TemporaryDirectory() as tmp:
            worker = BatchFitWorker(
                fitting_engine=UnexpectedPatternFit(),
                images=[np.zeros((5, 5), dtype=np.float32)],
                wavelengths=np.array([1.0]),
                fit_config=FullPatternFitConfig(),
                output_metadata={},
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
                work_directory=tmp,
            )

            worker.run()

            self.assertTrue(worker.invalid_center_mask[0, 0])

    def test_pattern_interpolation_leaves_outside_sample_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = np.zeros((5, 5), dtype=np.float32)
            for row, col in ((0, 0), (0, 4), (4, 0), (4, 4)):
                image[row, col] = 1.0
            worker = BatchFitWorker(
                fitting_engine=FittingEngine(),
                images=[image],
                wavelengths=np.array([1.0]),
                fit_config=FullPatternFitConfig(),
                output_metadata={},
                min_x=0,
                max_x=5,
                min_y=0,
                max_y=5,
                box_width=1,
                box_height=1,
                step_x=2,
                step_y=2,
                total_boxes=9,
                interpolation_enabled=True,
                work_directory=tmp,
            )
            array = np.full((5, 5), np.nan)
            array[0, 0], array[0, 4] = 0.0, 4.0
            array[4, 0], array[4, 4] = 4.0, 8.0
            worker.param_arrays = {"a": array.copy()}
            worker.param_unc_arrays = {"a": array.copy()}

            worker.interpolate_results()

            self.assertTrue(np.isnan(worker.param_arrays["a"][2, 2]))

    def test_individual_edge_worker_accepts_more_than_five_valid_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = _make_edges_worker(Path(tmp))
            template = worker.edge_configs[0]

            rebuilt = BatchFitEdgesWorker(
                fitting_engine=FittingEngine(),
                edge_configs=tuple(
                    replace(template, source_row=index, hkl=(index + 1, 1, 0))
                    for index in range(6)
                ),
                images=worker.images,
                wavelengths=worker.wavelengths,
                output_metadata=worker.metadata_snapshot,
                min_x=0,
                max_x=5,
                min_y=0,
                max_y=5,
                box_width=1,
                box_height=1,
                step_x=2,
                step_y=2,
                total_boxes=9,
                interpolation_enabled=False,
                work_directory=tmp,
            )

            self.assertEqual(rebuilt.num_edges, 6)
            self.assertEqual(rebuilt.a_array.shape, (5, 5, 6))

    def test_successful_worker_returns_actual_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = _make_edges_worker(Path(tmp))
            worker.fitting_engine = _SuccessfulIndividualFit(worker)
            worker.max_x = 1
            worker.max_y = 1
            worker.total_boxes = 1
            worker.interpolation_enabled = False
            payloads = []
            worker.finished.connect(payloads.append)

            worker.run()

            self.assertEqual(len(payloads), 1)
            output_paths = payloads[0].splitlines()
            self.assertEqual(len(output_paths), 2)
            self.assertTrue(all(Path(path).is_absolute() for path in output_paths))
            self.assertTrue(all(Path(path).is_file() for path in output_paths))

    def test_stop_during_final_box_does_not_save_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            worker = _make_edges_worker(folder)
            worker.fitting_engine = _CancellingIndividualFit(worker)
            worker.max_x = 1
            worker.max_y = 1
            worker.total_boxes = 1
            payloads = []
            worker.finished.connect(payloads.append)

            worker.run()

            self.assertEqual(payloads, [""])
            self.assertEqual(list(folder.glob("results_*.csv")), [])

    def test_individual_edge_interpolation_fills_inside_convex_hull(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = _make_edges_worker(Path(tmp))
            points = {
                (0, 0): 0.0,
                (0, 4): 4.0,
                (4, 0): 4.0,
                (4, 4): 8.0,
            }
            array_names = (
                "a_array",
                "s_array",
                "t_array",
                "eta_array",
                "width_array",
                "height_array",
                "a_unc_array",
                "s_unc_array",
                "t_unc_array",
                "eta_unc_array",
            )
            for name in array_names:
                array = getattr(worker, name)
                for (row, col), value in points.items():
                    array[row, col, 0] = value

            worker.interpolate_results()

            self.assertAlmostEqual(worker.a_array[2, 2, 0], 4.0)
            self.assertTrue(np.isfinite(worker.height_array[2, 2, 0]))

    def test_individual_edge_csv_column_names_capture_current_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            worker = _make_edges_worker(folder)
            worker.a_array[2, 2, 0] = 1.0

            worker.save_results_to_csv_ungrid()
            worker.save_results_to_csv()

            ungridded = next(folder.glob("results_edges_ungridded_*.csv"))
            gridded = next(folder.glob("results_edges_gridded_*.csv"))
            ungridded_text = ungridded.read_text(encoding="utf-8")
            gridded_text = gridded.read_text(encoding="utf-8")
            self.assertIn("\nx,y,d_110,s_110,t_110,eta_110", ungridded_text)
            self.assertIn("\nx,y,d_110,s_110,t_110,eta_110", gridded_text)
            self.assertNotIn("d_11,", ungridded_text)
            self.assertEqual(
                sum(1 for line in ungridded_text.splitlines() if line.startswith("0,0,")),
                1,
            )

    def test_unknown_csv_fallback_names_match_pre_refactor_writers(self):
        # Captured from main 4a1f572: the two legacy writers intentionally differ.
        with tempfile.TemporaryDirectory() as tmp:
            template = _make_edges_worker(Path(tmp)).edge_configs[0]
            unknown = replace(template, hkl=None, d=1.0, is_known_phase=False)
            worker = _make_edges_worker(Path(tmp), edge_configs=(unknown,))
            worker.a_array[2, 2, 0] = 1.0
            worker.save_results_to_csv_ungrid()
            worker.save_results_to_csv()
            ungridded = next(Path(tmp).glob("results_edges_ungridded_*.csv"))
            gridded = next(Path(tmp).glob("results_edges_gridded_*.csv"))
            header = lambda path: next(line for line in path.read_text().splitlines() if line.startswith("x,y,"))
            self.assertEqual(header(ungridded),
                "x,y,d_edge10,s_edge10,t_edge10,eta_edge10,fwhm_edge10,"
                "d_unc_edge10,s_unc_edge10,t_unc_edge10,eta_unc_edge10,height_edge10")
            self.assertEqual(header(gridded),
                "x,y,d_edge1,s_edge1,t_edge1,eta_edge1,fwhm_edge1,height_edge1,"
                "d_unc_edge1,s_unc_edge1,t_unc_edge1,eta_unc_edge1")

    def test_worker_is_gui_free_and_duplicate_hkls_keep_two_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = _make_edges_worker(Path(tmp)).edge_configs[0]
            configs = (replace(template, source_row=2), replace(template, source_row=5))
            worker = _make_edges_worker(Path(tmp), edge_configs=configs)
            self.assertNotIn("parent", worker.__dict__)  # QThread itself has parent().
            self.assertFalse(hasattr(worker, "fit_context"))
            source = inspect.getsource(BatchFitEdgesWorker)
            self.assertNotIn("self.parent", source)
            self.assertNotIn("fit_region(", source)
            self.assertEqual(worker.num_edges, 2)
            self.assertEqual(worker.a_array.shape[-1], 2)
            self.assertEqual(worker.hkl_list, [(1, 1, 0), (1, 1, 0)])
            fake = _SuccessfulIndividualFit()
            worker.fitting_engine = fake
            worker.max_x = worker.max_y = worker.total_boxes = 1
            worker.run()
            self.assertEqual([config.source_row for config in fake.calls], [2, 5])
            self.assertEqual(worker.a_array[0, 0, :].tolist(), [1.0, 1.0])
            # The pre-existing dict writer collides duplicate HKL column names.
            ungridded = next(Path(tmp).glob("results_edges_ungridded_*.csv"))
            header = next(line for line in ungridded.read_text().splitlines() if line.startswith("x,y,"))
            self.assertEqual(header.count("d_110"), 1)

    def test_batch_valid_but_service_invalid_rows_retain_nan_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = _make_edges_worker(Path(tmp)).edge_configs[0]
            missing_d = replace(template, source_row=3, hkl=None, d=None, is_known_phase=False)
            missing_lattice = replace(template, source_row=5, lattice_params={})
            valid = replace(template, source_row=8, hkl=(2, 1, 0))
            configs = (missing_d, missing_lattice, valid)
            worker = _make_edges_worker(Path(tmp), edge_configs=configs)
            calls = []

            class MixedService:
                def fit_individual_edge(self, wavelengths, intensities, config, **kwargs):
                    calls.append(config.source_row)
                    if config.source_row == 8:
                        return _SuccessfulIndividualFit().fit_individual_edge(
                            wavelengths, intensities, config, **kwargs)
                    return FittingEngine().fit_individual_edge(wavelengths, intensities, config, **kwargs)

            worker.fitting_engine = MixedService()
            worker.max_x = worker.max_y = worker.total_boxes = 1
            worker.run()
            self.assertEqual(calls, [3, 5, 8])
            self.assertEqual(worker.a_array.shape[-1], 3)
            self.assertTrue(np.isnan(worker.a_array[0, 0, 0]))
            self.assertTrue(np.isnan(worker.a_array[0, 0, 1]))
            self.assertEqual(worker.a_array[0, 0, 2], 1.0)

    def test_zero_valid_edges_preserves_worker_finish_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = _make_edges_worker(Path(tmp), edge_configs=())
            payloads = []
            worker.finished.connect(payloads.append)
            worker.run()
            self.assertEqual(payloads, ["No valid edges."])
            self.assertFalse(list(Path(tmp).glob("results_*.csv")))

    def test_stop_during_first_edge_finishes_current_box_then_saves_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = _make_edges_worker(Path(tmp)).edge_configs[0]
            worker = _make_edges_worker(Path(tmp), edge_configs=(template, replace(template, source_row=1)))
            fake = _CancellingIndividualFit(worker)
            worker.fitting_engine = fake
            worker.max_x = worker.max_y = worker.total_boxes = 1
            payloads = []
            worker.finished.connect(payloads.append)
            worker.run()
            self.assertEqual(len(fake.calls), 2)
            self.assertEqual(payloads, [""])
            self.assertFalse(list(Path(tmp).glob("results_*.csv")))

    def test_full_table_metadata_includes_unfitted_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = {"selected_phase": "Fe_bcc", "bragg_rows_text": ["valid row", "invalid row"]}
            worker = _make_edges_worker(Path(tmp), output_metadata=metadata)
            metadata["bragg_rows_text"].append("late mutation")
            worker.fitting_engine = _SuccessfulIndividualFit()
            worker.max_x = worker.max_y = worker.total_boxes = 1
            worker.run()
            ungridded = next(Path(tmp).glob("results_edges_ungridded_*.csv"))
            content = ungridded.read_text()
            self.assertIn("bragg_table_row_1,valid row", content)
            self.assertIn("bragg_table_row_2,invalid row", content)
            self.assertNotIn("late mutation", content)


if __name__ == "__main__":
    unittest.main()

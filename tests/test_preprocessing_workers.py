import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from NEAT.workers.batch import load_image_file
from NEAT.workers.preprocessing import (
    FilteringWorker,
    NormalisationWorker,
    OutlierFilteringWorker,
    OverlapCorrectionWorker,
    SummationWorker,
    validate_normalisation_windows,
)


def _write_shutter_count(folder: Path, values):
    rows = np.column_stack((np.arange(len(values)), np.asarray(values)))
    np.savetxt(folder / "fixture_ShutterCount.txt", rows)


class TestSummationWorker(unittest.TestCase):
    def test_sums_corresponding_images_shutter_counts_and_copies_spectra(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            run_b = root / "run_b"
            output = root / "output"
            run_a.mkdir()
            run_b.mkdir()

            _write_shutter_count(run_a, [10, 20])
            _write_shutter_count(run_b, [3, 4])
            (run_a / "a_Spectra.txt").write_text("1 2\n", encoding="utf-8")
            (run_b / "b_Spectra.txt").write_text("3 4\n", encoding="utf-8")

            runs = [
                {
                    "folder_path": str(run_a),
                    "images": {"00001": np.full((2, 2), 2, dtype=np.float32)},
                },
                {
                    "folder_path": str(run_b),
                    "images": {"00001": np.full((2, 2), 3, dtype=np.float32)},
                },
            ]
            worker = SummationWorker(runs, "sample", str(output))
            worker.run()

            np.testing.assert_allclose(
                load_image_file(output / "sample_Summed_00001.fits"),
                np.full((2, 2), 5, dtype=np.float32),
            )
            shutter = np.loadtxt(output / "sample_summed_ShutterCount.txt")
            np.testing.assert_array_equal(shutter[:, 1], [13, 24])
            self.assertTrue((output / "sample_1_Spectra.txt").exists())
            self.assertTrue((output / "sample_2_Spectra.txt").exists())
            self.assertTrue(worker.succeeded)

    def test_suffix_mismatch_does_not_write_summed_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            runs = [
                {"folder_path": str(root), "images": {"00001": np.ones((2, 2))}},
                {"folder_path": str(root), "images": {"00002": np.ones((2, 2))}},
            ]

            worker = SummationWorker(runs, "sample", str(output))
            worker.run()

            self.assertEqual(list(output.glob("*_Summed_*.fits")), [])
            self.assertFalse(worker.succeeded)

    def test_shutter_length_mismatch_aborts_before_writing_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            run_b = root / "run_b"
            output = root / "output"
            run_a.mkdir()
            run_b.mkdir()
            _write_shutter_count(run_a, [10, 20])
            _write_shutter_count(run_b, [3])
            runs = [
                {
                    "folder_path": str(run_a),
                    "images": {"frame": np.ones((2, 2), dtype=np.float32)},
                },
                {
                    "folder_path": str(run_b),
                    "images": {"frame": np.ones((2, 2), dtype=np.float32)},
                },
            ]

            worker = SummationWorker(runs, "sample", str(output))
            worker.run()

            self.assertFalse(worker.succeeded)
            self.assertEqual(list(output.glob("*")), [])

    def test_precombined_logical_run_is_not_summed_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            physical_a = root / "physical_a"
            physical_b = root / "physical_b"
            physical_a.mkdir()
            physical_b.mkdir()
            output = root / "output"
            _write_shutter_count(physical_a, [2])
            _write_shutter_count(physical_b, [3])
            (physical_a / "a_Spectra.txt").write_text("first spectrum\n", encoding="utf-8")
            (physical_b / "b_Spectra.txt").write_text("second spectrum\n", encoding="utf-8")
            progress = []
            finished = []
            messages = []
            worker = SummationWorker(
                [
                    {
                        "folder_path": str(root / "logical_sample"),
                        "images": {"00001": np.array([[7]], dtype=np.float32)},
                        "run_folders": [str(physical_a), str(physical_b)],
                    }
                ],
                "sample",
                str(output),
            )
            worker.progress_updated.connect(progress.append)
            worker.finished.connect(lambda: finished.append(True))
            worker.message.connect(messages.append)

            worker.run()

            np.testing.assert_array_equal(
                load_image_file(output / "sample_Summed_00001.fits"), [[7]]
            )
            self.assertEqual(np.loadtxt(output / "sample_summed_ShutterCount.txt")[1], 5)
            self.assertEqual(
                (output / "sample_1_Spectra.txt").read_text(encoding="utf-8"),
                "first spectrum\n",
            )
            self.assertTrue(worker.succeeded)
            self.assertEqual(worker.result.processed_count, 1)
            self.assertEqual(worker.result.expected_count, 1)
            self.assertEqual(progress[0], 0)
            self.assertEqual(progress[-1], 100)
            self.assertEqual(finished, [True])
            self.assertTrue(any("Summation complete" in message for message in messages))


class TestOutlierFilteringWorker(unittest.TestCase):
    def test_replaces_nonpositive_pixel_with_positive_neighbour_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            image = np.ones((5, 5), dtype=np.float32)
            image[2, 2] = 0
            run = {"folder_path": tmp, "images": {"00007": image}}

            OutlierFilteringWorker([run], str(output), "clean").run()

            cleaned = load_image_file(output / "clean_00007.fits")
            self.assertEqual(float(cleaned[2, 2]), 1.0)
            report = (output / "clean_outlier_report.csv").read_text(encoding="utf-8")
            self.assertIn("00007,2,2,0.0000,1.0000", report)

    def test_leaves_pixel_unchanged_when_no_positive_neighbour_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            image = np.zeros((3, 3), dtype=np.float32)
            run = {"folder_path": tmp, "images": {"00001": image}}

            OutlierFilteringWorker([run], str(output), "clean").run()

            cleaned = load_image_file(output / "clean_00001.fits")
            np.testing.assert_array_equal(cleaned, image)

    def test_expands_to_seven_by_seven_when_five_by_five_has_no_valid_neighbor(self):
        image = np.zeros((7, 7), dtype=np.float32)
        image[0, 0] = 6.0

        five_by_five = OutlierFilteringWorker._positive_neighbor_mean(
            image, 3, 3, radius=2
        )
        seven_by_seven = OutlierFilteringWorker._positive_neighbor_mean(
            image, 3, 3, radius=3
        )

        self.assertIsNone(five_by_five)
        self.assertEqual(seven_by_seven, 6.0)

    def test_replaces_positive_spike_at_ten_times_neighbor_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            image = np.ones((5, 5), dtype=np.float32)
            image[2, 2] = 10.0
            run = {"folder_path": tmp, "images": {"00002": image}}

            worker = OutlierFilteringWorker([run], str(output), "clean")
            worker.run()

            cleaned = load_image_file(output / "clean_00002.fits")
            self.assertEqual(float(cleaned[2, 2]), 1.0)
            report = (output / "clean_outlier_report.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("00002,2,2,10.0000,1.0000", report)

    def test_positive_spike_detection_handles_image_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            image = np.ones((3, 3), dtype=np.float32)
            image[0, 0] = 10.0
            run = {"folder_path": tmp, "images": {"00003": image}}

            OutlierFilteringWorker([run], str(output), "clean").run()

            cleaned = load_image_file(output / "clean_00003.fits")
            self.assertEqual(float(cleaned[0, 0]), 1.0)
            report = (output / "clean_outlier_report.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("00003,0,0,10.0000,1.0000", report)

    def test_adjacent_spikes_keep_sequential_raster_replacement_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            image = np.ones((7, 7), dtype=np.float32)
            image[3, 3] = 100.0
            image[3, 4] = 100.0
            run = {"folder_path": tmp, "images": {"00004": image}}

            OutlierFilteringWorker([run], str(output), "clean").run()

            cleaned = load_image_file(output / "clean_00004.fits")
            self.assertAlmostEqual(float(cleaned[3, 3]), 5.125)
            self.assertAlmostEqual(float(cleaned[3, 4]), 1.171875)


class TestNormalisationWorker(unittest.TestCase):
    def test_window_ranges_are_enforced(self):
        self.assertEqual(validate_normalisation_windows(10, 0), (10, 0))
        self.assertEqual(validate_normalisation_windows(100, 10), (100, 10))
        for n, m in ((-1, 0), (101, 0), (10, -1), (10, 11)):
            with self.subTest(n=n, m=m):
                with self.assertRaises(ValueError):
                    validate_normalisation_windows(n, m)

    def test_uses_local_open_beam_and_shutter_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_folder = root / "sample"
            open_beam_folder = root / "open_beam"
            output = root / "output"
            sample_folder.mkdir()
            open_beam_folder.mkdir()
            output.mkdir()
            _write_shutter_count(sample_folder, [2])
            _write_shutter_count(open_beam_folder, [4])

            sample_run = {
                "folder_path": str(sample_folder),
                "images": {"00001": np.full((3, 3), 10, dtype=np.float32)},
            }
            open_beam_run = {
                "folder_path": str(open_beam_folder),
                "images": {"00001": np.full((3, 3), 20, dtype=np.float32)},
            }
            worker = NormalisationWorker(
                [sample_run], [open_beam_run], str(output), "norm", 0, 0
            )

            with patch("NEAT.workers.preprocessing.QThread.sleep"):
                worker.run()

            np.testing.assert_allclose(
                load_image_file(output / "norm_00001.fits"),
                np.ones((3, 3), dtype=np.float32),
            )
            self.assertTrue(worker.succeeded)


class TestFilteringWorker(unittest.TestCase):
    def test_binary_mask_one_keeps_and_zero_discards(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            image = np.array([[1, 2], [3, 4]], dtype=np.float32)
            mask = np.array([[0, 1], [1, 0]], dtype=np.int16)
            run = {"folder_path": tmp, "images": {"00001": image}}

            worker = FilteringWorker([run], mask, str(output), "filtered")
            worker.run()

            np.testing.assert_array_equal(
                load_image_file(output / "filtered_00001.fits"),
                np.array([[0, 2], [3, 0]], dtype=np.float32),
            )
            self.assertTrue(worker.succeeded)

    def test_nonbinary_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            run = {
                "folder_path": tmp,
                "images": {"00001": np.ones((2, 2), dtype=np.float32)},
            }
            worker = FilteringWorker(
                [run],
                np.array([[0, 2], [1, 0]], dtype=np.float32),
                str(output),
                "filtered",
            )

            worker.run()

            self.assertFalse(worker.succeeded)
            self.assertFalse((output / "filtered_00001.fits").exists())

    def test_shape_mismatch_skips_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            run = {"folder_path": tmp, "images": {"00001": np.ones((2, 2))}}

            worker = FilteringWorker(
                [run], np.ones((3, 3)), str(output), "filtered"
            )
            worker.run()

            self.assertFalse((output / "filtered_00001.fits").exists())
            self.assertFalse(worker.succeeded)


class TestOverlapCorrectionWorker(unittest.TestCase):
    def test_applies_cumulative_probability_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            image = np.full((512, 512), 100, dtype=np.float32)
            run = {
                "folder_path": str(root),
                "images": {"00001": image},
                "spectra": np.array([[0.0, 1.0]], dtype=np.float32),
                "shutter_count": np.array([2000.0], dtype=np.float32),
            }

            worker = OverlapCorrectionWorker(run, "sample", str(output))
            worker.run()

            corrected = load_image_file(
                output / "Corrected_sample_00001.fits"
            )
            np.testing.assert_allclose(corrected, 100.0 / 0.95, rtol=1e-6)
            self.assertTrue(worker.succeeded)

    def test_rejects_non_512_image_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            run = {
                "folder_path": str(root),
                "images": {"00001": np.ones((2, 2), dtype=np.float32)},
                "spectra": np.array([[0.0, 1.0]], dtype=np.float32),
                "shutter_count": np.array([2000.0], dtype=np.float32),
            }

            worker = OverlapCorrectionWorker(run, "sample", str(output))
            worker.run()

            self.assertEqual(list(output.glob("*.fits")), [])
            self.assertFalse(worker.succeeded)


if __name__ == "__main__":
    unittest.main()

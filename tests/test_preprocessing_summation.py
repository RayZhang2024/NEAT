"""Headless regressions for the extracted classic summation service."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from NEAT.domain import LoadedImageRun, PreprocessingStatus
from NEAT.services import preprocessing_summation as summation
from NEAT.workers.batch import load_image_file


BASELINE_COMMIT = "e6e0e8cef9806d1f5000c8f38b7b2509c8a3618e"


def _write_shutter(folder: Path, values: list[float]) -> None:
    table = np.column_stack((np.arange(len(values)), values))
    np.savetxt(folder / "fixture_ShutterCount.txt", table, fmt="%.9g")


def _golden_fixture(root: Path) -> list[LoadedImageRun]:
    folders = {}
    for name in ("a1", "a2", "b1", "c1"):
        folder = root / name
        folder.mkdir()
        folders[name] = str(folder)

    # These values and expected outputs were captured by running the original
    # SummationWorker from baseline commit e6e0e8c with the same ordered runs.
    _write_shutter(root / "a1", [1e8, 1e8])
    _write_shutter(root / "a2", [1, -1e8])
    _write_shutter(root / "b1", [-1e8, 3])
    _write_shutter(root / "c1", [1, 4])
    (root / "a1" / "a1_Spectra.txt").write_text("spectrum A1\n", encoding="utf-8")
    (root / "a2" / "a2_Spectra.txt").write_text("spectrum A2\n", encoding="utf-8")
    (root / "b1" / "b1_Spectra.txt").write_text("spectrum B\n", encoding="utf-8")
    (root / "c1" / "c1_Spectra.txt").write_text("spectrum C\n", encoding="utf-8")

    return [
        LoadedImageRun(
            "logical-A",
            {
                "10": np.array([[1e8, 1], [10, -10]], dtype=np.float32),
                "2": np.array([[4, 3], [2, 1]], dtype=np.float32),
            },
            (folders["a1"], folders["a2"]),
        ),
        LoadedImageRun(
            "logical-B",
            {
                "10": np.array([[-1e8, 1], [1, 20]], dtype=np.float64),
                "2": np.array([[0.5, -2], [0.25, 4]], dtype=np.float64),
            },
            (folders["b1"],),
        ),
        LoadedImageRun(
            "logical-C",
            {
                "10": np.array([[1, -1], [-1, 3]], dtype=np.int16),
                "2": np.array([[-1, 5], [8, -1]], dtype=np.int16),
            },
            (folders["c1"],),
        ),
    ]


class TestPreprocessingSummationService(unittest.TestCase):
    def test_headless_service_matches_pre_refactor_golden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = _golden_fixture(root)
            output = root / "output"
            result = summation.sum_loaded_image_runs(runs, "gold", str(output))

            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(result.processed_count, 2)
            self.assertEqual(result.expected_count, 2)
            self.assertEqual(
                [item.role for item in result.outputs],
                [
                    "summed_image",
                    "summed_image",
                    "summed_shutter_count",
                    "spectrum_copy",
                    "spectrum_copy",
                    "spectrum_copy",
                ],
            )
            # Exact golden arrays captured from the original worker at BASELINE_COMMIT.
            np.testing.assert_array_equal(
                load_image_file(output / "gold_Summed_10.fits"),
                np.array([[1, 1], [10, 13]], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                load_image_file(output / "gold_Summed_2.fits"),
                np.array([[3.5, 6], [10.25, 4]], dtype=np.float32),
            )
            self.assertEqual(
                (output / "gold_summed_ShutterCount.txt").read_text(encoding="utf-8"),
                "0\t1\n1\t7\n",
            )
            self.assertEqual(
                (output / "gold_1_Spectra.txt").read_text(encoding="utf-8"),
                "spectrum A1\n",
            )
            self.assertEqual(
                (output / "gold_2_Spectra.txt").read_text(encoding="utf-8"),
                "spectrum B\n",
            )
            self.assertEqual(
                (output / "gold_3_Spectra.txt").read_text(encoding="utf-8"),
                "spectrum C\n",
            )
            self.assertEqual(
                [Path(item.path).name for item in result.outputs[:2]],
                ["gold_Summed_10.fits", "gold_Summed_2.fits"],
            )

    def test_float32_accumulation_is_sequential_and_does_not_mutate_inputs(self):
        first = np.array([[1e8]], dtype=np.float64)
        second = np.array([[-1e8]], dtype=np.float64)
        third = np.array([[1]], dtype=np.int16)
        result = summation.sum_corresponding_frames([first, second, third])

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.array([[1]], dtype=np.float32))
        np.testing.assert_array_equal(first, [[1e8]])

    def test_validation_failure_has_unknown_expected_count_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "run"
            folder.mkdir()
            # No shutter sidecar: all validation must fail before output creation.
            run = LoadedImageRun("logical-run", {"1": np.ones((2, 2))}, (str(folder),))
            output = root / "not-created"

            result = summation.sum_loaded_image_runs([run], "sample", str(output))

            self.assertEqual(result.status, PreprocessingStatus.FAILED)
            self.assertEqual(result.processed_count, 0)
            self.assertIsNone(result.expected_count)
            self.assertFalse(output.exists())

    def test_suffix_and_shape_validation_happen_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "run"
            folder.mkdir()
            _write_shutter(folder, [1])
            cases = (
                [
                    LoadedImageRun("a", {"1": np.ones((2, 2))}, (str(folder),)),
                    LoadedImageRun("b", {"2": np.ones((2, 2))}, (str(folder),)),
                ],
                [
                    LoadedImageRun("a", {"1": np.ones((2, 2))}, (str(folder),)),
                    LoadedImageRun("b", {"1": np.ones((3, 2))}, (str(folder),)),
                ],
                [
                    LoadedImageRun("a", {"1": np.ones((2, 2))}, (str(folder),)),
                    LoadedImageRun("b", {"1": np.ones((2, 2))}, (str(folder),), ("bad frame",)),
                ],
            )
            for idx, runs in enumerate(cases):
                output = root / f"output-{idx}"
                result = summation.sum_loaded_image_runs(runs, "sample", str(output))
                self.assertEqual(result.status, PreprocessingStatus.FAILED)
                self.assertEqual(result.processed_count, 0)
                self.assertIsNone(result.expected_count)
                self.assertFalse(output.exists())

    def test_additional_prewrite_validation_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid"
            malformed = root / "malformed"
            valid.mkdir()
            malformed.mkdir()
            _write_shutter(valid, [1])
            np.savetxt(
                malformed / "fixture_ShutterCount.txt", [[0, 1, 2]], fmt="%d"
            )
            cases = (
                ("no runs", []),
                ("empty first run", [LoadedImageRun("empty", {}, (str(valid),))]),
                (
                    "non-2-D frame",
                    [LoadedImageRun("one-dimensional", {"1": np.ones(2)}, (str(valid),))],
                ),
                (
                    "three-column ShutterCount",
                    [
                        LoadedImageRun(
                            "bad-sidecar",
                            {"1": np.ones((2, 2), dtype=np.float32)},
                            (str(malformed),),
                        )
                    ],
                ),
            )
            for index, (description, runs) in enumerate(cases):
                with self.subTest(description):
                    output = root / f"output-{index}"
                    result = summation.sum_loaded_image_runs(runs, "sample", str(output))
                    self.assertEqual(result.status, PreprocessingStatus.FAILED)
                    self.assertEqual(result.processed_count, 0)
                    self.assertIsNone(result.expected_count)
                    self.assertEqual(result.outputs, ())
                    self.assertFalse(output.exists())

    def test_failed_later_image_write_keeps_partial_results_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = _golden_fixture(root)
            output = root / "output"
            writer = summation.write_fits_image_file
            calls = 0

            def fail_second(path, data, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated second-frame failure")
                writer(path, data, **kwargs)

            with patch.object(summation, "write_fits_image_file", side_effect=fail_second):
                result = summation.sum_loaded_image_runs(runs, "gold", str(output))

            self.assertEqual(result.status, PreprocessingStatus.FAILED)
            self.assertEqual((result.processed_count, result.expected_count), (1, 2))
            self.assertEqual([item.role for item in result.outputs], ["summed_image"])
            self.assertTrue((output / "gold_Summed_10.fits").exists())
            self.assertFalse((output / "gold_summed_ShutterCount.txt").exists())

    def test_shutter_write_failure_keeps_all_completed_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = _golden_fixture(root)
            output = root / "output"
            with patch.object(summation.np, "savetxt", side_effect=OSError("denied")):
                result = summation.sum_loaded_image_runs(runs, "gold", str(output))

            self.assertEqual(result.status, PreprocessingStatus.FAILED)
            self.assertEqual((result.processed_count, result.expected_count), (2, 2))
            self.assertEqual(
                [item.role for item in result.outputs],
                ["summed_image", "summed_image"],
            )
            self.assertEqual(
                [Path(item.path).name for item in result.outputs],
                ["gold_Summed_10.fits", "gold_Summed_2.fits"],
            )
            self.assertTrue((output / "gold_Summed_10.fits").exists())
            self.assertTrue((output / "gold_Summed_2.fits").exists())
            self.assertFalse((output / "gold_summed_ShutterCount.txt").exists())
            self.assertEqual(list(output.glob("*_Spectra.txt")), [])

    def test_cancellation_after_image_keeps_only_completed_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = _golden_fixture(root)
            output = root / "output"
            cancelled = False

            def progress(value: int) -> None:
                nonlocal cancelled
                if value == 50:
                    cancelled = True

            result = summation.sum_loaded_image_runs(
                runs,
                "gold",
                str(output),
                progress_callback=progress,
                cancellation_check=lambda: cancelled,
            )

            self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
            self.assertEqual((result.processed_count, result.expected_count), (1, 2))
            self.assertEqual([item.role for item in result.outputs], ["summed_image"])
            self.assertFalse((output / "gold_summed_ShutterCount.txt").exists())
            self.assertFalse((output / "gold_1_Spectra.txt").exists())

    def test_spectra_copy_failure_is_a_warning_and_remains_nonfatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = _golden_fixture(root)
            output = root / "output"
            messages = []
            with patch.object(summation.shutil, "copyfile", side_effect=OSError("denied")):
                result = summation.sum_loaded_image_runs(
                    runs, "gold", str(output), message_callback=messages.append
                )

            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(result.processed_count, 2)
            # Each copy warning retains the legacy follow-up "No Spectra" message.
            self.assertEqual(len(result.warnings), 6)
            self.assertEqual(
                result.warnings[1::2],
                tuple(f"Run {index}: No Spectra file found." for index in (1, 2, 3)),
            )
            self.assertEqual(
                [message for message in messages if "No Spectra file found." in message],
                [f"Run {index}: No Spectra file found." for index in (1, 2, 3)],
            )
            self.assertEqual(
                [item.role for item in result.outputs],
                ["summed_image", "summed_image", "summed_shutter_count"],
            )

    def test_missing_spectra_uses_exact_legacy_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_folder = root / "run"
            run_folder.mkdir()
            _write_shutter(run_folder, [1])
            run = LoadedImageRun(
                "run", {"1": np.ones((2, 2), dtype=np.float32)}, (str(run_folder),)
            )
            messages = []

            result = summation.sum_loaded_image_runs(
                [run], "sample", str(root / "output"), message_callback=messages.append
            )

            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(result.warnings, ("Run 1: No Spectra file found.",))
            self.assertIn("Run 1: No Spectra file found.", messages)

    def test_service_import_does_not_load_qt_or_gui_modules(self):
        code = (
            "import sys\n"
            "from NEAT.services.preprocessing_summation import sum_loaded_image_runs\n"
            "assert not any(name == 'PyQt5' or name.startswith('PyQt5.') for name in sys.modules)\n"
            "assert 'NEAT.ui' not in sys.modules\n"
            "assert 'NEAT.workers' not in sys.modules\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()

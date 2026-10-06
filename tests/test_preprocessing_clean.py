"""Headless Clean regressions; golden values came from the original worker.

Reference: OutlierFilteringWorker at 56ddb9e8e10e9faaa8505ccdfac2a43531772268.
Run the ordered _golden_frames() mapping through that worker with base ``gold``.
The expected report digest and arrays below were captured before extraction.
Float32 output is compared exactly, including the order-sensitive spike values.
"""

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from NEAT.domain import LoadedImageRun, PreprocessingStatus
from NEAT.services.preprocessing_clean import (
    clean_loaded_image_runs,
    clean_outlier_frame,
    positive_neighbor_mean,
)
from NEAT.workers.batch import load_image_file


def _golden_frames():
    invalid = np.ones((5, 5), dtype=np.float32)
    invalid[2, 2] = 0
    invalid[0, 4] = np.nan
    invalid[4, 0] = -np.inf
    invalid[4, 4] = np.inf
    fallback = np.zeros((7, 7), dtype=np.float32)
    fallback[3, 3] = 6
    unresolved = np.zeros((1, 1), dtype=np.float32)
    spikes = np.ones((7, 7), dtype=np.float32)
    spikes[0, 0] = 10
    spikes[3, 3] = 100
    spikes[3, 4] = 100
    spikes[6, 6] = np.float32(9.999)
    return {
        "z_invalid": invalid,
        "a_fallback": fallback,
        "m_unresolved": unresolved,
        "b_spikes": spikes,
    }


def _run(source, frames, output, **callbacks):
    run = LoadedImageRun(str(source), frames)
    return clean_loaded_image_runs((run,), str(output), "gold", **callbacks)


class TestCleanTransform(unittest.TestCase):
    def test_original_worker_golden_arrays_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = _golden_frames()
            originals = {key: value.copy() for key, value in frames.items()}
            progress = []
            result = _run(root, frames, root / "out", progress_callback=progress.append)
            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual((result.processed_count, result.expected_count), (4, 4))
            self.assertEqual(progress, [25, 50, 75, 100])
            self.assertEqual(
                [Path(item.path).name for item in result.outputs],
                ["gold_outlier_report.csv"]
                + [f"gold_{suffix}.fits" for suffix in frames],
            )
            report = (root / "out" / "gold_outlier_report.csv").read_text(
                encoding="utf-8"
            )
            self.assertEqual(
                hashlib.sha256(report.encode("utf-8")).hexdigest(),
                "94817afb1ba39f557a4d00b43607e72f3a26dfabdb078dae86c7a3efa597caf8",
            )
            self.assertEqual(
                report.splitlines()[:5],
                [
                    "frame_idx,pixel_x,pixel_y,outlier_value,replace_value",
                    "z_invalid,4,0,nan,1.0000",
                    "z_invalid,2,2,0.0000,1.0000",
                    "z_invalid,0,4,-inf,1.0000",
                    "z_invalid,4,4,inf,1.0000",
                ],
            )
            expected = {
                "z_invalid": np.ones((5, 5), dtype=np.float32),
                "a_fallback": np.full((7, 7), 6, dtype=np.float32),
                "m_unresolved": np.zeros((1, 1), dtype=np.float32),
                "b_spikes": np.ones((7, 7), dtype=np.float32),
            }
            expected["b_spikes"][3, 3] = 5.125
            expected["b_spikes"][3, 4] = 1.171875
            expected["b_spikes"][6, 6] = np.float32(9.999)
            for suffix, array in expected.items():
                with self.subTest(suffix=suffix):
                    cleaned = load_image_file(root / "out" / f"gold_{suffix}.fits")
                    np.testing.assert_array_equal(cleaned, array)
                    np.testing.assert_array_equal(frames[suffix], originals[suffix])

    def test_pure_transform_has_no_input_mutation_and_records_in_order(self):
        frames = _golden_frames()
        for suffix, source in frames.items():
            before = source.copy()
            cleaned, records, count = clean_outlier_frame(source, suffix)
            self.assertEqual(cleaned.dtype, np.float32)
            self.assertIsNot(cleaned, source)
            np.testing.assert_array_equal(source, before)
            self.assertEqual(count, len(records))
        self.assertEqual(
            records,
            (
                "b_spikes,0,0,10.0000,1.0000",
                "b_spikes,3,3,100.0000,5.1250",
                "b_spikes,4,3,100.0000,1.1719",
            ),
        )
        self.assertIsNone(positive_neighbor_mean(frames["a_fallback"], 0, 0, radius=2))
        self.assertEqual(
            positive_neighbor_mean(frames["a_fallback"], 0, 0, radius=3), 6.0
        )
        self.assertEqual(
            clean_outlier_frame(frames["m_unresolved"], "x")[1], ("x,0,0,0.0000,nan",)
        )


class TestCleanService(unittest.TestCase):
    def test_empty_runs_and_zero_frame_run_succeed_with_report_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for runs in ((), (LoadedImageRun(str(root), {}),)):
                with self.subTest(runs=len(runs)):
                    progress, messages = [], []
                    result = clean_loaded_image_runs(
                        runs,
                        str(root / "out"),
                        "gold",
                        progress_callback=progress.append,
                        message_callback=messages.append,
                    )
                    self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
                    self.assertEqual(
                        (result.processed_count, result.expected_count), (0, 0)
                    )
                    self.assertEqual(
                        [item.role for item in result.outputs], ["outlier_report"]
                    )
                    self.assertEqual(progress, [])
                    self.assertIn("Processed 0 frame(s)", messages[-1])
                    self.assertEqual(
                        (root / "out" / "gold_outlier_report.csv").read_text(),
                        "frame_idx,pixel_x,pixel_y,outlier_value,replace_value\n",
                    )

    def test_ordered_frames_sidecars_primary_source_and_blank_record_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary, other, output = root / "primary", root / "other", root / "out"
            primary.mkdir()
            other.mkdir()
            (primary / "one_Spectra.txt").write_text("primary spectrum")
            (primary / "one_ShutterCount.txt").write_text("primary shutter")
            (other / "other_Spectra.txt").write_text("wrong source")
            run = LoadedImageRun(
                str(primary),
                {"10": np.ones((2, 2)), "2": np.ones((2, 2))},
                source_folders=(str(other),),
                load_errors=("ignored loader warning",),
            )
            progress, messages = [], []
            with patch(
                "NEAT.services.preprocessing_clean.shutil.copy2",
                wraps=__import__("shutil").copy2,
            ) as copy:
                result = clean_loaded_image_runs(
                    (run,),
                    str(output),
                    "gold",
                    progress_callback=progress.append,
                    message_callback=messages.append,
                )
            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual((result.processed_count, result.expected_count), (2, 2))
            self.assertEqual(copy.call_count, 2)
            self.assertEqual(progress, [50, 100])
            self.assertEqual(
                [item.role for item in result.outputs],
                [
                    "outlier_report",
                    "related_file_copy",
                    "related_file_copy",
                    "cleaned_image",
                    "cleaned_image",
                ],
            )
            self.assertEqual(
                [Path(item.path).name for item in result.outputs],
                [
                    "gold_outlier_report.csv",
                    "Run1_one_Spectra.txt",
                    "Run1_one_ShutterCount.txt",
                    "gold_10.fits",
                    "gold_2.fits",
                ],
            )
            self.assertEqual(
                (output / "gold_outlier_report.csv").read_text().count("\n"), 3
            )
            self.assertEqual(sum("no outliers detected" in m for m in messages), 2)
            self.assertFalse((output / "Run1_other_Spectra.txt").exists())

    def test_sidecar_first_match_uses_unsorted_listdir_and_copy_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "z_Spectra.txt",
                "a_Spectra.txt",
                "z_ShutterCount.txt",
                "a_ShutterCount.txt",
            ):
                (root / name).write_text(name)
            original_listdir = os.listdir

            def listed(path):
                if path == str(root):
                    return [
                        "z_Spectra.txt",
                        "a_Spectra.txt",
                        "z_ShutterCount.txt",
                        "a_ShutterCount.txt",
                    ]
                return original_listdir(path)

            with patch(
                "NEAT.services.preprocessing_clean.os.listdir", side_effect=listed
            ):
                result = _run(root, {}, root / "out")
            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(
                [Path(item.path).name for item in result.outputs],
                [
                    "gold_outlier_report.csv",
                    "Run1_z_Spectra.txt",
                    "Run1_z_ShutterCount.txt",
                ],
            )
            with patch(
                "NEAT.services.preprocessing_clean.shutil.copy2",
                side_effect=OSError("copy denied"),
            ):
                warned = _run(root, {}, root / "out2")
            self.assertEqual(warned.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(len(warned.warnings), 2)
            self.assertEqual([item.role for item in warned.outputs], ["outlier_report"])
            empty = root / "empty"
            empty.mkdir()
            silent = _run(empty, {}, root / "out3")
            self.assertEqual(silent.warnings, ())

    def test_middle_fits_failure_continues_and_retains_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = {
                key: np.zeros((1, 1), dtype=np.float32)
                for key in ("first", "middle", "last")
            }
            from NEAT.services.image_io import write_fits_image_file

            def write(path, data, overwrite=True):
                if str(path).endswith("gold_middle.fits"):
                    raise OSError("disk denied")
                return write_fits_image_file(path, data, overwrite=overwrite)

            progress, messages, failures = [], [], []
            with patch(
                "NEAT.services.preprocessing_clean.write_fits_image_file",
                side_effect=write,
            ):
                result = _run(
                    root,
                    frames,
                    root / "out",
                    progress_callback=progress.append,
                    message_callback=messages.append,
                    frame_failure_callback=failures.append,
                )
            self.assertEqual(result.status, PreprocessingStatus.FAILED)
            self.assertEqual((result.processed_count, result.expected_count), (2, 3))
            self.assertEqual(failures, ["middle"])
            self.assertIn("Frame middle: Cannot write FITS", result.errors[0])
            self.assertEqual(progress, [33, 66])
            self.assertEqual(
                [Path(item.path).name for item in result.outputs],
                ["gold_outlier_report.csv", "gold_first.fits", "gold_last.fits"],
            )
            self.assertIn(
                "middle,0,0,0.0000,nan",
                (root / "out" / "gold_outlier_report.csv").read_text(),
            )
            self.assertIn("Total cleaned pixels: 2", messages[-1])
            self.assertEqual(sum(m.startswith("[ERROR]") for m in messages), 1)

    def test_report_append_failure_is_warning_and_fits_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_open = open

            def opening(path, mode="r", *args, **kwargs):
                if mode == "a":
                    raise OSError("append denied")
                return real_open(path, mode, *args, **kwargs)

            with patch("builtins.open", side_effect=opening):
                result = _run(root, {"one": np.ones((1, 1))}, root / "out")
            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual((result.processed_count, result.expected_count), (1, 1))
            self.assertEqual(
                result.warnings, ("Could not append to report: append denied",)
            )
            self.assertTrue((root / "out" / "gold_one.fits").exists())

    def test_second_run_listdir_failure_is_fatal_with_partial_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, missing = root / "first", root / "missing"
            first.mkdir()
            (first / "a_Spectra.txt").write_text("a")
            runs = (
                LoadedImageRun(str(first), {"one": np.ones((1, 1))}),
                LoadedImageRun(str(missing), {"two": np.ones((1, 1))}),
            )
            messages = []
            result = clean_loaded_image_runs(
                runs, str(root / "out"), "gold", message_callback=messages.append
            )
            self.assertEqual(result.status, PreprocessingStatus.FAILED)
            self.assertEqual((result.processed_count, result.expected_count), (1, 2))
            self.assertEqual(
                [item.role for item in result.outputs],
                ["outlier_report", "related_file_copy", "cleaned_image"],
            )
            self.assertTrue(all(Path(item.path).exists() for item in result.outputs))
            self.assertFalse((root / "out" / "gold_two.fits").exists())
            self.assertEqual(sum(m.startswith("[FATAL]") for m in messages), 1)

    def test_cancel_after_output_and_after_earlier_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = {key: np.ones((1, 1)) for key in ("one", "two", "three")}
            stopped = [False]
            messages = []

            def progress(value):
                stopped[0] = True

            result = _run(
                root,
                frames,
                root / "out",
                progress_callback=progress,
                cancellation_check=lambda: stopped[0],
                message_callback=messages.append,
            )
            self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
            self.assertEqual((result.processed_count, result.expected_count), (1, 3))
            self.assertEqual(
                [Path(item.path).name for item in result.outputs],
                ["gold_outlier_report.csv", "gold_one.fits"],
            )
            self.assertFalse(any("Process stopped" in m for m in messages))
            self.assertIn("Processed 1 frame(s)", messages[-1])

            def on_failure(suffix):
                stopped[0] = True

            stopped[0] = False
            with patch(
                "NEAT.services.preprocessing_clean.write_fits_image_file",
                side_effect=OSError("fail"),
            ):
                failed_then_cancelled = _run(
                    root,
                    frames,
                    root / "out2",
                    cancellation_check=lambda: stopped[0],
                    frame_failure_callback=on_failure,
                )
            self.assertEqual(
                failed_then_cancelled.status, PreprocessingStatus.CANCELLED
            )
            self.assertEqual(
                (
                    failed_then_cancelled.processed_count,
                    failed_then_cancelled.expected_count,
                ),
                (0, 3),
            )
            self.assertEqual(len(failed_then_cancelled.errors), 1)

    def test_start_of_run_cancellation_message_and_empty_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            messages = []
            result = clean_loaded_image_runs(
                (LoadedImageRun(str(root), {}),),
                str(root / "out"),
                "gold",
                cancellation_check=lambda: True,
                message_callback=messages.append,
            )
            self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
            self.assertEqual(messages[0], "Process stopped by user")
            self.assertTrue(messages[-1].startswith("[SUCCESS] Processed 0"))
            messages.clear()
            empty = clean_loaded_image_runs(
                (),
                str(root / "out"),
                "gold",
                cancellation_check=lambda: True,
                message_callback=messages.append,
            )
            self.assertEqual(empty.status, PreprocessingStatus.CANCELLED)
            self.assertEqual(len(messages), 1)
            self.assertTrue(messages[0].startswith("[SUCCESS] Processed 0"))

    def test_fatal_report_creation_has_no_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "blocked").write_text("file, not folder")
            result = clean_loaded_image_runs((), str(root / "blocked" / "out"), "gold")
            self.assertEqual(result.status, PreprocessingStatus.FAILED)
            self.assertEqual((result.processed_count, result.expected_count), (0, 0))
            self.assertEqual(result.outputs, ())

    def test_fresh_subprocess_import_has_no_qt_ui_or_worker(self):
        code = (
            "import sys; from NEAT.services.preprocessing_clean import clean_loaded_image_runs; "
            "assert not any(k.startswith(('PyQt5', 'NEAT.ui', 'NEAT.workers')) for k in sys.modules); "
            "print('headless')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(completed.stdout.strip(), "headless")


if __name__ == "__main__":
    unittest.main()

"""Issue #33 regressions against the original worker at 0f3acf640925276d0cd8087e54fa2cd4eef092e5."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from NEAT.domain import LoadedImageRun, PreprocessingStatus
from NEAT.services import preprocessing_overlap as overlap
from NEAT.workers.batch import load_image_file
from NEAT.workers.preprocessing import OverlapCorrectionWorker


def frame(value: float) -> np.ndarray:
    image = np.full((512, 512), value, dtype=np.float32)
    image[511, 0] = value + 10
    return image


class OverlapServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        self.messages: list[str] = []
        self.progress: list[int] = []
        self.failed: list[str] = []

    def run_service(self, frames, spectra=None, shutter=None, **kwargs):
        if spectra is None:
            spectra = np.array([[0], [0.00001], [0.00002]], dtype=np.float32)
        if shutter is None:
            shutter = np.array([2000], dtype=np.float32)
        run = LoadedImageRun(str(self.source), frames, load_errors=("loader provenance",))
        return overlap.correct_loaded_image_run(
            run,
            spectra,
            shutter,
            "sample",
            str(self.output),
            message_callback=self.messages.append,
            progress_callback=self.progress.append,
            frame_failure_callback=self.failed.append,
            **kwargs,
        )

    def test_actual_worker_golden_multisegment(self):
        # Captured by executing OverlapCorrectionWorker at the issue baseline.
        frames = {str(n): frame(n) for n in (5, 2, 1, 4, 3)}
        frames["1"][0, 0] = 2000
        frames["2"][0, 0] = 1
        original_frames = {k: v.copy() for k, v in frames.items()}
        spectra = np.array(
            [[0], [0.00001], [0.001], [0.00102], [0.002]], dtype=np.float32
        )
        shutter = np.array([500, 2000, 900, 3000, 4000], dtype=np.float32)
        original_spectra = spectra.copy()
        original_shutter = shutter.copy()
        result = self.run_service(frames, spectra, shutter)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual((result.processed_count, result.expected_count), (5, 5))
        self.assertEqual(self.progress, [20, 40, 60, 80, 100])
        self.assertEqual(
            [Path(output.path).name for output in result.outputs],
            [f"Corrected_sample_{n}.fits" for n in range(1, 6)],
        )
        # Exact float32 FITS values from the original worker; no new-service math
        # is used to derive these expected values.
        expected = [
            (19999999655936.0, 1.0005003213882446, 11.060834884643555),
            (10000000000.0, 2.003004550933838, 12.139605522155762),
            (1.5015075206756592, 1.5015075206756592, 6.528315544128418),
            (2.004685640335083, 2.004685640335083, 7.063601016998291),
            (5.006258010864258, 5.006258010864258, 15.056462287902832),
        ]
        for n, (p00, p10, p5110) in enumerate(expected, start=1):
            path = self.output / f"Corrected_sample_{n}.fits"
            image = load_image_file(path)
            self.assertEqual((float(image[0, 0]), float(image[1, 0]), float(image[511, 0])), (p00, p10, p5110))
            raw = fits.getdata(path)
            self.assertEqual(float(raw[0, 0]), p5110)
            self.assertEqual(float(raw[511, 0]), p00)
        self.assertEqual(
            self.messages[:6],
            [
                "Extracted ToF values from Spectra data.",
                "Identified 3 segments based on ToF intervals.",
                "Selected 3 shutter counts for 3 segments.",
                "Segment intervals: [9.999999747378752e-06, 1.9999919459223747e-05, 1e-05]",
                "Reference interval (segment 1) = 0.0000100",
                "--- Starting Overlap Correction ---",
            ],
        )
        for key in frames:
            np.testing.assert_array_equal(frames[key], original_frames[key])
        np.testing.assert_array_equal(spectra, original_spectra)
        np.testing.assert_array_equal(shutter, original_shutter)

    def test_failed_write_still_advances_cumulative_golden(self):
        frames = {str(n): frame(n * 10) for n in (3, 1, 2)}
        original_write = overlap.write_fits_image_file

        def flaky(path, data, **kwargs):
            if str(path).endswith("_2.fits"):
                raise OSError("injected write failure")
            return original_write(path, data, **kwargs)

        with patch.object(overlap, "write_fits_image_file", side_effect=flaky):
            result = self.run_service(frames)
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual((result.processed_count, result.expected_count), (2, 3))
        self.assertEqual(self.failed, ["2"])
        self.assertEqual(self.progress, [33, 100])
        self.assertEqual(float(load_image_file(self.output / "Corrected_sample_3.fits")[1, 0]), 30.927833557128906)
        self.assertEqual(len(result.outputs), 2)

    def test_setup_aborts_without_sidecars_or_trailing_failure(self):
        (self.source / "x_Spectra.txt").write_text("sidecar")
        cases = [
            (np.array([1], dtype=np.float32), np.array([2000]), "Error extracting ToF values"),
            (np.array([[0]], dtype=np.float32), np.array([], dtype=np.float32), "Insufficient shutter counts"),
            (np.array([[0]], dtype=np.float32), np.array([1000], dtype=np.float32), "Only 0 shutter counts > 1000"),
        ]
        for spectra, shutter, prefix in cases:
            with self.subTest(prefix=prefix):
                self.messages.clear()
                result = self.run_service({"1": frame(1)}, spectra, shutter)
                self.assertEqual(result.status, PreprocessingStatus.FAILED)
                self.assertEqual((result.processed_count, result.expected_count), (0, 1))
                self.assertTrue(self.messages[-1].startswith(prefix))
                self.assertFalse(list(self.output.iterdir()))
        self.messages.clear()
        result = self.run_service({})
        self.assertEqual((result.processed_count, result.expected_count), (0, 0))
        self.assertEqual(self.messages[-1], "No images found in the run. Aborting.")

    def test_pure_intervals_and_strict_boundary(self):
        tof = np.array([0, 0.0001, 0.0002001], dtype=np.float32)
        cuts = np.where(np.diff(tof) > 0.0001)[0] + 1
        segments = np.split(np.arange(len(tof)), cuts)
        self.assertEqual(len(segments), 2)
        self.assertEqual(overlap.segment_mean_intervals(tof, [np.array([], dtype=int), np.array([0]), np.array([0, 1])])[:2], [0.0, 1e-5])
        self.assertEqual(overlap._numeric_key("abc"), -1)
        self.assertEqual(overlap._numeric_key("x01y2"), 12)

    def test_pure_correction_equation_without_qt_or_filesystem(self):
        image = np.array([[100.0]], dtype=np.float32)
        cumulative = image.copy()
        corrected = overlap.correct_overlap_frame(
            image, cumulative, np.float32(2000), 1e-5, 1e-5
        )
        # Original one-frame worker yields 100 / (1 - 100/2000).
        np.testing.assert_allclose(corrected, [[100.0 / 0.95]], rtol=1e-6, atol=0)
        np.testing.assert_array_equal(image, [[100.0]])
        np.testing.assert_array_equal(cumulative, [[100.0]])

    def test_first_insertion_shape_gate_and_later_shape_quirk(self):
        spectra = np.array([[0], [0.001]], dtype=np.float32)
        shutter = np.array([2000, 3000], dtype=np.float32)
        result = self.run_service({"2": np.ones((1, 1), dtype=np.float32), "1": frame(1)}, spectra, shutter)
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertIn("Image dimensions (1, 1)", self.messages[-1])
        self.messages.clear()
        result = self.run_service({"1": frame(1), "2": np.ones((1, 1), dtype=np.float32)}, spectra, shutter)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(fits.getdata(self.output / "Corrected_sample_2.fits").shape, (1, 1))

    def test_missing_segment_and_duplicate_digit_destinations(self):
        spectra = np.array([[0]], dtype=np.float32)
        frames = {"b1": frame(1), "a01": frame(2), "nodigits": frame(3)}
        result = self.run_service(frames, spectra)
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(self.failed, ["b1", "a01"])
        self.assertEqual(self.progress, [33])
        self.assertEqual(Path(result.outputs[0].path).name, "Corrected_sample_.fits")
        self.messages.clear()
        self.progress.clear()
        self.failed.clear()
        spectra = np.array([[0], [0.00001]], dtype=np.float32)
        result = self.run_service({"b1": frame(1), "a-1": frame(2)}, spectra)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual([item.path for item in result.outputs], [str(self.output / "Corrected_sample_1.fits")] * 2)
        self.assertEqual(self.progress, [50, 100])

    def test_nan_frame_fails_after_cumulative_update(self):
        first = frame(1)
        second = frame(2)
        second[1, 0] = np.nan
        third = frame(3)
        result = self.run_service({"1": first, "2": second, "3": third})
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(self.failed, ["2", "3"])
        self.assertEqual(result.processed_count, 1)
        self.assertEqual(self.progress, [33])

    def test_shape_broadcast_and_incompatible_shape_follow_numpy(self):
        spectra = np.array([[0], [0.00001]], dtype=np.float32)
        result = self.run_service(
            {"1": frame(1), "2": np.ones((1, 512), dtype=np.float32)}, spectra
        )
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(result.processed_count, 2)
        self.messages.clear()
        self.failed.clear()
        result = self.run_service(
            {"1": frame(1), "2": np.ones((2, 2), dtype=np.float32)}, spectra
        )
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(self.failed, ["2"])
        self.assertTrue(any("Error processing image '2':" in x for x in self.messages))

    def test_shutter_reference_and_malformed_setup(self):
        result = self.run_service({"1": frame(1)}, np.empty((0, 1), dtype=np.float32))
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertIn("First segment's interval is 0.", self.messages[-1])
        self.messages.clear()
        result = self.run_service({"1": frame(1)}, shutter=object())
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertTrue(self.messages[-1].startswith("Error processing shutter counts:"))

    def test_sidecar_copy_failure_retains_prior_copy_and_stops_sequence(self):
        for name in ("a_Spectra.txt", "b_Spectra.txt", "c_ShutterCount.txt"):
            (self.source / name).write_text(name)
        real_copy = overlap.shutil.copyfile
        calls = []

        def flaky(src, dst):
            calls.append(Path(src).name)
            if Path(src).name == "b_Spectra.txt":
                raise OSError("second copy failed")
            return real_copy(src, dst)

        with patch.object(overlap.shutil, "copyfile", side_effect=flaky):
            result = self.run_service({"1": frame(1)})
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(calls, ["a_Spectra.txt", "b_Spectra.txt"])
        self.assertEqual([Path(item.path).name for item in result.outputs], ["Corrected_sample_1.fits", "a_Spectra.txt"])
        self.assertEqual(len(result.warnings), 1)

    def test_missing_output_folder_is_not_upfront_setup_rejection(self):
        missing = self.root / "missing"
        run = LoadedImageRun(str(self.source), {"1": frame(1)})
        messages = []
        result = overlap.correct_loaded_image_run(
            run, np.array([[0]], dtype=np.float32),
            np.array([2000], dtype=np.float32), "sample", str(missing),
            message_callback=messages.append,
        )
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual((result.processed_count, result.expected_count), (0, 1))
        self.assertIn("--- Starting Overlap Correction ---", messages)
        self.assertTrue(any("Error processing image '1':" in message for message in messages))
        self.assertFalse(missing.exists())

    def test_missing_sidecar_messages_use_legacy_short_path(self):
        result = self.run_service({"1": frame(1)})
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        parts = os.path.normpath(str(self.source)).split(os.sep)
        short_path = os.path.join(parts[-2], parts[-1])
        self.assertEqual(
            self.messages[-2:],
            [
                f"No files ending with '_Spectra.txt' found in \\{short_path}.",
                f"No files ending with '_ShutterCount.txt' found in \\{short_path}.",
            ],
        )

    def test_cancel_after_one_frame_keeps_outputs_and_copies_sidecar(self):
        (self.source / "a_Spectra.txt").write_text("a")
        stopped = False

        def progress(value):
            nonlocal stopped
            self.progress.append(value)
            stopped = True

        run = LoadedImageRun(str(self.source), {"1": frame(1), "2": frame(2)})
        result = overlap.correct_loaded_image_run(
            run, np.array([[0], [0.00001]], dtype=np.float32),
            np.array([2000], dtype=np.float32), "sample", str(self.output),
            progress_callback=progress, message_callback=self.messages.append,
            cancellation_check=lambda: stopped,
        )
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual((result.processed_count, result.expected_count), (1, 2))
        self.assertEqual(self.progress, [50])
        self.assertEqual([item.role for item in result.outputs], ["corrected_image", "related_file_copy"])
        self.assertTrue((self.output / "a_Spectra.txt").exists())

    def test_sidecar_all_matches_two_listdirs_and_warning_nonfatal(self):
        names = ["b_Spectra.txt", "a_Spectra.txt", "z_ShutterCount.txt", "y_ShutterCount.txt"]
        for name in names:
            (self.source / name).write_text(name)
        real_listdir = os.listdir
        calls = []

        def listed(path):
            calls.append(path)
            return names if path == str(self.source) else real_listdir(path)

        with patch.object(overlap.os, "listdir", side_effect=listed):
            result = self.run_service({"1": frame(1)})
        self.assertEqual(calls, [str(self.source), str(self.source)])
        self.assertEqual([Path(item.path).name for item in result.outputs[1:]], names)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.messages.clear()
        with patch.object(overlap.shutil, "copyfile", side_effect=OSError("copy failed")):
            result = self.run_service({"1": frame(1)})
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(result.warnings, ("Error copying spectra or shuttercount files: copy failed",))
        self.assertEqual(len(result.outputs), 1)

    def test_cancellation_before_loop_and_after_last_frame_snapshot(self):
        result = self.run_service({"1": frame(1)}, cancellation_check=lambda: True)
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
        self.assertIn("--- Starting Overlap Correction ---", self.messages)
        self.assertIn("Overlap Correction process has been stopped by the user.", self.messages)
        self.assertEqual(self.messages[-1], "Overlap Correction did not complete successfully.")
        self.messages.clear()
        cancelled = False

        def progress(value):
            nonlocal cancelled
            cancelled = True
            self.progress.append(value)

        run = LoadedImageRun(str(self.source), {"1": frame(1)})
        result = overlap.correct_loaded_image_run(
            run, np.array([[0]], dtype=np.float32), np.array([2000], dtype=np.float32),
            "sample", str(self.output), progress_callback=progress,
            message_callback=self.messages.append, cancellation_check=lambda: cancelled,
        )
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual(result.processed_count, 1)
        self.assertNotIn("Overlap Correction process has been stopped by the user.", self.messages)

    def test_late_sidecar_cancellation_does_not_reclassify(self):
        (self.source / "a_Spectra.txt").write_text("a")
        cancelled = False
        real_copy = overlap.shutil.copyfile

        def copy_then_cancel(src, dst):
            nonlocal cancelled
            real_copy(src, dst)
            cancelled = True

        with patch.object(overlap.shutil, "copyfile", side_effect=copy_then_cancel):
            result = self.run_service({"1": frame(1)}, cancellation_check=lambda: cancelled)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(len(result.outputs), 2)

    def test_import_is_headless_in_fresh_process(self):
        code = "\n".join(
            [
                "import sys, tempfile, pathlib, numpy as np",
                "from NEAT.domain import LoadedImageRun, PreprocessingStatus",
                "from NEAT.services.preprocessing_overlap import correct_loaded_image_run",
                "with tempfile.TemporaryDirectory() as tmp:",
                "    root = pathlib.Path(tmp)",
                "    out = root / 'out'; out.mkdir()",
                "    run = LoadedImageRun(str(root), {'1': np.ones((512, 512), dtype=np.float32)})",
                "    result = correct_loaded_image_run(run, np.array([[0]], dtype=np.float32), np.array([2000], dtype=np.float32), 'sample', str(out))",
                "    assert result.status is PreprocessingStatus.SUCCEEDED",
                "assert not any(x.startswith(('PyQt5', 'NEAT.ui', 'NEAT.workers')) for x in sys.modules)",
            ]
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


class OverlapWorkerAdapterTests(unittest.TestCase):
    def test_success_failure_and_finished_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            output = Path(tmp) / "output"
            output.mkdir()
            run = {
                "folder_path": str(source), "images": {"1": frame(1)},
                "spectra": np.array([[0]], dtype=np.float32),
                "shutter_count": np.array([2000], dtype=np.float32),
                "load_errors": ["provenance only"],
            }
            worker = OverlapCorrectionWorker(run, "sample", str(output))
            finished = []
            progress = []
            worker.finished.connect(lambda: finished.append(True))
            worker.progress_updated.connect(progress.append)
            worker.run()
            self.assertTrue(worker.succeeded)
            self.assertEqual(worker.result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(progress, [100])
            self.assertEqual(len(finished), 1)
            worker = OverlapCorrectionWorker(run, "sample", str(output))
            worker.stop()
            worker.run()
            self.assertFalse(worker.succeeded)
            self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)

    def test_early_abort_and_unexpected_fatal_messages_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            output = Path(tmp) / "output"
            output.mkdir()
            run = {
                "folder_path": str(source), "images": {"1": frame(1)},
                "spectra": np.array([[0]], dtype=np.float32),
                "shutter_count": np.array([], dtype=np.float32),
            }
            worker = OverlapCorrectionWorker(run, "sample", str(output))
            messages = []
            finished = []
            worker.message.connect(messages.append)
            worker.finished.connect(lambda: finished.append(True))
            worker.run()
            self.assertEqual(len(finished), 1)
            self.assertEqual(messages[-1], "Insufficient shutter counts for 1 segments. Aborting run.")
            self.assertEqual(worker.result.expected_count, 1)
            run["shutter_count"] = np.array([2000], dtype=np.float32)
            worker = OverlapCorrectionWorker(run, "sample", str(output))
            messages.clear()
            finished.clear()
            worker.message.connect(messages.append)
            worker.finished.connect(lambda: finished.append(True))
            with patch.object(overlap, "segment_mean_intervals", side_effect=ValueError("injected fatal")):
                worker.run()
            self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
            self.assertEqual(worker.result.errors, ("injected fatal",))
            self.assertEqual(messages.count("Error in OverlapCorrectionWorker: injected fatal"), 1)
            self.assertEqual(messages[-1], "Overlap Correction did not complete successfully.")
            self.assertEqual(len(finished), 1)

    def test_frame_failure_relay_and_ordered_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            output = Path(tmp) / "output"
            output.mkdir()
            run = {
                "folder_path": str(source),
                "images": {"1": frame(1), "2": frame(2), "3": frame(3)},
                "spectra": np.array([[0]], dtype=np.float32),
                "shutter_count": np.array([2000], dtype=np.float32),
            }
            worker = OverlapCorrectionWorker(run, "sample", str(output))
            finished = []
            worker.finished.connect(lambda: finished.append(True))
            worker.run()
            self.assertFalse(worker.succeeded)
            self.assertEqual(worker.failed_frames, ["2", "3"])
            self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
            self.assertEqual(worker.result.processed_count, 1)
            self.assertEqual(len(finished), 1)


if __name__ == "__main__":
    unittest.main()

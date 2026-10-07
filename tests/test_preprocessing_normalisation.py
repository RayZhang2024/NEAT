"""Issue #35: golden and compatibility tests from worker 376be7d471bfe47caa53445a6391799898fe0ef1."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from NEAT.domain import LoadedImageRun, PreprocessingStatus
from NEAT.services import preprocessing_normalisation as service
from NEAT.workers.batch import load_image_file
from NEAT.workers.preprocessing import NormalisationWorker, validate_normalisation_windows


def grid(base: float) -> np.ndarray:
    return np.arange(base, base + 9, dtype=np.float32).reshape(3, 3)


class ClassicNormalisationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sample = self.root / "sample"
        self.beam = self.root / "beam"
        self.output = self.root / "output"
        for folder in (self.sample, self.beam, self.output):
            folder.mkdir()
        self.messages: list[str] = []
        self.progress: list[int] = []
        self.failures: list[str] = []

    def run_service(self, sample_frames=None, beam_frames=None, *, n=0, m=0,
                    samples=None, beams=None, **kwargs):
        if samples is None:
            samples = [LoadedImageRun(str(self.sample), sample_frames or {})]
        if beams is None:
            beams = [LoadedImageRun(str(self.beam), beam_frames or {})]
        return service.normalise_loaded_image_runs(
            samples, beams, str(self.output), "norm", n, m,
            message_callback=self.messages.append,
            progress_callback=self.progress.append,
            failed_frame_callback=self.failures.append,
            **kwargs,
        )

    def write_shutters(self, sample="0,2", beam="0,4"):
        if sample is not None:
            (self.sample / "a_ShutterCount.txt").write_text(sample + "\n9,9999\n")
        if beam is not None:
            (self.beam / "a_ShutterCount.txt").write_text(beam + "\n9,9999\n")

    def test_validator_exact_errors_and_worker_export(self):
        self.assertIs(validate_normalisation_windows, service.validate_normalisation_windows)
        self.assertEqual(validate_normalisation_windows("100", "10"), (100, 10))
        self.assertEqual(validate_normalisation_windows(0, 0), (0, 0))
        cases = [
            (None, 0, "Normalisation n and m must be integers."),
            (0, "x", "Normalisation n and m must be integers."),
            (-1, 0, "Normalisation n must be between 0 and 100."),
            (101, 0, "Normalisation n must be between 0 and 100."),
            (0, -1, "Normalisation m must be between 0 and 10."),
            (0, 11, "Normalisation m must be between 0 and 10."),
        ]
        for n, m, expected in cases:
            with self.subTest(n=n, m=m):
                with self.assertRaisesRegex(ValueError, "^" + expected.replace(".", r"\.") + "$"):
                    validate_normalisation_windows(n, m)
                with self.assertRaises(ValueError):
                    NormalisationWorker([], [], str(self.output), "norm", n, m)

    def test_shutter_reader_raw_first_match_first_line_second_token(self):
        (self.sample / "b_ShutterCount.txt").write_text("1,13\n2,999\n")
        (self.sample / "a_ShutterCount.txt").write_text("4 27\n8 999\n")
        with patch.object(service.os, "listdir", return_value=["b_ShutterCount.txt", "a_ShutterCount.txt"]):
            self.assertEqual(service.read_normalisation_shutter_count(str(self.sample)), 13)
            self.assertEqual(NormalisationWorker._read_shutter_count(str(self.sample)), 13)
        (self.sample / "b_ShutterCount.txt").write_text("broken\n1,13\n")
        with patch.object(service.os, "listdir", return_value=["b_ShutterCount.txt", "a_ShutterCount.txt"]):
            with self.assertRaisesRegex(ValueError, "cannot parse shutter count in b_ShutterCount.txt"):
                service.read_normalisation_shutter_count(str(self.sample))
        with patch.object(service.os, "listdir", return_value=[]):
            with self.assertRaisesRegex(FileNotFoundError, r"no \*_ShutterCount.txt found"):
                service.read_normalisation_shutter_count(str(self.sample))

    def test_pure_n0_m0_golden_and_no_input_mutation(self):
        sample = grid(2)
        beam = grid(4)
        original_sample, original_beam = sample.copy(), beam.copy()
        result = service.normalise_classic_frame(sample, {"x": beam}, ["x"], 0, 0, 0, np.float32(2))
        expected = [
            [1.0, 1.2000000476837158, 1.3333333730697632],
            [1.4285714626312256, 1.5, 1.5555555820465088],
            [1.600000023841858, 1.6363636255264282, 1.6666666269302368],
        ]
        np.testing.assert_array_equal(result, np.array(expected, dtype=np.float32))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(sample, original_sample)
        np.testing.assert_array_equal(beam, original_beam)

    def test_actual_worker_golden_spatial_temporal_and_orientation(self):
        # Original worker run before extraction: three 3x3 frames; common
        # lexicographic order 1,10,2; n=1, m=1; shutter scale=4/2.
        self.write_shutters()
        samples = {key: grid(base) for key, base in (("10", 10), ("2", 20), ("1", 1))}
        beams = {key: grid(base) for key, base in (("10", 30), ("2", 40), ("1", 5), ("extra", 100))}
        originals = {key: image.copy() for key, image in samples.items()}
        beam_originals = {key: image.copy() for key, image in beams.items()}
        result = self.run_service(samples, beams, n=1, m=1)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual((result.processed_count, result.expected_count), (3, 3))
        self.assertEqual(self.progress, [33, 66, 100])
        self.assertEqual([Path(item.path).name for item in result.outputs[:3]],
                         ["norm_1.fits", "norm_10.fits", "norm_2.fits"])
        expected = {
            "1": [[0.10256410390138626, 0.20000000298023224, 0.2926829159259796],
                  [0.380952388048172, 0.4651162922382355, 0.5454545617103577],
                  [0.6222222447395325, 0.695652186870575, 0.7659574747085571]],
            "10": [[0.7407407164573669, 0.800000011920929, 0.8571428656578064],
                   [0.9122806787490845, 0.9655172228813171, 1.01694917678833],
                   [1.0666667222976685, 1.1147540807724, 1.1612902879714966]],
            "2": [[1.0810810327529907, 1.1200000047683716, 1.1578947305679321],
                  [1.1948051452636719, 1.2307692766189575, 1.2658227682113647],
                  [1.2999999523162842, 1.3333333730697632, 1.3658536672592163]],
        }
        for suffix, values in expected.items():
            path = self.output / f"norm_{suffix}.fits"
            np.testing.assert_array_equal(load_image_file(path), np.array(values, dtype=np.float32))
            self.assertEqual(float(fits.getdata(path)[0, 0]), values[2][0])
        self.assertEqual(self.messages[:3], [
            "<b>--- Starting Normalisation ---</b>",
            "Using 3×3 spatial window and 3 frames.",
            "sample=2, open‐beam=4, scale=2.0000",
        ])
        for key, image in samples.items():
            np.testing.assert_array_equal(image, originals[key])
        for key, image in beams.items():
            np.testing.assert_array_equal(image, beam_originals[key])

    def test_only_common_open_beam_frames_contribute(self):
        samples = {"1": grid(1), "2": grid(2)}
        beams = {"1": grid(5), "2": grid(10), "extra": grid(10000)}
        result = service.normalise_classic_frame(samples["1"], beams, ["1", "2"], 0, 1, 1, np.float32(1))
        without_extra = service.normalise_classic_frame(samples["1"], {"1": beams["1"], "2": beams["2"]}, ["1", "2"], 0, 1, 1, np.float32(1))
        np.testing.assert_array_equal(result, without_extra)

    def test_zero_open_beam_and_nonfinite_result_become_zero(self):
        sample = np.array([[2]], dtype=np.float32)
        beam = {"x": np.array([[0]], dtype=np.float32)}
        with np.errstate(all="ignore"):
            result = service.normalise_classic_frame(sample, beam, ["x"], 0, 0, 0, np.float32(1))
        self.assertEqual(float(result[0, 0]), 0)
        with np.errstate(all="ignore"):
            result = service.normalise_classic_frame(sample, {"x": np.array([[4]], dtype=np.float32)}, ["x"], 0, 0, 0, np.float32(np.nan))
        self.assertEqual(float(result[0, 0]), 0)

    def test_shutter_fallback_and_nonfinite_golden(self):
        cases = [
            (None, None, 0.5, "shutter‐count error (no *_ShutterCount.txt found), scale=1.0"),
            ("oops", "0,4", 0.5, "shutter‐count error (cannot parse shutter count in a_ShutterCount.txt), scale=1.0"),
            ("0,0", "0,4", 0.5, "sample=0, open‐beam=4, scale=1.0000"),
            ("0,-2", "0,4", 0.5, "sample=-2, open‐beam=4, scale=1.0000"),
            ("0,nan", "0,4", 0.5, "sample=nan, open‐beam=4, scale=1.0000"),
            ("0,inf", "0,4", 0.0, "sample=inf, open‐beam=4, scale=0.0000"),
            ("0,2", "0,0", 0.0, "sample=2, open‐beam=0, scale=0.0000"),
            ("0,2", "0,nan", 0.0, "sample=2, open‐beam=nan, scale=nan"),
            ("0,2", "0,inf", 0.0, "sample=2, open‐beam=inf, scale=inf"),
        ]
        for sample_count, beam_count, expected, message in cases:
            with self.subTest(sample_count=sample_count, beam_count=beam_count):
                self.messages.clear()
                for path in (self.sample / "a_ShutterCount.txt", self.beam / "a_ShutterCount.txt"):
                    if path.exists():
                        path.unlink()
                self.write_shutters(sample_count, beam_count)
                with np.errstate(all="ignore"):
                    result = self.run_service({"x": np.full((1, 1), 2, dtype=np.float32)},
                                              {"x": np.full((1, 1), 4, dtype=np.float32)})
                self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
                self.assertEqual(self.messages[2], message)
                self.assertEqual(float(load_image_file(self.output / "norm_x.fits")[0, 0]), expected)

    def test_early_run_validation_and_no_summary(self):
        run = LoadedImageRun(str(self.sample), {"x": grid(1)})
        empty_result = self.run_service(samples=[], beams=[])
        self.assertEqual((empty_result.status, empty_result.expected_count),
                         (PreprocessingStatus.FAILED, 0))
        self.assertEqual(self.messages, ["No runs provided. Aborting."])
        self.messages.clear()
        result = self.run_service(samples=[run], beams=[])
        self.assertEqual((result.status, result.processed_count, result.expected_count),
                         (PreprocessingStatus.FAILED, 0, 1))
        self.assertEqual(self.messages, ["No runs provided. Aborting."])
        self.messages.clear()
        result = self.run_service(samples=[run, run], beams=[run])
        self.assertEqual((result.status, result.expected_count), (PreprocessingStatus.FAILED, 2))
        self.assertEqual(self.messages, ["Data vs. Open‐beam count mismatch. Aborting."])
        self.assertEqual(list(self.output.iterdir()), [])

    def test_missing_output_folder_fails_per_frame_not_during_setup(self):
        missing = self.root / "missing"
        sample = LoadedImageRun(str(self.sample), {"x": grid(1)}, load_errors=("provenance",))
        beam = LoadedImageRun(str(self.beam), {"x": grid(5)})
        result = service.normalise_loaded_image_runs(
            [sample], [beam], str(missing), "norm", 0, 0,
            message_callback=self.messages.append,
        )
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual((result.processed_count, result.expected_count), (0, 1))
        self.assertIn("<b>--- Starting Normalisation ---</b>", self.messages)
        self.assertTrue(any(text.startswith(" x: error (") for text in self.messages))
        self.assertFalse(missing.exists())

    def test_no_common_skips_cleanup_sidecars_and_preserves_mapping(self):
        (self.sample / "a_Spectra.txt").write_text("sidecar")
        samples = {"sample": grid(1)}
        worker = NormalisationWorker(
            [{"folder_path": str(self.sample), "images": samples}],
            [{"folder_path": str(self.beam), "images": {"beam": grid(5)}}],
            str(self.output), "norm", 0, 0,
        )
        worker.message.connect(self.messages.append)
        with patch("NEAT.workers.preprocessing.QThread.sleep") as sleep:
            worker.run()
        self.assertFalse(worker.succeeded)
        self.assertEqual(worker.failed_frames, ["run-1:no-matching-suffix"])
        self.assertEqual(list(samples), ["sample"])
        self.assertFalse((self.output / "Run1_a_Spectra.txt").exists())
        self.assertNotIn("Run done.", self.messages)
        sleep.assert_not_called()

    def test_unmatched_sample_and_extra_beam_suffixes(self):
        sample = {"a": grid(1), "b": grid(2)}
        beam = {"a": grid(5), "extra": grid(7)}
        result = self.run_service(sample, beam)
        self.assertEqual((result.status, result.processed_count, result.expected_count),
                         (PreprocessingStatus.FAILED, 1, 2))
        self.assertEqual(self.progress, [50])
        self.assertEqual(self.failures, [])
        self.assertEqual(len(result.errors), 1)
        self.messages.clear()
        self.progress.clear()
        result = self.run_service({"a": grid(1)}, beam)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(self.progress, [100])

    def test_shape_window_and_write_failures_continue_in_order(self):
        samples = {"a": grid(1), "b": np.ones((1, 1), dtype=np.float32),
                   "c": grid(3), "d": grid(4)}
        beams = {"a": np.ones((2, 2), dtype=np.float32),
                 "b": np.ones((1, 1), dtype=np.float32),
                 "c": grid(8), "d": grid(8)}
        original = service.write_fits_image_file

        def flaky(path, data, **kwargs):
            if str(path).endswith("_c.fits"):
                raise OSError("write failed")
            return original(path, data, **kwargs)

        with patch.object(service, "write_fits_image_file", side_effect=flaky):
            result = self.run_service(samples, beams, n=1)
        self.assertEqual(self.failures, ["a", "b", "c"])
        self.assertEqual(self.progress, [25])
        self.assertEqual((result.processed_count, result.expected_count), (1, 4))
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(len(result.outputs), 1)
        self.assertEqual(Path(result.outputs[0].path).name, "norm_d.fits")
        self.assertIn(" a: shape mismatch—skipping.", self.messages)
        self.assertIn(" b: too small for window—skipping.", self.messages)
        self.assertIn(" c: error (write failed)—skipping.", self.messages)

    def test_post_write_mutation_failure_keeps_artifact_not_count(self):
        callbacks = []

        def fail_first(run_idx, suffix):
            callbacks.append((run_idx, suffix))
            if suffix == "a":
                raise KeyError("deletion failed")

        result = self.run_service(
            {"a": grid(1), "b": grid(2)}, {"a": grid(5), "b": grid(6)},
            written_frame_callback=fail_first,
        )
        self.assertEqual(callbacks, [(1, "a"), (1, "b")])
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual((result.processed_count, result.expected_count), (1, 2))
        self.assertEqual(self.failures, ["a"])
        self.assertEqual(self.progress, [50])
        self.assertEqual([Path(item.path).name for item in result.outputs], ["norm_a.fits", "norm_b.fits"])
        self.assertTrue((self.output / "norm_a.fits").exists())

    def test_multi_run_overwrite_collision_golden(self):
        s2 = self.root / "sample2"
        o2 = self.root / "beam2"
        s2.mkdir()
        o2.mkdir()
        samples = [LoadedImageRun(str(self.sample), {"x": np.full((1, 1), 2, dtype=np.float32)}),
                   LoadedImageRun(str(s2), {"x": np.full((1, 1), 4, dtype=np.float32)})]
        beams = [LoadedImageRun(str(self.beam), {"x": np.full((1, 1), 4, dtype=np.float32)}),
                 LoadedImageRun(str(o2), {"x": np.full((1, 1), 4, dtype=np.float32)})]
        result = self.run_service(samples=samples, beams=beams)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual((result.processed_count, result.expected_count), (2, 2))
        self.assertEqual(self.progress, [50, 100])
        self.assertEqual([item.path for item in result.outputs], [str(self.output / "norm_x.fits")] * 2)
        self.assertEqual(float(load_image_file(self.output / "norm_x.fits")[0, 0]), 1.0)

    def test_sidecars_all_raw_order_one_listdir_and_warning(self):
        names = ["b_ShutterCount.txt", "b_Spectra.txt", "a_Spectra.txt", "a_ShutterCount.txt"]
        for name in names:
            (self.sample / name).write_text("0,2\n")
        real_listdir = os.listdir
        calls = []

        def listed(path):
            calls.append(path)
            return names if path == str(self.sample) else real_listdir(path)

        with patch.object(service.os, "listdir", side_effect=listed):
            outputs, warnings = service.copy_normalisation_related_files(
                2, str(self.sample), str(self.output), message_callback=self.messages.append
            )
        self.assertEqual(calls, [str(self.sample)])
        self.assertEqual(warnings, ())
        self.assertEqual([Path(item.path).name for item in outputs],
                         ["Run2_b_Spectra.txt", "Run2_a_Spectra.txt",
                          "Run2_b_ShutterCount.txt", "Run2_a_ShutterCount.txt"])
        self.assertEqual([item.role for item in outputs], ["related_file_copy"] * 4)
        self.messages.clear()
        real_copy = service.shutil.copyfile
        copied = []

        def flaky(src, dst):
            copied.append(Path(src).name)
            if len(copied) == 2:
                raise OSError("copy failed")
            return real_copy(src, dst)

        with patch.object(service.os, "listdir", return_value=names), patch.object(service.shutil, "copyfile", side_effect=flaky):
            outputs, warnings = service.copy_normalisation_related_files(
                3, str(self.sample), str(self.output), message_callback=self.messages.append
            )
        self.assertEqual(copied, ["b_Spectra.txt", "a_Spectra.txt"])
        self.assertEqual(len(outputs), 1)
        self.assertEqual(warnings, ("Error copying related files: copy failed",))

    def test_sidecar_warning_does_not_fail_service(self):
        (self.sample / "a_Spectra.txt").write_text("a")
        with patch.object(service.shutil, "copyfile", side_effect=OSError("copy failed")):
            result = self.run_service({"a": grid(1)}, {"a": grid(5)})
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(result.warnings, ("Error copying related files: copy failed",))
        self.assertEqual(result.processed_count, 1)

    def test_inner_cancellation_cleans_run_and_second_run_emits_boundary_message(self):
        s2 = self.root / "sample2"
        o2 = self.root / "beam2"
        s2.mkdir()
        o2.mkdir()
        samples = [LoadedImageRun(str(self.sample), {"a": grid(1), "b": grid(2)}),
                   LoadedImageRun(str(s2), {"a": grid(3)})]
        beams = [LoadedImageRun(str(self.beam), {"a": grid(5), "b": grid(6)}),
                 LoadedImageRun(str(o2), {"a": grid(7)})]
        cancelled = False
        cleaned = []

        def progress(value):
            nonlocal cancelled
            self.progress.append(value)
            cancelled = True

        result = service.normalise_loaded_image_runs(
            samples, beams, str(self.output), "norm", 0, 0,
            progress_callback=progress, message_callback=self.messages.append,
            cancellation_check=lambda: cancelled,
            run_cleanup_callback=cleaned.append,
        )
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual((result.processed_count, result.expected_count), (1, 3))
        self.assertEqual(cleaned, [1])
        self.assertEqual(self.messages.count("User stopped the process."), 1)
        self.assertIn("Run done.", self.messages)

    def test_final_run_inner_cancellation_no_outer_stop_message(self):
        cancelled = False

        def progress(value):
            nonlocal cancelled
            cancelled = True

        result = self.run_service(
            {"a": grid(1), "b": grid(2)}, {"a": grid(5), "b": grid(6)},
            cancellation_check=lambda: cancelled,
            written_frame_callback=lambda *_: progress(0),
        )
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
        self.assertNotIn("User stopped the process.", self.messages)
        self.assertIn("Run done.", self.messages)

    def test_cancelled_result_retains_prior_frame_error(self):
        cancelled = False

        def failed(_suffix):
            nonlocal cancelled
            cancelled = True

        sample = LoadedImageRun(str(self.sample), {"a": grid(1), "b": grid(2)})
        beam = LoadedImageRun(str(self.beam), {
            "a": np.ones((2, 2), dtype=np.float32), "b": grid(5)
        })
        result = service.normalise_loaded_image_runs(
            [sample], [beam], str(self.output), "norm", 0, 0,
            cancellation_check=lambda: cancelled,
            failed_frame_callback=failed,
            message_callback=self.messages.append,
        )
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual(result.errors, (" a: shape mismatch—skipping.",))
        self.assertIn("Run done.", self.messages)

    def test_cancellation_during_sidecar_or_pacing_changes_final_snapshot(self):
        (self.sample / "a_Spectra.txt").write_text("a")
        cancelled = False
        real_copy = service.shutil.copyfile

        def copy_then_stop(src, dst):
            nonlocal cancelled
            real_copy(src, dst)
            cancelled = True

        with patch.object(service.shutil, "copyfile", side_effect=copy_then_stop):
            result = self.run_service({"a": grid(1)}, {"a": grid(5)}, cancellation_check=lambda: cancelled)
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
        self.assertNotIn("User stopped the process.", self.messages)
        self.assertEqual(len(result.outputs), 2)
        self.messages.clear()
        cancelled = False

        def pace(_run_idx):
            nonlocal cancelled
            cancelled = True

        result = self.run_service({"a": grid(1)}, {"a": grid(5)},
                                  cancellation_check=lambda: cancelled,
                                  run_complete_callback=pace)
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)

    def test_no_recheck_after_final_snapshot(self):
        cancelled = False

        def message(text):
            nonlocal cancelled
            self.messages.append(text)
            if text.startswith("Normalisation completed:"):
                cancelled = True

        run = LoadedImageRun(str(self.sample), {"a": grid(1)})
        beam = LoadedImageRun(str(self.beam), {"a": grid(5)})
        result = service.normalise_loaded_image_runs(
            [run], [beam], str(self.output), "norm", 0, 0,
            message_callback=message, cancellation_check=lambda: cancelled,
        )
        self.assertTrue(cancelled)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)

    def test_fresh_process_headless_service_import_and_run(self):
        code = "\n".join([
            "import sys, tempfile, pathlib, numpy as np",
            "from NEAT.domain import LoadedImageRun, PreprocessingStatus",
            "from NEAT.services.preprocessing_normalisation import normalise_loaded_image_runs",
            "with tempfile.TemporaryDirectory() as tmp:",
            "    root = pathlib.Path(tmp); sample = root/'sample'; beam = root/'beam'; out = root/'out'",
            "    sample.mkdir(); beam.mkdir(); out.mkdir()",
            "    a = LoadedImageRun(str(sample), {'x': np.ones((1,1), dtype=np.float32)})",
            "    b = LoadedImageRun(str(beam), {'x': np.ones((1,1), dtype=np.float32)})",
            "    result = normalise_loaded_image_runs([a], [b], str(out), 'norm', 0, 0)",
            "    assert result.status is PreprocessingStatus.SUCCEEDED",
            "assert not any(x.startswith(('PyQt5', 'NEAT.ui', 'NEAT.workers')) for x in sys.modules)",
        ])
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class NormalisationAdapterTests(unittest.TestCase):
    def test_worker_mutation_cleanup_sidecars_pacing_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, beam, output = (root / name for name in ("sample", "beam", "output"))
            for folder in (sample, beam, output):
                folder.mkdir()
            (sample / "a_Spectra.txt").write_text("a")
            events = []

            class TrackedImages(dict):
                def __delitem__(self, key):
                    events.append("delete:" + key)
                    super().__delitem__(key)

                def clear(self):
                    events.append("clear")
                    super().clear()

            images = TrackedImages({"a": grid(1), "unmatched": grid(2)})
            unmatched_ref = weakref.ref(images["unmatched"])
            beam_images = {"a": grid(5)}
            worker = NormalisationWorker(
                [{"folder_path": str(sample), "images": images}],
                [{"folder_path": str(beam), "images": beam_images}],
                str(output), "norm", 0, 0,
            )
            messages, finished, progress = [], [], []
            worker.message.connect(lambda text: (messages.append(text), events.append(text)))
            worker.finished.connect(lambda: finished.append(True))
            worker.progress_updated.connect(progress.append)
            original_copy = service.copy_normalisation_related_files

            def copy(*args, **kwargs):
                events.append("copy")
                self.assertIsNone(unmatched_ref())
                return original_copy(*args, **kwargs)

            with patch.object(service, "copy_normalisation_related_files", side_effect=copy), \
                 patch("NEAT.workers.preprocessing.QThread.sleep", side_effect=lambda _: events.append("sleep")) as sleep, \
                 patch("NEAT.workers.preprocessing.gc.collect", side_effect=lambda: events.append("gc")):
                worker.run()
            self.assertFalse(worker.succeeded)  # unmatched sample frame
            self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
            self.assertEqual((worker.result.processed_count, worker.result.expected_count), (1, 2))
            self.assertEqual(images, {})
            self.assertEqual(list(beam_images), ["a"])
            self.assertEqual(progress, [50])
            self.assertEqual(len(finished), 1)
            self.assertEqual(sleep.call_count, 1)
            self.assertTrue(any(text.startswith("<b>Final memory usage:</b>") for text in messages))
            self.assertLess(events.index("delete:a"), events.index("clear"))
            self.assertLess(events.index("clear"), events.index("gc"))
            self.assertLess(events.index("gc"), events.index("copy"))
            self.assertLess(events.index("copy"), events.index("Run done."))
            self.assertLess(events.index("Run done."), events.index("sleep"))

    def test_worker_prestop_and_public_sidecar_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, beam, output = (root / name for name in ("sample", "beam", "output"))
            for folder in (sample, beam, output):
                folder.mkdir()
            (sample / "a_Spectra.txt").write_text("a")
            images = {"x": grid(1)}
            run = {"folder_path": str(sample), "images": images}
            worker = NormalisationWorker([run], [{"folder_path": str(beam), "images": {"x": grid(5)}}], str(output), "norm", 0, 0)
            messages, finished = [], []
            worker.message.connect(messages.append)
            worker.finished.connect(lambda: finished.append(True))
            worker.stop()
            with patch("NEAT.workers.preprocessing.QThread.sleep") as sleep:
                worker.run()
            self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)
            self.assertEqual(list(images), ["x"])
            self.assertIn("Stop signal received. Terminating Normalisation process.", messages)
            self.assertIn("<b>--- Starting Normalisation ---</b>", messages)
            self.assertIn("User stopped the process.", messages)
            self.assertEqual(len(finished), 1)
            sleep.assert_not_called()
            worker.copy_related_files(1, run)
            self.assertTrue((output / "Run1_a_Spectra.txt").exists())

    def test_worker_post_write_delete_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, beam, output = (root / name for name in ("sample", "beam", "output"))
            for folder in (sample, beam, output):
                folder.mkdir()

            class DeleteFailingImages(dict):
                def __delitem__(self, key):
                    if key == "a":
                        raise OSError("delete failed")
                    super().__delitem__(key)

            images = DeleteFailingImages({"a": grid(1), "b": grid(2)})
            worker = NormalisationWorker(
                [{"folder_path": str(sample), "images": images}],
                [{"folder_path": str(beam), "images": {"a": grid(5), "b": grid(6)}}],
                str(output), "norm", 0, 0,
            )
            progress, finished = [], []
            worker.progress_updated.connect(progress.append)
            worker.finished.connect(lambda: finished.append(True))
            with patch("NEAT.workers.preprocessing.QThread.sleep"):
                worker.run()
            self.assertEqual(worker.failed_frames, ["a"])
            self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
            self.assertEqual(worker.result.processed_count, 1)
            self.assertEqual(progress, [50])
            self.assertEqual([Path(item.path).name for item in worker.result.outputs], ["norm_a.fits", "norm_b.fits"])
            self.assertEqual(len(finished), 1)

    def test_worker_fatal_path_and_unchanged_memory_reporting_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, beam, output = (root / name for name in ("sample", "beam", "output"))
            for folder in (sample, beam, output):
                folder.mkdir()
            data = {"folder_path": str(sample), "images": {"x": grid(1)}}
            ob = {"folder_path": str(beam), "images": {"x": grid(5)}}
            worker = NormalisationWorker([data], [ob], str(output), "norm", 0, 0)
            messages, finished = [], []
            worker.message.connect(messages.append)
            worker.finished.connect(lambda: finished.append(True))
            with patch.object(worker, "_cleanup_sample_run", side_effect=OSError("cleanup failed")):
                worker.run()
            self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
            self.assertEqual(worker.result.errors, ("cleanup failed",))
            self.assertEqual(messages.count("Fatal error in normalisation: cleanup failed"), 1)
            self.assertEqual(len(finished), 1)
            # The original worker's finally block did not catch RSS-query errors;
            # preserving that behavior means finished is not emitted in this case.
            worker = NormalisationWorker([], [], str(output), "norm", 0, 0)
            finished.clear()
            worker.finished.connect(lambda: finished.append(True))
            with patch("NEAT.workers.preprocessing.psutil.Process", side_effect=OSError("RSS failed")):
                with self.assertRaisesRegex(OSError, "RSS failed"):
                    worker.run()
            self.assertEqual(finished, [])


if __name__ == "__main__":
    unittest.main()

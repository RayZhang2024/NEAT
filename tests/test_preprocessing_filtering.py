"""Headless Filtering regressions against the original worker at 44c62653.

The golden arrays, output order, messages, and progress below were captured by
executing FilteringWorker at 44c62653f27899506dbf7d19cdf1249fafe6ed11
before extraction. Exact float32 equality is required (rtol=0, atol=0).
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from NEAT.domain import LoadedImageRun, PreprocessingStatus
from NEAT.services.preprocessing_filtering import (
    apply_binary_mask,
    copy_filtering_related_files,
    filter_loaded_image_runs,
    short_filtering_path,
)
from NEAT.workers.batch import load_image_file

MASK = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int16)


def _golden_runs(root):
    source1, source2 = root / "run1", root / "run2"
    source1.mkdir()
    source2.mkdir()
    for folder, filename in (
        (source1, "r1_Spectra.txt"),
        (source1, "r1_ShutterCount.txt"),
        (source2, "r2_Spectra.txt"),
        (source2, "r2_ShutterCount.txt"),
    ):
        (folder / filename).write_text(filename, encoding="utf-8")
    run1 = LoadedImageRun(
        str(source1),
        {
            "z": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16),
            "a": np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float64),
        },
        source_folders=(str(source2),),
        load_errors=("ignored by Filtering",),
    )
    run2 = LoadedImageRun(
        str(source2),
        {"b": np.array([[-1, np.nan, np.inf], [4, 5, 6]], dtype=np.float32)},
    )
    return run1, run2


def _run(runs, mask, output, **callbacks):
    return filter_loaded_image_runs(runs, mask, str(output), "gold", **callbacks)


class TestPureFiltering(unittest.TestCase):
    def test_keep_discard_dtype_nan_inf_and_no_input_mutation(self):
        image = np.array([[np.nan, np.inf, 3], [4, 5, 6]], dtype=np.float64)
        mask = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.float32)
        image_before, mask_before = image.copy(), mask.copy()
        filtered = apply_binary_mask(image, mask)
        self.assertEqual(filtered.dtype, np.float32)
        self.assertTrue(np.isnan(filtered[0, 0]))
        self.assertTrue(np.isposinf(filtered[0, 1]))
        np.testing.assert_array_equal(filtered[1], [0, 5, 0])
        self.assertEqual(filtered[0, 2], 0)
        np.testing.assert_array_equal(image, image_before)
        np.testing.assert_array_equal(mask, mask_before)

    def test_no_extra_broadcast_policy_in_pure_transform(self):
        image = np.array([[2, 3]], dtype=np.float32)
        mask = np.array([[1], [0]], dtype=np.float32)
        np.testing.assert_array_equal(apply_binary_mask(image, mask), [[2, 3], [0, 0]])


class TestFilteringService(unittest.TestCase):
    def test_original_worker_golden_arrays_output_order_and_dual_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = _golden_runs(root)
            output = root / "output"
            output.mkdir()
            original_frames = [
                frame.copy() for run in runs for frame in run.frames.values()
            ]
            mask_before = MASK.copy()
            messages, progress = [], []
            result = _run(
                runs,
                MASK,
                output,
                message_callback=messages.append,
                progress_callback=progress.append,
            )
            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual((result.processed_count, result.expected_count), (3, 3))
            self.assertEqual(progress, [33, 66, 50, 100, 100])
            self.assertEqual(
                [item.role for item in result.outputs],
                [
                    "related_file_copy",
                    "related_file_copy",
                    "filtered_image",
                    "filtered_image",
                    "related_file_copy",
                    "related_file_copy",
                    "filtered_image",
                ],
            )
            self.assertEqual(
                [Path(item.path).name for item in result.outputs],
                [
                    "Run1_r1_Spectra.txt",
                    "Run1_r1_ShutterCount.txt",
                    "gold_a.fits",
                    "gold_z.fits",
                    "Run2_r2_Spectra.txt",
                    "Run2_r2_ShutterCount.txt",
                    "gold_b.fits",
                ],
            )
            self.assertEqual(
                messages,
                [
                    "Filtering started...",
                    "Copied 'r1_Spectra.txt' to 'Run1_r1_Spectra.txt'.",
                    "Copied 'r1_ShutterCount.txt' to 'Run1_r1_ShutterCount.txt'.",
                    "Copied 'r2_Spectra.txt' to 'Run2_r2_Spectra.txt'.",
                    "Copied 'r2_ShutterCount.txt' to 'Run2_r2_ShutterCount.txt'.",
                    f"Filtering completed. 3 of 3 images saved to {short_filtering_path(str(output))}.",
                ],
            )
            expected = {
                "gold_a.fits": np.array([[7, 0, 9], [0, 11, 0]], dtype=np.float32),
                "gold_z.fits": np.array([[1, 0, 3], [0, 5, 0]], dtype=np.float32),
                "gold_b.fits": np.array([[-1, 0, np.inf], [0, 5, 0]], dtype=np.float32),
            }
            for filename, array in expected.items():
                with self.subTest(filename=filename):
                    np.testing.assert_array_equal(
                        load_image_file(output / filename), array
                    )
                    np.testing.assert_array_equal(
                        fits.getdata(output / filename), np.flipud(array)
                    )
            np.testing.assert_array_equal(MASK, mask_before)
            for before, frame in zip(
                original_frames, (f for r in runs for f in r.frames.values())
            ):
                np.testing.assert_array_equal(frame, before)

    def test_initial_validation_order_messages_and_no_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = LoadedImageRun(str(root), {"x": np.ones((1, 1))})
            cases = (
                ((), None, "No filtering data runs to process.", 0),
                ((run,), None, "No mask image provided.", 1),
                (
                    (run,),
                    np.array([[np.nan]]),
                    "Error during filtering: Filtering mask contains NaN or infinite values.",
                    1,
                ),
                (
                    (run,),
                    np.array([[2]]),
                    "Error during filtering: Filtering mask must be binary and contain only 0 and 1.",
                    1,
                ),
            )
            for runs, mask, expected_message, expected_count in cases:
                with self.subTest(message=expected_message):
                    messages = []
                    result = _run(
                        runs, mask, root / "absent", message_callback=messages.append
                    )
                    self.assertEqual(result.status, PreprocessingStatus.FAILED)
                    self.assertEqual(
                        (result.processed_count, result.expected_count),
                        (0, expected_count),
                    )
                    self.assertEqual(result.outputs, ())
                    self.assertEqual(messages, [expected_message])
                    self.assertEqual(result.errors, (expected_message,))
                    self.assertFalse((root / "absent").exists())

    def test_accepted_numeric_masks_and_generic_setup_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = LoadedImageRun(str(root), {"x": np.ones((1, 1))})
            output = root / "out"
            output.mkdir()
            for mask in (
                np.array([[True]]),
                np.array([[1]], dtype=np.int64),
                np.array([[1.0]]),
            ):
                with self.subTest(dtype=mask.dtype):
                    result = _run((run,), mask, output)
                    self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            messages = []
            with patch(
                "NEAT.services.preprocessing_filtering.np.unique",
                side_effect=RuntimeError("setup broke"),
            ):
                result = _run(
                    (run,),
                    np.ones((1, 1)),
                    root / "absent",
                    message_callback=messages.append,
                )
            self.assertEqual(result.status, PreprocessingStatus.FAILED)
            self.assertEqual((result.processed_count, result.expected_count), (0, 1))
            self.assertEqual(result.outputs, ())
            self.assertEqual(messages, ["Error during filtering: setup broke"])
            self.assertFalse((root / "absent").exists())

    def test_keyboard_interrupt_and_system_exit_are_not_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = LoadedImageRun(tmp, {"x": np.ones((1, 1))})
            for exception in (KeyboardInterrupt(), SystemExit()):
                with (
                    self.subTest(exception=type(exception)),
                    patch(
                        "NEAT.services.preprocessing_filtering.np.unique",
                        side_effect=exception,
                    ),
                    self.assertRaises(type(exception)),
                ):
                    _run((run,), np.ones((1, 1)), Path(tmp))

    def test_one_zero_frame_run_still_copies_sidecars_and_emits_run_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a_Spectra.txt").write_text("s")
            output = root / "out"
            output.mkdir()
            progress, messages = [], []
            result = _run(
                (LoadedImageRun(str(root), {}),),
                np.array([[1]]),
                output,
                progress_callback=progress.append,
                message_callback=messages.append,
            )
            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual((result.processed_count, result.expected_count), (0, 0))
            self.assertEqual(progress, [100])
            self.assertEqual(
                [Path(item.path).name for item in result.outputs],
                ["Run1_a_Spectra.txt"],
            )
            self.assertIn(
                f"No shuttercount file (*_ShutterCount.txt) found in {root}.", messages
            )
            self.assertEqual(
                messages[-1],
                f"Filtering completed. 0 of 0 images saved to {short_filtering_path(str(output))}.",
            )

    def test_shape_and_fits_failures_continue_with_ordered_errors_and_partial_outputs(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            run = LoadedImageRun(
                str(root),
                {
                    "z": np.ones((2, 2)),
                    "a": np.ones((1, 1)),
                    "m": np.ones((2, 2)),
                    "q": np.ones((2, 2)),
                },
            )
            from NEAT.services.image_io import write_fits_image_file

            def write(path, data, overwrite=True):
                if str(path).endswith("gold_m.fits"):
                    raise OSError("disk full")
                return write_fits_image_file(path, data, overwrite=overwrite)

            progress, failures, messages = [], [], []
            with patch(
                "NEAT.services.preprocessing_filtering.write_fits_image_file",
                side_effect=write,
            ):
                result = _run(
                    (run,),
                    np.ones((2, 2)),
                    output,
                    progress_callback=progress.append,
                    frame_failure_callback=failures.append,
                    message_callback=messages.append,
                )
            self.assertEqual(result.status, PreprocessingStatus.FAILED)
            self.assertEqual((result.processed_count, result.expected_count), (2, 4))
            self.assertEqual(failures, ["a", "m"])
            self.assertEqual(
                result.errors[0],
                "Image a: Mask shape (2, 2) does not match image shape (1, 1). Skipping.",
            )
            self.assertEqual(
                result.errors[1],
                "Image m: Failed to save 'gold_m.fits': disk full. Skipping.",
            )
            self.assertEqual(progress, [25, 50, 100])
            self.assertEqual(
                [Path(item.path).name for item in result.outputs],
                ["gold_q.fits", "gold_z.fits"],
            )
            self.assertEqual(
                messages[-1],
                f"Filtering failed or incomplete. 2 of 4 images saved to {short_filtering_path(str(output))}.",
            )

    def test_unexpected_frame_exception_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            run = LoadedImageRun(
                str(root), {"a": np.ones((1, 1)), "b": np.ones((1, 1))}
            )
            original = apply_binary_mask

            def transform(image, mask):
                if len(calls) == 0:
                    calls.append(True)
                    raise RuntimeError("transform broke")
                return original(image, mask)

            calls = []
            with patch(
                "NEAT.services.preprocessing_filtering.apply_binary_mask",
                side_effect=transform,
            ):
                result = _run((run,), np.ones((1, 1)), output)
            self.assertEqual(result.status, PreprocessingStatus.FAILED)
            self.assertEqual((result.processed_count, result.expected_count), (1, 2))
            self.assertEqual(
                result.errors, ("Error filtering image a: transform broke",)
            )
            self.assertTrue((output / "gold_b.fits").exists())

    def test_invalid_source_missing_sidecars_and_no_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "missing-output"
            missing = root / "missing-source"
            messages = []
            run = LoadedImageRun(str(missing), {"x": np.ones((1, 1))})
            result = _run(
                (run,), np.ones((1, 1)), output, message_callback=messages.append
            )
            self.assertEqual(result.status, PreprocessingStatus.FAILED)
            self.assertEqual(result.warnings, ())
            self.assertIn(f"Data run folder not found or invalid: {missing}", messages)
            self.assertFalse(output.exists())
            self.assertIn("Failed to save 'gold_x.fits'", result.errors[0])
            output.mkdir()
            messages.clear()
            missing_source_only = _run(
                (run,), np.ones((1, 1)), output, message_callback=messages.append
            )
            self.assertEqual(missing_source_only.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(missing_source_only.warnings, ())
            self.assertIn(f"Data run folder not found or invalid: {missing}", messages)
            empty = root / "empty"
            empty.mkdir()
            messages.clear()
            result = _run(
                (LoadedImageRun(str(empty), {"x": np.ones((1, 1))}),),
                np.ones((1, 1)),
                output,
                message_callback=messages.append,
            )
            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(result.warnings, ())
            self.assertIn(
                f"No spectra file (*_Spectra.txt) found in {empty}.", messages
            )
            self.assertIn(
                f"No shuttercount file (*_ShutterCount.txt) found in {empty}.", messages
            )

    def test_one_unsorted_listdir_first_match_copyfile_and_primary_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary, other, output = root / "primary", root / "other", root / "out"
            primary.mkdir()
            other.mkdir()
            output.mkdir()
            names = [
                "z_Spectra.txt",
                "a_Spectra.txt",
                "z_ShutterCount.txt",
                "a_ShutterCount.txt",
            ]
            for name in names:
                (primary / name).write_text(name)
            (other / "other_Spectra.txt").write_text("wrong")
            run = LoadedImageRun(
                str(primary),
                {"x": np.ones((1, 1))},
                source_folders=(str(other),),
                load_errors=("ignored",),
            )
            original_listdir = os.listdir

            def listed(path):
                return names if path == str(primary) else original_listdir(path)

            with (
                patch(
                    "NEAT.services.preprocessing_filtering.os.listdir",
                    side_effect=listed,
                ) as listdir,
                patch(
                    "NEAT.services.preprocessing_filtering.shutil.copyfile",
                    wraps=shutil.copyfile,
                ) as copyfile,
            ):
                result = _run((run,), np.ones((1, 1)), output)
            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(listdir.call_count, 1)
            self.assertEqual(copyfile.call_count, 2)
            self.assertEqual(
                [Path(item.path).name for item in result.outputs],
                ["Run1_z_Spectra.txt", "Run1_z_ShutterCount.txt", "gold_x.fits"],
            )
            self.assertFalse((output / "Run1_other_Spectra.txt").exists())

    def test_spectra_copy_failure_blocks_shutter_and_sidecar_warning_nonfatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            (root / "a_Spectra.txt").write_text("s")
            (root / "b_ShutterCount.txt").write_text("c")
            with patch(
                "NEAT.services.preprocessing_filtering.shutil.copyfile",
                side_effect=OSError("copy denied"),
            ) as copy:
                result = _run(
                    (LoadedImageRun(str(root), {"x": np.ones((1, 1))}),),
                    np.ones((1, 1)),
                    output,
                )
            self.assertEqual(copy.call_count, 1)
            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(
                result.warnings, ("Error copying related files: copy denied",)
            )
            self.assertEqual([item.role for item in result.outputs], ["filtered_image"])

            original_copyfile = shutil.copyfile

            def copy_second(src, dst):
                if src.endswith("_ShutterCount.txt"):
                    raise OSError("shutter denied")
                return original_copyfile(src, dst)

            with patch(
                "NEAT.services.preprocessing_filtering.shutil.copyfile",
                side_effect=copy_second,
            ):
                partial = _run(
                    (LoadedImageRun(str(root), {}),), np.ones((1, 1)), output
                )
            self.assertEqual(partial.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(
                [item.role for item in partial.outputs], ["related_file_copy"]
            )
            self.assertEqual(
                partial.warnings, ("Error copying related files: shutter denied",)
            )

    def test_listdir_exception_is_warning_and_frames_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            with patch(
                "NEAT.services.preprocessing_filtering.os.listdir",
                side_effect=OSError("enumeration denied"),
            ):
                result = _run(
                    (LoadedImageRun(str(root), {"x": np.ones((1, 1))}),),
                    np.ones((1, 1)),
                    output,
                )
            self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
            self.assertEqual(
                result.warnings, ("Error copying related files: enumeration denied",)
            )
            self.assertTrue((output / "gold_x.fits").exists())

    def test_cancel_after_frame_retains_output_run_progress_and_repeated_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            runs = (
                LoadedImageRun(str(root), {"a": np.ones((1, 1)), "b": np.ones((1, 1))}),
                LoadedImageRun(str(root), {"c": np.ones((1, 1))}),
            )
            stopped = [False]
            progress, messages = [], []

            def on_progress(value):
                progress.append(value)
                if value == 33:
                    stopped[0] = True

            result = _run(
                runs,
                np.ones((1, 1)),
                output,
                progress_callback=on_progress,
                message_callback=messages.append,
                cancellation_check=lambda: stopped[0],
            )
            self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
            self.assertEqual((result.processed_count, result.expected_count), (1, 3))
            self.assertEqual(progress, [33, 50])
            self.assertEqual(
                messages.count("Filtering process has been stopped by the user."), 2
            )
            self.assertEqual(
                [Path(item.path).name for item in result.outputs], ["gold_a.fits"]
            )
            self.assertTrue(
                messages[-1].startswith("Filtering failed or incomplete. 1 of 3")
            )

    def test_cancel_after_failure_retains_error_and_run_progress_can_reach_100(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            stopped = [False]

            def failed(suffix):
                stopped[0] = True

            run = LoadedImageRun(
                str(root), {"bad": np.ones((2, 2)), "later": np.ones((1, 1))}
            )
            progress = []
            result = _run(
                (run,),
                np.ones((1, 1)),
                output,
                cancellation_check=lambda: stopped[0],
                frame_failure_callback=failed,
                progress_callback=progress.append,
            )
            self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
            self.assertEqual((result.processed_count, result.expected_count), (0, 2))
            self.assertEqual(progress, [100])
            self.assertEqual(len(result.errors), 1)
            self.assertFalse((output / "gold_later.fits").exists())

    def test_pre_stopped_starts_then_observes_stop_and_summarizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            messages = []
            result = _run(
                (LoadedImageRun(str(root), {"x": np.ones((1, 1))}),),
                np.ones((1, 1)),
                output,
                cancellation_check=lambda: True,
                message_callback=messages.append,
            )
            self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
            self.assertEqual(
                messages[0:2],
                [
                    "Filtering started...",
                    "Filtering process has been stopped by the user.",
                ],
            )
            self.assertTrue(
                messages[-1].startswith("Filtering failed or incomplete. 0 of 1")
            )

    def test_public_sidecar_helper_missing_folder_is_informational(self):
        messages = []
        outputs, warnings = copy_filtering_related_files(
            1, "Z:/not-a-real-neat-folder", "unused", message_callback=messages.append
        )
        self.assertEqual((outputs, warnings), ((), ()))
        self.assertEqual(
            messages,
            ["Data run folder not found or invalid: Z:/not-a-real-neat-folder"],
        )

    def test_fresh_subprocess_import_and_run_are_qt_free(self):
        code = (
            "import sys, tempfile, numpy as np; "
            "from NEAT.domain import LoadedImageRun; "
            "from NEAT.services.preprocessing_filtering import filter_loaded_image_runs; "
            "from pathlib import Path; "
            "tmp=tempfile.TemporaryDirectory(); p=Path(tmp.name); "
            "r=filter_loaded_image_runs((LoadedImageRun(str(p), {'x':np.ones((1,1))}),),np.ones((1,1)),str(p),'x'); "
            "assert r.processed_count==1; "
            "assert not any(k.startswith(('PyQt5','NEAT.ui','NEAT.workers')) for k in sys.modules); "
            "print('headless')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(completed.stdout.strip(), "headless")


if __name__ == "__main__":
    unittest.main()

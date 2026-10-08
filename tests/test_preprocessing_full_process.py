from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from PyQt5.QtCore import QThread

from NEAT.domain import (
    PreprocessingOperationResult,
    PreprocessingStatus,
    ProducedOutput,
)
from NEAT.services import preprocessing_full_process as full_process
from NEAT.services.image_io import load_image_file
from NEAT.workers.preprocessing import FullProcessWorker


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EPIC_BASELINE = "37a952a926f86244596bf9fdae73d3d147289205"


def _write_run(folder: Path, branch: str, run_index: int = 0) -> None:
    folder.mkdir(parents=True)
    tof = np.array([1.0, 1.00001, 2.0, 2.00001], dtype=np.float32)
    np.savetxt(folder / f"{branch}_Spectra.txt", np.column_stack((tof, tof)))
    np.savetxt(
        folder / f"{branch}_ShutterCount.txt",
        np.array([[0, 2500], [1, 2500]], dtype=np.int32),
        fmt="%d\t%d",
    )
    base = 100 if branch == "sample" else 300
    for frame_index in range(4):
        frame = np.full(
            (512, 512), base + run_index * 8 + frame_index, dtype=np.float32
        )
        fits.writeto(
            folder / f"{branch}_{frame_index:05}.fits",
            np.flipud(frame),
            overwrite=True,
        )


def _sha256_image(path: Path) -> str:
    data = np.ascontiguousarray(load_image_file(str(path)))
    return hashlib.sha256(data.tobytes()).hexdigest()


class FullProcessFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="neat-full-process-")
        self.root = Path(self.temporary.name)
        self.input_root = self.root / "input"
        self.sample = self.input_root / "sample_data"
        self.beam = self.input_root / "openbeam_data"
        self.output = self.root / "output"
        self.output.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def make_input(self, child_count: int = 0):
        if child_count:
            for index in range(child_count):
                _write_run(self.sample / f"sample-run-{index}", "sample", index)
                _write_run(self.beam / f"beam-run-{index}", "beam", index)
        else:
            _write_run(self.sample, "sample")
            _write_run(self.beam, "beam")

    def worker(self):
        return FullProcessWorker(
            str(self.sample), str(self.beam), str(self.output), "unused-base", 0, 0
        )


class TestFullProcessLoader(FullProcessFixture):
    def test_raw_order_extensions_suffixes_errors_and_progress(self):
        self.sample.mkdir(parents=True)
        filenames = [
            "z_nonnumeric.tif",
            "skip_00001.fts",
            "a_00001.FIT",
            "empty_.tiff",
            "first_dup.fit",
            "second_dup.tif",
            "broken_bad.tif",
        ]
        read_order = []

        def read(path):
            filename = os.path.basename(path)
            read_order.append(filename)
            if filename == "broken_bad.tif":
                raise OSError("bad image")
            return np.array([[1, 2], [3, 4]], dtype=np.uint16)

        progress = []
        messages = []
        with (
            patch.object(full_process.os, "listdir", return_value=filenames),
            patch.object(full_process, "load_image_file", side_effect=read),
        ):
            run = full_process.load_full_process_run(
                str(self.sample),
                progress_callback=progress.append,
                message_callback=messages.append,
            )

        self.assertEqual(read_order, [
            "z_nonnumeric.tif", "a_00001.FIT", "first_dup.fit", "broken_bad.tif"
        ])
        self.assertEqual(list(run.frames), ["nonnumeric", "00001", "dup"])
        self.assertTrue(all(array.dtype == np.float32 for array in run.frames.values()))
        self.assertEqual(progress, [16, 33, 50, 66, 83, 100])
        short_sample_path = os.path.join("input", "sample_data")
        self.assertEqual(
            run.load_errors,
            (
                "File 'empty_.tiff' has an empty frame suffix.",
                "File 'second_dup.tif' duplicates frame suffix 'dup'. Frame names must be unique.",
                f"Error loading file broken_bad.tif in \\{short_sample_path}: bad image",
            ),
        )
        self.assertEqual(messages, list(run.load_errors))

    def test_missing_folder_has_no_synthesized_load_error_or_progress(self):
        progress = []
        messages = []
        run = full_process.load_full_process_run(
            str(self.sample),
            progress_callback=progress.append,
            message_callback=messages.append,
        )
        self.assertEqual(dict(run.frames), {})
        self.assertEqual(run.load_errors, ())
        self.assertEqual(progress, [])
        self.assertEqual(messages, [f"Folder not found: {self.sample}"])

    def test_empty_folder_emits_no_load_progress(self):
        self.sample.mkdir(parents=True)
        progress = []
        self.assertEqual(
            full_process.load_full_process_run(
                str(self.sample), progress_callback=progress.append
            ).load_errors,
            (),
        )
        self.assertEqual(progress, [])

    def test_folder_enumeration_failure_is_recorded_and_emitted(self):
        self.sample.mkdir(parents=True)
        progress = []
        messages = []
        with patch.object(full_process.os, "listdir", side_effect=OSError("denied")):
            run = full_process.load_full_process_run(
                str(self.sample),
                progress_callback=progress.append,
                message_callback=messages.append,
            )
        short_sample_path = os.path.join("input", "sample_data")
        self.assertEqual(
            run.load_errors,
            (f"Error reading folder \\{short_sample_path}: denied",),
        )
        self.assertEqual(messages, list(run.load_errors))
        self.assertEqual(progress, [])

    def test_loader_does_not_stop_when_external_state_changes(self):
        self.sample.mkdir(parents=True)
        files = [f"frame_{index}.fit" for index in range(3)]
        stopped = False
        read = []

        def progress(_value):
            nonlocal stopped
            stopped = True

        def load(path):
            read.append(path)
            return np.ones((2, 2))

        with (
            patch.object(full_process.os, "listdir", return_value=files),
            patch.object(full_process, "load_image_file", side_effect=load),
        ):
            run = full_process.load_full_process_run(
                str(self.sample), progress_callback=progress
            )
        self.assertTrue(stopped)
        self.assertEqual(len(read), 3)
        self.assertEqual(len(run.frames), 3)


class TestFullProcessPipeline(FullProcessFixture):
    def run_pipeline(self):
        progress = []
        load_progress = []
        messages = []
        result = full_process.run_full_process(
            str(self.sample),
            str(self.beam),
            str(self.output),
            "overall-name-must-remain-unused",
            0,
            0,
            message_callback=messages.append,
            progress_callback=progress.append,
            load_progress_callback=load_progress.append,
            **self._pipeline_callbacks,
        )
        return result, messages, progress, load_progress

    def run_pipeline_with_patch(self, target, **patch_kwargs):
        with patch.object(full_process, target, **patch_kwargs):
            return self.run_pipeline()

    _pipeline_callbacks = {}

    def test_epic_baseline_two_run_golden_workflow_matches_exact_arrays(self):
        self.make_input(child_count=2)
        pacing = []
        self._pipeline_callbacks = {"normalisation_pacing": pacing.append}
        result, _messages, _progress, _load_progress = self.run_pipeline()
        self._pipeline_callbacks = {}
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(pacing, [5])
        self.assertEqual(
            [stage.stage.value for stage in result.stages],
            [
                "sample_summation",
                "open_beam_summation",
                "sample_clean",
                "open_beam_clean",
                "sample_overlap",
                "open_beam_overlap",
                "normalisation",
            ],
        )
        # Captured by running the original FullProcessWorker at Epic baseline
        # 37a952a926f86244596bf9fdae73d3d147289205 using the fixture above.
        golden = {
            "0_summed_sample_data/summed_sample_data_Summed_00000.fits":
                "6f91524a4b0bd4e52d93d142a60b22d1a0c7371f2a5b13339eb8a6e912cd03d2",
            "0_summed_openbeam_data/summed_openbeam_data_Summed_00000.fits":
                "304cc8fdea87d608bf6158edf59b72e7d30d85dd2957635ec7f4742894136215",
            "1_cleaned_0_summed_sample_data/cleaned_0_summed_sample_data_00000.fits":
                "6f91524a4b0bd4e52d93d142a60b22d1a0c7371f2a5b13339eb8a6e912cd03d2",
            "2_corrected_1_cleaned_0_summed_sample_data/Corrected_corrected_1_cleaned_0_summed_sample_data_00000.fits":
                "4569eda83e7caa86e8fb9ab6813aa7d63aba91cacaa916ff6f3b2f17286cee69",
            "2_corrected_1_cleaned_0_summed_openbeam_data/Corrected_corrected_1_cleaned_0_summed_openbeam_data_00000.fits":
                "1324c939a9b1542ccbef1513f1ab9c00b1cf012c6c66acd1b904a748d1641aea",
            "3_normalised_original/normalised_00000.fits":
                "5902146e457526482e38c72c264f2cc5a24f37ff5540613ffc35d15ab5863209",
            "3_normalised_original/normalised_00001.fits":
                "904b8547c0f8649b7afa61ffb02b5fb7c80fd2cdd02cb71f1e376c00f00f9cea",
            "3_normalised_original/normalised_00002.fits":
                "6178626f033262cf6f6930d1f140b582120c7582d66e87bc8b40e94aa1605281",
            "3_normalised_original/normalised_00003.fits":
                "de14b00e9900cba969a21fe376c2b2b2b3f37612fa2b5f5b905b7b8b68c9a2f1",
        }
        actual = {
            relative: _sha256_image(self.output / relative)
            for relative in golden
        }
        self.assertEqual(actual, golden)
        self.assertTrue(all(stage.operation_result is not None for stage in result.stages))
        self.assertTrue(all(not hasattr(stage, "frames") for stage in result.stages))
        normalisation = result.stages[-1]
        self.assertIsNone(normalisation.branch)
        self.assertEqual(normalisation.input_folder, str(self.output / "2_corrected_1_cleaned_0_summed_sample_data"))
        self.assertEqual(normalisation.secondary_input_folder, str(self.output / "2_corrected_1_cleaned_0_summed_openbeam_data"))
        self.assertEqual(
            [stage.outcome.value for stage in result.stages],
            ["succeeded"] * 7,
        )
        self.assertFalse(any("overall-name-must-remain-unused" in o.path for o in result.outputs))
        self.assertIn(EPIC_BASELINE, (REPOSITORY_ROOT / "docs/technical/preprocessing/full_process.md").read_text(encoding="utf-8"))

    def test_issue_baseline_no_summation_path_names_progress_and_hashes(self):
        self.make_input()
        worker = self.worker()
        messages, progress, loaded, finished = [], [], [], []
        worker.message.connect(messages.append)
        worker.progress_updated.connect(progress.append)
        worker.load_progress_updated.connect(loaded.append)
        worker.finished.connect(lambda: finished.append(True))
        sleep_calls = []
        with patch.object(QThread, "sleep", staticmethod(sleep_calls.append)):
            worker.run()
        self.assertTrue(worker.succeeded)
        self.assertEqual(worker.result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(finished, [True])
        self.assertEqual(sleep_calls, [5])
        short_sample_path = os.path.join("input", "sample_data")
        self.assertIn(
            f"0_sumation_Sample: No subfolders found in '\\{short_sample_path}'. Skipping Summation.",
            messages,
        )
        self.assertEqual(
            sorted(path.name for path in self.output.iterdir()),
            [
                "1_cleaned_openbeam_data",
                "1_cleaned_sample_data",
                "2_corrected_1_cleaned_openbeam_data",
                "2_corrected_1_cleaned_sample_data",
                "3_normalised_original",
            ],
        )
        self.assertEqual(
            progress,
            [0, 0, 25, 50, 75, 100, 0, 25, 50, 75, 100, 0,
             25, 50, 75, 100, 0, 25, 50, 75, 100, 0,
             25, 50, 75, 100, 0],
        )
        self.assertEqual(loaded, [25, 50, 75, 100] * 6)
        golden = {
            "1_cleaned_openbeam_data/cleaned_openbeam_data_00000.fits":
                "2ac60b6a21536b1c990729252fdbab39b190bd28b9cc3b7269822086006e5d11",
            "1_cleaned_sample_data/cleaned_sample_data_00000.fits":
                "cc67bd1414ca4117efe8b67cd3526c8c41630c2f0bcbb4fa2bcc7b6a60e1205b",
            "2_corrected_1_cleaned_openbeam_data/Corrected_corrected_1_cleaned_openbeam_data_00000.fits":
                "20a9a9522b22bf656a674a18860712505abb8b0c1731a6962c8c027e13bc900a",
            "2_corrected_1_cleaned_sample_data/Corrected_corrected_1_cleaned_sample_data_00000.fits":
                "128502b0526e91bc39b066b244609f2e42660f9e76d422729c4a85573a44e579",
            "3_normalised_original/normalised_00000.fits":
                "adc9c1866fbdc43aa8dfd0651df63a2fef89f779366a4293154fd5d4924ac595",
        }
        self.assertEqual(
            {path: _sha256_image(self.output / path) for path in golden}, golden
        )
        self.assertEqual(messages[-1], "=== <b>Full Process Completed Successfully</b> ===")

    def test_issue_baseline_one_child_still_runs_summation(self):
        self.make_input(child_count=1)
        result, messages, _progress, _load = self.run_pipeline()
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertTrue(any("Found 1 subfolder(s)" in message for message in messages))
        self.assertTrue((self.output / "0_summed_sample_data").is_dir())
        self.assertEqual(result.stages[0].artifact_folder, str(self.output / "0_summed_sample_data"))

    def test_missing_overlap_sidecars_stops_before_normalisation(self):
        self.make_input()
        (self.sample / "sample_Spectra.txt").unlink()
        result, messages, _progress, _load = self.run_pipeline()
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(result.failed_stage, full_process.FullProcessStage.SAMPLE_OVERLAP)
        self.assertNotIn("normalisation", [stage.stage.value for stage in result.stages])
        self.assertTrue(any("missing Spectra sidecar" in message for message in messages))
        self.assertFalse(
            any("Error in OverlapCorrectionWorker:" in message for message in messages)
        )
        overlap = result.stages[-1]
        self.assertEqual(overlap.outcome.value, "failed")
        self.assertIsNotNone(overlap.operation_result)
        self.assertEqual(overlap.operation_result.outputs, ())

    def test_clean_setup_failure_has_no_artifact_folder(self):
        self.sample.mkdir(parents=True)
        self.beam.mkdir(parents=True)
        result, _messages, _progress, _load = self.run_pipeline()
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        stage = next(stage for stage in result.stages if stage.stage.value == "sample_clean")
        self.assertEqual(stage.outcome.value, "failed")
        self.assertIsNone(stage.operation_result)
        self.assertIsNone(stage.artifact_folder)
        self.assertFalse((self.output / "1_cleaned_sample_data").exists())
        self.assertFalse(any(message.startswith("[FATAL]") for message in _messages))
        self.assertFalse(any(message.startswith("<b>Final memory usage:</b>") for message in _messages))

    def test_all_empty_summation_children_fail_after_reusing_created_folder(self):
        self.sample.mkdir(parents=True)
        (self.sample / "empty-one").mkdir()
        (self.sample / "unrelated-empty").mkdir()
        self.beam.mkdir(parents=True)
        result, messages, _progress, _load = self.run_pipeline()
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(result.failed_stage, full_process.FullProcessStage.SAMPLE_SUMMATION)
        stage = result.stages[0]
        self.assertEqual(stage.outcome.value, "failed")
        self.assertIsNone(stage.operation_result)
        self.assertEqual(stage.artifact_folder, str(self.output / "0_summed_sample_data"))
        self.assertTrue(Path(stage.artifact_folder).is_dir())
        self.assertFalse(any(message.startswith("[FATAL] Summation aborted:") for message in messages))

    def test_summation_passes_loaded_frames_with_errors_to_service(self):
        child = self.sample / "run-one"
        child.mkdir(parents=True)
        self.beam.mkdir(parents=True)
        bad_run = full_process.LoadedImageRun(
            str(child), {"00001": np.ones((2, 2), dtype=np.float32)},
            load_errors=("a frame failed to load",),
        )
        with patch.object(full_process, "load_full_process_run", return_value=bad_run) as loader:
            result, _messages, _progress, _load = self.run_pipeline()
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        stage = result.stages[0]
        self.assertEqual(stage.outcome.value, "failed")
        self.assertEqual(stage.operation_result.status, PreprocessingStatus.FAILED)
        self.assertTrue(any("load error" in error.message.lower() for error in result.errors))

    def test_stale_summation_image_is_not_cleaned_or_passed_to_overlap(self):
        self.make_input(child_count=2)
        stale_folder = self.output / "0_summed_sample_data"
        first_result, _first_messages, _progress, _load = self.run_pipeline()
        self.assertEqual(first_result.status, PreprocessingStatus.SUCCEEDED)
        stale = np.full((512, 512), 77, dtype=np.float32)
        fits.writeto(stale_folder / "stale_99999.fits", np.flipud(stale), overwrite=True)
        result, messages, _progress, _load = self.run_pipeline()
        self.assertTrue((stale_folder / "stale_99999.fits").exists())
        sum_paths = [
            os.path.normcase(output.path)
            for output in result.stages[0].operation_result.outputs
        ]
        self.assertNotIn(os.path.normcase(str(stale_folder / "stale_99999.fits")), sum_paths)
        self.assertFalse(
            (self.output / "1_cleaned_0_summed_sample_data" / "cleaned_0_summed_sample_data_99999.fits").exists()
        )
        self.assertTrue(any("unmanifested image" in message for message in messages))

    def test_service_outputs_keep_duplicate_paths_without_directory_scanning(self):
        self.make_input(child_count=1)
        duplicate = ProducedOutput(str(self.output / "existing.fits"), "test")
        operation = PreprocessingOperationResult(
            PreprocessingStatus.SUCCEEDED,
            1,
            outputs=(duplicate, duplicate),
            expected_count=1,
            warnings=("legacy warning",),
        )
        pipeline = full_process.FullProcessPipeline(
            str(self.sample), str(self.beam), str(self.output), "unused", 0, 0
        )
        with patch.object(full_process, "sum_loaded_image_runs", return_value=operation):
            pipeline.maybe_do_summation(str(self.sample), "Sample")
        result = pipeline.result()
        self.assertEqual(result.outputs, (duplicate, duplicate))
        self.assertEqual(
            result.warnings,
            (full_process.FullProcessDiagnostic(full_process.FullProcessStage.SAMPLE_SUMMATION, "legacy warning"),),
        )
        self.assertFalse((self.output / "existing.fits").exists())

    def test_partial_service_failure_is_attributed_and_stops_later_stages(self):
        self.make_input(child_count=1)
        partial = ProducedOutput(str(self.output / "partial.fits"), "summed_image")
        failed = PreprocessingOperationResult(
            PreprocessingStatus.FAILED,
            0,
            outputs=(partial,),
            expected_count=1,
            errors=("synthetic operation failure",),
        )
        messages = []
        pipeline = full_process.FullProcessPipeline(
            str(self.sample),
            str(self.beam),
            str(self.output),
            "unused",
            0,
            0,
            message_callback=messages.append,
        )
        with patch.object(full_process, "sum_loaded_image_runs", return_value=failed):
            result = pipeline.run()
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(result.failed_stage, full_process.FullProcessStage.SAMPLE_SUMMATION)
        self.assertEqual([stage.stage for stage in result.stages], [full_process.FullProcessStage.SAMPLE_SUMMATION])
        self.assertEqual(result.outputs, (partial,))
        self.assertEqual(
            result.errors,
            (
                full_process.FullProcessDiagnostic(
                    full_process.FullProcessStage.SAMPLE_SUMMATION,
                    "synthetic operation failure",
                ),
                full_process.FullProcessDiagnostic(
                    full_process.FullProcessStage.SAMPLE_SUMMATION,
                    "0_summation_Sample failed.",
                ),
            ),
        )
        self.assertTrue(messages[-1].startswith("[ERROR] 0_summation_Sample failed."))

    def test_unexpected_summation_exception_preserves_child_and_parent_diagnostics(self):
        self.make_input(child_count=1)
        result, messages, _progress, _load_progress = self.run_pipeline_with_patch(
            "sum_loaded_image_runs", side_effect=RuntimeError("boom")
        )
        stage = full_process.FullProcessStage.SAMPLE_SUMMATION
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(result.failed_stage, stage)
        self.assertEqual(
            messages[-2:],
            ["[FATAL] Summation aborted: boom", "[ERROR] 0_summation_Sample failed."],
        )
        self.assertEqual(
            result.errors,
            (
                full_process.FullProcessDiagnostic(stage, "boom"),
                full_process.FullProcessDiagnostic(stage, "0_summation_Sample failed."),
            ),
        )
        self.assertNotIn("[FATAL] Summation aborted: boom", [e.message for e in result.errors])

    def test_unexpected_summation_exception_with_parent_stop_is_cancelled(self):
        self.make_input(child_count=1)
        worker = self.worker()
        messages = []
        worker.message.connect(messages.append)

        def stop_then_raise(*_args, **_kwargs):
            worker.stop()
            raise RuntimeError("boom")

        with patch.object(
            full_process, "sum_loaded_image_runs", side_effect=stop_then_raise
        ):
            worker.run()
        stage = full_process.FullProcessStage.SAMPLE_SUMMATION
        self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual(worker.result.cancelled_stage, stage)
        self.assertEqual(
            worker.result.errors,
            (
                full_process.FullProcessDiagnostic(stage, "boom"),
                full_process.FullProcessDiagnostic(stage, "0_summation_Sample failed."),
            ),
        )
        self.assertEqual(
            messages[-2:],
            ["[FATAL] Summation aborted: boom", "[ERROR] 0_summation_Sample failed."],
        )

    def test_unexpected_clean_exception_uses_parent_wrapper_without_child_fatal(self):
        self.make_input()
        result, messages, _progress, _load_progress = self.run_pipeline_with_patch(
            "clean_loaded_image_runs", side_effect=RuntimeError("boom")
        )
        stage = full_process.FullProcessStage.SAMPLE_CLEAN
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(result.failed_stage, stage)
        self.assertEqual(messages[-1], "[ERROR] 1_clean_Sample failed or skipped frames.")
        self.assertFalse(any(message.startswith("[FATAL]") for message in messages))
        self.assertEqual(
            result.errors,
            (
                full_process.FullProcessDiagnostic(stage, "boom"),
                full_process.FullProcessDiagnostic(stage, "1_clean_Sample failed or skipped frames."),
            ),
        )

    def test_unexpected_overlap_exception_preserves_child_before_parent_wrapper(self):
        self.make_input()
        result, messages, _progress, _load_progress = self.run_pipeline_with_patch(
            "correct_loaded_image_run", side_effect=RuntimeError("boom")
        )
        stage = full_process.FullProcessStage.SAMPLE_OVERLAP
        child = "Error in OverlapCorrectionWorker: boom"
        parent = "[ERROR] 2_correction_Sample failed."
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(result.failed_stage, stage)
        self.assertEqual(messages[-3:], [child, "Overlap Correction did not complete successfully.", parent])
        self.assertEqual(
            result.errors,
            (
                full_process.FullProcessDiagnostic(stage, "boom"),
                full_process.FullProcessDiagnostic(stage, "2_correction_Sample failed."),
            ),
        )
        self.assertNotIn(child, [error.message for error in result.errors])

    def test_unexpected_normalisation_exception_preserves_child_and_parent_diagnostics(self):
        self.make_input()
        result, messages, _progress, _load_progress = self.run_pipeline_with_patch(
            "normalise_loaded_image_runs", side_effect=RuntimeError("boom")
        )
        stage = full_process.FullProcessStage.NORMALISATION
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(result.failed_stage, stage)
        self.assertEqual(
            messages[-2:],
            [
                "Fatal error in normalisation: boom",
                "[ERROR] 3_normalisation failed or skipped one or more frames.",
            ],
        )
        self.assertEqual(
            result.errors,
            (
                full_process.FullProcessDiagnostic(stage, "boom"),
                full_process.FullProcessDiagnostic(
                    stage, "3_normalisation failed or skipped one or more frames."
                ),
            ),
        )
        self.assertNotIn(
            "Fatal error in normalisation: boom", [error.message for error in result.errors]
        )

    def test_unexpected_pipeline_exception_without_builder_is_structured(self):
        pipeline = full_process.FullProcessPipeline(
            str(self.sample), str(self.beam), str(self.output), "unused", 0, 0
        )
        with patch.object(
            pipeline, "maybe_do_summation", side_effect=RuntimeError("before stage")
        ):
            result = pipeline.run()
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(
            result.errors,
            (full_process.FullProcessDiagnostic(None, "before stage"),),
        )

    def test_normalisation_callback_failure_keeps_operation_result_and_worker_state(self):
        self.make_input()
        worker = self.worker()
        finished = []
        messages = []
        collections = []
        worker.finished.connect(lambda: finished.append(True))
        worker.message.connect(messages.append)
        produced = ProducedOutput(
            str(self.output / "normalised_00000.fits"), "normalised_image"
        )
        operation_result = PreprocessingOperationResult(
            PreprocessingStatus.SUCCEEDED,
            1,
            outputs=(produced,),
            expected_count=1,
        )

        def fail_memory_report():
            raise RuntimeError("memory reporting failed")

        worker._normalisation_finished = fail_memory_report
        with (
            patch.object(
                full_process,
                "normalise_loaded_image_runs",
                return_value=operation_result,
            ),
            patch(
                "NEAT.workers.preprocessing.gc.collect",
                side_effect=lambda: collections.append(True),
            ),
            patch.object(QThread, "sleep", staticmethod(lambda _seconds: None)),
        ):
            worker.run()
        self.assertIsNotNone(worker.result)
        self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
        self.assertFalse(worker.succeeded)
        self.assertEqual(finished, [True])
        self.assertGreaterEqual(len(collections), 1)
        normalisation = worker.result.stages[-1]
        self.assertIs(normalisation.operation_result, operation_result)
        self.assertEqual(worker.result.outputs[-1], produced)
        stage = full_process.FullProcessStage.NORMALISATION
        self.assertEqual(
            worker.result.errors[-2:],
            (
                full_process.FullProcessDiagnostic(stage, "memory reporting failed"),
                full_process.FullProcessDiagnostic(
                    stage, "3_normalisation failed or skipped one or more frames."
                ),
            ),
        )
        self.assertEqual(
            messages[-1],
            "[ERROR] 3_normalisation failed or skipped one or more frames.",
        )
        self.assertFalse(any(message.startswith("[WARN] Worker finalization:") for message in messages))

    def test_ambiguous_overlap_sidecars_fail_preflight_without_fallback(self):
        self.make_input()
        malformed = self.output / "1_cleaned_sample_data" / "a_Spectra.txt"
        malformed.parent.mkdir()
        malformed.write_text("not numeric", encoding="utf-8")
        original_listdir = os.listdir
        selected = []

        def ordered_listdir(path):
            if os.path.normcase(path) == os.path.normcase(str(malformed.parent)):
                actual = original_listdir(path)
                images = [name for name in actual if name.lower().endswith(".fits")]
                sidecars = [name for name in actual if name.endswith("_Spectra.txt")]
                return images + ["a_Spectra.txt"] + [name for name in sidecars if name != "a_Spectra.txt"] + [
                    name for name in actual if name.endswith("_ShutterCount.txt")
                ]
            return original_listdir(path)

        original_loadtxt = np.loadtxt

        def track_loadtxt(path, *args, **kwargs):
            selected.append(os.path.basename(path))
            return original_loadtxt(path, *args, **kwargs)

        pipeline = full_process.FullProcessPipeline(
            str(self.sample), str(self.beam), str(self.output), "unused", 0, 0
        )
        with (
            patch.object(full_process.os, "listdir", side_effect=ordered_listdir),
            patch.object(full_process.np, "loadtxt", side_effect=track_loadtxt),
        ):
            result = pipeline.run()
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(result.failed_stage, full_process.FullProcessStage.SAMPLE_OVERLAP)
        self.assertNotIn("a_Spectra.txt", selected)
        self.assertNotIn("Run1_sample_Spectra.txt", selected)
        overlap = result.stages[-1]
        self.assertEqual(overlap.outcome.value, "failed")
        self.assertIsNotNone(overlap.operation_result)
        self.assertEqual(overlap.operation_result.outputs, ())
        self.assertTrue(
            any("ambiguous Spectra sidecars" in error.message for error in result.errors)
        )
        self.assertIsNone(overlap.artifact_folder)

    def test_stop_during_loader_uses_fresh_operation_token_then_stops_at_boundary(self):
        self.make_input(child_count=1)
        worker = self.worker()
        messages, loaded = [], []
        started = []
        original_started = worker._operation_started

        def operation_started(stage, token):
            started.append((stage, token.is_set()))
            original_started(stage, token)

        def load_progress(_value):
            if worker._active_operation_token is None and not loaded:
                loaded.append("stop")
                worker.stop()

        worker._operation_started = operation_started
        worker.message.connect(messages.append)
        worker.load_progress_updated.connect(load_progress)
        with patch.object(QThread, "sleep", staticmethod(lambda _seconds: None)):
            worker.run()
        self.assertEqual(started[0][0], full_process.FullProcessStage.SAMPLE_SUMMATION)
        self.assertFalse(started[0][1])
        self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)
        sample_sum = worker.result.stages[0]
        self.assertEqual(sample_sum.outcome.value, "succeeded")
        self.assertEqual(sample_sum.propagation_folder, str(self.sample))
        self.assertEqual(messages.count("FullProcessWorker: Stop signal received."), 1)
        self.assertIn("Stopped during sample summation.", messages)

    def test_stop_before_run_reaches_first_stage_boundary(self):
        self.make_input(child_count=1)
        worker = self.worker()
        messages, finished = [], []
        worker.message.connect(messages.append)
        worker.finished.connect(lambda: finished.append(True))
        worker.stop()
        worker.run()
        self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual(worker.result.cancelled_stage, full_process.FullProcessStage.SAMPLE_SUMMATION)
        self.assertEqual([stage.stage.value for stage in worker.result.stages], ["sample_summation"])
        self.assertIsNone(worker.result.stages[0].operation_result)
        self.assertIn("Stopped during sample summation.", messages)
        self.assertEqual(messages.count("=== <b>Full Process Pipeline Started</b> ==="), 1)
        self.assertEqual(messages.count("FullProcessWorker: Stop signal received."), 1)
        self.assertEqual(finished, [True])

    def test_adapter_exception_still_sets_result_collects_and_finishes_once(self):
        worker = self.worker()
        messages, finished, collections = [], [], []
        worker.message.connect(messages.append)
        worker.finished.connect(lambda: finished.append(True))
        with (
            patch.object(worker, "_make_pipeline", side_effect=RuntimeError("adapter broke")),
            patch("NEAT.workers.preprocessing.gc.collect", side_effect=lambda: collections.append(True)),
        ):
            worker.run()
        self.assertIsNotNone(worker.result)
        self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
        self.assertEqual(worker.succeeded, False)
        self.assertTrue(messages[-1].startswith("[ERROR] adapter broke"))
        self.assertEqual(finished, [True])
        self.assertGreaterEqual(len(collections), 1)

    def test_setup_stop_at_clean_overlap_and_normalisation_loads_uses_fresh_tokens(self):
        for stop_at, expected_stage, phase in (
            (1, "sample_clean", "sample cleaning"),
            (9, "sample_overlap", "sample overlap"),
            (17, "normalisation", "normalisation"),
        ):
            with self.subTest(stage=expected_stage):
                self.temporary.cleanup()
                self.setUp()
                self.make_input(child_count=0)
                worker = self.worker()
                count = []
                started = []
                messages = []
                original_started = worker._operation_started

                def on_started(stage, token):
                    started.append((stage, token.is_set()))
                    original_started(stage, token)

                def on_load(_value):
                    count.append(1)
                    if len(count) == stop_at:
                        worker.stop()

                worker._operation_started = on_started
                worker.load_progress_updated.connect(on_load)
                worker.message.connect(messages.append)
                with patch.object(QThread, "sleep", staticmethod(lambda _seconds: None)):
                    worker.run()
                self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)
                stage = next(s for s in worker.result.stages if s.stage.value == expected_stage)
                self.assertEqual(stage.outcome.value, "succeeded")
                self.assertIsNotNone(stage.operation_result)
                started_stage = next(pair for pair in started if pair[0].value == expected_stage)
                self.assertFalse(started_stage[1])
                self.assertIn(f"Stopped during {phase}.", messages)
                self.assertEqual(messages.count("FullProcessWorker: Stop signal received."), 1)

    def test_active_clean_stop_emits_only_parent_stop_diagnostic(self):
        self.make_input()
        worker = self.worker()
        messages, finished = [], []
        worker.message.connect(messages.append)
        worker.finished.connect(lambda: finished.append(True))
        actual = full_process.clean_loaded_image_runs

        def stop_then_clean(*args, **kwargs):
            worker.stop()
            return actual(*args, **kwargs)

        with patch.object(full_process, "clean_loaded_image_runs", side_effect=stop_then_clean):
            worker.run()
        self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual(worker.result.cancelled_stage, full_process.FullProcessStage.SAMPLE_CLEAN)
        self.assertEqual(messages.count("FullProcessWorker: Stop signal received."), 1)
        self.assertFalse(any("Terminating" in message or "cancelling at next safe point" in message for message in messages))
        self.assertEqual(finished, [True])

    def test_active_overlap_stop_emits_child_then_parent_diagnostic(self):
        self.make_input()
        worker = self.worker()
        messages = []
        worker.message.connect(messages.append)
        actual = full_process.correct_loaded_image_run

        def stop_then_correct(*args, **kwargs):
            worker.stop()
            return actual(*args, **kwargs)

        with (
            patch.object(full_process, "correct_loaded_image_run", side_effect=stop_then_correct),
            patch.object(QThread, "sleep", staticmethod(lambda _seconds: None)),
        ):
            worker.run()
        self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual(worker.result.cancelled_stage, full_process.FullProcessStage.SAMPLE_OVERLAP)
        child = "Stop signal received. Terminating Overlap Correction process."
        parent = "FullProcessWorker: Stop signal received."
        self.assertLess(messages.index(child), messages.index(parent))

    def test_active_normalisation_stop_keeps_memory_diagnostic_and_order(self):
        self.make_input()
        worker = self.worker()
        messages = []
        worker.message.connect(messages.append)
        actual = full_process.normalise_loaded_image_runs

        def stop_then_normalise(*args, **kwargs):
            worker.stop()
            return actual(*args, **kwargs)

        with (
            patch.object(full_process, "normalise_loaded_image_runs", side_effect=stop_then_normalise),
            patch.object(QThread, "sleep", staticmethod(lambda _seconds: None)),
        ):
            worker.run()
        self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual(worker.result.cancelled_stage, full_process.FullProcessStage.NORMALISATION)
        child = "Stop signal received. Terminating Normalisation process."
        parent = "FullProcessWorker: Stop signal received."
        self.assertLess(messages.index(child), messages.index(parent))
        self.assertTrue(any(message.startswith("<b>Final memory usage:</b>") for message in messages))

    def test_active_summation_stop_order_and_structured_cancellation(self):
        self.make_input(child_count=1)
        worker = self.worker()
        messages = []
        stopped = []
        worker.message.connect(messages.append)

        def stop_on_start(message):
            if message == "--- Starting image summation ---" and not stopped:
                stopped.append(True)
                worker.stop()

        worker.message.connect(stop_on_start)
        worker.run()
        self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual(worker.succeeded, False)
        child = "Stop signal received – cancelling at next safe point."
        parent = "FullProcessWorker: Stop signal received."
        self.assertLess(messages.index(child), messages.index(parent))
        self.assertTrue(messages[-1].startswith("[ERROR] 0_summation_Sample failed."))
        self.assertEqual(worker.result.cancelled_stage, full_process.FullProcessStage.SAMPLE_SUMMATION)


class TestFullProcessIsolation(unittest.TestCase):
    def test_service_import_does_not_load_qt_ui_or_worker_modules(self):
        code = (
            "import sys; import NEAT.services.preprocessing_full_process; "
            "assert not any(n == 'PyQt5' or n.startswith('PyQt5.') for n in sys.modules); "
            "assert not any(n == 'NEAT.ui' or n.startswith('NEAT.ui.') for n in sys.modules); "
            "assert not any(n == 'NEAT.workers' or n.startswith('NEAT.workers.') for n in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_worker_class_has_no_nested_preprocessing_workers_or_event_loop(self):
        source = (REPOSITORY_ROOT / "NEAT/workers/preprocessing.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        worker_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FullProcessWorker"
        )
        identifiers = {node.id for node in ast.walk(worker_class) if isinstance(node, ast.Name)}
        self.assertTrue(
            identifiers.isdisjoint(
                {"SummationWorker", "OutlierFilteringWorker", "OverlapCorrectionWorker", "NormalisationWorker", "QEventLoop"}
            )
        )


if __name__ == "__main__":
    unittest.main()

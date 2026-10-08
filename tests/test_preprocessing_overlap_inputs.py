"""Shared deterministic preparation regressions for overlap correction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from NEAT.domain import PreprocessingStatus
from NEAT.services import preprocessing_full_process as full_process
from NEAT.services import preprocessing_overlap as overlap
from NEAT.services import preprocessing_overlap_inputs as overlap_inputs
from NEAT.workers import batch as batch_workers
from NEAT.workers.batch import ImageLoadWorker
from NEAT.workers.preprocessing import OverlapCorrectionWorker


class OverlapInputFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="neat-overlap-inputs-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()

    def write_sidecars(self, folder: Path, rows: int) -> None:
        tof = np.arange(rows, dtype=np.float32) * np.float32(1e-5)
        np.savetxt(folder / "sample_Spectra.txt", np.column_stack((tof, tof)))
        np.savetxt(
            folder / "sample_ShutterCount.txt",
            np.full((1, max(1, rows)), 2500, dtype=np.int32),
            fmt="%d",
        )

    def write_images(self, folder: Path, suffixes: tuple[str, ...]) -> None:
        for position, suffix in enumerate(suffixes, start=1):
            data = np.full((512, 512), position, dtype=np.float32)
            fits.writeto(
                folder / f"sample_{suffix}.fits",
                np.flipud(data),
                overwrite=True,
            )


class TestOverlapInputPreparation(OverlapInputFixture):
    def test_numeric_suffixes_define_shared_frame_order(self):
        self.write_sidecars(self.source, 3)
        self.write_images(self.source, ("00002", "00000", "00001"))

        prepared = overlap_inputs.prepare_overlap_inputs(str(self.source))

        self.assertEqual(prepared.errors, ())
        self.assertEqual(tuple(prepared.run.frames), ("00000", "00001", "00002"))
        self.assertEqual(
            tuple(Path(path).name for path in prepared.image_paths),
            ("sample_00000.fits", "sample_00001.fits", "sample_00002.fits"),
        )
        self.assertEqual(prepared.expected_count, 3)
        self.assertEqual(prepared.spectra_filename, "sample_Spectra.txt")
        self.assertEqual(
            {Path(path).name for path in prepared.related_files},
            {"sample_Spectra.txt", "sample_ShutterCount.txt"},
        )

    def test_numeric_gap_and_collision_are_preflight_errors(self):
        cases = (
            (("sample_00000.fits", "sample_00002.fits"), "Frame suffix gap"),
            (("sample_1.fits", "other_01.fit"), "Frame suffix collision"),
        )
        for filenames, expected in cases:
            with self.subTest(expected=expected):
                for path in self.source.iterdir():
                    path.unlink()
                for filename in filenames:
                    (self.source / filename).write_bytes(b"not-read")
                self.write_sidecars(self.source, len(filenames))
                with patch.object(overlap_inputs, "load_image_file") as load_image:
                    prepared = overlap_inputs.prepare_overlap_inputs(str(self.source))
                self.assertTrue(any(expected in error for error in prepared.errors))
                self.assertEqual(prepared.run.frames, {})
                load_image.assert_not_called()

    def test_ambiguous_sidecars_and_malformed_metadata_are_rejected(self):
        self.write_images(self.source, ("00000",))
        self.write_sidecars(self.source, 1)
        (self.source / "other_Spectra.txt").write_text("0 0\n", encoding="utf-8")

        ambiguous = overlap_inputs.prepare_overlap_inputs(str(self.source))
        self.assertTrue(any("ambiguous Spectra sidecars" in x for x in ambiguous.errors))
        self.assertEqual(ambiguous.run.frames, {})

        (self.source / "other_Spectra.txt").unlink()
        (self.source / "sample_Spectra.txt").write_text("not numeric\n", encoding="utf-8")
        malformed = overlap_inputs.prepare_overlap_inputs(str(self.source))
        self.assertTrue(any("invalid Spectra sidecar" in x for x in malformed.errors))
        self.assertEqual(malformed.run.frames, {})

        (self.source / "sample_Spectra.txt").write_text("0 0\n", encoding="utf-8")
        (self.source / "sample_ShutterCount.txt").write_text("nan\n", encoding="utf-8")
        invalid_shutter = overlap_inputs.prepare_overlap_inputs(str(self.source))
        self.assertTrue(any("invalid ShutterCount sidecar" in x for x in invalid_shutter.errors))

    def test_manifest_excludes_and_reports_stale_images(self):
        self.write_sidecars(self.source, 2)
        self.write_images(self.source, ("00000", "00001", "99999"))
        selected = (str(self.source / "sample_00000.fits"), str(self.source / "sample_00001.fits"))

        prepared = overlap_inputs.prepare_overlap_inputs(
            str(self.source), image_paths=selected, stage="Full Process overlap"
        )

        self.assertEqual(prepared.errors, ())
        self.assertEqual(tuple(prepared.run.frames), ("00000", "00001"))
        self.assertEqual(prepared.expected_count, 2)
        self.assertTrue(any("ignored 1 unmanifested image" in warning for warning in prepared.warnings))

    def test_nonnumeric_auxiliary_is_excluded_for_standalone_and_manifested_inputs(self):
        count = 2925
        self.write_sidecars(self.source, count)
        for index in range(count):
            (self.source / f"sample_{index:05}.fits").touch()
        auxiliary = self.source / "cleaned_0_sample_run1_SummedImg.fits"
        auxiliary.write_bytes(b"auxiliary image, not a correction frame")
        manifested_paths = tuple(
            str(path) for path in sorted(self.source.glob("*.fits"))
        )

        standalone_payloads = []
        standalone_messages = []
        standalone = ImageLoadWorker(str(self.source), overlap_inputs=True)
        standalone.run_loaded.connect(
            lambda _folder, payload: standalone_payloads.append(payload)
        )
        standalone.message.connect(standalone_messages.append)
        full_process_messages = []
        with patch.object(
            overlap_inputs,
            "load_image_file",
            return_value=np.ones((1, 1), dtype=np.float32),
        ) as image_loader:
            standalone.run()
            manifested = full_process.prepare_overlap_inputs(
                str(self.source),
                image_paths=manifested_paths,
                stage="Full Process overlap",
                message_callback=full_process_messages.append,
            )

        self.assertEqual(len(standalone_payloads), 1)
        payload = standalone_payloads[0]
        self.assertEqual(payload["preflight_errors"], ())
        self.assertEqual(payload["expected_count"], count)
        self.assertEqual(len(payload["images"]), count)
        self.assertEqual(len(payload["image_paths"]), count)
        self.assertTrue(
            any(
                "excluded 1 image(s)" in warning and auxiliary.name in warning
                for warning in payload["preflight_warnings"]
            )
        )
        self.assertTrue(any(auxiliary.name in message for message in standalone_messages))
        self.assertEqual(manifested.errors, ())
        self.assertEqual(manifested.expected_count, count)
        self.assertEqual(len(manifested.run.frames), count)
        self.assertEqual(len(manifested.image_paths), count)
        self.assertTrue(
            any(
                "excluded 1 image(s)" in warning and auxiliary.name in warning
                for warning in manifested.warnings
            )
        )
        self.assertTrue(any(auxiliary.name in message for message in full_process_messages))
        self.assertEqual(image_loader.call_count, count * 2)
        self.assertEqual(
            auxiliary.read_bytes(), b"auxiliary image, not a correction frame"
        )
        self.assertNotIn(auxiliary.name, {Path(path).name for path in manifested.image_paths})

    def test_only_nonnumeric_images_fail_preflight_without_loading(self):
        self.write_sidecars(self.source, 1)
        auxiliary = self.source / "sample_SummedImg.fits"
        auxiliary.write_bytes(b"not a correction frame")

        with patch.object(overlap_inputs, "load_image_file") as image_loader:
            prepared = overlap_inputs.prepare_overlap_inputs(str(self.source))

        self.assertEqual(prepared.expected_count, 0)
        self.assertEqual(prepared.image_paths, ())
        self.assertEqual(prepared.run.frames, {})
        self.assertTrue(
            any(
                "no numeric-suffix FITS/TIFF correction frames were selected" in error
                for error in prepared.errors
            )
        )
        self.assertTrue(any(auxiliary.name in warning for warning in prepared.warnings))
        image_loader.assert_not_called()

        payload = {
            "folder_path": str(self.source),
            "images": dict(prepared.run.frames),
            "spectra": prepared.spectra_data,
            "shutter_count": prepared.shutter_count_data,
            "preflight_errors": prepared.errors,
            "preflight_warnings": prepared.warnings,
            "expected_count": prepared.expected_count,
            "spectra_filename": prepared.spectra_filename,
            "related_files": prepared.related_files,
            "image_paths": prepared.image_paths,
        }
        worker = OverlapCorrectionWorker(
            payload, "standalone", str(self.root / "no-numeric-output")
        )
        worker.run()
        self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
        self.assertEqual(worker.result.expected_count, 0)
        self.assertEqual(worker.result.processed_count, 0)
        self.assertEqual(worker.result.outputs, ())
        self.assertFalse((self.root / "no-numeric-output").exists())

    def test_nonnumeric_auxiliary_does_not_mask_a_real_numeric_count_mismatch(self):
        self.write_sidecars(self.source, 1)
        self.write_images(self.source, ("00000", "00001"))
        (self.source / "sample_SummedImg.fits").write_bytes(b"not a correction frame")

        with patch.object(overlap_inputs, "load_image_file") as image_loader:
            prepared = overlap_inputs.prepare_overlap_inputs(str(self.source))

        self.assertEqual(prepared.expected_count, 2)
        self.assertTrue(
            any(
                "images=2, ToF rows=1" in error
                and "extra/unmatched image suffix '00001'" in error
                for error in prepared.errors
            )
        )
        self.assertFalse(any("IndexError" in error for error in prepared.errors))
        image_loader.assert_not_called()

    def test_standalone_preflight_exception_reports_source_and_finishes(self):
        worker = ImageLoadWorker(str(self.source), overlap_inputs=True)
        messages = []
        finished = []
        worker.message.connect(messages.append)
        worker.finished.connect(lambda: finished.append(True))

        with patch.object(
            batch_workers,
            "prepare_overlap_inputs",
            side_effect=RuntimeError("injected preflight failure"),
        ):
            worker.run()

        self.assertEqual(finished, [True])
        self.assertTrue(
            any(
                str(self.source) in message
                and "injected preflight failure" in message
                for message in messages
            )
        )
        self.assertFalse(any("UnboundLocalError" in message for message in messages))

    def test_seven_tof_segments_keep_legacy_shutter_selection(self):
        segment_tof = [
            value
            for index in range(7)
            for value in (index * 0.01, index * 0.01 + 0.00001)
        ]
        np.savetxt(
            self.source / "sample_Spectra.txt",
            np.column_stack((segment_tof, segment_tof)),
        )
        shutter_values = [0, 500, 1500, 900, 2000, 100, 2500, 300, 3000, 0, 3500, 600, 4000, 800, 4500]
        np.savetxt(self.source / "sample_ShutterCount.txt", shutter_values)
        self.write_images(self.source, tuple(f"{index:05}" for index in range(14)))
        prepared = overlap_inputs.prepare_overlap_inputs(str(self.source))
        selected_counts = []
        original = overlap.correct_overlap_frame

        def capture_count(image, cumulative, count, interval, reference):
            selected_counts.append(float(count))
            return original(image, cumulative, count, interval, reference)

        messages = []
        with patch.object(overlap, "correct_overlap_frame", side_effect=capture_count):
            result = overlap.correct_loaded_image_run(
                prepared.run,
                prepared.spectra_data,
                prepared.shutter_count_data,
                "sample",
                str(self.root / "seven-segment-output"),
                message_callback=messages.append,
                preflight_errors=prepared.errors,
                expected_count=prepared.expected_count,
                spectra_filename=prepared.spectra_filename,
                related_files=prepared.related_files,
            )

        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(result.processed_count, 14)
        self.assertTrue(any(message.startswith("Identified 7 segments") for message in messages))
        self.assertIn("Selected 7 shutter counts for 7 segments.", messages)
        self.assertEqual(
            selected_counts,
            [1500.0, 1500.0, 2000.0, 2000.0, 2500.0, 2500.0, 3000.0, 3000.0,
             3500.0, 3500.0, 4000.0, 4000.0, 4500.0, 4500.0],
        )

    def test_reported_2926_vs_2925_fails_before_image_loading_on_both_routes(self):
        count = 2926
        self.write_sidecars(self.source, count - 1)
        for index in range(count):
            (self.source / f"sample_{index:05}.fits").touch()

        with patch.object(overlap_inputs, "load_image_file") as image_loader:
            standalone_payloads = []
            standalone_progress = []
            loader = ImageLoadWorker(str(self.source), overlap_inputs=True)
            loader.run_loaded.connect(lambda _folder, payload: standalone_payloads.append(payload))
            loader.progress_updated.connect(standalone_progress.append)
            loader.run()

            self.assertEqual(len(standalone_payloads), 1)
            self.assertEqual(standalone_progress, [])
            payload = standalone_payloads[0]
            self.assertEqual(payload["expected_count"], count)
            self.assertTrue(
                any(
                    "images=2926, ToF rows=2925" in error
                    and "sample_Spectra.txt" in error
                    and "extra/unmatched image suffix '02925'" in error
                    for error in payload["preflight_errors"]
                )
            )
            worker = OverlapCorrectionWorker(payload, "standalone", str(self.root / "standalone"))
            correction_progress = []
            worker.progress_updated.connect(correction_progress.append)
            worker.run()
            self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
            self.assertEqual(worker.result.processed_count, 0)
            self.assertEqual(worker.result.expected_count, count)
            self.assertEqual(worker.result.outputs, ())
            self.assertEqual(correction_progress, [])
            self.assertFalse((self.root / "standalone").exists())

            full_progress = []
            pipeline = full_process.FullProcessPipeline(
                str(self.source),
                str(self.source),
                str(self.root / "full"),
                "unused",
                0,
                0,
                progress_callback=full_progress.append,
            )
            pipeline._stage_sidecar_manifests[
                full_process._path_key(str(self.source))
            ] = (
                str(self.source / "sample_Spectra.txt"),
                str(self.source / "sample_ShutterCount.txt"),
            )
            with self.assertRaises(full_process._StageAbort):
                pipeline.do_overlap_correction(str(self.source), "Sample")
            result = pipeline.result()
            self.assertEqual(result.status, PreprocessingStatus.FAILED)
            self.assertEqual(result.failed_stage, full_process.FullProcessStage.SAMPLE_OVERLAP)
            stage = result.stages[0]
            self.assertEqual(stage.outcome, full_process.FullProcessStageOutcome.FAILED)
            self.assertIsNotNone(stage.operation_result)
            self.assertEqual(stage.operation_result.expected_count, count)
            self.assertEqual(stage.operation_result.outputs, ())
            self.assertEqual(full_progress, [])
            self.assertFalse((self.root / "full").exists())
            image_loader.assert_not_called()

    def test_standalone_worker_and_full_process_produce_identical_overlap_arrays(self):
        self.write_sidecars(self.source, 3)
        self.write_images(self.source, ("00002", "00000", "00001"))
        auxiliary = self.source / "sample_SummedImg.fits"
        auxiliary.write_bytes(b"not a correction frame")

        payloads = []
        loader = ImageLoadWorker(str(self.source), overlap_inputs=True)
        loader.run_loaded.connect(lambda _folder, payload: payloads.append(payload))
        loader.run()
        payload = payloads[0]
        self.assertEqual(tuple(payload["images"]), ("00000", "00001", "00002"))
        self.assertTrue(
            any(auxiliary.name in warning for warning in payload["preflight_warnings"])
        )

        standalone = OverlapCorrectionWorker(
            payload, "standalone", str(self.root / "standalone")
        )
        standalone.run()
        self.assertEqual(standalone.result.status, PreprocessingStatus.SUCCEEDED)

        output = self.root / "full"
        output.mkdir()
        pipeline = full_process.FullProcessPipeline(
            str(self.source), str(self.source), str(output), "unused", 0, 0
        )
        pipeline._stage_image_manifests[
            full_process._path_key(str(self.source))
        ] = (*tuple(payload["image_paths"]), str(auxiliary))
        pipeline._stage_sidecar_manifests[
            full_process._path_key(str(self.source))
        ] = (
            str(self.source / "sample_Spectra.txt"),
            str(self.source / "sample_ShutterCount.txt"),
        )
        pipeline.do_overlap_correction(str(self.source), "Sample")
        full_result = pipeline.result().stages[0].operation_result
        self.assertEqual(full_result.status, PreprocessingStatus.SUCCEEDED)
        self.assertTrue(
            any(auxiliary.name in warning for warning in full_result.warnings)
        )
        self.assertFalse(
            any("extra/unmatched" in warning for warning in full_result.warnings)
        )
        self.assertFalse(
            any(auxiliary.name in Path(item.path).name for item in full_result.outputs)
        )
        self.assertEqual(auxiliary.read_bytes(), b"not a correction frame")
        self.assertEqual(full_result.processed_count, standalone.result.processed_count)
        self.assertEqual(full_result.expected_count, standalone.result.expected_count)
        self.assertEqual(
            [item.role for item in full_result.outputs],
            [item.role for item in standalone.result.outputs],
        )

        def corrected_arrays(result):
            paths = [item.path for item in result.outputs if item.role == "corrected_image"]
            return [fits.getdata(path) for path in paths]

        for full_array, standalone_array in zip(
            corrected_arrays(full_result), corrected_arrays(standalone.result)
        ):
            np.testing.assert_array_equal(full_array, standalone_array)
        self.assertEqual(
            {Path(item.path).name for item in full_result.outputs if item.role == "related_file_copy"},
            {Path(item.path).name for item in standalone.result.outputs if item.role == "related_file_copy"},
        )


if __name__ == "__main__":
    unittest.main()

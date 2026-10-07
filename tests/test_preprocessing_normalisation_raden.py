"""RADEN normalisation golden and compatibility tests.

Golden arrays were captured by running the original RadenNormalisationWorker
at 04bb285383de172dca308e0c9512c6640ded3ca2 with the fixtures below.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from NEAT.domain import PreprocessingStatus
from NEAT.services import preprocessing_normalisation_raden as raden
from NEAT.services.preprocessing_normalisation import normalise_classic_frame
from NEAT.services.preprocessing_normalisation_kernel import (
    normalise_local_open_beam_frame,
)
from NEAT.workers.preprocessing import RadenNormalisationWorker

SAMPLE_FRAMES = tuple(
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    + 10 * index
    for index in range(3)
)
OPEN_BEAM_FRAMES = tuple(
    np.array([[2, 1, 3], [4, 6, 5], [7, 8, 9]], dtype=np.float32)
    * (index + 1)
    for index in range(3)
)

GOLDEN_N1_M1_SCALE3 = np.array(
    [
        [
            [0.6153846383094788, 1.1428571939468384, 1.6000001430511475],
            [1.7142858505249023, 2.0, 2.25],
            [2.240000009536743, 2.461538553237915, 2.5714285373687744],
        ],
        [
            [5.076923370361328, 5.142857074737549, 5.199999809265137],
            [4.5, 4.5, 4.5],
            [4.079999923706055, 4.153846263885498, 4.071428298950195],
        ],
        [
            [7.753846168518066, 7.5428571701049805, 7.360000133514404],
            [6.171428680419922, 6.0, 5.850000381469727],
            [5.184000015258789, 5.169230937957764, 4.971428871154785],
        ],
    ],
    dtype=np.float32,
)

GOLDEN_N0_M0_SCALE1 = np.array(
    [
        [
            [0.5, 2.0, 1.0],
            [1.0, 0.8333333134651184, 1.2000000476837158],
            [1.0, 1.0, 1.0],
        ],
        [
            [2.75, 6.0, 2.1666667461395264],
            [1.75, 1.25, 1.600000023841858],
            [1.2142857313156128, 1.125, 1.0555555820465088],
        ],
        [
            [3.5, 7.333333492279053, 2.555555582046509],
            [2.0, 1.3888888359069824, 1.7333333492279053],
            [1.2857142686843872, 1.1666666269302368, 1.0740740299224854],
        ],
    ],
    dtype=np.float32,
)

GOLDEN_CANCELLED_PAGE_ZERO = np.array(
    [
        [1.5, 6.0, 3.0],
        [3.0, 2.5, 3.6000001430511475],
        [3.0, 3.0, 3.0],
    ],
    dtype=np.float32,
)

GOLDEN_NONFINITE_PAGE = np.array(
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    dtype=np.float32,
)


def write_stack(
    folder: Path,
    stem: str,
    frames: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    pulse_meta: dict | None = None,
    tof: dict | None = None,
    metadata_path: str | None = None,
) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{stem}.tiff"
    pages = [Image.fromarray(np.asarray(frame, dtype=np.float32)) for frame in frames]
    pages[0].save(file_path, save_all=True, append_images=pages[1:])
    return {
        "file_path": str(file_path),
        "n_frames": len(frames),
        "image_shape": tuple(frames[0].shape),
        "metadata_path": metadata_path,
        "axes": {
            "tof": tof
            or {"bins": len(frames), "min": 0, "max": len(frames), "units": "ms"},
            "meta": pulse_meta or {},
        },
    }


def read_stack(path: str | Path) -> np.ndarray:
    pages = []
    with Image.open(path) as image:
        for index in range(image.n_frames):
            image.seek(index)
            pages.append(np.array(image, copy=True))
    return np.asarray(pages, dtype=np.float32)


class _RadenFixtureMixin:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sample_folder = self.root / "sample"
        self.open_beam_folder = self.root / "open_beam"
        self.output_folder = self.root / "out"
        self.sample_info = write_stack(
            self.sample_folder,
            "sample",
            SAMPLE_FRAMES,
            pulse_meta={"pulses_with_data": 4, "pulses": 9},
        )
        self.open_beam_info = write_stack(
            self.open_beam_folder,
            "beam",
            OPEN_BEAM_FRAMES,
            pulse_meta={"pulses_with_data": 12, "pulses": 15},
        )
        self.messages: list[str] = []
        self.progress: list[int] = []

    def tearDown(self):
        self.temp.cleanup()

    def run_service(self, n=1, m=1, **kwargs):
        message_callback = kwargs.pop("message_callback", self.messages.append)
        progress_callback = kwargs.pop("progress_callback", self.progress.append)
        return raden.normalise_raden_tiff_stack(
            self.sample_info,
            self.open_beam_info,
            str(self.output_folder),
            kwargs.pop("base_name", "foo"),
            n,
            m,
            message_callback=message_callback,
            progress_callback=progress_callback,
            **kwargs,
        )

    def worker(self, *, base_name="foo", n=1, m=1):
        return RadenNormalisationWorker(
            {"folder_path": str(self.sample_folder), "kind": "raden_tiff_stack", "info": self.sample_info},
            {"folder_path": str(self.open_beam_folder), "kind": "raden_tiff_stack", "info": self.open_beam_info},
            str(self.output_folder),
            base_name,
            n,
            m,
        )


class TestRadenNormalisationService(_RadenFixtureMixin, unittest.TestCase):
    def test_original_worker_golden_border_temporal_pulses_and_progress(self):
        result = self.run_service(n=1, m=1)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual((result.processed_count, result.expected_count), (3, 3))
        self.assertEqual(self.progress, [33, 66, 100])
        self.assertEqual(
            self.messages[:2],
            [
                "RADEN pulse scale: sample=4, open-beam=12, scale=3.0000",
                "<b>--- Starting RADEN stack normalisation ---</b> 3 frames, output='foo_sample.tiff'",
            ],
        )
        self.assertEqual(len(result.outputs), 1)
        self.assertEqual(result.outputs[0].role, "normalised_stack")
        np.testing.assert_array_equal(read_stack(result.outputs[0].path), GOLDEN_N1_M1_SCALE3)
        # Nonsymmetric values prove output orientation is unchanged (unflipped).
        self.assertEqual(float(read_stack(result.outputs[0].path)[0, 0, 1]), 1.1428571939468384)

    def test_original_worker_golden_n0_m0_unavailable_pulses(self):
        self.sample_info["axes"]["meta"] = {}
        self.open_beam_info["axes"]["meta"] = {}
        result = self.run_service(n=0, m=0, base_name="zero")
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        np.testing.assert_array_equal(
            read_stack(result.outputs[0].path), GOLDEN_N0_M0_SCALE1
        )
        self.assertEqual(
            self.messages[0], "RADEN pulse metadata unavailable; using scale=1.0"
        )

    def test_input_nonfinite_values_are_sanitised_before_kernel(self):
        values = [
            np.array([[np.nan, np.inf, -np.inf], [1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
        ]
        beam = [
            np.array([[np.nan, np.inf, -np.inf], [2, 4, 6], [8, 10, 12]], dtype=np.float32),
            np.full((3, 3), 2, dtype=np.float32),
        ]
        self.sample_info = write_stack(self.sample_folder, "sample", values)
        self.open_beam_info = write_stack(self.open_beam_folder, "beam", beam)
        result = self.run_service(n=0, m=0)
        np.testing.assert_array_equal(read_stack(result.outputs[0].path)[0], GOLDEN_NONFINITE_PAGE)
        # The source page remains as written; sanitisation is on a copied read.
        with Image.open(self.sample_info["file_path"]) as source:
            source.seek(0)
            self.assertTrue(np.isnan(np.array(source)[0, 0]))

    def test_temporal_pages_clip_at_both_edges_and_accumulate_as_float64(self):
        seen = []
        original = raden.normalise_raden_frame

        def record(sample, open_beam_sum, frame_count, window_half, scale):
            seen.append((frame_count, open_beam_sum.dtype))
            return original(sample, open_beam_sum, frame_count, window_half, scale)

        with patch.object(raden, "normalise_raden_frame", side_effect=record):
            self.run_service(n=1, m=1)
        self.assertEqual(seen, [(2, np.dtype("float64")), (3, np.dtype("float64")), (2, np.dtype("float64"))])

    def test_shared_kernel_and_classic_golden_remain_exact(self):
        sample = SAMPLE_FRAMES[0].astype(np.float32)
        open_sum = OPEN_BEAM_FRAMES[0].astype(np.float64) + OPEN_BEAM_FRAMES[1]
        # The Issue #37 baseline classic function produced these exact values
        # for this sample/open-beam pair, before the shared-kernel extraction.
        classic_baseline = np.array(
            [
                [0.6153846383094788, 1.1428571939468384, 1.6000001430511475],
                [1.7142858505249023, 2.0, 2.25],
                [2.240000009536743, 2.461538553237915, 2.5714285373687744],
            ],
            dtype=np.float32,
        )
        actual_kernel = normalise_local_open_beam_frame(
            sample, open_sum, 2, 1, np.float32(3)
        )
        np.testing.assert_array_equal(actual_kernel, classic_baseline)

        classic_sample = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        classic_ob = {"a": np.array([[2, 1, 3], [4, 6, 5], [7, 8, 9]], dtype=np.float32),
                      "b": np.array([[4, 2, 6], [8, 12, 10], [14, 16, 18]], dtype=np.float32)}
        classic = normalise_classic_frame(
            classic_sample, classic_ob, ["a", "b"], 0, 1, 1, 3.0
        )
        np.testing.assert_array_equal(classic, classic_baseline)
        self.assertEqual(classic.dtype, np.float32)

    def test_kernel_preserves_raden_specific_shape_and_window_errors(self):
        with self.assertRaisesRegex(ValueError, "^sample/open-beam frame shape mismatch$"):
            raden.normalise_raden_frame(np.ones((2, 2)), np.ones((2, 3)), 1, 0, 1)
        with self.assertRaisesRegex(ValueError, "^frame is too small for the selected spatial window$"):
            raden.normalise_raden_frame(np.ones((2, 2)), np.ones((2, 2)), 1, 1, 1)

    def test_pulse_preference_fallback_and_invalid_values(self):
        self.assertEqual(
            RadenNormalisationWorker._pulse_count(
                {"axes": {"meta": {"pulses_with_data": 4, "pulses": 99}}}
            ),
            4.0,
        )
        self.assertEqual(
            RadenNormalisationWorker._pulse_count(
                {"axes": {"meta": {"pulses_with_data": np.nan, "pulses": 7}}}
            ),
            7.0,
        )
        for value in (None, 0, -2, np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.assertIsNone(
                    RadenNormalisationWorker._pulse_count(
                        {"axes": {"meta": {"pulses_with_data": value}}}
                    )
                )
        with self.assertRaises(TypeError):
            RadenNormalisationWorker._pulse_count(
                {"axes": {"meta": {"pulses_with_data": "not-a-number"}}}
            )

    def test_service_falls_back_to_pulses_and_keeps_exact_scale_message(self):
        self.sample_info["axes"]["meta"] = {"pulses_with_data": np.nan, "pulses": 7}
        self.open_beam_info["axes"]["meta"] = {"pulses_with_data": 0, "pulses": 14}
        result = self.run_service(n=0, m=0)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(
            self.messages[0],
            "RADEN pulse scale: sample=7, open-beam=14, scale=2.0000",
        )

    def test_invalid_pulse_type_is_fatal_and_expected_count_is_not_invented(self):
        self.sample_info["axes"]["meta"] = {"pulses_with_data": "bad"}
        result = self.run_service()
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(result.expected_count, None)
        self.assertIn("Fatal error in RADEN normalisation:", self.messages[-1])

    def test_tof_default_tolerance_units_and_whitespace(self):
        sample = {"axes": {"tof": {"bins": 3, "min": 0, "max": 3, "units": "MS"}}}
        close = {"axes": {"tof": {"bins": 3, "min": 1e-9, "max": 3, "units": "ms"}}}
        self.assertTrue(RadenNormalisationWorker._same_tof_axis(sample, close))
        close["axes"]["tof"]["units"] = " ms "
        self.assertFalse(RadenNormalisationWorker._same_tof_axis(sample, close))
        del close["axes"]["tof"]["min"]
        self.assertFalse(RadenNormalisationWorker._same_tof_axis(sample, close))

    def test_validation_order_exact_errors_and_output_folder_timing(self):
        self.open_beam_info["n_frames"] = 4
        self.open_beam_info["image_shape"] = (4, 4)
        self.open_beam_info["axes"]["tof"]["units"] = "s"
        result = self.run_service()
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertIsNone(result.expected_count)
        self.assertEqual(
            result.errors,
            ("Frame count mismatch: sample=3, open beam=4.",),
        )
        self.assertTrue(self.output_folder.is_dir())
        self.assertEqual(list(self.output_folder.iterdir()), [])

        self.open_beam_info["n_frames"] = 3
        result = self.run_service()
        self.assertEqual(
            result.errors,
            ("Image shape mismatch: sample=(3, 3), open beam=(4, 4).",),
        )
        self.open_beam_info["image_shape"] = (3, 3)
        result = self.run_service()
        self.assertEqual(
            result.errors,
            ("Sample and open-beam RADEN TOF axes do not match.",),
        )
        self.assertEqual(list(self.output_folder.iterdir()), [])

    def test_expected_count_becomes_known_after_total_and_tiff_open_failure(self):
        self.sample_info["file_path"] = str(self.root / "missing.tiff")
        result = self.run_service(n=0, m=0)
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual(result.expected_count, 3)
        self.assertEqual(result.processed_count, 0)
        self.assertEqual(result.outputs, ())

    def test_output_directory_creation_failure_has_unknown_expected_count(self):
        with patch.object(raden.os, "makedirs", side_effect=OSError("mkdir failure")):
            result = self.run_service(n=0, m=0)
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertIsNone(result.expected_count)
        self.assertEqual(result.errors, ("mkdir failure",))
        self.assertEqual(result.outputs, ())

    def test_output_names_cover_blank_plain_and_tiff_extensions(self):
        for base, expected in (
            ("", "normalised_sample.tiff"),
            ("foo", "foo_sample.tiff"),
            ("foo.tif", "foo.tiff"),
            ("foo.TIFF", "foo.tiff"),
        ):
            with self.subTest(base=base):
                folder = self.root / f"name-{base or 'blank'}"
                result = raden.normalise_raden_tiff_stack(
                    self.sample_info,
                    self.open_beam_info,
                    str(folder),
                    base,
                    0,
                    0,
                )
                self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
                self.assertEqual(Path(result.outputs[0].path).name, expected)

    def test_page_cancellation_retains_one_page_and_skips_sidecars(self):
        (self.sample_folder / "sample.stat").write_text("sample sidecar")
        running = [True]

        def progress(value):
            self.progress.append(value)
            if value == 33:
                running[0] = False

        result = self.run_service(
            n=0,
            m=0,
            cancellation_check=lambda: not running[0],
            progress_callback=progress,
        )
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual((result.processed_count, result.expected_count), (1, 3))
        self.assertEqual(self.progress, [33])
        self.assertEqual([output.role for output in result.outputs], ["normalised_stack"])
        np.testing.assert_array_equal(read_stack(result.outputs[0].path), GOLDEN_CANCELLED_PAGE_ZERO[None, ...])
        self.assertFalse((self.output_folder / "foo_sample.stat").exists())
        self.assertIn("User stopped RADEN normalisation.", self.messages)
        self.assertEqual(self.messages[-1], "RADEN normalisation incomplete: 1 of 3 frames written.")

    def test_cancellation_after_last_page_skips_sidecars(self):
        (self.sample_folder / "sample.stat").write_text("sample sidecar")
        running = [True]

        def progress(value):
            self.progress.append(value)
            if value == 100:
                running[0] = False

        result = self.run_service(
            n=0,
            m=0,
            cancellation_check=lambda: not running[0],
            progress_callback=progress,
        )
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual((result.processed_count, result.expected_count), (3, 3))
        self.assertEqual(self.progress, [33, 66, 100])
        self.assertEqual([output.role for output in result.outputs], ["normalised_stack"])
        self.assertFalse((self.output_folder / "foo_sample.stat").exists())

    def test_prestopped_service_creates_directory_but_reports_no_phantom_tiff(self):
        result = self.run_service(n=0, m=0, cancellation_check=lambda: True)
        self.assertTrue(self.output_folder.is_dir())
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual((result.processed_count, result.expected_count), (0, 3))
        self.assertEqual(result.outputs, ())
        output_tiff = self.output_folder / "foo_sample.tiff"
        # Baseline Pillow writer creates a zero-byte placeholder before page 0.
        self.assertTrue(output_tiff.exists())
        self.assertEqual(output_tiff.stat().st_size, 0)
        self.assertEqual(self.messages[1], "<b>--- Starting RADEN stack normalisation ---</b> 3 frames, output='foo_sample.tiff'")
        self.assertIn("User stopped RADEN normalisation.", self.messages)

    def test_later_calculation_failure_retains_partial_tiff_and_count(self):
        original = raden.normalise_raden_frame
        calls = [0]

        def fail_second(*args):
            calls[0] += 1
            if calls[0] == 2:
                raise ValueError("fixture page failure")
            return original(*args)

        with patch.object(raden, "normalise_raden_frame", side_effect=fail_second):
            result = self.run_service(n=0, m=0)
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual((result.processed_count, result.expected_count), (1, 3))
        self.assertEqual(result.errors, ("fixture page failure",))
        self.assertEqual([output.role for output in result.outputs], ["normalised_stack"])
        np.testing.assert_array_equal(read_stack(result.outputs[0].path), GOLDEN_CANCELLED_PAGE_ZERO[None, ...])
        self.assertTrue(self.messages[-1].startswith("Fatal error in RADEN normalisation:"))
        self.assertFalse(any("incomplete" in message for message in self.messages))

    def test_later_page_write_failure_retains_prior_page_and_is_fatal(self):
        real_fromarray = Image.fromarray
        calls = [0]

        class FailOnSave:
            def save(self, *_args, **_kwargs):
                raise OSError("fixture page write failure")

        def fromarray(array, *args, **kwargs):
            calls[0] += 1
            return FailOnSave() if calls[0] == 2 else real_fromarray(array, *args, **kwargs)

        with patch.object(raden.Image, "fromarray", side_effect=fromarray):
            result = self.run_service(n=0, m=0)
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual((result.processed_count, result.expected_count), (1, 3))
        self.assertEqual(result.errors, ("fixture page write failure",))
        self.assertEqual([o.role for o in result.outputs], ["normalised_stack"])
        np.testing.assert_array_equal(read_stack(result.outputs[0].path), GOLDEN_CANCELLED_PAGE_ZERO[None, ...])

    def test_sidecars_metadata_precedence_raw_order_and_output_order(self):
        metadata = self.sample_folder / "preferred.STAT"
        metadata.write_text("metadata wins")
        (self.sample_folder / "first.stat").write_text("unselected stat")
        (self.sample_folder / "one.JSON").write_text("json")
        (self.sample_folder / "one.log").write_text("log")
        (self.open_beam_folder / "beam.json").write_text("must not copy")
        self.sample_info["metadata_path"] = str(metadata)
        real_listdir = os.listdir
        calls = []

        def listdir(path):
            calls.append(os.fspath(path))
            if os.path.normcase(os.fspath(path)) == os.path.normcase(str(self.sample_folder)):
                return ["first.stat", "preferred.STAT", "one.JSON", "one.log", "sample.tiff"]
            return real_listdir(path)

        with patch.object(raden.os, "listdir", side_effect=listdir):
            result = self.run_service(n=0, m=0)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(calls, [str(self.sample_folder), str(self.sample_folder)])
        self.assertEqual(
            [Path(output.path).name for output in result.outputs],
            ["foo_sample.tiff", "foo_sample.stat", "foo_sample.json", "foo_sample.log"],
        )
        self.assertEqual((self.output_folder / "foo_sample.stat").read_text(), "metadata wins")
        self.assertEqual((self.output_folder / "foo_sample.json").read_text(), "json")
        self.assertEqual((self.output_folder / "foo_sample.log").read_text(), "log")
        self.assertFalse((self.output_folder / "beam.json").exists())

    def test_unusual_metadata_extension_is_preserved_in_lowercase(self):
        metadata = self.sample_folder / "instrument.MetadataX"
        metadata.write_text("metadata")
        self.sample_info["metadata_path"] = str(metadata)
        result = self.run_service(n=0, m=0)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(Path(result.outputs[1].path).name, "foo_sample.metadatax")
        self.assertEqual((self.output_folder / "foo_sample.metadatax").read_text(), "metadata")

    def test_sidecar_scans_separately_in_stat_json_log_order_and_uses_first_match(self):
        (self.sample_folder / "first.STAT").write_text("first")
        (self.sample_folder / "second.stat").write_text("second")
        (self.sample_folder / "note.JSON").write_text("json")
        (self.sample_folder / "note.log").write_text("log")
        real_listdir = os.listdir
        seen = []

        def listdir(path):
            if os.path.normcase(os.fspath(path)) == os.path.normcase(str(self.sample_folder)):
                seen.append(path)
                return ["second.stat", "first.STAT", "note.JSON", "note.log"]
            return real_listdir(path)

        with patch.object(raden.os, "listdir", side_effect=listdir):
            self.output_folder.mkdir()
            outputs = raden.copy_raden_sidecars(self.sample_info, "C:/out/result.tiff", str(self.output_folder))
        self.assertEqual(len(seen), 3)
        self.assertEqual([Path(o.path).name for o in outputs], ["result.stat", "result.json", "result.log"])
        self.assertEqual((self.output_folder / "result.stat").read_text(), "second")

    def test_sidecar_scan_requires_a_regular_file(self):
        (self.sample_folder / "directory.stat").mkdir()
        (self.sample_folder / "real.stat").write_text("real sidecar")
        self.output_folder.mkdir()
        outputs = raden.copy_raden_sidecars(
            self.sample_info, "C:/out/regular.tiff", str(self.output_folder)
        )
        self.assertEqual([Path(output.path).name for output in outputs], ["regular.stat"])
        self.assertEqual((self.output_folder / "regular.stat").read_text(), "real sidecar")

    def test_missing_sidecars_are_silent_and_nonfatal(self):
        result = self.run_service(n=0, m=0)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(len(result.outputs), 1)
        self.assertEqual(result.warnings, ())

    def test_fatal_sidecar_failure_keeps_tiff_and_prior_sidecar_outputs(self):
        (self.sample_folder / "a.stat").write_text("stat")
        (self.sample_folder / "a.json").write_text("json")
        real_copy = raden.shutil.copyfile

        def fail_json(src, dst):
            if os.path.splitext(src)[1].lower() == ".json":
                raise OSError("fixture sidecar failure")
            return real_copy(src, dst)

        with patch.object(raden.shutil, "copyfile", side_effect=fail_json):
            result = self.run_service(n=0, m=0)
        self.assertEqual(result.status, PreprocessingStatus.FAILED)
        self.assertEqual((result.processed_count, result.expected_count), (3, 3))
        self.assertEqual(
            [Path(output.path).name for output in result.outputs],
            ["foo_sample.tiff", "foo_sample.stat"],
        )
        self.assertEqual(result.errors, ("fixture sidecar failure",))
        self.assertTrue((self.output_folder / "foo_sample.tiff").exists())
        self.assertTrue((self.output_folder / "foo_sample.stat").exists())
        self.assertFalse(any("complete:" in message or "incomplete:" in message for message in self.messages))
        self.assertTrue(self.messages[-1].startswith("Fatal error in RADEN normalisation:"))

    def test_cancellation_during_started_sidecar_copy_finishes_copy_then_snapshots(self):
        (self.sample_folder / "a.stat").write_text("stat")
        (self.sample_folder / "a.json").write_text("json")
        running = [True]
        real_copy = raden.shutil.copyfile
        copies = []

        def stop_during_copy(src, dst):
            copies.append(Path(dst).suffix)
            out = real_copy(src, dst)
            if len(copies) == 1:
                running[0] = False
            return out

        with patch.object(raden.shutil, "copyfile", side_effect=stop_during_copy):
            result = self.run_service(
                n=0,
                m=0,
                cancellation_check=lambda: not running[0],
            )
        self.assertEqual(copies, [".stat", ".json"])
        self.assertEqual(result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual([Path(o.path).suffix for o in result.outputs], [".tiff", ".stat", ".json"])

    def test_no_cancellation_recheck_after_final_status_snapshot(self):
        calls = [0]
        running = [True]

        def cancellation_check():
            calls[0] += 1
            if calls[0] == 5:
                running[0] = False
                return False
            return not running[0]

        result = self.run_service(n=0, m=0, cancellation_check=cancellation_check)
        self.assertEqual(calls[0], 5)
        self.assertEqual(result.status, PreprocessingStatus.SUCCEEDED)

    def test_service_import_and_run_in_fresh_process_without_qt_ui_or_workers(self):
        script = r'''
import json, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from NEAT.services.preprocessing_normalisation_raden import normalise_raden_tiff_stack
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    infos = []
    for folder_name, name, factor in (("sample", "sample", 2), ("beam", "beam", 4)):
        folder = root / folder_name
        folder.mkdir()
        path = folder / f"{name}.tiff"
        pages = [Image.fromarray(np.full((3, 3), factor + i, dtype=np.float32)) for i in range(2)]
        pages[0].save(path, save_all=True, append_images=pages[1:])
        infos.append({"file_path": str(path), "n_frames": 2, "image_shape": (3, 3),
            "axes": {"tof": {"bins": 2, "min": 0, "max": 2, "units": "ms"}, "meta": {}}})
    result = normalise_raden_tiff_stack(infos[0], infos[1], str(root / "out"), "", 0, 0)
    assert result.status.value == "succeeded"
    forbidden = [name for name in sys.modules if name == "PyQt5" or name.startswith("PyQt5.")
        or name == "NEAT.ui" or name.startswith("NEAT.ui.")
        or name == "NEAT.workers" or name.startswith("NEAT.workers.")]
    assert not forbidden, forbidden
    print(json.dumps({"status": result.status.value, "outputs": len(result.outputs), "forbidden": forbidden}))
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(completed.stdout), {"status": "succeeded", "outputs": 1, "forbidden": []})


class TestRadenNormalisationWorkerAdapter(_RadenFixtureMixin, unittest.TestCase):
    def test_constructor_validation_helpers_and_success_result(self):
        with self.assertRaisesRegex(ValueError, "Normalisation n must be between 0 and 100"):
            self.worker(n=-1)
        worker = self.worker(n=0, m=0)
        finished = []
        messages = []
        progress = []
        worker.finished.connect(lambda: finished.append(True))
        worker.message.connect(messages.append)
        worker.progress_updated.connect(progress.append)
        worker.run()
        self.assertTrue(worker.succeeded)
        self.assertEqual(worker.result.status, PreprocessingStatus.SUCCEEDED)
        self.assertEqual(progress, [33, 66, 100])
        self.assertEqual(len(finished), 1)
        self.assertTrue(any("RADEN stack normalisation complete:" in value for value in messages))

    def test_worker_helpers_delegate_and_preserve_errors(self):
        self.assertEqual(RadenNormalisationWorker._pulse_count(self.sample_info), 4.0)
        self.assertTrue(RadenNormalisationWorker._same_tof_axis(self.sample_info, self.open_beam_info))
        worker = self.worker(n=1, m=0)
        worker._validate(self.sample_info, self.open_beam_info)
        frame = np.ones((3, 3), dtype=np.float32)
        self.assertEqual(worker._normalise_frame(frame, frame.astype(np.float64), 1, 1).dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "sample/open-beam frame shape mismatch"):
            worker._normalise_frame(np.ones((2, 2)), np.ones((2, 3)), 1, 1)
        with self.assertRaisesRegex(ValueError, "frame is too small for the selected spatial window"):
            worker._normalise_frame(np.ones((2, 2)), np.ones((2, 2)), 1, 1)
        (self.sample_folder / "worker.stat").write_text("worker sidecar")
        self.output_folder.mkdir()
        worker._copy_sidecars(self.sample_info, str(self.output_folder / "compat.tiff"))
        self.assertEqual((self.output_folder / "compat.stat").read_text(), "worker sidecar")

    def test_worker_stop_after_first_page_maps_cancelled_and_emits_finished_once(self):
        worker = self.worker(n=0, m=0)
        messages = []
        finished = []
        progress = []
        worker.message.connect(messages.append)
        worker.finished.connect(lambda: finished.append(True))
        worker.progress_updated.connect(lambda value: (progress.append(value), worker.stop() if value == 33 else None))
        worker.run()
        self.assertFalse(worker.succeeded)
        self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual((worker.result.processed_count, worker.result.expected_count), (1, 3))
        self.assertEqual(progress, [33])
        self.assertEqual(len(finished), 1)
        np.testing.assert_array_equal(read_stack(worker.result.outputs[0].path), GOLDEN_CANCELLED_PAGE_ZERO[None, ...])
        self.assertIn("Stop signal received. Terminating RADEN normalisation process.", messages)

    def test_worker_prestopped_still_validates_and_creates_empty_output_folder(self):
        worker = self.worker(n=0, m=0)
        worker.stop()
        messages = []
        worker.message.connect(messages.append)
        worker.run()
        self.assertEqual(worker.result.status, PreprocessingStatus.CANCELLED)
        self.assertEqual((worker.result.processed_count, worker.result.expected_count), (0, 3))
        self.assertTrue(self.output_folder.is_dir())
        self.assertEqual(worker.result.outputs, ())
        self.assertEqual(messages[0], "RADEN pulse scale: sample=4, open-beam=12, scale=3.0000")
        self.assertIn("<b>--- Starting RADEN stack normalisation ---</b> 3 frames, output='foo_sample.tiff'", messages)

    def test_worker_info_resolution_fatal_retains_empty_directory_and_unknown_expected(self):
        worker = RadenNormalisationWorker(
            {"folder_path": str(self.sample_folder)},
            {"folder_path": str(self.open_beam_folder)},
            str(self.output_folder),
            "foo",
            0,
            0,
        )
        finished = []
        worker.finished.connect(lambda: finished.append(True))
        with patch("NEAT.workers.preprocessing.get_raden_tiff_stack_info", side_effect=ValueError("metadata failure")):
            worker.run()
        self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
        self.assertIsNone(worker.result.expected_count)
        self.assertEqual(worker.result.errors, ("metadata failure",))
        self.assertEqual(worker.result.outputs, ())
        self.assertTrue(self.output_folder.is_dir())
        self.assertEqual(len(finished), 1)

    def test_worker_final_gc_occurs_after_page_gc_and_finished_once_on_fatal(self):
        worker = self.worker(n=0, m=0)
        event_progress = []
        gc_progress = []
        finished = []
        worker.progress_updated.connect(event_progress.append)
        worker.finished.connect(lambda: finished.append(True))
        with patch("NEAT.workers.preprocessing.gc.collect", side_effect=lambda: gc_progress.append(len(event_progress))):
            worker.run()
        self.assertEqual(gc_progress, [1, 3])  # page 0, then worker-finally
        self.assertEqual(len(finished), 1)

    def test_worker_final_gc_cadence_includes_indices_zero_and_one_hundred(self):
        frames = tuple(np.full((1, 1), index + 1, dtype=np.float32) for index in range(101))
        self.sample_info = write_stack(self.sample_folder, "many", frames)
        self.open_beam_info = write_stack(self.open_beam_folder, "beam-many", frames)
        worker = self.worker(n=0, m=0)
        progress = []
        gc_progress = []
        worker.progress_updated.connect(progress.append)
        with patch("NEAT.workers.preprocessing.gc.collect", side_effect=lambda: gc_progress.append(len(progress))):
            worker.run()
        self.assertTrue(worker.succeeded)
        self.assertEqual(progress[0], 0)  # int(100 * 1 / 101)
        self.assertEqual(progress[100], 100)
        self.assertEqual(gc_progress, [1, 101, 101])  # indices 0, 100, then final worker GC

    def test_page_boundary_gc_failure_remains_an_in_service_failure(self):
        worker = self.worker(n=0, m=0)
        messages, finished = [], []
        collections = []
        worker.message.connect(messages.append)
        worker.finished.connect(lambda: finished.append(True))

        def collect():
            collections.append(True)
            if len(collections) == 1:
                raise RuntimeError("page GC failed")

        with patch("NEAT.workers.preprocessing.gc.collect", side_effect=collect):
            worker.run()

        self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
        self.assertFalse(worker.succeeded)
        self.assertIn("page GC failed", worker.result.errors)
        self.assertFalse(any(message.startswith("[WARN] Worker finalization:") for message in messages))
        self.assertEqual(collections, [True, True])
        self.assertEqual(finished, [True])

    def test_worker_reports_fatal_sidecar_after_completed_tiff_without_summary(self):
        (self.sample_folder / "a.json").write_text("json")
        worker = self.worker(n=0, m=0)
        messages = []
        worker.message.connect(messages.append)
        with patch("NEAT.workers.preprocessing.shutil.copyfile", side_effect=OSError("copy failed")):
            worker.run()
        self.assertFalse(worker.succeeded)
        self.assertEqual(worker.result.status, PreprocessingStatus.FAILED)
        self.assertEqual((worker.result.processed_count, worker.result.expected_count), (3, 3))
        self.assertEqual([Path(o.path).name for o in worker.result.outputs], ["foo_sample.tiff"])
        self.assertTrue(any(message == "Fatal error in RADEN normalisation: copy failed" for message in messages))
        self.assertFalse(any("complete:" in message or "incomplete:" in message for message in messages))


if __name__ == "__main__":
    unittest.main()

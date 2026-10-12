"""Offline, image-bearing tests for the local visual-review boundary."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import unittest
from dataclasses import replace
from io import BytesIO
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication  # noqa: E402

from NEAT.domain.agent_model import AgentModelSettings
from NEAT.domain.agent_preflight import FrameEvidence, PreflightResult, assess_dataset
from NEAT.domain.agent_vision import VisionSettings, VisionStatus
from NEAT.domain.observation import (
    DatasetMode,
    DatasetRevisionClock,
    ImageViewSpec,
    WavelengthAxisStatus,
)
from NEAT.services.agent_vision import (
    LocalVisionInspector,
    VisionCancelled,
    VisionError,
    VisionSettingsRepository,
    VisionStale,
    VisionUnavailable,
    _roi_from_response,
    render_roi_overlay,
    run_visual_rehearsal,
    trusted_phase_edges,
    validate_vision_settings,
)
from NEAT.services.scientific_observation import (
    create_dataset_handle,
    get_image_preview,
    inspect_dataset,
)
from NEAT.ui.agent_model_settings_dialog import AgentModelSettingsDialog


def _fixture(*, ambiguous=False):
    wavelengths = np.linspace(1.0, 5.0, 64)
    frames = []
    for index, wavelength in enumerate(wavelengths):
        image = np.zeros((64, 64), dtype=float)
        if not ambiguous:
            image[18:50, 16:52] = 20.0 + (wavelength >= 2.4) * 3.0 + index * 0.01
        frames.append(image)
    clock = DatasetRevisionClock()
    handle = create_dataset_handle(
        mode=DatasetMode.IMAGES,
        images=frames,
        wavelengths=wavelengths,
        revision_clock=clock,
    )
    dataset = inspect_dataset(handle)
    evidence = tuple(
        FrameEvidence(
            index,
            float(wavelengths[index]),
            get_image_preview(handle, index, ImageViewSpec(max_dimension=32)),
        )
        for index in (0, 31, 63)
    )
    preflight = PreflightResult(
        "preflight",
        handle.dataset_id,
        handle.revision,
        dataset,
        assess_dataset(
            dataset,
            None,
            evidence_references=tuple(f"image:frame:{e.frame_index}" for e in evidence),
        ),
        None,
        evidence,
        None,
        None,
    )
    return handle, preflight


class FakeVision:
    def __init__(
        self,
        *,
        bad_roi=None,
        bad_phase=None,
        bad_hkl=None,
        bad_window=None,
        uncertain=False,
        no_roi=False,
        no_phase=False,
    ):
        self.calls = []
        self.bad_roi = bad_roi
        self.bad_phase = bad_phase
        self.bad_hkl = bad_hkl
        self.bad_window = bad_window
        self.uncertain = uncertain
        self.no_roi = no_roi
        self.no_phase = no_phase
        self.cancelled = False

    def inspect(self, stage, images, prompt):
        self.calls.append((stage, images, prompt))
        for png in images:
            assert png.startswith(b"\x89PNG\r\n\x1a\n")
            with Image.open(BytesIO(png)) as image:
                image.verify()
        if stage == "sample_roi":
            if self.no_roi:
                return {
                    "roi": None,
                    "reason": "The frame does not show a reliable interior.",
                    "confidence": "low",
                    "ambiguity_flags": ["sample boundary unclear"],
                }
            return {
                "roi": self.bad_roi
                or {"x_min": 20, "x_max": 36, "y_min": 24, "y_max": 40},
                "reason": "Visually inside the bright band; background remains ambiguous.",
                "confidence": "medium",
                "ambiguity_flags": ["brightness alone is not sample proof"],
            }
        if stage == "phase_shortlist":
            return {"candidate_phase_ids": [] if self.no_phase else ["Fe_bcc", "CeO2"]}
        if self.uncertain:
            return {
                "status": "uncertain",
                "phase_id": None,
                "hkls": [],
                "edge_reasons": [],
                "display_min": None,
                "display_max": None,
                "reason": "Overlays overlap ambiguously.",
                "alternatives": ["Fe_bcc", "CeO2"],
                "unresolved": ["Review another ROI"],
            }
        return {
            "status": "hypothesis",
            "phase_id": self.bad_phase or "Fe_bcc",
            "hkls": [self.bad_hkl or [2, 1, 1]],
            "edge_reasons": ["Marker appears near a visible step."],
            "display_min": 2.0,
            "display_max": self.bad_window or 4.5,
            "reason": "A visible step appears near a trusted marker; not identified.",
            "alternatives": ["CeO2"],
            "unresolved": ["Compare fitting uncertainty later"],
        }

    def cancel(self):
        self.cancelled = True


class VisionServiceTests(unittest.TestCase):
    def test_actual_pngs_roi_bootstrap_overlay_and_no_scientific_writes(self):
        handle, preflight = _fixture()
        model = FakeVision()
        original = tuple(frame.copy() for frame in handle.frames)
        result = run_visual_rehearsal("run", handle, preflight, model)
        self.assertEqual(result.status, VisionStatus.NEEDS_REVIEW)
        self.assertEqual(result.roi.x_min, 20)
        self.assertEqual(result.phase_id, "Fe_bcc")
        self.assertEqual(result.hkls, ((2, 1, 1),))
        self.assertEqual(result.display_window, (2.0, 4.5))
        self.assertEqual(result.inspected_phases, ("Fe_bcc", "CeO2"))
        self.assertEqual(
            result.request_stages, ("sample_roi", "phase_shortlist", "phase_overlay")
        )
        self.assertEqual([call[0] for call in model.calls], list(result.request_stages))
        self.assertEqual([len(call[1]) for call in model.calls], [3, 1, 2])
        self.assertIsNotNone(result.roi_preview_png)
        self.assertIsNotNone(result.spectrum_png)
        self.assertEqual(len(result.phase_overlays), 2)
        with Image.open(BytesIO(result.roi_preview_png)) as painted:
            self.assertEqual(painted.size, (32, 32))
            self.assertEqual(painted.convert("RGB").getpixel((10, 12)), (255, 255, 0))
        for before, after in zip(original, handle.frames):
            np.testing.assert_array_equal(before, after)

    def test_preview_original_coordinate_mapping_is_preserved(self):
        handle, preflight = _fixture()
        frame = preflight.frames[0].image
        roi, *_ = _roi_from_response(
            {
                "roi": {"x_min": 20, "x_max": 36, "y_min": 24, "y_max": 40},
                "reason": "Visible interior",
                "confidence": "low",
                "ambiguity_flags": [],
            },
            (64, 64),
        )
        png = render_roi_overlay(frame.png_bytes, frame.coordinate_mapping, roi)
        with Image.open(BytesIO(png)) as image:
            self.assertEqual(image.convert("RGB").getpixel((10, 12)), (255, 255, 0))
        self.assertEqual(
            frame.coordinate_mapping.preview_pixel_to_original(10, 12), (21, 25)
        )

    def test_rejects_false_coordinates_phase_hkl_and_display_interval(self):
        for kwargs in (
            {"bad_roi": {"x_min": True, "x_max": 36, "y_min": 24, "y_max": 40}},
            {"bad_roi": {"x_min": 60, "x_max": 70, "y_min": 24, "y_max": 40}},
            {"bad_phase": "unsupported_phase"},
            {"bad_hkl": [9, 9, 9]},
            {"bad_window": 9.0},
        ):
            with self.subTest(kwargs=kwargs):
                handle, preflight = _fixture()
                with self.assertRaises(VisionError):
                    run_visual_rehearsal("run", handle, preflight, FakeVision(**kwargs))

    def test_uncertainty_is_not_fabricated_as_phase_selection(self):
        handle, preflight = _fixture()
        result = run_visual_rehearsal(
            "run", handle, preflight, FakeVision(uncertain=True)
        )
        self.assertEqual(result.status, VisionStatus.NEEDS_REVIEW)
        self.assertEqual(result.phase_id, "")
        self.assertEqual(result.hkls, ())
        self.assertEqual(result.inspected_phases, ("Fe_bcc", "CeO2"))

    def test_missing_roi_or_phase_returns_honest_review_state(self):
        for variant, expected_calls in (({"no_roi": True}, 1), ({"no_phase": True}, 2)):
            with self.subTest(variant=variant):
                handle, preflight = _fixture()
                model = FakeVision(**variant)
                result = run_visual_rehearsal("run", handle, preflight, model)
                self.assertEqual(result.status, VisionStatus.NEEDS_REVIEW)
                self.assertEqual(result.phase_id, "")
                self.assertEqual(len(model.calls), expected_calls)
        handle, preflight = _fixture(ambiguous=True)
        result = run_visual_rehearsal("run", handle, preflight, FakeVision(no_roi=True))
        self.assertIsNone(result.roi)
        self.assertEqual(result.phase_id, "")

    def test_stale_and_cancelled_stop_before_following_model_calls(self):
        handle, preflight = _fixture()
        model = FakeVision()
        with self.assertRaises(VisionCancelled):
            run_visual_rehearsal(
                "run", handle, preflight, model, cancelled=lambda: True
            )
        self.assertEqual(model.calls, [])
        handle, preflight = _fixture()
        model = FakeVision()
        original = model.inspect

        def mutate(stage, images, prompt):
            result = original(stage, images, prompt)
            if stage == "sample_roi":
                handle._revision_clock.invalidate()
            return result

        model.inspect = mutate
        with self.assertRaises(VisionStale):
            run_visual_rehearsal("run", handle, preflight, model)
        self.assertEqual(len(model.calls), 1)

    def test_profile_and_invalid_axis_do_not_claim_visual_phase(self):
        handle, preflight = _fixture()
        profile = replace(
            preflight, dataset=replace(preflight.dataset, mode=DatasetMode.PROFILE)
        )
        with self.assertRaises(VisionUnavailable):
            run_visual_rehearsal("run", handle, profile, FakeVision())
        invalid = replace(
            preflight,
            dataset=replace(
                preflight.dataset,
                wavelength_axis_status=WavelengthAxisStatus.NON_MONOTONIC,
            ),
        )
        with self.assertRaises(VisionError):
            run_visual_rehearsal("run", handle, invalid, FakeVision())

    def test_endpoint_is_numeric_loopback_only(self):
        for endpoint in (
            "https://api.openai.com/v1",
            "http://example.com/v1",
            "http://localhost:11434/v1",
            "http://127.0.0.1:11434/redirect",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_vision_settings(VisionSettings("vision-model", endpoint))
        self.assertEqual(
            validate_vision_settings(
                VisionSettings("vision-model", "http://127.0.0.1:11434/v1")
            ).model_id,
            "vision-model",
        )

    def test_builtin_phase_markers_are_trusted_neat_values(self):
        edges = trusted_phase_edges("Fe_bcc")
        self.assertIn((2, 1, 1), [hkl for hkl, _ in edges])
        with self.assertRaises(VisionError):
            trusted_phase_edges("user_custom")

    def test_local_multimodal_request_contains_pngs_and_no_cloud_or_credential(self):
        handle, preflight = _fixture()
        received = []

        async def respond(request):
            received.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]
                },
            )

        with patch(
            "httpx.AsyncHTTPTransport", return_value=httpx.MockTransport(respond)
        ):
            inspector = LocalVisionInspector(
                VisionSettings("vision-model", "http://127.0.0.1:11434/v1")
            )
            self.assertEqual(
                inspector.inspect(
                    "sample_roi",
                    (preflight.frames[0].image.png_bytes,),
                    "Inspect this frame",
                ),
                {},
            )
        self.assertEqual(len(received), 1)
        self.assertEqual(
            str(received[0].url), "http://127.0.0.1:11434/v1/chat/completions"
        )
        self.assertNotIn("authorization", received[0].headers)
        body = json.loads(received[0].content)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertTrue(
            body["messages"][1]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )
        self.assertNotIn("api.openai.com", received[0].content.decode("ascii"))

    def test_active_local_request_is_cooperatively_cancelled(self):
        _, preflight = _fixture()
        entered = threading.Event()
        errors = []

        async def wait_forever(_request):
            entered.set()
            await asyncio.sleep(60)
            return httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]
                },
            )

        with patch(
            "httpx.AsyncHTTPTransport", return_value=httpx.MockTransport(wait_forever)
        ):
            inspector = LocalVisionInspector(
                VisionSettings("vision-model", "http://127.0.0.1:11434/v1")
            )

            def run():
                try:
                    inspector.inspect(
                        "sample_roi",
                        (preflight.frames[0].image.png_bytes,),
                        "Inspect this frame",
                    )
                except VisionCancelled as exc:
                    errors.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(entered.wait(2))
            inspector.cancel()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)

    def test_agent_settings_save_and_cancel_local_vision_independently(self):
        app = QApplication.instance() or QApplication([])

        class MemoryRepo:
            def __init__(self, value):
                self.value = value

            def load(self):
                return self.value

            def save(self, value):
                self.value = value

            def clear(self):
                self.value = VisionSettings()

        planning_repo = MemoryRepo(AgentModelSettings())
        vision_repo = MemoryRepo(VisionSettings())
        dialog = AgentModelSettingsDialog(
            settings_repository=planning_repo,
            vision_settings_repository=vision_repo,
        )
        dialog.vision_model_edit.setText("local-vision")
        dialog.vision_endpoint_edit.setText("http://127.0.0.1:11434/v1")
        dialog.reject()
        self.assertEqual(vision_repo.load(), VisionSettings())
        saved = AgentModelSettingsDialog(
            settings_repository=planning_repo,
            vision_settings_repository=vision_repo,
        )
        saved.vision_model_edit.setText("local-vision")
        saved.vision_endpoint_edit.setText("http://127.0.0.1:11434/v1")
        saved.save_and_accept()
        self.assertEqual(
            vision_repo.load(),
            VisionSettings("local-vision", "http://127.0.0.1:11434/v1"),
        )
        self.assertEqual(planning_repo.load().mode.value, "offline_deterministic")
        app.processEvents()


if __name__ == "__main__":
    unittest.main()

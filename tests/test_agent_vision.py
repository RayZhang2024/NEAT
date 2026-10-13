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
from PyQt5.QtWidgets import QApplication

from NEAT.domain.agent_model import AgentModelMode, AgentModelSettings
from NEAT.domain.agent_preflight import FrameEvidence, PreflightResult, assess_dataset
from NEAT.domain.agent_vision import VisionStatus
from NEAT.domain.observation import (
    DatasetMode,
    DatasetRevisionClock,
    ImageViewSpec,
    WavelengthAxisStatus,
)
from NEAT.services.agent_vision import (
    MAX_VISION_COMPLETION_TOKENS,
    OpenAIVisionInspector,
    VisionCancelled,
    VisionError,
    VisionStale,
    VisionUnavailable,
    _message_text_content,
    _roi_from_response,
    _visual_response_schema,
    render_roi_overlay,
    run_visual_rehearsal,
    trusted_phase_edges,
)
from NEAT.services.scientific_observation import (
    create_dataset_handle,
    get_image_preview,
    inspect_dataset,
)
from NEAT.ui.agent_model_settings_dialog import AgentModelSettingsDialog
from NEAT.ui.agent_vision_worker import VisionWorker


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
        candidate_hkls=None,
        bad_window=None,
        uncertain=False,
        no_roi=False,
        no_phase=False,
        roi_reason="Visually inside the bright band; background remains ambiguous.",
    ):
        self.calls = []
        self.bad_roi = bad_roi
        self.bad_phase = bad_phase
        self.bad_hkl = bad_hkl
        self.candidate_hkls = candidate_hkls
        self.bad_window = bad_window
        self.uncertain = uncertain
        self.no_roi = no_roi
        self.no_phase = no_phase
        self.roi_reason = roi_reason
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
                "reason": self.roi_reason,
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
            "hkls": (
                self.candidate_hkls
                if self.candidate_hkls is not None
                else [self.bad_hkl or [2, 1, 1]]
            ),
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
        self.assertIn("between 3 and 128 original pixels inclusive", model.calls[0][2])
        self.assertIn("x coordinates for columns and y coordinates for rows", model.calls[0][2])
        self.assertIn("choose 1 to 5 unique HKLs", model.calls[2][2])
        self.assertIn("set status to uncertain", model.calls[2][2])
        self.assertIsNotNone(result.roi_preview_png)
        self.assertIsNotNone(result.spectrum_png)
        self.assertEqual(len(result.phase_overlays), 2)
        with Image.open(BytesIO(result.roi_preview_png)) as painted:
            self.assertEqual(painted.size, (32, 32))
            self.assertEqual(painted.convert("RGB").getpixel((10, 12)), (255, 255, 0))
        for before, after in zip(original, handle.frames):
            np.testing.assert_array_equal(before, after)

    def test_preview_original_coordinate_mapping_is_preserved(self):
        _handle, preflight = _fixture()
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

    def test_rejects_false_coordinates_and_uninspected_phase(self):
        for kwargs in (
            {"bad_roi": {"x_min": True, "x_max": 36, "y_min": 24, "y_max": 40}},
            {"bad_roi": {"x_min": 60, "x_max": 70, "y_min": 24, "y_max": 40}},
        ):
            with self.subTest(kwargs=kwargs):
                handle, preflight = _fixture()
                with self.assertRaises(VisionError):
                    run_visual_rehearsal("run", handle, preflight, FakeVision(**kwargs))

    def test_invalid_candidate_edge_lists_preserve_evidence_for_manual_review(self):
        for hkls in ([], [[2, 1, 1]] * 6):
            with self.subTest(hkls=hkls):
                handle, preflight = _fixture()
                model = FakeVision(candidate_hkls=hkls)
                result = run_visual_rehearsal("run", handle, preflight, model)
                self.assertEqual(result.status, VisionStatus.NEEDS_REVIEW)
                self.assertEqual(result.phase_id, "")
                self.assertEqual(result.hkls, ())
                self.assertIn("1 to 5 edges", result.message)
                self.assertIn("did not accept a phase or edge suggestion", result.message)
                self.assertIsNotNone(result.roi_preview_png)
                self.assertIsNotNone(result.spectrum_png)
                self.assertEqual(len(result.phase_overlays), 2)

    def test_unsupported_hkl_and_display_interval_are_not_presented_as_valid(self):
        for kwargs in (
            {"bad_phase": "unsupported_phase"},
            {"bad_hkl": [9, 9, 9]},
        ):
            with self.subTest(kwargs=kwargs):
                handle, preflight = _fixture()
                result = run_visual_rehearsal("run", handle, preflight, FakeVision(**kwargs))
                self.assertEqual(result.status, VisionStatus.NEEDS_REVIEW)
                self.assertEqual(result.phase_id, "")
                self.assertEqual(result.hkls, ())
                self.assertIn("phase", result.message.lower())
                self.assertIsNotNone(result.roi_preview_png)
                self.assertIsNotNone(result.spectrum_png)

        handle, preflight = _fixture()
        result = run_visual_rehearsal(
            "run", handle, preflight, FakeVision(bad_window=9.0)
        )
        self.assertEqual(result.status, VisionStatus.NEEDS_REVIEW)
        self.assertEqual(result.phase_id, "Fe_bcc")
        self.assertEqual(result.hkls, ((2, 1, 1),))
        self.assertIsNotNone(result.display_window)
        assert result.display_window is not None
        self.assertLess(result.display_window[0], 2.3352)
        self.assertGreater(result.display_window[1], 2.3352)
        self.assertIn("Model proposed 2–9 Å", result.display_interval_diagnostic)
        self.assertIn("above the measured maximum", result.display_interval_diagnostic)
        self.assertIn("NEAT shows its locally calculated edge-flank envelope", result.message)

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

    def test_openai_visual_review_requires_cloud_mode_and_fresh_consent(self):
        settings = AgentModelSettings(
            mode=AgentModelMode.OPENAI_CLOUD, model_id="gpt-6-luna"
        )
        with self.assertRaises(VisionUnavailable):
            OpenAIVisionInspector(
                settings, api_key="test-key", cloud_egress_authorized=False
            )
        with self.assertRaises(VisionUnavailable):
            OpenAIVisionInspector(
                AgentModelSettings(), api_key="test-key", cloud_egress_authorized=True
            )
        with self.assertRaises(VisionUnavailable):
            OpenAIVisionInspector(
                settings, api_key=None, cloud_egress_authorized=True
            )

    def test_builtin_phase_markers_are_trusted_neat_values(self):
        edges = trusted_phase_edges("Fe_bcc")
        self.assertIn((2, 1, 1), [hkl for hkl, _ in edges])
        with self.assertRaises(VisionError):
            trusted_phase_edges("user_custom")

    def test_openai_multimodal_request_uses_shared_model_and_actual_bounded_png(self):
        _handle, preflight = _fixture()
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
            inspector = OpenAIVisionInspector(
                AgentModelSettings(
                    mode=AgentModelMode.OPENAI_CLOUD, model_id="gpt-6-luna"
                ),
                api_key="temporary-test-key",
                cloud_egress_authorized=True,
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
            str(received[0].url), "https://api.openai.com/v1/chat/completions"
        )
        self.assertEqual(
            received[0].headers["authorization"], "Bearer temporary-test-key"
        )
        body = json.loads(received[0].content)
        self.assertEqual(
            body["max_completion_tokens"], MAX_VISION_COMPLETION_TOKENS
        )
        response_format = body["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(
            response_format["json_schema"]["name"], "neat_visual_sample_roi"
        )
        self.assertEqual(
            set(response_format["json_schema"]["schema"]["required"]),
            {"roi", "reason", "confidence", "ambiguity_flags"},
        )
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertIs(body["store"], False)
        self.assertTrue(
            body["messages"][1]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )
        self.assertEqual(inspector._api_key, "temporary-test-key")
        inspector.release_credential()
        self.assertIsNone(inspector._api_key)

    def test_each_visual_stage_has_a_closed_strict_output_schema(self):
        for stage, required in (
            (
                "sample_roi",
                {"roi", "reason", "confidence", "ambiguity_flags"},
            ),
            ("phase_shortlist", {"candidate_phase_ids"}),
            (
                "phase_overlay",
                {
                    "status",
                    "phase_id",
                    "hkls",
                    "edge_reasons",
                    "display_min",
                    "display_max",
                    "reason",
                    "alternatives",
                    "unresolved",
                },
            ),
        ):
            schema = _visual_response_schema(stage)
            self.assertEqual(schema["type"], "object")
            self.assertEqual(set(schema["required"]), required)
            self.assertIs(schema["additionalProperties"], False)
            self.assertEqual(set(schema["properties"]), required)

    def test_openai_visual_response_accepts_text_only_content_blocks(self):
        self.assertEqual(
            _message_text_content(
                {"content": [{"type": "text", "text": '{"ok":true}'}]}
            ),
            '{"ok":true}',
        )
        with self.assertRaisesRegex(VisionError, "non-text content"):
            _message_text_content(
                {"content": [{"type": "image", "data": "private"}]}
            )
        with self.assertRaisesRegex(VisionError, "missing content"):
            _message_text_content({"content": None})

    def test_worker_exposes_safe_validator_diagnostic_without_model_text(self):
        handle, preflight = _fixture()
        model = FakeVision(
            bad_roi={"x_min": -1, "x_max": 10, "y_min": 10, "y_max": 20},
            roi_reason="private model response must not be shown",
        )
        results = []
        worker = VisionWorker(
            "diagnostic-run",
            handle,
            preflight,
            AgentModelSettings(),
            inspector=model,
        )
        worker.completed.connect(results.append)
        worker.run()
        self.assertEqual(results[0].status, VisionStatus.BLOCKED)
        self.assertIn("ROI is outside the current image", results[0].message)
        self.assertNotIn("private model response must not be shown", results[0].message)

    def test_active_openai_request_is_cooperatively_cancelled(self):
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
            inspector = OpenAIVisionInspector(
                AgentModelSettings(
                    mode=AgentModelMode.OPENAI_CLOUD, model_id="gpt-6-luna"
                ),
                api_key="test-key",
                cloud_egress_authorized=True,
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

    def test_vision_worker_requires_consent_and_drops_resolved_credential(self):
        import NEAT.ui.agent_vision_worker as worker_module

        handle, preflight = _fixture()
        settings = AgentModelSettings(
            mode=AgentModelMode.OPENAI_CLOUD, model_id="shared-model"
        )

        class Resolver:
            reads = 0

            def get(self):
                self.reads += 1
                return "ephemeral-agent-key"

        no_consent_resolver = Resolver()
        no_consent_results = []
        denied = VisionWorker(
            "denied-run",
            handle,
            preflight,
            settings,
            credential_resolver=no_consent_resolver,
            cloud_egress_authorized=False,
        )
        denied.completed.connect(no_consent_results.append)
        denied.run()
        self.assertEqual(no_consent_resolver.reads, 0)
        self.assertEqual(no_consent_results[0].status, VisionStatus.UNAVAILABLE)

        handle, preflight = _fixture()
        resolver = Resolver()
        retained = []

        class ReleasableFake(FakeVision):
            released = False

            def release_credential(self):
                self.released = True

        fake = ReleasableFake()

        def make_inspector(settings_arg, *, api_key, cloud_egress_authorized):
            retained.append((settings_arg.model_id, api_key, cloud_egress_authorized))
            return fake

        worker = VisionWorker(
            "authorized-run",
            handle,
            preflight,
            settings,
            credential_resolver=resolver,
            cloud_egress_authorized=True,
        )
        results = []
        worker.completed.connect(results.append)
        with patch.object(
            worker_module, "OpenAIVisionInspector", side_effect=make_inspector
        ):
            worker.run()
        self.assertEqual(resolver.reads, 1)
        self.assertEqual(retained, [("shared-model", "ephemeral-agent-key", True)])
        self.assertTrue(fake.released)
        self.assertIsNone(worker._credential_resolver)
        self.assertIsNone(worker._handle)
        self.assertNotIn("ephemeral-agent-key", repr(vars(worker)))
        self.assertEqual(results[0].status, VisionStatus.NEEDS_REVIEW)

    def test_agent_settings_use_one_model_configuration_for_planning_and_vision(self):
        app = QApplication.instance() or QApplication([])

        class MemoryRepo:
            def __init__(self, value):
                self.value = value

            def load(self):
                return self.value

            def save(self, value):
                self.value = value

        planning_repo = MemoryRepo(AgentModelSettings())
        saved = AgentModelSettingsDialog(
            settings_repository=planning_repo,
        )
        self.assertFalse(hasattr(saved, "vision_model_edit"))
        self.assertFalse(hasattr(saved, "vision_endpoint_edit"))
        saved.mode_combo.setCurrentIndex(
            saved.mode_combo.findData(AgentModelMode.OPENAI_CLOUD.value)
        )
        saved.model_edit.setText("gpt-6-luna")
        saved.save_and_accept()
        self.assertEqual(planning_repo.load().model_id, "gpt-6-luna")
        self.assertEqual(planning_repo.load().mode, AgentModelMode.OPENAI_CLOUD)
        app.processEvents()


if __name__ == "__main__":
    unittest.main()

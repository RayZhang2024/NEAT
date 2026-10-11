"""Offline LangGraph planning contract and transition tests."""

from __future__ import annotations

import json
from importlib.util import find_spec
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from NEAT.domain.agent_planning import (
    MAX_PLAN_STEPS,
    FuturePermissionClass,
    PlanStep,
    PlanStepType,
    PlanningContext,
    PlanningErrorCode,
    PlanningStatus,
    PlanningValidationError,
    context_to_state,
)
from NEAT.domain.agent_preflight import PreflightReadiness
from NEAT.domain.observation import DatasetMode, WavelengthAxisStatus
from NEAT.services.agent_planning import (
    PlanningCancelled,
    PlanningDependencyUnavailable,
    PlanningStale,
    _default_proposal,
    run_planning,
)


_HAS_LANGGRAPH = find_spec("langgraph") is not None


def _context(
    *,
    mode: DatasetMode = DatasetMode.IMAGES,
    preflight_complete: bool = True,
    readiness: PreflightReadiness = PreflightReadiness.READY,
    axis: WavelengthAxisStatus = WavelengthAxisStatus.VALID,
    roi: bool = True,
    evidence: tuple[str, ...] = (
        "image:frame:0",
        "image:frame:2",
        "spectrum:full-source",
        "spectrum:plot",
    ),
    provenance: bool = True,
    nonfinite: bool = False,
    observations: tuple[str, ...] = ("Dataset mode: images.", "Wavelength axis: valid."),
    warnings: tuple[str, ...] = (),
) -> PlanningContext:
    return PlanningContext(
        run_id=uuid4().hex,
        preflight_run_id=uuid4().hex,
        preflight_complete=preflight_complete,
        dataset_id=uuid4().hex,
        revision=4,
        readiness=readiness,
        mode=mode,
        wavelength_axis_status=axis,
        images_available=mode is DatasetMode.IMAGES,
        spectrum_available=mode is not DatasetMode.EMPTY,
        mapping_applicable=mode is DatasetMode.IMAGES,
        roi_available=roi,
        calibration_provenance_known=provenance,
        spectrum_has_nonfinite=nonfinite,
        evidence_references=evidence,
        observations=observations,
        warnings=warnings,
    )


class AgentPlanningEvidenceWordingTests(unittest.TestCase):
    def test_roi_overlay_and_spectrum_plot_wording_requires_matching_evidence(self):
        context = _context(
            roi=False,
            evidence=("image:frame:0", "spectrum:full-source"),
        )
        steps = _default_proposal(context_to_state(context))
        rationales = {step.step_type: step.rationale for step in steps}

        self.assertIn(PlanStepType.REVIEW_IMAGE_PREVIEWS, rationales)
        self.assertNotIn("overlay", rationales[PlanStepType.REVIEW_IMAGE_PREVIEWS].lower())
        self.assertIn(PlanStepType.REVIEW_MEASURED_SPECTRUM, rationales)
        self.assertIn("unavailable", rationales[PlanStepType.REVIEW_MEASURED_SPECTRUM].lower())

        context_with_visual_evidence = _context(
            roi=True,
            evidence=("image:frame:0", "spectrum:full-source", "spectrum:plot"),
        )
        steps_with_visual_evidence = _default_proposal(
            context_to_state(context_with_visual_evidence)
        )
        visual_rationales = {
            step.step_type: step.rationale for step in steps_with_visual_evidence
        }
        self.assertIn("overlay", visual_rationales[PlanStepType.REVIEW_IMAGE_PREVIEWS].lower())
        self.assertIn("rendered plot", visual_rationales[PlanStepType.REVIEW_MEASURED_SPECTRUM].lower())


@unittest.skipUnless(_HAS_LANGGRAPH, "requires the optional NEAT [agent] extra")
class AgentPlanningGraphTests(unittest.TestCase):
    def test_image_dataset_runs_real_bounded_graph_transitions(self):
        context = _context()
        outcome = run_planning(context)
        self.assertIn(
            outcome.status,
            {PlanningStatus.READY_PLAN, PlanningStatus.NEEDS_INPUT},
        )
        self.assertEqual(
            outcome.transitions,
            (
                "validate_preflight",
                "propose_read_only_plan",
                "validate_plan",
                "terminal",
            ),
        )
        self.assertLessEqual(len(outcome.steps), MAX_PLAN_STEPS)
        self.assertTrue(
            all(step.step_type is not PlanStepType.STOP_NEEDS_MORE_EVIDENCE for step in outcome.steps)
        )

    def test_valid_plan_and_profile_only_input_are_distinguished(self):
        context = _context(
            mode=DatasetMode.PROFILE,
            evidence=("spectrum:full-source", "spectrum:plot"),
            roi=False,
        )
        outcome = run_planning(context)
        self.assertEqual(outcome.status, PlanningStatus.READY_PLAN)
        self.assertIn(
            PlanStepType.REVIEW_MEASURED_SPECTRUM,
            {step.step_type for step in outcome.steps},
        )
        self.assertNotIn(
            PlanStepType.REVIEW_IMAGE_PREVIEWS,
            {step.step_type for step in outcome.steps},
        )

    def test_missing_roi_invalid_axis_unknown_calibration_and_nonfinite_data_need_input(self):
        scenarios = (
            (_context(roi=False), PlanStepType.REQUEST_ROI),
            (
                _context(axis=WavelengthAxisStatus.NON_MONOTONIC),
                PlanStepType.REQUEST_CALIBRATION_REVIEW,
            ),
            (
                _context(provenance=False),
                PlanStepType.REQUEST_CALIBRATION_REVIEW,
            ),
            (
                _context(nonfinite=True, provenance=True),
                PlanStepType.STOP_NEEDS_MORE_EVIDENCE,
            ),
        )
        for context, expected_step in scenarios:
            with self.subTest(expected_step=expected_step):
                outcome = run_planning(context)
                self.assertEqual(outcome.status, PlanningStatus.NEEDS_INPUT)
                self.assertIn(expected_step, {step.step_type for step in outcome.steps})

    def test_invalid_dataset_and_missing_evidence_take_finite_blocked_route(self):
        for context in (
            _context(readiness=PreflightReadiness.INVALID),
            _context(evidence=()),
            _context(preflight_complete=False),
        ):
            outcome = run_planning(context)
            self.assertEqual(outcome.status, PlanningStatus.BLOCKED_INVALID_DATA)
            self.assertEqual(outcome.transitions, ("validate_preflight", "terminal"))
            self.assertFalse(outcome.steps)

    def test_validator_rejects_unissued_evidence_and_unsupported_step_vocabulary(self):
        context = _context()
        invalid_ref = PlanStep(
            PlanStepType.REVIEW_MEASURED_SPECTRUM,
            "Review the existing measured spectrum.",
            ("spectrum:forged",),
            True,
            False,
            # The type remains a read-only proposal; provenance must still match.
            FuturePermissionClass.LOCAL_READ_ONLY,
        )
        outcome = run_planning(context, proposer=lambda _state: (invalid_ref,))
        self.assertEqual(outcome.status, PlanningStatus.FAILED)
        self.assertEqual(outcome.error_code, PlanningErrorCode.PLAN_REJECTED)
        self.assertIn("validate_plan", outcome.transitions)

        unsupported = PlanStep(
            "run_fitting",  # type: ignore[arg-type]
            "Run a fit.",
            (),
            True,
            False,
            invalid_ref.permission_classification,
        )
        rejected = run_planning(context, proposer=lambda _state: (unsupported,))
        self.assertEqual(rejected.status, PlanningStatus.FAILED)
        self.assertEqual(rejected.error_code, PlanningErrorCode.PLAN_REJECTED)

    def test_oversized_plan_is_rejected_without_truncation(self):
        context = _context()
        step = PlanStep(
            PlanStepType.INSPECT_DATASET_SUMMARY,
            "Review the validated summary.",
            (),
            True,
            False,
            FuturePermissionClass.LOCAL_READ_ONLY,
        )
        outcome = run_planning(context, proposer=lambda _state: (step,) * (MAX_PLAN_STEPS + 1))
        self.assertEqual(outcome.status, PlanningStatus.FAILED)
        self.assertEqual(outcome.error_code, PlanningErrorCode.PLAN_REJECTED)
        self.assertFalse(outcome.steps)

    def test_memory_checkpoint_contains_only_bounded_json_compatible_state(self):
        from langgraph.checkpoint.memory import InMemorySaver

        context = _context()
        checkpointer = InMemorySaver()
        outcome = run_planning(context, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": context.run_id}}
        checkpoint = checkpointer.get(config)
        self.assertIsNotNone(checkpoint)
        values = checkpoint["channel_values"]
        encoded = json.dumps(values)
        self.assertNotIn("png_bytes", encoded)
        self.assertNotIn("frames", encoded)
        self.assertNotIn("C:/", encoded)
        self.assertEqual(values["run_id"], context.run_id)
        self.assertEqual(values["dataset_id"], context.dataset_id)
        self.assertEqual(outcome.status, PlanningStatus.READY_PLAN)

    def test_cancellation_and_dataset_change_are_checked_cooperatively(self):
        context = _context()
        with self.assertRaises(PlanningCancelled):
            run_planning(context, cancelled=lambda: True)

        calls = 0

        def changed_token():
            nonlocal calls
            calls += 1
            return (
                (context.dataset_id, context.revision)
                if calls < 4
                else (context.dataset_id, context.revision + 1)
            )

        with self.assertRaises(PlanningStale):
            run_planning(context, current_dataset_token=changed_token)

    def test_global_tracing_configuration_cannot_send_planning_state(self):
        import socket

        context = _context()
        with (
            patch.dict(
                os.environ,
                {
                    "LANGSMITH_TRACING": "true",
                    "LANGCHAIN_TRACING_V2": "true",
                    "LANGSMITH_API_KEY": "must-not-be-used",
                },
            ),
            patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("planning attempted network access"),
            ),
        ):
            outcome = run_planning(context)
        self.assertIn(
            outcome.status,
            {PlanningStatus.READY_PLAN, PlanningStatus.NEEDS_INPUT},
        )

    def test_planning_uses_no_filesystem_and_imports_no_model_provider(self):
        import builtins
        import sys

        context = _context()
        import_before = set(sys.modules)
        run_planning(context)  # Load LangGraph before guarding Python file access.
        provider_prefixes = (
            "langchain_openai",
            "langchain_anthropic",
            "langchain_google",
            "langchain_ollama",
        )
        self.assertFalse(
            any(
                name.startswith(provider_prefixes) and name not in import_before
                for name in sys.modules
            ),
            "The local planner imported a model provider.",
        )
        with patch.object(
            builtins, "open", side_effect=AssertionError("planning opened a file")
        ):
            outcome = run_planning(context)
        self.assertIn(
            outcome.status,
            {PlanningStatus.READY_PLAN, PlanningStatus.NEEDS_INPUT},
        )


class AgentPlanningOptionalDependencyTests(unittest.TestCase):
    def test_import_is_lazy_and_missing_extra_is_reported(self):
        import NEAT.services.agent_planning as planner

        context = _context()
        with patch.object(planner, "_load_langgraph", side_effect=PlanningDependencyUnavailable):
            with self.assertRaises(PlanningDependencyUnavailable):
                planner.run_planning(context)

    def test_safe_state_rejects_private_or_oversized_payloads(self):
        for private_path in (
            "C:\\private\\sample.fits",
            "/private/sample.fits",
            "Data file: ./sample.fits",
        ):
            with self.subTest(private_path=private_path):
                context = _context(observations=(private_path,))
                with self.assertRaises(PlanningValidationError):
                    context_to_state(context)
        context = _context(observations=("x" * 241,))
        with self.assertRaises(PlanningValidationError):
            context_to_state(context)

    def test_planning_worker_reports_missing_extra_and_releases_context(self):
        import threading

        from PyQt5.QtCore import Qt
        from NEAT.domain.observation import DatasetRevisionClock
        from NEAT.ui.agent_planning_worker import PlanningWorker

        import NEAT.ui.agent_planning_worker as worker_module

        completed = []
        event = threading.Event()
        worker = PlanningWorker(_context(), DatasetRevisionClock())
        worker.completed.connect(
            lambda result: (completed.append(result), event.set()),
            Qt.DirectConnection,
        )
        with patch.object(
            worker_module,
            "run_planning",
            side_effect=PlanningDependencyUnavailable("missing"),
        ):
            worker.start()
            self.assertTrue(event.wait(3))
            self.assertTrue(worker.wait(3000))
        self.assertEqual(completed[0].status, PlanningStatus.UNAVAILABLE)
        self.assertIsNone(worker._context)


if __name__ == "__main__":
    unittest.main()

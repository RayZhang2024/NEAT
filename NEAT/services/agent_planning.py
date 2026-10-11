"""Offline LangGraph planner for validated, read-only preflight evidence.

LangGraph is imported only when a planning run starts so the normal NEAT
installation remains independent of the optional ``agent`` extra.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Protocol, TypedDict, cast

from ..domain.agent_planning import (
    MAX_EVIDENCE_REFERENCES_PER_STEP,
    MAX_GRAPH_NODES,
    MAX_PLAN_STEPS,
    MAX_SUMMARY_ITEMS,
    MAX_TEXT_LENGTH,
    FuturePermissionClass,
    PlanStep,
    PlanStepType,
    PlanningContext,
    PlanningErrorCode,
    PlanningOutcome,
    PlanningStatus,
    PlanningValidationError,
    context_to_state,
    validate_bounded_text,
)
from ..domain.agent_preflight import PreflightReadiness
from ..domain.observation import DatasetMode, WavelengthAxisStatus


class PlanningDependencyUnavailable(RuntimeError):
    """Raised when the optional LangGraph extra is not installed."""


class PlanningCancelled(RuntimeError):
    """Cooperative stop request observed between bounded graph nodes."""


class PlanningStale(RuntimeError):
    """The live dataset token changed before planning completed."""


class GraphState(TypedDict):
    run_id: str
    preflight_run_id: str
    preflight_complete: bool
    dataset_id: str
    revision: int
    readiness: str
    mode: str
    wavelength_axis_status: str
    capabilities: dict[str, bool]
    roi_available: bool
    calibration_provenance_known: bool
    spectrum_has_nonfinite: bool
    evidence_references: list[str]
    observations: list[str]
    warnings: list[str]
    steps: list[dict[str, object]]
    route: str
    error_code: str
    transitions: list[str]


class PlanProposer(Protocol):
    """Injectable deterministic proposal seam; never an execution interface."""

    def __call__(self, state: GraphState) -> Sequence[PlanStep]: ...


_STATE_KEYS = frozenset(
    {
        "run_id",
        "preflight_run_id",
        "preflight_complete",
        "dataset_id",
        "revision",
        "readiness",
        "mode",
        "wavelength_axis_status",
        "capabilities",
        "roi_available",
        "calibration_provenance_known",
        "spectrum_has_nonfinite",
        "evidence_references",
        "observations",
        "warnings",
        "steps",
        "route",
        "error_code",
        "transitions",
    }
)
_ALLOWED_TRANSITIONS = frozenset(
    {"validate_preflight", "propose_read_only_plan", "validate_plan", "terminal"}
)
_TERMINAL_STATUSES = {status.value for status in PlanningStatus}


def _check_live(
    context: PlanningContext,
    cancelled: Callable[[], bool] | None,
    current_dataset_token: Callable[[], tuple[str, int]] | None,
) -> None:
    if cancelled is not None and cancelled():
        raise PlanningCancelled("Planning was stopped.")
    if current_dataset_token is not None and current_dataset_token() != (
        context.dataset_id,
        context.revision,
    ):
        raise PlanningStale("Dataset changed during planning.")


def _append_transition(state: GraphState, node: str) -> list[str]:
    transitions = list(state["transitions"])
    if len(transitions) >= MAX_GRAPH_NODES or node not in _ALLOWED_TRANSITIONS:
        raise PlanningValidationError("Graph transition bound was exceeded.")
    transitions.append(node)
    return transitions


def _safe_state(state: Mapping[str, object]) -> GraphState:
    if set(state) != _STATE_KEYS:
        raise PlanningValidationError("Planning state contains missing or unknown fields.")
    if not all(
        isinstance(state[key], str)
        for key in (
            "run_id",
            "preflight_run_id",
            "dataset_id",
            "readiness",
            "mode",
            "wavelength_axis_status",
            "route",
            "error_code",
        )
    ):
        raise PlanningValidationError("Planning identity fields must be strings.")
    if (
        not isinstance(state["revision"], int)
        or isinstance(state["revision"], bool)
        or state["revision"] < 0
    ):
        raise PlanningValidationError("Planning revision must be a non-negative integer.")
    for key in (
        "preflight_complete",
        "roi_available",
        "calibration_provenance_known",
        "spectrum_has_nonfinite",
    ):
        if not isinstance(state[key], bool):
            raise PlanningValidationError(f"Planning field {key} must be boolean.")
    caps = state["capabilities"]
    if not isinstance(caps, dict) or set(caps) != {
        "images_available",
        "spectrum_available",
        "mapping_applicable",
    } or not all(isinstance(value, bool) for value in caps.values()):
        raise PlanningValidationError("Dataset capabilities are malformed.")
    refs = state["evidence_references"]
    if not isinstance(refs, list) or len(refs) > 16 or not all(
        isinstance(ref, str) and len(ref) <= 80 for ref in refs
    ):
        raise PlanningValidationError("Evidence references are malformed or oversized.")
    for key in ("observations", "warnings"):
        values = state[key]
        if (
            not isinstance(values, list)
            or len(values) > MAX_SUMMARY_ITEMS
            or not all(
                isinstance(value, str) and 0 < len(value) <= MAX_TEXT_LENGTH
                for value in values
            )
        ):
            raise PlanningValidationError(f"Planning field {key} is malformed or oversized.")
    steps = state["steps"]
    transitions = state["transitions"]
    if not isinstance(steps, list) or len(steps) > MAX_PLAN_STEPS or not all(
        isinstance(step, dict) for step in steps
    ):
        raise PlanningValidationError("Proposed plan exceeds its step limit.")
    if not isinstance(transitions, list) or len(transitions) > MAX_GRAPH_NODES or not all(
        isinstance(value, str) and value in _ALLOWED_TRANSITIONS for value in transitions
    ):
        raise PlanningValidationError("Graph transition history is malformed.")
    try:
        if state["readiness"] not in {item.value for item in PreflightReadiness}:
            raise PlanningValidationError("Unknown preflight readiness.")
        if state["mode"] not in {item.value for item in DatasetMode}:
            raise PlanningValidationError("Unknown dataset mode.")
        if state["wavelength_axis_status"] not in {
            item.value for item in WavelengthAxisStatus
        }:
            raise PlanningValidationError("Unknown wavelength-axis status.")
    except TypeError as exc:
        raise PlanningValidationError("Planning enum state is malformed.") from exc
    return cast(GraphState, dict(state))


def _assert_context_coherence(state: GraphState, context: PlanningContext) -> None:
    expected = context_to_state(context)
    for key in (
        "run_id",
        "preflight_run_id",
        "preflight_complete",
        "dataset_id",
        "revision",
        "readiness",
        "mode",
        "wavelength_axis_status",
        "capabilities",
        "roi_available",
        "calibration_provenance_known",
        "spectrum_has_nonfinite",
        "evidence_references",
        "observations",
        "warnings",
    ):
        if state[key] != expected[key]:
            raise PlanningValidationError(
                "Planning state no longer matches its completed preflight context."
            )


def _serialize_step(step: PlanStep) -> dict[str, object]:
    if not isinstance(step, PlanStep) or not isinstance(step.step_type, PlanStepType):
        raise PlanningValidationError("A proposed step is not an allowlisted typed step.")
    if not isinstance(step.permission_classification, FuturePermissionClass):
        raise PlanningValidationError("A proposed step has an unknown permission class.")
    validate_bounded_text(step.rationale, "Proposed rationale")
    if len(step.evidence_ids) > MAX_EVIDENCE_REFERENCES_PER_STEP:
        raise PlanningValidationError("A proposed step cites too many evidence IDs.")
    if not all(isinstance(ref, str) for ref in step.evidence_ids):
        raise PlanningValidationError("A proposed step cites malformed evidence IDs.")
    if not isinstance(step.supported, bool) or not isinstance(step.requires_human_input, bool):
        raise PlanningValidationError("A proposed step is missing support or input classification.")
    return {
        "step_type": step.step_type.value,
        "rationale": step.rationale,
        "evidence_ids": list(step.evidence_ids),
        "supported": step.supported,
        "requires_human_input": step.requires_human_input,
        "permission_classification": step.permission_classification.value,
    }


def _deserialize_steps(raw_steps: list[dict[str, object]]) -> tuple[PlanStep, ...]:
    output: list[PlanStep] = []
    for raw in raw_steps:
        if set(raw) != {
            "step_type",
            "rationale",
            "evidence_ids",
            "supported",
            "requires_human_input",
            "permission_classification",
        }:
            raise PlanningValidationError("A proposed step contains unknown fields.")
        try:
            step_type = PlanStepType(str(raw["step_type"]))
            permission = FuturePermissionClass(str(raw["permission_classification"]))
        except ValueError as exc:
            raise PlanningValidationError("A proposed step is not allowlisted.") from exc
        evidence = raw["evidence_ids"]
        if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            raise PlanningValidationError("A proposed step cites malformed evidence IDs.")
        if (
            not isinstance(raw["rationale"], str)
            or not isinstance(raw["supported"], bool)
            or not isinstance(raw["requires_human_input"], bool)
        ):
            raise PlanningValidationError("A proposed step has malformed fields.")
        output.append(
            PlanStep(
                step_type=step_type,
                rationale=raw["rationale"],
                evidence_ids=tuple(evidence),
                supported=raw["supported"],
                requires_human_input=raw["requires_human_input"],
                permission_classification=permission,
            )
        )
    return tuple(output)


def _default_proposal(state: GraphState) -> tuple[PlanStep, ...]:
    evidence = state["evidence_references"]
    image_evidence = tuple(ref for ref in evidence if ref.startswith("image:frame:"))
    spectrum_evidence = tuple(
        ref for ref in evidence if ref in {"spectrum:full-source", "spectrum:plot"}
    )
    steps: list[PlanStep] = []

    def add(
        step_type: PlanStepType,
        rationale: str,
        evidence_ids: tuple[str, ...] = (),
        *,
        human: bool = False,
    ) -> None:
        steps.append(
            PlanStep(
                step_type,
                rationale,
                evidence_ids[:MAX_EVIDENCE_REFERENCES_PER_STEP],
                True,
                human,
                FuturePermissionClass.HUMAN_INPUT_REQUIRED
                if human
                else FuturePermissionClass.LOCAL_READ_ONLY,
            )
        )

    add(
        PlanStepType.INSPECT_DATASET_SUMMARY,
        "Review the validated dataset summary and its stated limitations.",
        tuple(evidence[:MAX_EVIDENCE_REFERENCES_PER_STEP]),
    )

    if state["mode"] == DatasetMode.IMAGES.value and image_evidence:
        add(
            PlanStepType.REVIEW_IMAGE_PREVIEWS,
            "Review the existing bounded image previews and their selected ROI overlay.",
            image_evidence,
        )
    if spectrum_evidence:
        add(
            PlanStepType.REVIEW_MEASURED_SPECTRUM,
            "Review the existing measured spectrum and its rendered plot.",
            spectrum_evidence,
        )

    if state["mode"] == DatasetMode.IMAGES.value and not state["roi_available"]:
        add(
            PlanStepType.REQUEST_ROI,
            "Choose an explicit image ROI before a measured ROI spectrum can be reviewed.",
            image_evidence,
            human=True,
        )
    needs_calibration_review = (
        state["wavelength_axis_status"] != WavelengthAxisStatus.VALID.value
        or not state["calibration_provenance_known"]
    )
    if needs_calibration_review:
        add(
            PlanStepType.REQUEST_CALIBRATION_REVIEW,
            (
                "Review wavelength calibration and provenance before relying on spectral positions."
                if state["wavelength_axis_status"] == WavelengthAxisStatus.VALID.value
                else "Review wavelength calibration because the preflight axis is not valid."
            ),
            tuple(ref for ref in evidence if ref.startswith("spectrum:")),
            human=True,
        )
    if state["spectrum_has_nonfinite"] and not needs_calibration_review:
        add(
            PlanStepType.STOP_NEEDS_MORE_EVIDENCE,
            "The measured spectrum contains non-finite intensities and needs review.",
            spectrum_evidence,
            human=True,
        )
    if (
        state["capabilities"]["spectrum_available"]
        and not spectrum_evidence
        and state["mode"] in {DatasetMode.IMAGES.value, DatasetMode.PROFILE.value}
        and (state["roi_available"] or state["mode"] == DatasetMode.PROFILE.value)
    ):
        add(
            PlanStepType.STOP_NEEDS_MORE_EVIDENCE,
            "The current preflight did not produce the expected measured spectrum evidence.",
            (),
            human=True,
        )
    if not steps or not evidence:
        return ()
    if len(steps) > MAX_PLAN_STEPS:
        raise PlanningValidationError("Deterministic proposal exceeded the step limit.")
    return tuple(steps)


def _validate_steps(state: GraphState, steps: tuple[PlanStep, ...]) -> None:
    if not steps:
        raise PlanningValidationError("No evidence is available for a proposed plan.")
    if len(steps) > MAX_PLAN_STEPS:
        raise PlanningValidationError("Plan contains too many steps.")
    refs = set(state["evidence_references"])
    for step in steps:
        _serialize_step(step)
        if any(reference not in refs for reference in step.evidence_ids):
            raise PlanningValidationError("A proposed step cites evidence outside the preflight.")
        if step.requires_human_input != (
            step.permission_classification is FuturePermissionClass.HUMAN_INPUT_REQUIRED
        ):
            raise PlanningValidationError("Human-input and permission classifications disagree.")
        if not step.supported:
            raise PlanningValidationError("Unsupported proposals are not valid M3 plan steps.")


def _load_langgraph():
    try:
        from langgraph.graph import END, START, StateGraph
        from langsmith.run_helpers import tracing_context
    except (ImportError, OSError) as exc:
        raise PlanningDependencyUnavailable(
            "Offline LangGraph planning requires NEAT's optional [agent] extra."
        ) from exc
    return END, START, StateGraph, tracing_context


def run_planning(
    context: PlanningContext,
    *,
    cancelled: Callable[[], bool] | None = None,
    current_dataset_token: Callable[[], tuple[str, int]] | None = None,
    proposer: PlanProposer | None = None,
    checkpointer: object | None = None,
) -> PlanningOutcome:
    """Run the fixed, offline StateGraph and validate its read-only proposal."""
    _check_live(context, cancelled, current_dataset_token)
    END, START, StateGraph, tracing_context = _load_langgraph()
    initial = context_to_state(context)
    _check_live(context, cancelled, current_dataset_token)

    def validate_preflight(state: GraphState) -> dict[str, object]:
        _check_live(context, cancelled, current_dataset_token)
        safe = _safe_state(state)
        _assert_context_coherence(safe, context)
        readiness = PreflightReadiness(safe["readiness"])
        if not safe["run_id"] or not safe["preflight_run_id"] or not safe["dataset_id"]:
            return {
                "route": PlanningStatus.BLOCKED_INVALID_DATA.value,
                "error_code": PlanningErrorCode.INVALID_PREFLIGHT.value,
                "transitions": _append_transition(safe, "validate_preflight"),
            }
        if not safe["preflight_complete"] or readiness is PreflightReadiness.INVALID:
            return {
                "route": PlanningStatus.BLOCKED_INVALID_DATA.value,
                "error_code": PlanningErrorCode.INVALID_PREFLIGHT.value,
                "transitions": _append_transition(safe, "validate_preflight"),
            }
        if not safe["evidence_references"]:
            return {
                "route": PlanningStatus.BLOCKED_INVALID_DATA.value,
                "error_code": PlanningErrorCode.NO_EVIDENCE.value,
                "transitions": _append_transition(safe, "validate_preflight"),
            }
        return {
            "route": "propose",
            "error_code": "",
            "transitions": _append_transition(safe, "validate_preflight"),
        }

    def route_after_preflight(state: GraphState) -> str:
        return "propose_read_only_plan" if state["route"] == "propose" else "terminal"

    def propose_read_only_plan(state: GraphState) -> dict[str, object]:
        _check_live(context, cancelled, current_dataset_token)
        safe = _safe_state(state)
        _assert_context_coherence(safe, context)
        proposals = (
            tuple(proposer(cast(GraphState, deepcopy(safe))))
            if proposer is not None
            else _default_proposal(safe)
        )
        if len(proposals) > MAX_PLAN_STEPS:
            return {
                "steps": [],
                "route": PlanningStatus.FAILED.value,
                "error_code": PlanningErrorCode.PLAN_REJECTED.value,
                "transitions": _append_transition(safe, "propose_read_only_plan"),
            }
        try:
            steps = [_serialize_step(step) for step in proposals]
        except PlanningValidationError:
            return {
                "steps": [],
                "route": PlanningStatus.FAILED.value,
                "error_code": PlanningErrorCode.PLAN_REJECTED.value,
                "transitions": _append_transition(safe, "propose_read_only_plan"),
            }
        needs_input = any(step.requires_human_input for step in proposals)
        return {
            "steps": steps,
            "route": PlanningStatus.NEEDS_INPUT.value
            if needs_input
            else PlanningStatus.READY_PLAN.value,
            "error_code": PlanningErrorCode.NO_EVIDENCE.value if not proposals else "",
            "transitions": _append_transition(safe, "propose_read_only_plan"),
        }

    def validate_plan(state: GraphState) -> dict[str, object]:
        _check_live(context, cancelled, current_dataset_token)
        safe = _safe_state(state)
        _assert_context_coherence(safe, context)
        try:
            steps = _deserialize_steps(safe["steps"])
            _validate_steps(safe, steps)
        except (PlanningValidationError, ValueError, TypeError):
            return {
                "route": PlanningStatus.FAILED.value,
                "error_code": PlanningErrorCode.PLAN_REJECTED.value,
                "steps": [],
                "transitions": _append_transition(safe, "validate_plan"),
            }
        if safe["route"] not in {
            PlanningStatus.READY_PLAN.value,
            PlanningStatus.NEEDS_INPUT.value,
        }:
            return {
                "route": PlanningStatus.FAILED.value,
                "error_code": PlanningErrorCode.INVALID_STATE.value,
                "steps": [],
                "transitions": _append_transition(safe, "validate_plan"),
            }
        return {"transitions": _append_transition(safe, "validate_plan")}

    def terminal(state: GraphState) -> dict[str, object]:
        _check_live(context, cancelled, current_dataset_token)
        safe = _safe_state(state)
        _assert_context_coherence(safe, context)
        if safe["route"] not in _TERMINAL_STATUSES:
            return {
                "route": PlanningStatus.FAILED.value,
                "error_code": PlanningErrorCode.INVALID_STATE.value,
                "steps": [],
                "transitions": _append_transition(safe, "terminal"),
            }
        return {"transitions": _append_transition(safe, "terminal")}

    builder = StateGraph(GraphState)
    builder.add_node("validate_preflight", validate_preflight)
    builder.add_node("propose_read_only_plan", propose_read_only_plan)
    builder.add_node("validate_plan", validate_plan)
    builder.add_node("terminal", terminal)
    builder.add_edge(START, "validate_preflight")
    builder.add_conditional_edges(
        "validate_preflight",
        route_after_preflight,
        {"propose_read_only_plan": "propose_read_only_plan", "terminal": "terminal"},
    )
    builder.add_edge("propose_read_only_plan", "validate_plan")
    builder.add_edge("validate_plan", "terminal")
    builder.add_edge("terminal", END)
    app = builder.compile(checkpointer=checkpointer)
    config = {
        "callbacks": [],
        "configurable": {"thread_id": context.run_id},
        "recursion_limit": MAX_GRAPH_NODES + 1,
        "metadata": {"workflow": "neat_read_only_plan", "run_id": context.run_id},
    }
    try:
        # A user may have enabled LangSmith tracing globally for other tools.
        # This local scientific workflow explicitly suppresses that context.
        with tracing_context(enabled=False):
            raw = app.invoke(initial, config=config)
    except (PlanningCancelled, PlanningStale):
        raise
    except PlanningValidationError as exc:
        return PlanningOutcome(
            context.run_id,
            context.preflight_run_id,
            context.dataset_id,
            context.revision,
            PlanningStatus.FAILED,
            error_code=PlanningErrorCode.INVALID_STATE,
            message=str(exc)[:MAX_TEXT_LENGTH],
        )
    _check_live(context, cancelled, current_dataset_token)
    state = _safe_state(raw)
    try:
        status = PlanningStatus(state["route"])
        error = PlanningErrorCode(state["error_code"]) if state["error_code"] else None
    except ValueError:
        status = PlanningStatus.FAILED
        error = PlanningErrorCode.INVALID_STATE
    steps = _deserialize_steps(state["steps"])
    return PlanningOutcome(
        run_id=context.run_id,
        preflight_run_id=context.preflight_run_id,
        dataset_id=context.dataset_id,
        revision=context.revision,
        status=status,
        steps=steps,
        transitions=tuple(state["transitions"]),
        error_code=error,
        message=("Read-only plan proposal is available." if status is PlanningStatus.READY_PLAN
                 else "More user input or evidence is required." if status is PlanningStatus.NEEDS_INPUT
                 else "Preflight did not provide valid data and evidence." if status is PlanningStatus.BLOCKED_INVALID_DATA
                 else "Planning did not complete safely."),
    )


__all__ = [
    "PlanningCancelled",
    "PlanningDependencyUnavailable",
    "PlanningStale",
    "run_planning",
]

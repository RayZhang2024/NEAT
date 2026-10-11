"""Typed, bounded M4 model-selection and egress contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from .agent_planning import (
    PlannerSource,
    PlanningContext,
    PlanningObjective,
    PlanningWarningCode,
    PlanStepType,
)

SCHEMA_VERSION = "neat-agent-planning-v1"
MAX_CANONICAL_REQUEST_BYTES = 4096
# Conservative ceiling: the complete ASCII request body, including all prompt
# and schema text, is also capped at 4096 bytes. Actual provider tokenization
# can only be lower because one token cannot represent an empty byte sequence.
MAX_INPUT_TOKENS = 4096
MAX_STRUCTURED_RESPONSE_BYTES = 4096
MAX_OUTPUT_TOKENS = 384
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_PLANNING_DURATION_SECONDS = 15.0


class AgentModelMode(str, Enum):
    OFFLINE = "offline_deterministic"
    LOCAL = "local_model"
    OPENAI_CLOUD = "openai_cloud"


@dataclass(frozen=True)
class AgentModelSettings:
    """Persistable, non-secret Agent settings; never stores consent or keys."""

    mode: AgentModelMode = AgentModelMode.OFFLINE
    model_id: str = ""
    endpoint: str = ""
    allow_environment_credential: bool = False


@dataclass(frozen=True)
class AgentRunSnapshot:
    """Owner-thread snapshot for one run. It contains no credential material."""

    settings: AgentModelSettings
    objective: PlanningObjective
    cloud_egress_consent: bool = False
    deterministic_fallback: bool = False
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class CanonicalPlanningPayload:
    """Exact allowlisted model-visible data, with a stable canonical wire form."""

    schema_version: str
    objective: PlanningObjective
    dataset_mode: str
    capabilities: tuple[tuple[str, bool], ...]
    wavelength_axis_valid: bool
    roi_present: bool
    measured_spectrum_evidence_available: bool
    calibration_provenance_known: bool
    nonfinite_intensity_present: bool
    warning_codes: tuple[PlanningWarningCode, ...]
    available_plan_step_types: tuple[PlanStepType, ...]

    def to_wire_dict(self) -> dict[str, object]:
        """Create the only supported model request payload representation."""
        return {
            "schema_version": self.schema_version,
            "objective": self.objective.value,
            "dataset_mode": self.dataset_mode,
            "capabilities": {key: value for key, value in self.capabilities},
            "wavelength_axis_valid": self.wavelength_axis_valid,
            "roi_present": self.roi_present,
            "measured_spectrum_evidence_available": self.measured_spectrum_evidence_available,
            "calibration_provenance_known": self.calibration_provenance_known,
            "nonfinite_intensity_present": self.nonfinite_intensity_present,
            "warning_codes": [item.value for item in self.warning_codes],
            "available_plan_step_types": [item.value for item in self.available_plan_step_types],
        }

    def to_json(self) -> str:
        encoded = json.dumps(
            self.to_wire_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        if len(encoded.encode("ascii")) > MAX_CANONICAL_REQUEST_BYTES:
            raise ValueError("Canonical planning request exceeds its byte limit.")
        return encoded


def canonical_planning_schema() -> dict[str, object]:
    """Return the schema that both the request serializer and UI preview describe."""
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "objective": {
                "type": "string",
                "enum": [item.value for item in PlanningObjective],
            },
            "dataset_mode": {
                "type": "string",
                "enum": ["empty", "images", "profile"],
            },
            "capabilities": {
                "type": "object",
                "properties": {
                    "images_available": {"type": "boolean"},
                    "spectrum_available": {"type": "boolean"},
                    "mapping_applicable": {"type": "boolean"},
                },
                "required": [
                    "images_available",
                    "spectrum_available",
                    "mapping_applicable",
                ],
                "additionalProperties": False,
            },
            "wavelength_axis_valid": {"type": "boolean"},
            "roi_present": {"type": "boolean"},
            "measured_spectrum_evidence_available": {"type": "boolean"},
            "calibration_provenance_known": {"type": "boolean"},
            "nonfinite_intensity_present": {"type": "boolean"},
            "warning_codes": {
                "type": "array",
                "maxItems": len(PlanningWarningCode),
                "items": {
                    "type": "string",
                    "enum": [item.value for item in PlanningWarningCode],
                },
            },
            "available_plan_step_types": {
                "type": "array",
                "maxItems": 5,
                "items": {"type": "string", "enum": [item.value for item in PlanStepType]},
            },
        },
        "required": [
            "schema_version",
            "objective",
            "dataset_mode",
            "capabilities",
            "wavelength_axis_valid",
            "roi_present",
            "measured_spectrum_evidence_available",
            "calibration_provenance_known",
            "nonfinite_intensity_present",
            "warning_codes",
            "available_plan_step_types",
        ],
        "additionalProperties": False,
    }


def canonical_planning_payload(
    context: PlanningContext,
    available_step_types: tuple[PlanStepType, ...],
) -> CanonicalPlanningPayload:
    """Reduce trusted preflight facts to the fixed metadata-only wire schema."""
    if not isinstance(context.objective, PlanningObjective):
        raise TypeError("Planning objective is not allowlisted.")
    if len(available_step_types) > 5 or not all(
        isinstance(item, PlanStepType) for item in available_step_types
    ):
        raise ValueError("Available plan-step vocabulary is not bounded.")
    warnings: list[PlanningWarningCode] = []
    from .agent_preflight import PreflightReadiness
    from .observation import DatasetMode, WavelengthAxisStatus

    if context.readiness is PreflightReadiness.LIMITED:
        warnings.append(PlanningWarningCode.PREFLIGHT_LIMITED)
    if context.mode is DatasetMode.IMAGES and not context.roi_available:
        warnings.append(PlanningWarningCode.ROI_MISSING)
    if context.wavelength_axis_status is not WavelengthAxisStatus.VALID:
        warnings.append(PlanningWarningCode.WAVELENGTH_AXIS_INVALID)
    if not context.calibration_provenance_known:
        warnings.append(PlanningWarningCode.CALIBRATION_PROVENANCE_UNKNOWN)
    spectrum_evidence = "spectrum:full-source" in context.evidence_references
    if not spectrum_evidence:
        warnings.append(PlanningWarningCode.MEASURED_SPECTRUM_UNAVAILABLE)
    if context.spectrum_has_nonfinite:
        warnings.append(PlanningWarningCode.NONFINITE_INTENSITY)

    return CanonicalPlanningPayload(
        schema_version=SCHEMA_VERSION,
        objective=context.objective,
        dataset_mode=context.mode.value,
        capabilities=(
            ("images_available", bool(context.images_available)),
            ("spectrum_available", bool(context.spectrum_available)),
            ("mapping_applicable", bool(context.mapping_applicable)),
        ),
        wavelength_axis_valid=(
            context.wavelength_axis_status is WavelengthAxisStatus.VALID
        ),
        roi_present=bool(context.roi_available),
        measured_spectrum_evidence_available=spectrum_evidence,
        calibration_provenance_known=bool(context.calibration_provenance_known),
        nonfinite_intensity_present=bool(context.spectrum_has_nonfinite),
        warning_codes=tuple(warnings),
        available_plan_step_types=available_step_types,
    )


__all__ = [
    "MAX_CANONICAL_REQUEST_BYTES",
    "MAX_INPUT_TOKENS",
    "MAX_OUTPUT_TOKENS",
    "MAX_PLANNING_DURATION_SECONDS",
    "MAX_STRUCTURED_RESPONSE_BYTES",
    "REQUEST_TIMEOUT_SECONDS",
    "SCHEMA_VERSION",
    "AgentModelMode",
    "AgentModelSettings",
    "AgentRunSnapshot",
    "CanonicalPlanningPayload",
    "PlannerSource",
    "PlanningObjective",
    "PlanningWarningCode",
    "canonical_planning_payload",
    "canonical_planning_schema",
]

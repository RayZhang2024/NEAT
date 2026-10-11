"""Smoke-test an installed NEAT wheel with its opt-in LangGraph extra."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import NEAT
from NEAT.domain.agent_planning import PlanningContext, PlanningStatus
from NEAT.domain.agent_preflight import PreflightReadiness
from NEAT.domain.observation import DatasetMode, WavelengthAxisStatus
from NEAT.services.agent_planning import run_planning


checkout = Path(os.environ["NEAT_CHECKOUT_ROOT"]).resolve()
for entry in sys.path:
    if entry and Path(entry).resolve().is_relative_to(checkout):
        raise AssertionError(f"Checkout leaked onto sys.path: {entry}")
if Path(NEAT.__file__).resolve().is_relative_to(checkout):
    raise AssertionError("NEAT imported from checkout instead of the installed wheel")

context = PlanningContext(
    run_id="a" * 32,
    preflight_run_id="b" * 32,
    preflight_complete=True,
    dataset_id="c" * 32,
    revision=1,
    readiness=PreflightReadiness.READY,
    mode=DatasetMode.PROFILE,
    wavelength_axis_status=WavelengthAxisStatus.VALID,
    images_available=False,
    spectrum_available=True,
    mapping_applicable=False,
    roi_available=False,
    calibration_provenance_known=True,
    spectrum_has_nonfinite=False,
    evidence_references=("spectrum:full-source", "spectrum:plot"),
    observations=("Imported spectrum-only input.",),
    warnings=(),
)
outcome = run_planning(context)
if outcome.status is not PlanningStatus.READY_PLAN:
    raise AssertionError(f"Installed graph returned {outcome.status}")
if not outcome.transitions or outcome.transitions[-1] != "terminal":
    raise AssertionError(f"Installed graph did not reach its terminal node: {outcome.transitions}")
if any(name.startswith(("langchain_openai", "langchain_anthropic", "langchain_google")) for name in sys.modules):
    raise AssertionError("Offline planner imported a model provider")
print(f"Installed NEAT={NEAT.__file__}; graph transitions={outcome.transitions}")

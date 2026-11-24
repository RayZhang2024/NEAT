"""Local mapping preparation for the autonomous Agent workflow."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from ..domain.agent_fitting import (
    AgentFitOutcome,
    AgentFitPlan,
    AgentFitCellState,
    AgentMacroPixelSize,
)
from ..domain.fitting import HKL


@dataclass(frozen=True)
class MappingShapeParameterMean:
    """Mean fixed shape parameters from valid location fits for one HKL."""

    hkl: HKL
    t: float
    eta: float
    valid_locations: int


def build_mapping_calibration_plan(
    plan: AgentFitPlan,
    pixel_size: int,
    *,
    hkls: tuple[HKL, ...] | None = None,
) -> AgentFitPlan:
    """Clone a validated sweep plan for a seven-location fit at one exact size."""
    if isinstance(pixel_size, bool) or not isinstance(pixel_size, int) or pixel_size <= 0:
        raise ValueError("The mapping pixel size must be a positive integer.")
    selected_hkls = set(hkls) if hkls is not None else None
    edges = tuple(
        edge for edge in plan.edges
        if selected_hkls is None or edge.hkl in selected_hkls
    )
    if not edges:
        raise ValueError("Mapping calibration needs at least one selected edge.")
    return AgentFitPlan(
        run_id=plan.run_id,
        dataset_id=plan.dataset_id,
        revision=plan.revision,
        spatial_rois=plan.spatial_rois,
        phase_id=plan.phase_id,
        structure_type=plan.structure_type,
        lattice_params=plan.lattice_params,
        edges=edges,
        fitting_parameter_bounds=plan.fitting_parameter_bounds,
        macro_pixel_sizes=(AgentMacroPixelSize(0, pixel_size, pixel_size),),
    )


def mean_mapping_shape_parameters(
    outcome: AgentFitOutcome,
    *,
    size_index: int = 0,
    hkls: tuple[HKL, ...] | None = None,
) -> tuple[MappingShapeParameterMean, ...]:
    """Average finite t/eta values from successful location fits only."""
    requested = set(hkls) if hkls is not None else {edge.hkl for edge in outcome.edges}
    values: dict[HKL, list[tuple[float, float]]] = {hkl: [] for hkl in requested}
    for cell in outcome.cells:
        if (
            cell.hkl not in values
            or cell.size_index != size_index
            or cell.state is not AgentFitCellState.SUCCEEDED
            or cell.attempt is None
            or cell.attempt.result is None
        ):
            continue
        t = float(cell.attempt.result.t_fit)
        eta = float(cell.attempt.result.eta_fit)
        if isfinite(t) and isfinite(eta):
            values[cell.hkl].append((t, eta))

    ordered_hkls = [edge.hkl for edge in outcome.edges if edge.hkl in requested]
    result: list[MappingShapeParameterMean] = []
    for hkl in ordered_hkls:
        pairs = values[hkl]
        if not pairs:
            continue
        mean_t, mean_eta = np.mean(np.asarray(pairs, dtype=float), axis=0)
        result.append(
            MappingShapeParameterMean(
                hkl=hkl,
                t=float(mean_t),
                eta=float(mean_eta),
                valid_locations=len(pairs),
            )
        )
    return tuple(result)


__all__ = [
    "MappingShapeParameterMean",
    "build_mapping_calibration_plan",
    "mean_mapping_shape_parameters",
]

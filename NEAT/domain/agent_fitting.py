"""Bounded run and result contracts for autonomous seven-ROI edge fitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType

from .agent_vision import SpatialRoiCandidate
from .fitting import HKL, FittingParameterBounds, WavelengthRegion
from .individual_edge import IndividualEdgeFitAttempt
from .observation import RoiBounds

MAX_SPATIAL_ROIS = 7
MAX_SELECTED_EDGES = 5
MAX_AUTONOMOUS_FITS = MAX_SPATIAL_ROIS * MAX_SELECTED_EDGES
MAX_STORED_FIT_CURVE_POINTS = 1_000_000
SPATIAL_ROI_LABELS = ("Reference", "A", "B", "C", "D", "E", "F")


class AgentFitCellState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class AgentFitRunStatus(str, Enum):
    COMPLETE = "completed"
    COMPLETE_WITH_ISSUES = "completed_with_issues"
    BLOCKED = "blocked"
    STOPPED = "stopped"
    STALE = "stale"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentEdgeFitPlan:
    """One trusted HKL and common wavelength windows for every spatial ROI."""

    source_row: int
    hkl: HKL
    edge_wavelength: float
    region1: WavelengthRegion
    region2: WavelengthRegion
    region3: WavelengthRegion
    s: float = 0.001
    t: float = 0.01
    eta: float = 0.5

    def __post_init__(self) -> None:
        if type(self.source_row) is not int or self.source_row < 0:
            raise ValueError("The source edge row must be a non-negative integer.")
        values = tuple(self.hkl)
        if len(values) != 3 or any(type(value) is not int for value in values):
            raise ValueError("An edge plan needs a three-integer HKL.")
        object.__setattr__(self, "hkl", values)
        edge = float(self.edge_wavelength)
        if not isfinite(edge) or edge <= 0:
            raise ValueError("Theoretical edge wavelength must be positive and finite.")
        object.__setattr__(self, "edge_wavelength", edge)
        for name in ("s", "t", "eta"):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"Initial {name} must be finite.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class AgentFitPlan:
    """Immutable, UI-captured inputs for one bounded Agent fitting run."""

    run_id: str
    dataset_id: str
    revision: int
    spatial_rois: tuple[SpatialRoiCandidate, ...]
    phase_id: str
    structure_type: str
    lattice_params: Mapping[str, float]
    edges: tuple[AgentEdgeFitPlan, ...]
    fitting_parameter_bounds: FittingParameterBounds = field(
        default_factory=FittingParameterBounds
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "spatial_rois", tuple(self.spatial_rois))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "lattice_params", MappingProxyType(dict(self.lattice_params)))
        if not isinstance(self.fitting_parameter_bounds, FittingParameterBounds):
            raise TypeError("Fitting bounds must be FittingParameterBounds.")


@dataclass(frozen=True)
class AgentFitCellResult:
    """Fit state and actual numerical evidence for one ROI × HKL cell."""

    roi_index: int
    roi_label: str
    roi: RoiBounds
    hkl: HKL
    state: AgentFitCellState
    attempt: IndividualEdgeFitAttempt | None = None
    message: str = ""
    review_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentFitProgress:
    stage: str
    message: str
    total_fits: int
    attempted: int
    succeeded: int
    failed: int
    needs_review: int
    current_roi_index: int | None = None
    current_roi_label: str = ""
    current_hkl: HKL | None = None
    cell: AgentFitCellResult | None = None


@dataclass(frozen=True)
class AgentFitOutcome:
    run_id: str
    dataset_id: str
    revision: int
    status: AgentFitRunStatus
    phase_id: str
    edges: tuple[AgentEdgeFitPlan, ...]
    cells: tuple[AgentFitCellResult, ...]
    total_fits: int
    attempted: int
    succeeded: int
    failed: int
    needs_review: int
    skipped: int
    cancelled: int
    message: str = ""


__all__ = [
    "MAX_AUTONOMOUS_FITS",
    "MAX_SELECTED_EDGES",
    "MAX_SPATIAL_ROIS",
    "MAX_STORED_FIT_CURVE_POINTS",
    "SPATIAL_ROI_LABELS",
    "AgentEdgeFitPlan",
    "AgentFitCellResult",
    "AgentFitCellState",
    "AgentFitOutcome",
    "AgentFitPlan",
    "AgentFitProgress",
    "AgentFitRunStatus",
]

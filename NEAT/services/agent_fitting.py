"""Bounded local execution for M5A-2 seven-spatial-ROI edge fitting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fractions import Fraction
from math import isfinite
from typing import TypedDict

import numpy as np

from ..core import PHASE_DATA
from ..core.fitting import initial_value_within_bounds
from ..domain.agent_fitting import (
    MACRO_PIXEL_SCALE_FACTOR,
    MACRO_PIXEL_SIZE_LEVELS,
    MAX_SELECTED_EDGES,
    MAX_SPATIAL_ROIS,
    MAX_STORED_FIT_CURVE_POINTS,
    SPATIAL_ROI_LABELS,
    AgentEdgeFitPlan,
    AgentFitCellResult,
    AgentFitCellState,
    AgentFitOutcome,
    AgentFitPlan,
    AgentFitProgress,
    AgentFitRunStatus,
    AgentMacroPixelSize,
)
from ..domain.agent_vision import SpatialRoiCandidate, VisionOutcome
from ..domain.fitting import (
    FittingParameterBounds,
    WavelengthRegion,
    default_bragg_edge_windows,
)
from ..domain.individual_edge import IndividualEdgeFitConfig
from ..domain.observation import (
    DatasetMode,
    ObservationStatus,
    RoiBounds,
    SpectrumObservation,
)
from .agent_tracing import agent_langsmith_context
from .agent_vision import trusted_phase_edges
from .fitting_engine import FittingEngine
from .scientific_observation import (
    ObservationErrorBase,
    ScientificDatasetHandle,
    StaleDatasetError,
    get_roi_spectrum,
)

MIN_ROI_WIDTH = 3
MAX_ROI_WIDTH = 128
MIN_FIT_SAMPLES_REGION1 = 3
MIN_FIT_SAMPLES_REGION2 = 3
MIN_FIT_SAMPLES_REGION3 = 9


class FitGraphState(TypedDict):
    extraction_index: int
    fit_index: int
    completed: int
    attempted: int
    succeeded: int
    failed: int
    needs_review: int
    stopped: bool
    blocked: bool


class AgentFitBlocked(ValueError):
    """A deterministic input or resource check prevented local fitting."""


def build_macro_pixel_sizes(width: int, height: int) -> tuple[AgentMacroPixelSize, ...]:
    """Build three smaller, baseline, and three larger pixel sizes.

    The exact 5/4 ratio avoids floating-point boundary surprises. Smaller
    dimensions round down; larger dimensions round up.
    """
    if (
        isinstance(width, bool) or not isinstance(width, int) or width <= 0
        or isinstance(height, bool) or not isinstance(height, int) or height <= 0
    ):
        raise AgentFitBlocked("Baseline macro-pixel width and height must be positive integers.")
    sizes: list[AgentMacroPixelSize] = []
    for level in range(-MACRO_PIXEL_SIZE_LEVELS, MACRO_PIXEL_SIZE_LEVELS + 1):
        factor = Fraction(str(MACRO_PIXEL_SCALE_FACTOR)) ** level

        def dimension(
            value: int,
            *,
            factor: Fraction = factor,
            level: int = level,
        ) -> int:
            numerator = value * factor.numerator
            denominator = factor.denominator
            if level < 0:
                return numerator // denominator
            if level > 0:
                return -(-numerator // denominator)
            return value

        sizes.append(AgentMacroPixelSize(level, dimension(width), dimension(height)))
    if any(size.width <= 0 or size.height <= 0 for size in sizes):
        raise AgentFitBlocked("The 1.25 macro-pixel size series contains a non-positive size.")
    dimensions = [(size.width, size.height) for size in sizes]
    if len(set(dimensions)) != len(dimensions):
        raise AgentFitBlocked(
            "The 1.25 macro-pixel size series repeats a rounded size; choose a larger baseline."
        )
    return tuple(sizes)


def roi_for_macro_pixel_size(
    roi: RoiBounds, size: AgentMacroPixelSize
) -> RoiBounds:
    """Return a half-open box at the requested size, centered as closely as pixels allow."""
    # The source rectangle's geometric center is represented in doubled
    # coordinates. Integer half-open bounds can preserve it exactly when the
    # old and new dimensions share parity; otherwise the nearest box differs
    # by at most half a pixel.
    x_min = (roi.x_min + roi.x_max - size.width) // 2
    y_min = (roi.y_min + roi.y_max - size.height) // 2
    return RoiBounds(x_min, x_min + size.width, y_min, y_min + size.height)


def _roi_overlap(left: RoiBounds, right: RoiBounds) -> bool:
    return (
        min(left.x_max, right.x_max) > max(left.x_min, right.x_min)
        and min(left.y_max, right.y_max) > max(left.y_min, right.y_min)
    )


def validate_spatial_roi_candidates(
    candidates: tuple[SpatialRoiCandidate, ...],
    image_shape: tuple[int, int],
) -> tuple[RoiBounds, ...]:
    """Independently validate seven original-coordinate, non-overlapping ROIs."""
    if len(candidates) != MAX_SPATIAL_ROIS:
        raise AgentFitBlocked(
            f"Seven valid sample locations are required; NEAT received {len(candidates)}."
        )
    if tuple(item.label for item in candidates) != SPATIAL_ROI_LABELS:
        raise AgentFitBlocked("ROI labels must be Reference followed by A–F.")
    height, width = image_shape
    accepted: list[RoiBounds] = []
    for candidate in candidates:
        roi = candidate.roi
        if not candidate.accepted or roi is None:
            reason = candidate.validation_error or "The ROI is not validated."
            raise AgentFitBlocked(f"ROI {candidate.label}: {reason}")
        values = (roi.x_min, roi.x_max, roi.y_min, roi.y_max)
        if any(type(value) is not int for value in values):
            raise AgentFitBlocked(f"ROI {candidate.label} coordinates must be integers.")
        if not (0 <= roi.x_min < roi.x_max <= width and 0 <= roi.y_min < roi.y_max <= height):
            raise AgentFitBlocked(f"ROI {candidate.label} is outside the current image.")
        if not (
            MIN_ROI_WIDTH <= roi.width <= MAX_ROI_WIDTH
            and MIN_ROI_WIDTH <= roi.height <= MAX_ROI_WIDTH
        ):
            raise AgentFitBlocked(f"ROI {candidate.label} exceeds the 3–128 pixel bounds.")
        if candidate.confidence not in {"high", "user-selected"} or candidate.ambiguity_flags:
            raise AgentFitBlocked(f"ROI {candidate.label} still has unresolved visual ambiguity.")
        if any(_roi_overlap(roi, other) for other in accepted):
            raise AgentFitBlocked(f"ROI {candidate.label} overlaps another sample location.")
        accepted.append(roi)
    return tuple(accepted)


def build_agent_fit_plan(
    *,
    run_id: str,
    dataset_id: str,
    revision: int,
    image_shape: tuple[int, int],
    visual_outcome: VisionOutcome,
    shape_parameter_initials: Mapping[tuple[int, int, int], tuple[float, float, float]] | None = None,
    fitting_parameter_bounds: FittingParameterBounds | None = None,
) -> AgentFitPlan:
    """Build trusted common per-edge windows without reading or writing Qt state."""
    phase_id = visual_outcome.phase_id
    if not isinstance(phase_id, str) or phase_id not in PHASE_DATA or phase_id == "Unknown_Phase":
        raise AgentFitBlocked("A provisional built-in phase is required before fitting.")
    phase_value = PHASE_DATA[phase_id]
    if not isinstance(phase_value, Mapping):
        raise AgentFitBlocked("Trusted phase metadata is unavailable.")
    phase = phase_value
    structure_type = phase.get("structure")
    if structure_type not in {"fcc", "bcc"}:
        raise AgentFitBlocked(
            f"The individual-edge Region 3 fit does not support {structure_type!r} phase structure."
        )
    lattice_params = phase.get("lattice_params")
    if not isinstance(lattice_params, Mapping) or not isfinite(float(lattice_params.get("a", 0.0))) or float(lattice_params.get("a", 0.0)) <= 0:
        raise AgentFitBlocked("Trusted positive cubic lattice parameter is unavailable.")
    hkls = tuple(visual_outcome.hkls)
    if not 1 <= len(hkls) <= MAX_SELECTED_EDGES or len(set(hkls)) != len(hkls):
        raise AgentFitBlocked("Select one to five distinct trusted Bragg edges.")
    edge_positions = dict(trusted_phase_edges(phase_id))
    if any(hkl not in edge_positions for hkl in hkls):
        raise AgentFitBlocked("A selected HKL is not in the trusted phase catalogue.")

    rois = tuple(visual_outcome.spatial_roi_candidates)
    validate_spatial_roi_candidates(rois, image_shape)
    reference_roi = rois[0].roi
    if reference_roi is None:
        raise AgentFitBlocked("A validated Reference location is required for the baseline size.")
    macro_pixel_sizes = build_macro_pixel_sizes(reference_roi.width, reference_roi.height)
    all_phase_edges = sorted(
        trusted_phase_edges(phase_id), key=lambda item: (item[1], item[0])
    )
    positions_by_hkl = {hkl: position for hkl, position in all_phase_edges}
    ordered_positions = [position for _hkl, position in all_phase_edges]
    initials = shape_parameter_initials or {}
    bounds = fitting_parameter_bounds or FittingParameterBounds()
    specs: list[AgentEdgeFitPlan] = []
    for source_row, hkl in enumerate(hkls):
        edge = positions_by_hkl[hkl]
        all_index = next(
            index for index, (candidate_hkl, _position) in enumerate(all_phase_edges)
            if candidate_hkl == hkl
        )
        lower_midpoint = (
            (ordered_positions[all_index - 1] + edge) / 2.0
            if all_index > 0
            else None
        )
        upper_midpoint = (
            (edge + ordered_positions[all_index + 1]) / 2.0
            if all_index + 1 < len(ordered_positions)
            else None
        )
        windows = default_bragg_edge_windows(edge, lower_midpoint, upper_midpoint)
        if not windows["valid"]:
            raise AgentFitBlocked(
                f"NEAT's adjacent-edge rules cannot create valid fitting windows for {hkl}."
            )
        region2 = WavelengthRegion(
            float(windows["lower_min"]), float(windows["lower_max"])
        )
        region1 = WavelengthRegion(
            float(windows["upper_min"]), float(windows["upper_max"])
        )
        region3 = WavelengthRegion(
            min(region1.min_wavelength, region2.min_wavelength),
            max(region1.max_wavelength, region2.max_wavelength),
        )
        raw_s, raw_t, raw_eta = initials.get(hkl, (0.001, 0.01, 0.5))
        s = initial_value_within_bounds(raw_s, bounds.s)
        t = initial_value_within_bounds(raw_t, bounds.t)
        eta = initial_value_within_bounds(raw_eta, bounds.eta)
        specs.append(
            AgentEdgeFitPlan(
                source_row=source_row,
                hkl=hkl,
                edge_wavelength=edge,
                region1=region1,
                region2=region2,
                region3=region3,
                s=s,
                t=t,
                eta=eta,
            )
        )
    return AgentFitPlan(
        run_id=run_id,
        dataset_id=dataset_id,
        revision=revision,
        spatial_rois=rois,
        phase_id=phase_id,
        structure_type=str(structure_type),
        lattice_params=dict(lattice_params),
        edges=tuple(specs),
        fitting_parameter_bounds=bounds,
        macro_pixel_sizes=macro_pixel_sizes,
    )


def _new_cell(
    plan: AgentFitPlan,
    roi_index: int,
    size_index: int,
    edge: AgentEdgeFitPlan,
    state: AgentFitCellState,
    *,
    attempt=None,
    message: str = "",
    review_flags: tuple[str, ...] = (),
) -> AgentFitCellResult:
    roi_candidate = plan.spatial_rois[roi_index]
    assert roi_candidate.roi is not None
    size = plan.macro_pixel_sizes[size_index]
    return AgentFitCellResult(
        roi_index=roi_index,
        roi_label=roi_candidate.label,
        roi=roi_for_macro_pixel_size(roi_candidate.roi, size),
        hkl=edge.hkl,
        state=state,
        attempt=attempt,
        message=message,
        review_flags=review_flags,
        size_index=size_index,
        macro_pixel_width=size.width,
        macro_pixel_height=size.height,
    )


def _empty_cells(
    plan: AgentFitPlan,
    state: AgentFitCellState,
    message: str,
) -> tuple[AgentFitCellResult, ...]:
    return tuple(
        _new_cell(plan, roi_index, size_index, edge, state, message=message)
        for roi_index in range(len(plan.spatial_rois))
        for size_index in range(len(plan.macro_pixel_sizes))
        for edge in plan.edges
    )


def _make_outcome(
    plan: AgentFitPlan,
    status: AgentFitRunStatus,
    cells: tuple[AgentFitCellResult, ...],
    *,
    attempted: int,
    message: str,
) -> AgentFitOutcome:
    succeeded = sum(
        cell.state in {AgentFitCellState.SUCCEEDED, AgentFitCellState.NEEDS_REVIEW}
        for cell in cells
    )
    failed = sum(cell.state is AgentFitCellState.FAILED for cell in cells)
    needs_review = sum(cell.state is AgentFitCellState.NEEDS_REVIEW for cell in cells)
    skipped = sum(cell.state is AgentFitCellState.SKIPPED for cell in cells)
    cancelled = sum(cell.state is AgentFitCellState.CANCELLED for cell in cells)
    total = len(plan.spatial_rois) * len(plan.macro_pixel_sizes) * len(plan.edges)
    return AgentFitOutcome(
        plan.run_id,
        plan.dataset_id,
        plan.revision,
        status,
        plan.phase_id,
        plan.edges,
        cells,
        total,
        attempted,
        succeeded,
        failed,
        needs_review,
        skipped,
        cancelled,
        message,
        plan.macro_pixel_sizes,
    )


def _validate_spectrum_edge(
    spectrum: SpectrumObservation,
    roi: RoiBounds,
    edge: AgentEdgeFitPlan,
) -> tuple[int, str | None]:
    if (
        spectrum.status is not ObservationStatus.OK
        or spectrum.wavelengths is None
        or spectrum.intensities is None
        or spectrum.valid_pair_mask is None
        or spectrum.roi_effective != roi
    ):
        return 0, "NEAT could not extract a valid measured spectrum for this location and size."
    wavelengths = np.asarray(spectrum.wavelengths)
    intensities = np.asarray(spectrum.intensities)
    valid = np.asarray(spectrum.valid_pair_mask, dtype=bool)
    if wavelengths.shape != intensities.shape or valid.shape != wavelengths.shape:
        return 0, "Spectrum arrays are inconsistent for this location and size."
    if not np.isfinite(wavelengths[valid]).all() or not np.isfinite(intensities[valid]).all():
        return 0, "Spectrum contains invalid finite-pair data."
    required = (
        ("Region 1", "region1", MIN_FIT_SAMPLES_REGION1),
        ("Region 2", "region2", MIN_FIT_SAMPLES_REGION2),
        ("Region 3", "region3", MIN_FIT_SAMPLES_REGION3),
    )
    points = 0
    for label, region_name, minimum_count in required:
        region = getattr(edge, region_name)
        mask = (
            valid
            & (wavelengths >= region.min_wavelength)
            & (wavelengths <= region.max_wavelength)
        )
        count = int(np.count_nonzero(mask))
        if count < minimum_count:
            return 0, (
                f"{label} has {count} finite wavelength samples; at least "
                f"{minimum_count} are required."
            )
        points += count
    return points, None


def _validate_fit_plan(plan: AgentFitPlan) -> None:
    if plan.phase_id not in PHASE_DATA or plan.phase_id == "Unknown_Phase":
        raise AgentFitBlocked("The phase is not in NEAT's trusted catalogue.")
    phase_value = PHASE_DATA[plan.phase_id]
    if not isinstance(phase_value, Mapping):
        raise AgentFitBlocked("Trusted phase metadata is unavailable.")
    phase = phase_value
    if phase.get("structure") not in {"fcc", "bcc"} or plan.structure_type != phase.get("structure"):
        raise AgentFitBlocked("The selected phase structure is not supported by this fit path.")
    if dict(plan.lattice_params) != phase.get("lattice_params"):
        raise AgentFitBlocked("The fit lattice parameters do not match the trusted phase catalogue.")
    if not 1 <= len(plan.edges) <= MAX_SELECTED_EDGES:
        raise AgentFitBlocked("One to five selected edges are required.")
    if len(plan.macro_pixel_sizes) != MACRO_PIXEL_SIZE_LEVELS * 2 + 1:
        raise AgentFitBlocked("The run requires exactly seven macro-pixel sizes.")
    reference = plan.spatial_rois[0].roi
    if reference is None:
        raise AgentFitBlocked("The Reference location has no validated bounds.")
    expected_sizes = build_macro_pixel_sizes(reference.width, reference.height)
    if plan.macro_pixel_sizes != expected_sizes:
        raise AgentFitBlocked("Macro-pixel sizes differ from the validated 1.25 scale series.")
    if len({edge.hkl for edge in plan.edges}) != len(plan.edges):
        raise AgentFitBlocked("Selected edges must be distinct trusted HKLs.")
    if len({edge.source_row for edge in plan.edges}) != len(plan.edges):
        raise AgentFitBlocked("Each selected edge must have a distinct result row.")
    all_phase_edges = sorted(
        trusted_phase_edges(plan.phase_id), key=lambda item: (item[1], item[0])
    )
    positions_by_hkl = {hkl: position for hkl, position in all_phase_edges}
    for edge in plan.edges:
        if edge.hkl not in positions_by_hkl or not np.isclose(
            edge.edge_wavelength, positions_by_hkl[edge.hkl], rtol=0, atol=1e-12
        ):
            raise AgentFitBlocked(f"Edge {edge.hkl} is not a trusted phase marker.")
        index = next(i for i, (hkl, _position) in enumerate(all_phase_edges) if hkl == edge.hkl)
        position = positions_by_hkl[edge.hkl]
        lower_midpoint = (
            (all_phase_edges[index - 1][1] + position) / 2 if index > 0 else None
        )
        upper_midpoint = (
            (position + all_phase_edges[index + 1][1]) / 2
            if index + 1 < len(all_phase_edges)
            else None
        )
        expected = default_bragg_edge_windows(position, lower_midpoint, upper_midpoint)
        if not expected["valid"]:
            raise AgentFitBlocked(f"NEAT cannot create valid default windows for {edge.hkl}.")
        upper_min = float(expected["upper_min"])
        upper_max = float(expected["upper_max"])
        lower_min = float(expected["lower_min"])
        lower_max = float(expected["lower_max"])
        expected_regions = (
            WavelengthRegion(upper_min, upper_max),
            WavelengthRegion(lower_min, lower_max),
            WavelengthRegion(
                min(upper_min, lower_min),
                max(upper_max, lower_max),
            ),
        )
        for actual, expected_region in zip(
            (edge.region1, edge.region2, edge.region3), expected_regions
        ):
            if not np.allclose(
                (actual.min_wavelength, actual.max_wavelength),
                (expected_region.min_wavelength, expected_region.max_wavelength),
                rtol=0,
                atol=1e-12,
            ):
                raise AgentFitBlocked(f"Fit windows for {edge.hkl} differ from NEAT's defaults.")
        for name, value in (
            ("s", edge.s), ("t", edge.t), ("eta", edge.eta)
        ):
            lower, upper = plan.fitting_parameter_bounds[name]
            if not isfinite(value) or not lower <= value <= upper:
                raise AgentFitBlocked(f"Initial {name} for edge {edge.hkl} is outside its bounds.")


def _review_flags(result, bounds: FittingParameterBounds) -> tuple[str, ...]:
    flags = []
    for name, value, uncertainty in (
        ("d_fit", result.d_fit, result.d_unc),
        ("s", result.s_fit, result.s_unc),
        ("t", result.t_fit, result.t_unc),
        ("eta", result.eta_fit, result.eta_unc),
    ):
        if not isfinite(float(value)):
            flags.append(f"{name} is non-finite")
        if not isfinite(float(uncertainty)):
            flags.append(f"{name} uncertainty is unavailable or non-finite")
    if not isfinite(float(result.rms)):
        flags.append("RMS is non-finite")
    for name, value in (("s", result.s_fit), ("t", result.t_fit), ("eta", result.eta_fit)):
        lower, upper = bounds[name]
        if isfinite(float(value)) and min(abs(value - lower), abs(upper - value)) <= 0.01 * (upper - lower):
            flags.append(f"{name} is within 1% of a configured fit bound")
    return tuple(flags)


def run_agent_fitting(
    handle: ScientificDatasetHandle,
    plan: AgentFitPlan,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[AgentFitProgress], None] = lambda _event: None,
    langsmith_enabled: bool = False,
    langsmith_api_key: str | None = None,
    langsmith_agent_id: str = "neat-agentic-roi",
    langsmith_environment: str = "local",
    langsmith_region: str = "eu",
    langsmith_workspace_id: str = "",
) -> AgentFitOutcome:
    """Run the seven-location, seven-size exploratory edge-fit graph locally."""
    total = len(plan.spatial_rois) * len(plan.macro_pixel_sizes) * len(plan.edges)
    if total < 1:
        raise AgentFitBlocked("The fit plan does not contain any location, size, and edge combinations.")
    if (handle.dataset_id, handle.revision) != (plan.dataset_id, plan.revision):
        return _make_outcome(
            plan, AgentFitRunStatus.STALE, (), attempted=0,
            message="Dataset changed before fitting began.",
        )
    if handle.mode is not DatasetMode.IMAGES or handle.image_shape is None:
        raise AgentFitBlocked("Seven spatial image ROIs require an image-stack dataset.")
    validate_spatial_roi_candidates(plan.spatial_rois, handle.image_shape)
    if plan.structure_type not in {"fcc", "bcc"} or "a" not in plan.lattice_params:
        raise AgentFitBlocked("The local Region 3 fit requires a trusted fcc/bcc lattice parameter.")
    if not isinstance(plan.fitting_parameter_bounds, FittingParameterBounds):
        raise AgentFitBlocked("Configured fitting bounds are invalid.")
    _validate_fit_plan(plan)

    cells_by_key: dict[tuple[int, int, int], AgentFitCellResult] = {}
    spectra: dict[tuple[int, int], SpectrumObservation] = {}
    geometry: dict[tuple[int, int], RoiBounds] = {}
    geometry_errors: dict[tuple[int, int], str] = {}
    for size_index, size in enumerate(plan.macro_pixel_sizes):
        for roi_index, candidate in enumerate(plan.spatial_rois):
            assert candidate.roi is not None
            roi = roi_for_macro_pixel_size(candidate.roi, size)
            key = (roi_index, size_index)
            if not (
                0 <= roi.x_min < roi.x_max <= handle.image_shape[1]
                and 0 <= roi.y_min < roi.y_max <= handle.image_shape[0]
            ):
                geometry_errors[key] = (
                    f"The {size.label} box at location {candidate.label} extends outside the image."
                )
            else:
                geometry[key] = roi
    for size_index, size in enumerate(plan.macro_pixel_sizes):
        for left_index in range(len(plan.spatial_rois)):
            left_key = (left_index, size_index)
            left = geometry.get(left_key)
            if left is None:
                continue
            for right_index in range(left_index + 1, len(plan.spatial_rois)):
                right_key = (right_index, size_index)
                right = geometry.get(right_key)
                if right is not None and _roi_overlap(left, right):
                    left_label = plan.spatial_rois[left_index].label
                    right_label = plan.spatial_rois[right_index].label
                    reason = (
                        f"The {size.label} boxes at locations {left_label} and {right_label} overlap."
                    )
                    geometry_errors[left_key] = reason
                    geometry_errors[right_key] = reason
                    geometry.pop(left_key, None)
                    geometry.pop(right_key, None)
    for (roi_index, size_index), reason in geometry_errors.items():
        for edge_index, edge in enumerate(plan.edges):
            cells_by_key[(roi_index, size_index, edge_index)] = _new_cell(
                plan, roi_index, size_index, edge, AgentFitCellState.SKIPPED,
                message=reason,
            )

    extraction_jobs = tuple(
        key for key in geometry if key not in geometry_errors
    )
    fit_tasks: list[tuple[int, int, int]] = []
    fit_points = 0

    def add_skipped(roi_index: int, size_index: int, message: str) -> None:
        for edge_index, edge in enumerate(plan.edges):
            cells_by_key[(roi_index, size_index, edge_index)] = _new_cell(
                plan, roi_index, size_index, edge, AgentFitCellState.SKIPPED,
                message=message,
            )

    def add_cancelled(roi_index: int, size_index: int) -> None:
        for edge_index, edge in enumerate(plan.edges):
            key = (roi_index, size_index, edge_index)
            if key not in cells_by_key:
                cells_by_key[key] = _new_cell(
                    plan, roi_index, size_index, edge, AgentFitCellState.CANCELLED,
                    message="Stopped before this size-specific fit was attempted.",
                )

    def extract_next_spectrum(state: FitGraphState) -> dict[str, object]:
        nonlocal fit_points
        index = state["extraction_index"]
        if index >= len(extraction_jobs):
            return {"extraction_index": index, "completed": len(cells_by_key)}
        if cancelled():
            for roi_index, size_index, _edge_index in fit_tasks:
                add_cancelled(roi_index, size_index)
            for roi_index, size_index in extraction_jobs[index:]:
                add_cancelled(roi_index, size_index)
            return {
                "extraction_index": len(extraction_jobs),
                "completed": len(cells_by_key),
                "stopped": True,
            }
        roi_index, size_index = extraction_jobs[index]
        candidate = plan.spatial_rois[roi_index]
        size = plan.macro_pixel_sizes[size_index]
        roi = geometry[(roi_index, size_index)]
        handle.assert_current()
        try:
            spectrum = get_roi_spectrum(handle, roi, display_window=None)
        except StaleDatasetError:
            raise
        except (ObservationErrorBase, ValueError) as exc:
            add_skipped(
                roi_index,
                size_index,
                f"Spectrum extraction failed for {size.label} ({type(exc).__name__}).",
            )
        else:
            handle.assert_current()
            spectra[(roi_index, size_index)] = spectrum
            for edge_index, edge in enumerate(plan.edges):
                points, reason = _validate_spectrum_edge(spectrum, roi, edge)
                key = (roi_index, size_index, edge_index)
                if reason:
                    cells_by_key[key] = _new_cell(
                        plan, roi_index, size_index, edge, AgentFitCellState.SKIPPED,
                        message=f"{size.label}: {reason}",
                    )
                else:
                    fit_points += points
                    fit_tasks.append((roi_index, size_index, edge_index))
        handle.assert_current()
        progress(
            AgentFitProgress(
                "extract_spectra",
                f"Prepared {size.label} spectrum for {candidate.label} · "
                f"location {roi_index + 1}/7, size {size_index + 1}/7.",
                total,
                state["attempted"],
                state["succeeded"],
                state["failed"],
                state["needs_review"],
                current_roi_index=roi_index,
                current_roi_label=candidate.label,
                completed=len(cells_by_key),
                current_size_index=size_index,
                current_macro_pixel_width=size.width,
                current_macro_pixel_height=size.height,
            )
        )
        return {
            "extraction_index": index + 1,
            "completed": len(cells_by_key),
        }

    def route_after_extraction(state: FitGraphState) -> str:
        if state["stopped"]:
            return "finish"
        if state["extraction_index"] < len(extraction_jobs):
            return "extract_more"
        return "check_fit_budget"

    def check_fit_evidence_budget(state: FitGraphState) -> dict[str, object]:
        if fit_points > MAX_STORED_FIT_CURVE_POINTS:
            message = (
                "Selected fitting windows exceed NEAT's bounded in-memory fit-evidence limit; "
                "no local fits were started."
            )
            for roi_index, size_index, _edge_index in fit_tasks:
                for edge_index, edge in enumerate(plan.edges):
                    cells_by_key[(roi_index, size_index, edge_index)] = _new_cell(
                        plan, roi_index, size_index, edge, AgentFitCellState.SKIPPED,
                        message=message,
                    )
            fit_tasks.clear()
            progress(
                AgentFitProgress(
                    "validate_fit_evidence", message, total, 0, 0, 0, 0,
                    completed=len(cells_by_key),
                )
            )
            return {"blocked": True, "completed": len(cells_by_key)}
        if not fit_tasks:
            return {
                "blocked": True,
                "completed": len(cells_by_key),
            }
        return {"blocked": False, "completed": len(cells_by_key)}

    def route_after_fit_budget(state: FitGraphState) -> str:
        return "fit" if not state["blocked"] else "finish"

    def fit_next_cell(state: FitGraphState) -> dict[str, object]:
        index = state["fit_index"]
        attempted = state["attempted"]
        succeeded = state["succeeded"]
        failed = state["failed"]
        needs_review = state["needs_review"]
        if index >= len(fit_tasks):
            return {"fit_index": index, "completed": len(cells_by_key)}
        if cancelled():
            for roi_index, size_index, _edge_index in fit_tasks[index:]:
                add_cancelled(roi_index, size_index)
            return {
                "fit_index": len(fit_tasks),
                "completed": len(cells_by_key),
                "stopped": True,
            }
        roi_index, size_index, edge_index = fit_tasks[index]
        candidate = plan.spatial_rois[roi_index]
        size = plan.macro_pixel_sizes[size_index]
        spectrum = spectra[(roi_index, size_index)]
        edge = plan.edges[edge_index]
        running = _new_cell(
            plan, roi_index, size_index, edge, AgentFitCellState.RUNNING,
            message=f"Fit {attempted + 1} of {total} is running.",
        )
        cells_by_key[(roi_index, size_index, edge_index)] = running
        progress(
            AgentFitProgress(
                "fit_edges",
                f"Fitting {candidate.label} at {size.label}, edge {edge.hkl} · "
                f"{attempted + 1}/{total} attempts.",
                total,
                attempted,
                succeeded,
                failed,
                needs_review,
                current_roi_index=roi_index,
                current_roi_label=candidate.label,
                current_hkl=edge.hkl,
                cell=running,
                completed=len(cells_by_key),
                current_size_index=size_index,
                current_macro_pixel_width=size.width,
                current_macro_pixel_height=size.height,
            )
        )
        config = IndividualEdgeFitConfig(
            source_row=edge.source_row,
            hkl=edge.hkl,
            d=None,
            # The legacy typed contract is deliberately R2, R1, then R3.
            regions=(edge.region2, edge.region1, edge.region3),
            s=edge.s,
            t=edge.t,
            eta=edge.eta,
            is_known_phase=True,
            structure_type=plan.structure_type,
            lattice_params=plan.lattice_params,
            fitting_parameter_bounds=plan.fitting_parameter_bounds,
        )
        attempted += 1
        try:
            assert spectrum.wavelengths is not None
            assert spectrum.intensities is not None
            fit_attempt = FittingEngine().fit_individual_edge(
                spectrum.wavelengths,
                spectrum.intensities,
                config,
                fix_s=False,
                fix_t=False,
                fix_eta=False,
            )
        except Exception as exc:  # noqa: BLE001 - independent fits must fail per cell
            cell = _new_cell(
                plan, roi_index, size_index, edge, AgentFitCellState.FAILED,
                message=f"Fit failed ({type(exc).__name__}).",
            )
            failed += 1
        else:
            if fit_attempt.result is None:
                failure_detail = fit_attempt.error_message or "Fit did not return a numerical result."
                if fit_attempt.error_stage:
                    failure_detail = f"{fit_attempt.error_stage}: {failure_detail}"
                cell = _new_cell(
                    plan, roi_index, size_index, edge, AgentFitCellState.FAILED,
                    attempt=fit_attempt,
                    message=failure_detail[:400],
                )
                failed += 1
            else:
                flags = _review_flags(fit_attempt.result, plan.fitting_parameter_bounds)
                cell_state = AgentFitCellState.NEEDS_REVIEW if flags else AgentFitCellState.SUCCEEDED
                cell = _new_cell(
                    plan, roi_index, size_index, edge, cell_state,
                    attempt=fit_attempt,
                    message="Fit completed." if not flags else "Fit completed; inspect the flagged values.",
                    review_flags=flags,
                )
                succeeded += 1
                needs_review += bool(flags)
        cells_by_key[(roi_index, size_index, edge_index)] = cell
        handle.assert_current()
        progress(
            AgentFitProgress(
                "fit_edges",
                f"{candidate.label}, {size.label}, edge {edge.hkl}: "
                f"{cell.state.value.replace('_', ' ')} · {attempted}/{total} attempts.",
                total,
                attempted,
                succeeded,
                failed,
                needs_review,
                current_roi_index=roi_index,
                current_roi_label=candidate.label,
                current_hkl=edge.hkl,
                cell=cell,
                completed=len(cells_by_key),
                current_size_index=size_index,
                current_macro_pixel_width=size.width,
                current_macro_pixel_height=size.height,
            )
        )
        return {
            "fit_index": index + 1,
            "attempted": attempted,
            "succeeded": succeeded,
            "failed": failed,
            "needs_review": needs_review,
            "completed": len(cells_by_key),
        }

    def route_after_fit(state: FitGraphState) -> str:
        if state["stopped"] or state["fit_index"] >= len(fit_tasks):
            return "finish"
        return "fit_more"

    try:
        from langgraph.graph import END, START, StateGraph
    except (ImportError, OSError) as exc:
        raise AgentFitBlocked(
            "Local fitting requires NEAT's optional [agent] extra."
        ) from exc
    fit_graph = StateGraph(FitGraphState)
    fit_graph.add_node("extract_next_location_size_spectrum", extract_next_spectrum)
    fit_graph.add_node("check_fit_evidence_budget", check_fit_evidence_budget)
    fit_graph.add_node("fit_next_location_size_edge", fit_next_cell)
    fit_graph.add_edge(START, "extract_next_location_size_spectrum")
    fit_graph.add_conditional_edges(
        "extract_next_location_size_spectrum",
        route_after_extraction,
        {
            "extract_more": "extract_next_location_size_spectrum",
            "check_fit_budget": "check_fit_evidence_budget",
            "finish": END,
        },
    )
    fit_graph.add_conditional_edges(
        "check_fit_evidence_budget",
        route_after_fit_budget,
        {"fit": "fit_next_location_size_edge", "finish": END},
    )
    fit_graph.add_conditional_edges(
        "fit_next_location_size_edge",
        route_after_fit,
        {"fit_more": "fit_next_location_size_edge", "finish": END},
    )
    compiled_fit_graph = fit_graph.compile()
    with agent_langsmith_context(
        enabled=langsmith_enabled,
        api_key=langsmith_api_key,
        agent_id=langsmith_agent_id,
        environment=langsmith_environment,
        region=langsmith_region,
        workspace_id=langsmith_workspace_id,
        workflow="neat_macro_pixel_size_sweep",
    ):
        fit_state = compiled_fit_graph.invoke(
            {
                "extraction_index": 0,
                "fit_index": 0,
                "completed": len(cells_by_key),
                "attempted": 0,
                "succeeded": 0,
                "failed": 0,
                "needs_review": 0,
                "stopped": False,
                "blocked": False,
            },
            config={
                "recursion_limit": (
                    len(plan.spatial_rois) * len(plan.macro_pixel_sizes) + total + 10
                )
            },
        )
    cells = tuple(
        cells_by_key[(roi_index, size_index, edge_index)]
        for roi_index in range(len(plan.spatial_rois))
        for size_index in range(len(plan.macro_pixel_sizes))
        for edge_index in range(len(plan.edges))
        if (roi_index, size_index, edge_index) in cells_by_key
    )
    attempted = fit_state["attempted"]
    failed = fit_state["failed"]
    needs_review = fit_state["needs_review"]
    if fit_state["stopped"]:
        return _make_outcome(
            plan, AgentFitRunStatus.STOPPED, cells, attempted=attempted,
            message="Stopped safely between spectrum extractions or individual-edge fits.",
        )
    if fit_state["blocked"]:
        return _make_outcome(
            plan, AgentFitRunStatus.BLOCKED, cells, attempted=attempted,
            message=(
                "No location, size, and edge combination had enough valid spectrum data to fit."
                if not fit_tasks and fit_points <= MAX_STORED_FIT_CURVE_POINTS
                else "The bounded fit-evidence memory limit prevented local fitting."
            ),
        )

    status = (
        AgentFitRunStatus.COMPLETE_WITH_ISSUES
        if failed or needs_review or any(cell.state is AgentFitCellState.SKIPPED for cell in cells)
        else AgentFitRunStatus.COMPLETE
    )
    return _make_outcome(
        plan,
        status,
        tuple(cells),
        attempted=attempted,
        message=(
            "Every safe location, size, and edge fit was attempted. Review skipped or flagged results and scientific limitations."
            if status is AgentFitRunStatus.COMPLETE_WITH_ISSUES
            else "All selected location, size, and edge fits completed. Numerical convergence is not scientific validation."
        ),
    )


__all__ = [
    "AgentFitBlocked",
    "build_agent_fit_plan",
    "build_macro_pixel_sizes",
    "roi_for_macro_pixel_size",
    "run_agent_fitting",
    "validate_spatial_roi_candidates",
]

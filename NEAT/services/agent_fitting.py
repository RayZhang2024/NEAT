"""Bounded local execution for M5A-2 seven-spatial-ROI edge fitting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite

import numpy as np

from ..core import PHASE_DATA
from ..core.fitting import initial_value_within_bounds
from ..domain.agent_fitting import (
    MAX_AUTONOMOUS_FITS,
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
)
from ..domain.agent_vision import SpatialRoiCandidate, VisionOutcome
from ..domain.fitting import (
    FittingParameterBounds,
    WavelengthRegion,
    default_bragg_edge_windows,
)
from ..domain.individual_edge import IndividualEdgeFitConfig
from ..domain.observation import DatasetMode, ObservationStatus, RoiBounds
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


class AgentFitBlocked(ValueError):
    """A deterministic input or resource check prevented local fitting."""


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
    )


def _new_cell(
    plan: AgentFitPlan,
    roi_index: int,
    edge: AgentEdgeFitPlan,
    state: AgentFitCellState,
    *,
    attempt=None,
    message: str = "",
    review_flags: tuple[str, ...] = (),
) -> AgentFitCellResult:
    roi_candidate = plan.spatial_rois[roi_index]
    assert roi_candidate.roi is not None
    return AgentFitCellResult(
        roi_index,
        roi_candidate.label,
        roi_candidate.roi,
        edge.hkl,
        state,
        attempt,
        message,
        review_flags,
    )


def _empty_cells(
    plan: AgentFitPlan,
    state: AgentFitCellState,
    message: str,
) -> tuple[AgentFitCellResult, ...]:
    return tuple(
        _new_cell(plan, roi_index, edge, state, message=message)
        for roi_index in range(len(plan.spatial_rois))
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
    total = len(plan.spatial_rois) * len(plan.edges)
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
    )


def _validate_spectra(plan: AgentFitPlan, spectra) -> None:
    points = 0
    required = (
        ("Region 1", "region1", MIN_FIT_SAMPLES_REGION1),
        ("Region 2", "region2", MIN_FIT_SAMPLES_REGION2),
        ("Region 3", "region3", MIN_FIT_SAMPLES_REGION3),
    )
    for roi_candidate, spectrum in zip(plan.spatial_rois, spectra):
        if roi_candidate.roi is None:
            raise AgentFitBlocked(f"ROI {roi_candidate.label} has no validated bounds.")
        if (
            spectrum.status is not ObservationStatus.OK
            or spectrum.wavelengths is None
            or spectrum.intensities is None
            or spectrum.valid_pair_mask is None
            or spectrum.roi_effective != roi_candidate.roi
        ):
            raise AgentFitBlocked(f"NEAT could not extract a valid spectrum for ROI {roi_candidate.label}.")
        wavelengths = np.asarray(spectrum.wavelengths)
        intensities = np.asarray(spectrum.intensities)
        valid = np.asarray(spectrum.valid_pair_mask, dtype=bool)
        if wavelengths.shape != intensities.shape or valid.shape != wavelengths.shape:
            raise AgentFitBlocked(f"Spectrum arrays for ROI {roi_candidate.label} are inconsistent.")
        if not np.isfinite(wavelengths[valid]).all() or not np.isfinite(intensities[valid]).all():
            raise AgentFitBlocked(f"Spectrum for ROI {roi_candidate.label} contains invalid finite-pair data.")
        for edge in plan.edges:
            for label, region_name, minimum_count in required:
                region = getattr(edge, region_name)
                mask = (
                    valid
                    & (wavelengths >= region.min_wavelength)
                    & (wavelengths <= region.max_wavelength)
                )
                count = int(np.count_nonzero(mask))
                if count < minimum_count:
                    raise AgentFitBlocked(
                        f"ROI {roi_candidate.label}, edge {edge.hkl}: {label} has {count} "
                        f"finite samples; at least {minimum_count} are required."
                    )
                points += count
    if points > MAX_STORED_FIT_CURVE_POINTS:
        raise AgentFitBlocked(
            "Selected fitting windows exceed NEAT's bounded in-memory fit-evidence limit."
        )


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
    if not 1 <= len(plan.edges) <= MAX_SELECTED_EDGES or len(plan.edges) > MAX_AUTONOMOUS_FITS:
        raise AgentFitBlocked("One to five selected edges are required.")
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
) -> AgentFitOutcome:
    """Extract seven real ROI spectra and fit each trusted edge locally once."""
    total = len(plan.spatial_rois) * len(plan.edges)
    if total < 1 or total > MAX_AUTONOMOUS_FITS:
        raise AgentFitBlocked("The fit plan exceeds the 7 ROI × 5 edge limit.")
    if (handle.dataset_id, handle.revision) != (plan.dataset_id, plan.revision):
        return _make_outcome(
            plan, AgentFitRunStatus.STALE, (), attempted=0,
            message="Dataset changed before fitting began.",
        )
    if handle.mode is not DatasetMode.IMAGES or handle.image_shape is None:
        raise AgentFitBlocked("Seven spatial image ROIs require an image-stack dataset.")
    validated_rois = validate_spatial_roi_candidates(plan.spatial_rois, handle.image_shape)
    if plan.structure_type not in {"fcc", "bcc"} or "a" not in plan.lattice_params:
        raise AgentFitBlocked("The local Region 3 fit requires a trusted fcc/bcc lattice parameter.")
    if not isinstance(plan.fitting_parameter_bounds, FittingParameterBounds):
        raise AgentFitBlocked("Configured fitting bounds are invalid.")
    _validate_fit_plan(plan)

    spectra = []
    for roi_index, (candidate, roi) in enumerate(zip(plan.spatial_rois, validated_rois)):
        if cancelled():
            terminal_cells = _empty_cells(
                plan, AgentFitCellState.CANCELLED, "Stopped before spectrum extraction."
            )
            return _make_outcome(
                plan, AgentFitRunStatus.STOPPED, terminal_cells, attempted=0,
                message="Stopped before local fitting began.",
            )
        handle.assert_current()
        try:
            spectrum = get_roi_spectrum(handle, roi, display_window=None)
        except StaleDatasetError:
            raise
        except (ObservationErrorBase, ValueError) as exc:
            terminal_cells = _empty_cells(
                plan, AgentFitCellState.SKIPPED,
                f"Spectrum extraction blocked: {type(exc).__name__}.",
            )
            return _make_outcome(
                plan, AgentFitRunStatus.BLOCKED, terminal_cells, attempted=0,
                message=f"Could not extract the measured spectrum for ROI {candidate.label}.",
            )
        spectra.append(spectrum)
        handle.assert_current()
        progress(
            AgentFitProgress(
                "extract_spectra",
                f"Extracted spectrum for ROI {candidate.label} ({roi_index + 1}/7).",
                total,
                0,
                0,
                0,
                0,
                current_roi_index=roi_index,
                current_roi_label=candidate.label,
            )
        )

    try:
        _validate_spectra(plan, spectra)
    except AgentFitBlocked as exc:
        terminal_cells = _empty_cells(plan, AgentFitCellState.SKIPPED, str(exc))
        return _make_outcome(
            plan, AgentFitRunStatus.BLOCKED, terminal_cells, attempted=0,
            message=str(exc),
        )

    cells: list[AgentFitCellResult] = []
    attempted = succeeded = failed = needs_review = 0
    for roi_index, (candidate, spectrum) in enumerate(zip(plan.spatial_rois, spectra)):
        for edge_index, edge in enumerate(plan.edges):
            handle.assert_current()
            if cancelled():
                cells.extend(
                    _new_cell(plan, remaining_roi, remaining_edge, AgentFitCellState.CANCELLED,
                              message="Stopped before this fit was attempted.")
                    for remaining_roi in range(roi_index, len(plan.spatial_rois))
                    for remaining_edge in (
                        plan.edges[edge_index:] if remaining_roi == roi_index else plan.edges
                    )
                )
                return _make_outcome(
                    plan, AgentFitRunStatus.STOPPED, tuple(cells), attempted=attempted,
                    message="Stopped safely between individual-edge fits.",
                )
            running = _new_cell(
                plan, roi_index, edge, AgentFitCellState.RUNNING,
                message=f"Fit {attempted + 1} of {total} is running.",
            )
            progress(
                AgentFitProgress(
                    "fit_edges",
                    f"Fitting ROI {candidate.label}, edge {edge.hkl} · {attempted + 1}/{total} fits",
                    total,
                    attempted,
                    succeeded,
                    failed,
                    needs_review,
                    current_roi_index=roi_index,
                    current_roi_label=candidate.label,
                    current_hkl=edge.hkl,
                    cell=running,
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
                    plan, roi_index, edge, AgentFitCellState.FAILED,
                    message=f"Fit failed ({type(exc).__name__}).",
                )
                failed += 1
            else:
                if fit_attempt.result is None:
                    failure_detail = fit_attempt.error_message or "Fit did not return a numerical result."
                    if fit_attempt.error_stage:
                        failure_detail = f"{fit_attempt.error_stage}: {failure_detail}"
                    cell = _new_cell(
                        plan, roi_index, edge, AgentFitCellState.FAILED,
                        attempt=fit_attempt,
                        message=failure_detail[:400],
                    )
                    failed += 1
                else:
                    flags = _review_flags(fit_attempt.result, plan.fitting_parameter_bounds)
                    state = AgentFitCellState.NEEDS_REVIEW if flags else AgentFitCellState.SUCCEEDED
                    cell = _new_cell(
                        plan, roi_index, edge, state,
                        attempt=fit_attempt,
                        message="Fit completed." if not flags else "Fit completed; inspect the flagged values.",
                        review_flags=flags,
                    )
                    succeeded += 1
                    needs_review += bool(flags)
            cells.append(cell)
            handle.assert_current()
            progress(
                AgentFitProgress(
                    "fit_edges",
                    f"ROI {candidate.label}, edge {edge.hkl}: {cell.state.value.replace('_', ' ')} · "
                    f"{attempted}/{total} fits",
                    total,
                    attempted,
                    succeeded,
                    failed,
                    needs_review,
                    current_roi_index=roi_index,
                    current_roi_label=candidate.label,
                    current_hkl=edge.hkl,
                    cell=cell,
                )
            )

    status = (
        AgentFitRunStatus.COMPLETE_WITH_ISSUES
        if failed or needs_review
        else AgentFitRunStatus.COMPLETE
    )
    return _make_outcome(
        plan,
        status,
        tuple(cells),
        attempted=attempted,
        message=(
            "All selected edge fits were attempted. Review flagged results and scientific limitations."
            if failed or needs_review
            else "All selected edge fits completed. Numerical convergence is not scientific validation."
        ),
    )


__all__ = [
    "AgentFitBlocked",
    "build_agent_fit_plan",
    "run_agent_fitting",
    "validate_spatial_roi_candidates",
]

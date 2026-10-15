"""Local M5A-2 validation and seven-location fitting tests."""

from __future__ import annotations

from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from NEAT.domain.agent_fitting import AgentFitCellState, AgentFitRunStatus
from NEAT.domain.agent_vision import SpatialRoiCandidate, VisionOutcome, VisionStatus
from NEAT.domain.individual_edge import (
    IndividualEdgeFitAttempt,
    IndividualEdgeFitResult,
    RegionFitCurve,
)
from NEAT.domain.observation import DatasetMode, DatasetRevisionClock, RoiBounds
from NEAT.services.agent_fitting import (
    AgentFitBlocked,
    build_agent_fit_plan,
    run_agent_fitting,
    validate_spatial_roi_candidates,
)
from NEAT.services.agent_vision import trusted_phase_edges
from NEAT.services.scientific_observation import (
    StaleDatasetError,
    create_dataset_handle,
    get_roi_spectrum,
)

ROI_BOUNDS = (
    RoiBounds(0, 10, 0, 10),
    RoiBounds(15, 25, 0, 10),
    RoiBounds(30, 40, 0, 10),
    RoiBounds(45, 55, 0, 10),
    RoiBounds(0, 10, 15, 25),
    RoiBounds(15, 25, 15, 25),
    RoiBounds(30, 40, 15, 25),
)


def _make_run(edge_count: int = 3):
    wavelengths = np.linspace(0.5, 5.0, 800)
    images = []
    for frame_index in range(wavelengths.size):
        image = np.zeros((60, 60), dtype=np.float32)
        for roi_index, roi in enumerate(ROI_BOUNDS):
            image[roi.y_min : roi.y_max, roi.x_min : roi.x_max] = (
                roi_index + frame_index / 100.0
            )
        images.append(image)
    clock = DatasetRevisionClock()
    handle = create_dataset_handle(
        mode=DatasetMode.IMAGES,
        images=images,
        wavelengths=wavelengths,
        revision_clock=clock,
    )
    candidates = tuple(
        SpatialRoiCandidate(
            label,
            roi,
            "Visible sample interior.",
            "high",
            (),
            (roi.x_min, roi.x_max, roi.y_min, roi.y_max),
        )
        for label, roi in zip(("Reference", "A", "B", "C", "D", "E", "F"), ROI_BOUNDS)
    )
    edge_list = tuple(hkl for hkl, _wavelength in trusted_phase_edges("Fe_bcc")[:edge_count])
    initials = {edge_list[0]: (0.0015, 0.012, 0.45)} if edge_list else {}
    visual = VisionOutcome(
        "run",
        handle.dataset_id,
        handle.revision,
        VisionStatus.NEEDS_REVIEW,
        "Provisional visual proposal.",
        roi=ROI_BOUNDS[0],
        phase_id="Fe_bcc",
        hkls=edge_list,
        spatial_roi_candidates=candidates,
    )
    plan = build_agent_fit_plan(
        run_id="run",
        dataset_id=handle.dataset_id,
        revision=handle.revision,
        image_shape=handle.image_shape,
        visual_outcome=visual,
        shape_parameter_initials=initials,
    )
    return handle, plan, clock


def _successful_attempt(config):
    def curve(low, high):
        x = np.linspace(low, high, 12)
        y = np.linspace(0.1, 0.2, 12)
        return RegionFitCurve(x, y, y.copy())

    result = IndividualEdgeFitResult(
        source_row=config.source_row,
        hkl=config.hkl,
        region1=curve(*config.window(1)),
        region2=curve(*config.window(0)),
        region3=curve(*config.window(2)),
        d_fit=2.34,
        d_unc=0.0002,
        s_fit=0.005,
        s_unc=0.00001,
        t_fit=0.05,
        t_unc=0.0001,
        eta_fit=0.5,
        eta_unc=0.001,
        edge_height=0.1,
        edge_width=0.02,
        rms=0.003,
        baseline_params={"a0": 0.1},
    )
    return IndividualEdgeFitAttempt(
        region1=result.region1,
        region2=result.region2,
        region3=result.region3,
        result=result,
    )


class AgentFittingTests(TestCase):
    def test_plan_uses_seven_distinct_spatial_rois_and_trusted_windows(self):
        handle, plan, _clock = _make_run()
        validated = validate_spatial_roi_candidates(plan.spatial_rois, handle.image_shape)

        self.assertEqual(validated, ROI_BOUNDS)
        self.assertEqual(tuple(item.label for item in plan.spatial_rois), (
            "Reference", "A", "B", "C", "D", "E", "F"
        ))
        self.assertEqual(len(plan.edges), 3)
        edge = plan.edges[0]
        self.assertLess(edge.region2.max_wavelength, edge.edge_wavelength)
        self.assertGreater(edge.region1.min_wavelength, edge.edge_wavelength)
        self.assertEqual(edge.region3.min_wavelength, edge.region2.min_wavelength)
        self.assertEqual(edge.region3.max_wavelength, edge.region1.max_wavelength)
        self.assertEqual((edge.s, edge.t, edge.eta), (0.0015, 0.012, 0.45))

    def test_rejects_missing_ambiguous_overlapping_and_out_of_bounds_rois(self):
        _handle, plan, _clock = _make_run()
        variants = (
            plan.spatial_rois[:-1],
            plan.spatial_rois[:6] + (
                SpatialRoiCandidate("F", ROI_BOUNDS[-1], "uncertain", "medium"),
            ),
            plan.spatial_rois[:6] + (
                SpatialRoiCandidate("F", ROI_BOUNDS[-2], "overlap", "high"),
            ),
            plan.spatial_rois[:6] + (
                SpatialRoiCandidate("F", RoiBounds(55, 65, 15, 25), "outside", "high"),
            ),
        )
        for candidates in variants:
            with (
                self.subTest(candidates=candidates[-1]),
                self.assertRaises(AgentFitBlocked),
            ):
                validate_spatial_roi_candidates(candidates, (60, 60))

    def test_three_edges_fit_all_seven_rois_using_legacy_window_order_and_free_shape(self):
        handle, plan, _clock = _make_run(3)
        source_snapshot = tuple(frame.copy() for frame in handle.frames)
        calls = []
        progress = []

        def fit(wavelengths, intensities, config, **kwargs):
            calls.append((wavelengths.copy(), intensities.copy(), config, kwargs))
            return _successful_attempt(config)

        engine = type("FitSpy", (), {"fit_individual_edge": staticmethod(fit)})()
        with patch("NEAT.services.agent_fitting.FittingEngine", return_value=engine):
            outcome = run_agent_fitting(handle, plan, progress=progress.append)

        self.assertEqual(outcome.status, AgentFitRunStatus.COMPLETE)
        self.assertEqual((outcome.total_fits, outcome.attempted, outcome.succeeded), (21, 21, 21))
        self.assertEqual(len(calls), 21)
        self.assertEqual(len({(cell.roi_label, cell.hkl) for cell in outcome.cells}), 21)
        for _wavelengths, _intensities, config, kwargs in calls:
            edge = next(edge for edge in plan.edges if edge.hkl == config.hkl)
            self.assertEqual(config.window(0), (edge.region2.min_wavelength, edge.region2.max_wavelength))
            self.assertEqual(config.window(1), (edge.region1.min_wavelength, edge.region1.max_wavelength))
            self.assertEqual(config.window(2), (edge.region3.min_wavelength, edge.region3.max_wavelength))
            self.assertEqual(kwargs, {"fix_s": False, "fix_t": False, "fix_eta": False})
        expected_reference = get_roi_spectrum(handle, ROI_BOUNDS[0]).intensities
        np.testing.assert_array_equal(calls[0][1], expected_reference)
        np.testing.assert_allclose(expected_reference, np.arange(800) / 100.0, atol=2e-6)
        self.assertTrue(all(not frame.flags.writeable for frame in handle.frames))
        for before, after in zip(source_snapshot, handle.frames):
            np.testing.assert_array_equal(before, after)
        self.assertTrue(any(event.attempted == 21 for event in progress))

    def test_five_edges_are_bounded_to_thirty_five_attempts(self):
        handle, plan, _clock = _make_run(5)
        calls = []

        def fit(_wavelengths, _intensities, config, **_kwargs):
            calls.append(config.hkl)
            return _successful_attempt(config)

        engine = type("FitSpy", (), {"fit_individual_edge": staticmethod(fit)})()
        with patch("NEAT.services.agent_fitting.FittingEngine", return_value=engine):
            outcome = run_agent_fitting(handle, plan)

        self.assertEqual((outcome.total_fits, outcome.attempted, len(calls)), (35, 35, 35))
        self.assertEqual(outcome.status, AgentFitRunStatus.COMPLETE)

    def test_duplicate_edge_is_rejected_before_local_fit(self):
        handle, plan, _clock = _make_run(1)
        duplicate = replace(plan, edges=(plan.edges[0], replace(plan.edges[0], source_row=1)))
        fit_engine = type("FitSpy", (), {"fit_individual_edge": lambda *_args, **_kwargs: self.fail("must not fit")})()
        with (
            patch("NEAT.services.agent_fitting.FittingEngine", return_value=fit_engine),
            self.assertRaisesRegex(AgentFitBlocked, "distinct trusted HKLs"),
        ):
            run_agent_fitting(handle, duplicate)

    def test_per_cell_numerical_failure_does_not_stop_other_fits(self):
        handle, plan, _clock = _make_run(1)
        count = 0

        def fit(_wavelengths, _intensities, config, **_kwargs):
            nonlocal count
            count += 1
            if count == 2:
                return IndividualEdgeFitAttempt(error_stage="Region 3 fit", error_message="synthetic failure")
            return _successful_attempt(config)

        engine = type("FitSpy", (), {"fit_individual_edge": staticmethod(fit)})()
        with patch("NEAT.services.agent_fitting.FittingEngine", return_value=engine):
            outcome = run_agent_fitting(handle, plan)

        self.assertEqual(outcome.status, AgentFitRunStatus.COMPLETE_WITH_ISSUES)
        self.assertEqual((outcome.attempted, outcome.succeeded, outcome.failed), (7, 6, 1))
        self.assertEqual(outcome.cells[1].state, AgentFitCellState.FAILED)

    def test_stop_is_checked_between_fits_and_marks_unattempted_cells(self):
        handle, plan, _clock = _make_run(3)
        calls = []

        def fit(_wavelengths, _intensities, config, **_kwargs):
            calls.append(config.hkl)
            return _successful_attempt(config)

        engine = type("FitSpy", (), {"fit_individual_edge": staticmethod(fit)})()
        with patch("NEAT.services.agent_fitting.FittingEngine", return_value=engine):
            outcome = run_agent_fitting(
                handle,
                plan,
                cancelled=lambda: len(calls) >= 2,
            )

        self.assertEqual(outcome.status, AgentFitRunStatus.STOPPED)
        self.assertEqual(outcome.attempted, 2)
        self.assertEqual(outcome.cancelled, 19)
        self.assertEqual(len(calls), 2)

    def test_dataset_invalidation_during_a_fit_aborts_stale_evidence(self):
        handle, plan, clock = _make_run(1)
        count = 0

        def fit(_wavelengths, _intensities, config, **_kwargs):
            nonlocal count
            count += 1
            clock.invalidate()
            return _successful_attempt(config)

        engine = type("FitSpy", (), {"fit_individual_edge": staticmethod(fit)})()
        with (
            patch("NEAT.services.agent_fitting.FittingEngine", return_value=engine),
            self.assertRaisesRegex(StaleDatasetError, "dataset changed"),
        ):
            run_agent_fitting(handle, plan)

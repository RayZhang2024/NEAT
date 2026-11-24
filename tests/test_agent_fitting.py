"""Local M5A-2 validation and seven-location fitting tests."""

from __future__ import annotations

from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from NEAT.domain.agent_fitting import (
    AgentEdgeFitPlan,
    AgentFitCellResult,
    AgentFitCellState,
    AgentFitOutcome,
    AgentFitRunStatus,
    AgentMacroPixelSize,
)
from NEAT.domain.agent_vision import SpatialRoiCandidate, VisionOutcome, VisionStatus
from NEAT.domain.fitting import WavelengthRegion
from NEAT.domain.individual_edge import (
    IndividualEdgeFitAttempt,
    IndividualEdgeFitResult,
    RegionFitCurve,
)
from NEAT.domain.observation import DatasetMode, DatasetRevisionClock, RoiBounds
from NEAT.services.agent_fitting import (
    AgentFitBlocked,
    analyze_macro_pixel_selection,
    build_agent_fit_plan,
    build_macro_pixel_sizes,
    roi_for_macro_pixel_size,
    run_agent_fitting,
    validate_spatial_roi_candidates,
)
from NEAT.services.agent_mapping import (
    build_mapping_calibration_plan,
    mean_mapping_shape_parameters,
)
from NEAT.services.agent_vision import trusted_phase_edges
from NEAT.services.scientific_observation import (
    StaleDatasetError,
    create_dataset_handle,
    get_roi_spectrum,
)
from NEAT.ui.agent_fitting_worker import AgentFitWorker

ROI_BOUNDS = (
    RoiBounds(15, 35, 15, 35),
    RoiBounds(40, 60, 15, 35),
    RoiBounds(65, 85, 15, 35),
    RoiBounds(15, 35, 40, 60),
    RoiBounds(40, 60, 40, 60),
    RoiBounds(65, 85, 40, 60),
    RoiBounds(40, 60, 65, 85),
)


def _make_run(edge_count: int = 3):
    wavelengths = np.linspace(0.5, 5.0, 800)
    images = []
    for frame_index in range(wavelengths.size):
        image = np.zeros((100, 100), dtype=np.float32)
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


def _macro_pixel_outcome(
    uncertainty_curves,
    *,
    status=AgentFitRunStatus.COMPLETE,
    failed_cells=frozenset(),
    vary_by_location=False,
):
    sizes = tuple(
        AgentMacroPixelSize(level, width, width)
        for level, width in enumerate((10, 12, 16, 20, 25, 32, 40), start=-3)
    )
    edges = tuple(
        AgentEdgeFitPlan(
            source_row=index,
            hkl=(index + 1, 1, 0),
            edge_wavelength=2.0 + index,
            region1=WavelengthRegion(2.5 + index, 3.0 + index),
            region2=WavelengthRegion(1.0 + index, 1.5 + index),
            region3=WavelengthRegion(1.0 + index, 3.0 + index),
        )
        for index in range(len(uncertainty_curves))
    )
    x = np.linspace(0.1, 0.2, 3)
    curve = RegionFitCurve(x, x.copy(), x.copy())
    cells = []
    for roi_index, roi_label in enumerate(("Reference", "A", "B", "C", "D", "E", "F")):
        for size_index, size in enumerate(sizes):
            for edge_index, edge in enumerate(edges):
                key = (roi_index, size_index, edge_index)
                if key in failed_cells:
                    cells.append(
                        AgentFitCellResult(
                            roi_index, roi_label, ROI_BOUNDS[roi_index], edge.hkl,
                            AgentFitCellState.FAILED, size_index=size_index,
                            macro_pixel_width=size.width, macro_pixel_height=size.height,
                        )
                    )
                    continue
                relative_percent = uncertainty_curves[edge_index][size_index]
                if vary_by_location:
                    relative_percent *= 1 + (roi_index - 3) * 0.01
                d_fit = 2.34
                result = IndividualEdgeFitResult(
                    source_row=edge.source_row,
                    hkl=edge.hkl,
                    region1=curve,
                    region2=curve,
                    region3=curve,
                    d_fit=d_fit,
                    d_unc=d_fit * relative_percent / 100,
                    s_fit=0.001,
                    s_unc=0.0001,
                    t_fit=0.01,
                    t_unc=0.001,
                    eta_fit=0.5,
                    eta_unc=0.01,
                    edge_height=0.1,
                    edge_width=0.02,
                    rms=0.003,
                    baseline_params={},
                )
                attempt = IndividualEdgeFitAttempt(
                    region1=curve, region2=curve, region3=curve, result=result
                )
                cells.append(
                    AgentFitCellResult(
                        roi_index, roi_label, ROI_BOUNDS[roi_index], edge.hkl,
                        AgentFitCellState.SUCCEEDED, attempt=attempt,
                        size_index=size_index, macro_pixel_width=size.width,
                        macro_pixel_height=size.height,
                    )
                )
    failed_count = len(failed_cells)
    return AgentFitOutcome(
        run_id="macro-pixel-test",
        dataset_id="macro-pixel-test",
        revision=1,
        status=status,
        phase_id="Fe_bcc",
        edges=edges,
        cells=tuple(cells),
        total_fits=len(cells),
        attempted=len(cells),
        succeeded=len(cells) - failed_count,
        failed=failed_count,
        needs_review=0,
        skipped=0,
        cancelled=0,
        macro_pixel_sizes=sizes,
    )


class AgentFittingTests(TestCase):
    def test_mapping_calibration_uses_one_exact_size_with_the_same_locations(self):
        handle, plan, _clock = _make_run(2)
        selected_hkl = plan.edges[1].hkl

        calibration = build_mapping_calibration_plan(
            plan, 19, hkls=(selected_hkl,)
        )

        self.assertEqual(calibration.spatial_rois, plan.spatial_rois)
        self.assertEqual(tuple(edge.hkl for edge in calibration.edges), (selected_hkl,))
        self.assertEqual(
            tuple((size.width, size.height) for size in calibration.macro_pixel_sizes),
            ((19, 19),),
        )
        self.assertEqual((calibration.dataset_id, calibration.revision), (handle.dataset_id, handle.revision))

    def test_mapping_calibration_fits_all_seven_locations_at_the_exact_size(self):
        handle, plan, _clock = _make_run(1)
        calibration = build_mapping_calibration_plan(plan, 19)
        calls = []

        def fit(_wavelengths, _intensities, config, **_kwargs):
            calls.append(config.hkl)
            return _successful_attempt(config)

        engine = type("FitSpy", (), {"fit_individual_edge": staticmethod(fit)})()
        with patch("NEAT.services.agent_fitting.FittingEngine", return_value=engine):
            outcome = run_agent_fitting(
                handle, calibration, mapping_calibration=True
            )

        self.assertEqual(outcome.total_fits, 7)
        self.assertEqual(outcome.succeeded, 7)
        self.assertEqual(len(calls), 7)
        self.assertEqual({cell.macro_pixel_width for cell in outcome.cells}, {19})

    def test_mapping_parameter_means_use_only_successful_finite_fits(self):
        outcome = _macro_pixel_outcome(((0.01,) * 7,))
        cells = []
        successful_values = []
        for cell in outcome.cells:
            if cell.size_index != 4:
                cells.append(replace(cell, state=AgentFitCellState.FAILED, attempt=None))
                continue
            if cell.roi_index == 0:
                cells.append(replace(cell, state=AgentFitCellState.NEEDS_REVIEW))
                continue
            if cell.roi_index == 6:
                cells.append(replace(cell, state=AgentFitCellState.FAILED, attempt=None))
                continue
            t_value = 0.01 * (cell.roi_index + 1)
            eta_value = 0.1 * (cell.roi_index + 1)
            result = replace(
                cell.attempt.result,
                t_fit=t_value,
                eta_fit=eta_value,
            )
            cells.append(replace(cell, attempt=replace(cell.attempt, result=result)))
            successful_values.append((t_value, eta_value))
        outcome = replace(outcome, cells=tuple(cells))

        means = mean_mapping_shape_parameters(outcome, size_index=4)

        self.assertEqual(len(means), 1)
        self.assertEqual(means[0].valid_locations, 5)
        self.assertAlmostEqual(means[0].t, np.mean([value[0] for value in successful_values]))
        self.assertAlmostEqual(means[0].eta, np.mean([value[1] for value in successful_values]))

    def test_macro_pixel_size_series_uses_requested_rounding_and_rectangles(self):
        expected_square = ((6, 6), (8, 8), (10, 10), (13, 13), (17, 17), (21, 21), (26, 26))
        sizes = build_macro_pixel_sizes(13, 13)
        self.assertEqual(tuple((item.width, item.height) for item in sizes), expected_square)
        self.assertEqual(tuple(item.scale_level for item in sizes), (-3, -2, -1, 0, 1, 2, 3))

        rectangular = build_macro_pixel_sizes(13, 8)
        self.assertEqual(
            tuple((item.width, item.height) for item in rectangular),
            ((6, 4), (8, 5), (10, 6), (13, 8), (17, 10), (21, 13), (26, 16)),
        )

    def test_macro_pixel_size_series_rejects_nonpositive_or_rounded_duplicates(self):
        for dimensions in ((0, 13), (13, -1), (3, 3)):
            with self.subTest(dimensions=dimensions), self.assertRaises(AgentFitBlocked):
                build_macro_pixel_sizes(*dimensions)

    def test_macro_pixel_uncertainty_selects_smallest_rounded_target_crossing(self):
        outcome = _macro_pixel_outcome(
            (
                (0.03, 0.015, 0.012, 0.009, 0.007, 0.005, 0.004),
                (0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06),
            ),
            vary_by_location=True,
        )

        selection = analyze_macro_pixel_selection(outcome)

        first_edge, flagged_edge = selection.edges
        self.assertTrue(first_edge.complete_location_coverage)
        self.assertEqual(first_edge.points[0].valid_locations, 7)
        self.assertGreater(first_edge.points[0].standard_deviation_percent, 0)
        self.assertAlmostEqual(first_edge.target_crossing_width, 11.1699, places=3)
        self.assertEqual(first_edge.rounded_target_width, 12)
        self.assertTrue(flagged_edge.excluded)
        self.assertEqual(flagged_edge.above_flag_threshold, 49)
        self.assertEqual(selection.selected_width, 12)
        self.assertEqual(selection.selected_hkls, (first_edge.hkl,))

    def test_macro_pixel_uncertainty_extrapolates_on_log_axis(self):
        outcome = _macro_pixel_outcome(
            ((0.05, 0.045, 0.04, 0.035, 0.03, 0.027, 0.024),)
        )

        edge = analyze_macro_pixel_selection(outcome).edges[0]

        self.assertEqual(edge.crossing_kind, "extrapolated")
        self.assertGreater(edge.target_crossing_width, 40)
        self.assertEqual(edge.rounded_target_width, int(np.ceil(edge.target_crossing_width)))

    def test_macro_pixel_uncertainty_does_not_recommend_incomplete_sweep(self):
        outcome = _macro_pixel_outcome(
            ((0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004),),
            status=AgentFitRunStatus.COMPLETE_WITH_ISSUES,
            failed_cells={(0, 0, 0)},
        )

        selection = analyze_macro_pixel_selection(outcome)

        self.assertTrue(selection.run_complete)
        self.assertFalse(selection.edges[0].complete_location_coverage)
        self.assertIsNone(selection.edges[0].target_crossing_width)
        self.assertIsNone(selection.selected_width)

    def test_scaled_boxes_keep_each_location_center_with_at_most_half_pixel_rounding(self):
        _plan_handle, plan, _clock = _make_run(1)
        for candidate in plan.spatial_rois:
            original = candidate.roi
            self.assertIsNotNone(original)
            for size in plan.macro_pixel_sizes:
                scaled = roi_for_macro_pixel_size(original, size)
                self.assertEqual((scaled.width, scaled.height), (size.width, size.height))
                self.assertLessEqual(abs((scaled.x_min + scaled.x_max - original.x_min - original.x_max) / 2), 0.5)
                self.assertLessEqual(abs((scaled.y_min + scaled.y_max - original.y_min - original.y_max) / 2), 0.5)
            self.assertEqual(roi_for_macro_pixel_size(original, plan.macro_pixel_sizes[3]), original)

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

    def test_visual_cautions_and_overlap_are_allowed_but_invalid_geometry_is_rejected(self):
        _handle, plan, _clock = _make_run()
        uncertain = plan.spatial_rois[:6] + (
            SpatialRoiCandidate(
                "F", ROI_BOUNDS[-1], "Near the sample boundary.", "low",
                ("Boundary proximity is uncertain.",),
            ),
        )
        overlapping = plan.spatial_rois[:6] + (
            SpatialRoiCandidate("F", ROI_BOUNDS[-2], "overlap", "high"),
        )
        self.assertEqual(
            validate_spatial_roi_candidates(uncertain, _handle.image_shape),
            ROI_BOUNDS,
        )
        self.assertEqual(
            validate_spatial_roi_candidates(overlapping, _handle.image_shape),
            ROI_BOUNDS[:-1] + (ROI_BOUNDS[-2],),
        )

        invalid = (
            plan.spatial_rois[:-1],
            plan.spatial_rois[:6] + (
                SpatialRoiCandidate("F", RoiBounds(96, 106, 56, 66), "outside", "high"),
            ),
        )
        for candidates in invalid:
            with (
                self.subTest(candidates=candidates[-1]),
                self.assertRaises(AgentFitBlocked),
            ):
                validate_spatial_roi_candidates(candidates, _handle.image_shape)

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
        self.assertEqual((outcome.total_fits, outcome.attempted, outcome.succeeded), (147, 147, 147))
        self.assertEqual(len(calls), 147)
        self.assertEqual(
            len({(cell.roi_label, cell.size_index, cell.hkl) for cell in outcome.cells}),
            147,
        )
        for _wavelengths, _intensities, config, kwargs in calls:
            edge = next(edge for edge in plan.edges if edge.hkl == config.hkl)
            self.assertEqual(config.window(0), (edge.region2.min_wavelength, edge.region2.max_wavelength))
            self.assertEqual(config.window(1), (edge.region1.min_wavelength, edge.region1.max_wavelength))
            self.assertEqual(config.window(2), (edge.region3.min_wavelength, edge.region3.max_wavelength))
            self.assertEqual(kwargs, {"fix_s": False, "fix_t": False, "fix_eta": False})
        expected_reference = get_roi_spectrum(handle, ROI_BOUNDS[0]).intensities
        call_index = 0
        baseline_reference_call = None
        for size_index, size in enumerate(plan.macro_pixel_sizes):
            for candidate in plan.spatial_rois:
                scaled_roi = roi_for_macro_pixel_size(candidate.roi, size)
                expected_spectrum = get_roi_spectrum(handle, scaled_roi).intensities
                for _edge in plan.edges:
                    np.testing.assert_allclose(calls[call_index][1], expected_spectrum, atol=1e-6)
                    if size_index == 3 and candidate.label == "Reference":
                        baseline_reference_call = calls[call_index][1]
                    call_index += 1
        self.assertEqual(call_index, len(calls))
        np.testing.assert_allclose(baseline_reference_call, expected_reference, atol=1e-6)
        np.testing.assert_allclose(expected_reference, np.arange(800) / 100.0, atol=2e-6)
        self.assertTrue(all(not frame.flags.writeable for frame in handle.frames))
        for before, after in zip(source_snapshot, handle.frames):
            np.testing.assert_array_equal(before, after)
        self.assertTrue(any(event.attempted == 147 for event in progress))

    def test_unexpected_worker_error_reports_detail_and_logs_traceback(self):
        handle, plan, _clock = _make_run(1)
        worker = AgentFitWorker(handle, plan)
        outcomes = []
        worker.completed.connect(outcomes.append)

        with (
            patch(
                "NEAT.ui.agent_fitting_worker.run_agent_fitting",
                side_effect=TypeError("unexpected graph state payload"),
            ),
            self.assertLogs("NEAT.ui.agent_fitting_worker", level="ERROR") as logs,
        ):
            worker.run()

        self.assertEqual(len(outcomes), 1)
        self.assertIn("TypeError: unexpected graph state payload", outcomes[0].message)
        self.assertIn("launch console", outcomes[0].message)
        self.assertIn("Traceback", "\n".join(logs.output))

    def test_five_edges_run_every_safe_location_size_combination(self):
        handle, plan, _clock = _make_run(5)
        calls = []

        def fit(_wavelengths, _intensities, config, **_kwargs):
            calls.append(config.hkl)
            return _successful_attempt(config)

        engine = type("FitSpy", (), {"fit_individual_edge": staticmethod(fit)})()
        with patch("NEAT.services.agent_fitting.FittingEngine", return_value=engine):
            outcome = run_agent_fitting(handle, plan)

        self.assertEqual((outcome.total_fits, outcome.attempted, len(calls)), (245, 245, 245))
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
        self.assertEqual((outcome.attempted, outcome.succeeded, outcome.failed), (49, 48, 1))
        failed_cells = [cell for cell in outcome.cells if cell.state is AgentFitCellState.FAILED]
        self.assertEqual(len(failed_cells), 1)
        self.assertIn("synthetic failure", failed_cells[0].message)

    def test_out_of_bounds_scaled_boxes_are_skipped_without_stopping_safe_combinations(self):
        handle, plan, _clock = _make_run(1)
        shifted_reference = replace(
            plan.spatial_rois[0],
            roi=RoiBounds(1, 9, 6, 14),
            proposed_coordinates=(1, 9, 6, 14),
        )
        plan = replace(plan, spatial_rois=(shifted_reference,) + plan.spatial_rois[1:])
        calls = []

        def fit(_wavelengths, _intensities, config, **_kwargs):
            calls.append(config.hkl)
            return _successful_attempt(config)

        engine = type("FitSpy", (), {"fit_individual_edge": staticmethod(fit)})()
        with patch("NEAT.services.agent_fitting.FittingEngine", return_value=engine):
            outcome = run_agent_fitting(handle, plan)

        self.assertEqual(outcome.total_fits, 49)
        self.assertEqual((outcome.attempted, outcome.skipped, len(calls)), (43, 6, 43))
        self.assertEqual(outcome.status, AgentFitRunStatus.COMPLETE_WITH_ISSUES)
        skipped = [cell for cell in outcome.cells if cell.state is AgentFitCellState.SKIPPED]
        self.assertEqual({cell.size_index for cell in skipped}, {1, 2, 3, 4, 5, 6})
        self.assertTrue(all(cell.roi_label == "Reference" for cell in skipped))
        self.assertTrue(all("extends outside the image" in cell.message for cell in skipped))

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
        self.assertEqual(outcome.cancelled, 145)
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

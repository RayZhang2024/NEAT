"""Local agent-window preflight and Qt worker lifetime regressions."""

from __future__ import annotations

import gc
import os
import threading
import time
import unittest
import weakref
from dataclasses import replace
from importlib.util import find_spec
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Preserve the application import order: ONNX Runtime is initialized before Qt.
# isort: off
import NEAT  # noqa: F401
# isort: on

import numpy as np
from PyQt5.QtCore import QEventLoop, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QWidget,
)

from NEAT.domain.agent_model import (
    AgentModelMode,
    AgentModelSettings,
    PlanningObjective,
)
from NEAT.domain.agent_planning import PlannerSource, PlanningOutcome, PlanningStatus
from NEAT.domain.agent_preflight import PreflightReadiness, representative_frame_indices
from NEAT.domain.agent_vision import SpatialRoiCandidate, VisionOutcome, VisionStatus
from NEAT.domain.individual_edge import (
    IndividualEdgeFitAttempt,
    IndividualEdgeFitResult,
    RegionFitCurve,
)
from NEAT.domain.observation import (
    DatasetMode,
    ObservationStatus,
    RoiBounds,
    WavelengthAxisStatus,
)
from NEAT.ui.agent_model_settings_dialog import (
    AgentModelConnectionWorker,
    AgentModelSettingsDialog,
)
from NEAT.ui.agent_preflight_worker import PreflightWorker
from NEAT.ui.agent_window import AgentWindow
from NEAT.ui.main_window import FitsViewer
from NEAT.ui.scientific_observation_adapter import ScientificObservationAdapter


class _Host(FitsViewer):
    """Production observation and shutdown helpers without application startup."""

    def __init__(self):
        QMainWindow.__init__(self)
        self._initialize_shutdown_lifecycle()
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.agent_window = None
        self.agent_preflight_worker = None
        self.assistant_dock = _Assistant()
        self.preproc_message_box = QTextEdit(self)
        self.message_box = QTextEdit(self)
        self.image_slider = QSlider(self)
        self.images = [np.arange(36, dtype=float).reshape(6, 6) + i for i in range(4)]
        self.wavelengths = np.array([1.0, 2.0, 3.0, 4.0])
        self.fitting_data_source = "images"
        self.selected_area = (1, 4, 2, 5)
        self.manual_wavelength_mode = True
        self.current_fitting_input = "C:/private/sample.fits"
        self.scientific_observation_api = ScientificObservationAdapter(self)
        self._startup_update_timer = QTimer(self)
        self.display_image = Mock()
        self.save_user_settings = Mock()
        self._hide_load_progress_dialog = Mock()
        self._set_fits_image_button_states = Mock()
        self.show()


class _Assistant:
    worker = None

    @staticmethod
    def shutdown():
        return True


class _FakeSpatialVisionWorker(QThread):
    """Eventless offline visual fixture used to verify the real Qt handoff."""

    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)

    def __init__(self, run_id, handle, preflight, settings, *, parent=None, **_kwargs):
        super().__init__(parent)
        self.run_id = run_id
        self._stopped = False
        bounds = (
            RoiBounds(8, 13, 8, 13),
            RoiBounds(18, 23, 8, 13),
            RoiBounds(28, 33, 8, 13),
            RoiBounds(38, 43, 8, 13),
            RoiBounds(8, 13, 20, 25),
            RoiBounds(18, 23, 20, 25),
            RoiBounds(28, 33, 20, 25),
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
            for label, roi in zip(("Reference", "A", "B", "C", "D", "E", "F"), bounds)
        )
        self.outcome = VisionOutcome(
            run_id,
            handle.dataset_id,
            handle.revision,
            VisionStatus.NEEDS_REVIEW,
            "Offline test visual proposal.",
            roi=bounds[0],
            phase_id="Fe_bcc",
            hkls=((2, 1, 1),),
            edge_reasons=("Fixture edge rationale.",),
            spatial_roi_candidates=candidates,
        )

    def stop(self):
        self._stopped = True

    def run(self):
        self.progress.emit(80, "Offline visual fixture complete")
        self.completed.emit(self.outcome)


def _successful_fit(config):
    def curve(region):
        x = np.linspace(region.min_wavelength, region.max_wavelength, 20)
        y = np.linspace(0.1, 0.2, 20)
        return RegionFitCurve(x, y, y.copy())

    region1 = curve(config.regions[1])
    region2 = curve(config.regions[0])
    region3 = curve(config.regions[2])
    result = IndividualEdgeFitResult(
        config.source_row,
        config.hkl,
        region1,
        region2,
        region3,
        2.34,
        0.0002,
        0.005,
        0.0001,
        0.05,
        0.001,
        0.5,
        0.01,
        0.1,
        0.02,
        0.003,
        {"a0": 0.1},
    )
    return IndividualEdgeFitAttempt(region1, region2, region3, result)


class AgentPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.host = _Host()
        self.window = AgentWindow(self.host)
        # Keep these tests independent of a developer's saved Agent provider.
        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.LOCAL,
            model_id="injected-test-model",
            endpoint="http://127.0.0.1:11434/v1",
        )
        self.window._update_model_readiness()
        self.host.agent_window = self.window
        self.window.show()

    def tearDown(self):
        for worker in (
            self.window._worker,
            self.window._planning_worker,
            self.window._vision_worker,
            self.window._fit_worker,
        ):
            if worker is not None:
                worker.stop()
                worker.wait(5000)
        self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)
        self.host.close()
        self.app.processEvents(QEventLoop.AllEvents, 10)

    def _wait_for(self, predicate, timeout=6):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents(QEventLoop.AllEvents, 10)
            if predicate():
                return
            QTest.qWait(5)
        self.fail("Qt state did not settle before timeout")

    def _prepare_spatial_fit_host(self):
        wavelengths = np.linspace(0.5, 5.0, 400)
        images = []
        for index in range(wavelengths.size):
            image = np.zeros((64, 64), dtype=np.float32)
            image[5:55, 5:55] = 10.0 + index / 100.0
            images.append(image)
        self.host.images = images
        self.host.wavelengths = wavelengths
        self.host.selected_area = None
        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.OPENAI_CLOUD,
            model_id="injected-test-model",
        )
        self.window._update_model_readiness()
        return images, wavelengths

    def test_one_click_preflight_visual_validation_and_seven_roi_fit(self):
        from NEAT.domain.agent_fitting import AgentFitRunStatus

        images, wavelengths = self._prepare_spatial_fit_host()
        fitting_table = QTableWidget(1, 11, self.host)
        self.host.bragg_table = fitting_table
        row_values = (
            "(2, 1, 1)", "1.234", "2.100", "2.200", "2.400", "2.500",
            "2.100", "2.500", "0.0017", "0.0140", "0.4700",
        )
        for column, value in enumerate(row_values):
            fitting_table.setItem(0, column, QTableWidgetItem(value))
        fitting_table_snapshot = tuple(
            fitting_table.item(0, column).text()
            for column in range(fitting_table.columnCount())
        )
        fit_engine = type(
            "FitSpy",
            (),
            {
                "fit_individual_edge": staticmethod(
                    lambda _wave, _signal, config, **_kwargs: _successful_fit(config)
                )
            },
        )()
        with (
            patch("NEAT.ui.agent_window.VisionWorker", _FakeSpatialVisionWorker),
            patch("NEAT.services.agent_fitting.FittingEngine", return_value=fit_engine),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(
                lambda: self.window._fit_outcome is not None
                and self.window._fit_worker is None
                and not self.host._shutdown_worker_inventory.workers,
                timeout=20,
            )

        self.assertEqual(self.window._fit_outcome.status, AgentFitRunStatus.COMPLETE)
        self.assertEqual((self.window._fit_outcome.total_fits, self.window._fit_outcome.attempted), (7, 7))
        self.assertEqual(self.window.fit_matrix.rowCount(), 7)
        self.assertEqual(self.window.fit_matrix.columnCount(), 1)
        self.assertEqual(len(self.window.spatial_roi_preview._candidates), 7)
        self.assertTrue(self.window.summary_page.isVisible())
        self.assertEqual(self.window.summary_page.heading.text(), "Analysis complete")
        self.assertIn("Fe_bcc", self.window.summary_page.rows[1].conclusion_label.text())
        self.assertIn("7 sample locations", self.window.summary_page.rows[2].conclusion_label.text())
        self.assertEqual(self.window.summary_page.rows[3].conclusion_label.text(), "7 fits completed")
        self.assertFalse(self.window.summary_page.review_button.isVisible())
        self.assertFalse(self.window.fit_matrix.isVisible())
        self.window.summary_page.results_button.click()
        self.assertTrue(self.window.fit_matrix.isVisible())
        self.assertIn("7/7 attempted", self.window.fit_matrix_summary.text())
        self.assertIn("d_fit", self.window.fit_detail_label.text())
        self.assertIs(
            self.window.fit_curve_widgets[0]._curve,
            self.window._fit_outcome.cells[0].attempt.region1,
        )
        self.window.fit_matrix.setCurrentCell(3, 0)
        self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertIn("C · edge", self.window.fit_detail_label.text())
        self.assertTrue(self.window.start_autonomous_button.isEnabled())
        self.assertFalse(self.window.stop_button.isVisible())
        self.assertIs(self.host.images, images)
        self.assertIs(self.host.wavelengths, wavelengths)
        self.assertIsNone(self.host.selected_area)
        self.assertEqual(
            tuple(
                fitting_table.item(0, column).text()
                for column in range(fitting_table.columnCount())
            ),
            fitting_table_snapshot,
        )
        self.assertEqual(self.window.splitter.count(), 2)
        self.assertEqual(self.window.evidence_splitter.orientation(), Qt.Vertical)
        self.window.close()
        self.window.show()
        self.assertTrue(self.window.summary_page.isVisible())
        self.assertEqual(self.window.summary_page.heading.text(), "Explore your dataset")
        self.assertFalse(self.window.summary_page.results_button.isVisible())
        self.assertIsNone(self.window._fit_outcome)

    def _complete_spatial_fit_for_invalidation(self):
        self._prepare_spatial_fit_host()
        fit_engine = Mock()
        fit_engine.fit_individual_edge.side_effect = (
            lambda _wave, _signal, config, **_kwargs: _successful_fit(config)
        )
        with (
            patch("NEAT.ui.agent_window.VisionWorker", _FakeSpatialVisionWorker),
            patch("NEAT.services.agent_fitting.FittingEngine", return_value=fit_engine),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(
                lambda: self.window._fit_outcome is not None
                and not self.host._shutdown_worker_inventory.workers,
                timeout=20,
            )
        self.window.fit_matrix.setCurrentCell(3, 0)
        self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertIn("d_fit", self.window.fit_detail_label.text())
        self.assertTrue(all(widget._curve is not None for widget in self.window.fit_curve_widgets))

    def _assert_fit_evidence_withdrawn(self):
        self.assertIsNone(self.window._fit_plan)
        self.assertIsNone(self.window._fit_outcome)
        self.assertIsNone(self.window._pending_fit_plan)
        self.assertFalse(self.window._fit_cells)
        self.assertEqual(self.window.fit_matrix.currentRow(), -1)
        self.assertFalse(self.window.fit_matrix.selectedItems())
        for row in range(self.window.fit_matrix.rowCount()):
            self.assertEqual(self.window.fit_matrix.item(row, 0).text(), "—")
        self.assertNotIn("d_fit", self.window.fit_detail_label.text())
        self.assertTrue(all(widget._curve is None for widget in self.window.fit_curve_widgets))
        self.assertFalse(self.window.spatial_roi_preview._candidates)
        self.assertIsNone(self.window.spatial_roi_preview._selected_index)
        self.assertTrue(self.window.spatial_roi_preview._pixmap.isNull())
        # Neither a stale matrix cell nor an old ROI selection can resurrect results.
        self.window._select_fit_cell(3, 0)
        self.assertNotIn("d_fit", self.window.fit_detail_label.text())

    def test_adjusting_completed_fit_roi_withdraws_all_old_fit_evidence(self):
        self._complete_spatial_fit_for_invalidation()
        for edit, value in zip(self.window.roi_edits, (10, 20, 10, 20)):
            edit.setValue(value)
        self.window.preview_adjusted_roi()
        self._assert_fit_evidence_withdrawn()
        self.assertIsNone(self.window._vision_outcome)
        self.assertEqual(self.window._adjusted_roi, RoiBounds(10, 20, 10, 20))
        self.assertIn("Adjusted ROI saved", self.window.status_label.text())
        self.assertEqual(self.window.summary_page.heading.text(), "ROI adjustment saved")
        self.assertEqual(self.window.summary_page.rows[3].conclusion_label.text(), "Earlier fits withdrawn")

    def test_rejecting_completed_fit_proposal_withdraws_all_old_fit_evidence(self):
        self._complete_spatial_fit_for_invalidation()
        self.window.reject_visual_proposal()
        self._assert_fit_evidence_withdrawn()
        self.assertIsNone(self.window._vision_outcome)
        self.assertIn("Visual proposal rejected", self.window.status_label.text())
        self.assertEqual(self.window.summary_page.heading.text(), "Proposal withdrawn")

    def test_dataset_revision_withdraws_completed_fit_numeric_summary(self):
        self._complete_spatial_fit_for_invalidation()
        self.host.scientific_observation_api.revision_clock.invalidate()
        self.window._check_revision()
        self._assert_fit_evidence_withdrawn()
        self.assertTrue(self.window._stale)
        self.assertEqual(self.window.summary_page.heading.text(), "Dataset changed")

    def test_invalid_adjustment_preserves_completed_fit_evidence(self):
        self._complete_spatial_fit_for_invalidation()
        outcome = self.window._fit_outcome
        self.window.roi_edits[0].setValue(20)
        self.window.roi_edits[1].setValue(10)
        self.window.preview_adjusted_roi()
        self.assertIs(self.window._fit_outcome, outcome)
        self.assertIn("d_fit", self.window.fit_detail_label.text())
        self.assertIn("outside the current image or empty", self.window.status_label.text())

    def test_stop_during_fit_waits_for_worker_retirement(self):
        from NEAT.domain.agent_fitting import AgentFitRunStatus

        self._prepare_spatial_fit_host()
        fit_entered = threading.Event()
        release_fit = threading.Event()

        def paused_fit(_wave, _signal, config, **_kwargs):
            fit_entered.set()
            release_fit.wait(5)
            return _successful_fit(config)

        fit_engine = type(
            "PausedFitSpy",
            (),
            {"fit_individual_edge": staticmethod(paused_fit)},
        )()
        with (
            patch("NEAT.ui.agent_window.VisionWorker", _FakeSpatialVisionWorker),
            patch("NEAT.services.agent_fitting.FittingEngine", return_value=fit_engine),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(fit_entered.is_set, timeout=20)
            worker = self.window._fit_worker
            self.assertIsNotNone(worker)
            self.assertTrue(self.window.stop_button.isVisible())
            self.assertTrue(self.window.stop_button.isEnabled())
            self.window.stop_button.click()
            self.assertFalse(self.window.stop_button.isEnabled())
            self.assertIs(self.window._fit_worker, worker)
            self.assertFalse(self.window.start_autonomous_button.isEnabled())
            release_fit.set()
            self._wait_for(
                lambda: self.window._fit_worker is None
                and not self.host._shutdown_worker_inventory.workers,
                timeout=20,
            )

        self.assertEqual(self.window._fit_outcome.status, AgentFitRunStatus.STOPPED)
        self.assertEqual(self.window._fit_outcome.attempted, 1)
        self.assertEqual(self.window._fit_outcome.cancelled, 6)
        self.assertEqual(self.window._stage, "cancelled")
        self.assertIn("stopped", self.window.status_label.text().lower())
        self.assertIsNone(worker._handle)
        self.assertIsNone(worker._plan)

    def test_partial_fit_failure_is_visible_and_successful_cell_remains_selectable(self):
        from NEAT.domain.agent_fitting import AgentFitRunStatus

        self._prepare_spatial_fit_host()
        calls = 0

        def partially_failing_fit(_wave, _signal, config, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return IndividualEdgeFitAttempt(
                    error_stage="Region 3 fit",
                    error_message="synthetic fit failure",
                )
            return _successful_fit(config)

        fit_engine = type(
            "PartialFitSpy",
            (),
            {"fit_individual_edge": staticmethod(partially_failing_fit)},
        )()
        with (
            patch("NEAT.ui.agent_window.VisionWorker", _FakeSpatialVisionWorker),
            patch("NEAT.services.agent_fitting.FittingEngine", return_value=fit_engine),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(
                lambda: self.window._fit_outcome is not None
                and self.window._fit_worker is None
                and not self.host._shutdown_worker_inventory.workers,
                timeout=20,
            )

        outcome = self.window._fit_outcome
        self.assertEqual(outcome.status, AgentFitRunStatus.COMPLETE_WITH_ISSUES)
        self.assertEqual((outcome.attempted, outcome.succeeded, outcome.failed), (7, 6, 1))
        self.assertEqual(self.window.fit_matrix.item(0, 0).text(), "Failed")
        self.assertIn("1 failed", self.window.fit_matrix_summary.text())
        self.assertIn("1 failed", self.window.summary_page.rows[3].conclusion_label.text())
        self.window.summary_page.review_button.click()
        self.assertTrue(self.window.review_page.isVisible())
        self.assertIn("Reference", self.window.review_identity.text())
        self.assertTrue(self.window.review_reason.text())
        self.assertFalse(self.window.fit_matrix.isVisible())
        self.window.inspect_flagged_button.click()
        self.assertTrue(self.window.fit_matrix.isVisible())
        self.assertIn("Region 3 fit", self.window.fit_detail_label.text())
        self.window.fit_matrix.setCurrentCell(1, 0)
        self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertIn("d_fit", self.window.fit_detail_label.text())
        self.assertIs(
            self.window.fit_curve_widgets[0]._curve,
            outcome.cells[1].attempt.region1,
        )

    def test_ambiguous_spatial_proposal_blocks_all_fits(self):
        self._prepare_spatial_fit_host()

        class AmbiguousSpatialWorker(_FakeSpatialVisionWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                candidates = list(self.outcome.spatial_roi_candidates)
                candidates[3] = replace(candidates[3], confidence="medium")
                self.outcome = replace(
                    self.outcome, spatial_roi_candidates=tuple(candidates)
                )

        fit_engine = Mock()
        with (
            patch("NEAT.ui.agent_window.VisionWorker", AmbiguousSpatialWorker),
            patch("NEAT.services.agent_fitting.FittingEngine", return_value=fit_engine),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(
                lambda: self.window._stage == "fitting_needs_input"
                and not self.host._shutdown_worker_inventory.workers,
                timeout=20,
            )

        fit_engine.fit_individual_edge.assert_not_called()
        self.assertIsNone(self.window._fit_outcome)
        self.assertIn("ROI C", self.window.status_label.text())
        self.assertIn("no local fit was started", self.window.fit_matrix_summary.text())

    def test_visual_response_error_and_uncertainty_have_distinct_pause_messages(self):
        self._prepare_spatial_fit_host()
        for diagnostic in ("Visual phase explanation is empty after whitespace normalization.", ""):
            with self.subTest(diagnostic=diagnostic):
                class UnacceptedPhaseWorker(_FakeSpatialVisionWorker):
                    def __init__(self, *args, _diagnostic=diagnostic, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.outcome = replace(
                            self.outcome,
                            phase_id="", hkls=(), edge_reasons=(),
                            message="The visual model marked the overlay comparison as uncertain.",
                            response_validation_error=_diagnostic,
                            proposed_phase_id="Fe_bcc" if _diagnostic else "",
                        )

                fit_engine = Mock()
                with (
                    patch("NEAT.ui.agent_window.VisionWorker", UnacceptedPhaseWorker),
                    patch("NEAT.services.agent_fitting.FittingEngine", return_value=fit_engine),
                ):
                    self.window.start_autonomous_analysis()
                    self._wait_for(
                        lambda: self.window._stage == "fitting_needs_input"
                        and not self.host._shutdown_worker_inventory.workers,
                        timeout=20,
                    )
                fit_engine.fit_individual_edge.assert_not_called()
                self.assertIsNone(self.window._fit_outcome)
                message = self.window.status_label.text()
                self.assertIn("No fitting was started", message)
                if diagnostic:
                    self.assertIn(diagnostic, message)
                    self.assertNotIn("uncertain", message)
                else:
                    self.assertIn("marked the overlay comparison as uncertain", message)
                    self.assertNotIn("response validation failed", message)

    def test_frame_selection_is_bounded_and_deduplicated(self):
        self.assertEqual(representative_frame_indices(0), ())
        self.assertEqual(representative_frame_indices(1), (0,))
        self.assertEqual(representative_frame_indices(2), (0, 1))
        self.assertEqual(representative_frame_indices(7), (0, 3, 6))
        with self.assertRaises(ValueError):
            representative_frame_indices(-1)

    def test_image_preflight_uses_selected_roi_and_bounded_evidence(self):
        old_images = self.host.images
        old_roi = self.host.selected_area
        old_revision = self.host.scientific_observation_api.revision_clock.current()
        self.window.start_preflight()
        self._wait_for(lambda: self.window.result is not None)
        result = self.window.result
        self.assertEqual(result.dataset.mode, DatasetMode.IMAGES)
        self.assertEqual(result.assessment.readiness, PreflightReadiness.READY)
        self.assertEqual(result.roi, self.host.scientific_observation_api.capture_selected_roi(
            self.host.scientific_observation_api.capture_dataset()
        ))
        self.assertEqual([frame.frame_index for frame in result.frames], [0, 1, 3])
        self.assertTrue(all(max(frame.image.preview_shape) <= 512 for frame in result.frames))
        expected = [float(np.mean(image[2:5, 1:4])) for image in old_images]
        np.testing.assert_array_equal(result.spectrum.intensities, expected)
        self.assertEqual(result.plot.roi_effective, result.roi)
        self.assertLess(
            result.frames[0].image.preview_pixels[0, 0],
            result.frames[0].image.preview_pixels[-1, -1],
        )
        self.assertEqual(result.frames[0].image.coordinate_mapping.origin, "upper")
        overlay = self.window._preview_pixmap(result.frames[0].image, result.roi).toImage()
        self.assertEqual(overlay.pixelColor(1, 2).name(), "#ffff00")
        self.assertNotIn("C:/private", self.window.dataset_summary_label.text())
        self.assertIn("Companion metadata", self.window.dataset_summary_label.text())
        self.assertIs(self.host.images, old_images)
        self.assertEqual(self.host.selected_area, old_roi)
        self.assertEqual(self.host.scientific_observation_api.revision_clock.current(), old_revision)
        self._wait_for(lambda: self.window._worker is None)
        self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)
        self.assertFalse(self.window.stop_button.isVisible())
        self.assertFalse(self.window.progress_bar.isVisible())

    def test_agent_entry_is_independent_and_reuses_hidden_window(self):
        menu = self.host.menuBar()
        self.host._add_agent_menu(menu)
        self.assertEqual(menu.actions()[-1].text(), "AI Agent")
        self.window.hide()
        self.host.open_ai_agent()
        self.assertIs(self.host.agent_window, self.window)
        self.assertTrue(self.window.isVisible())
        self.window.close()
        self.assertFalse(self.window.isVisible())
        self.host.open_ai_agent()
        self.assertIs(self.host.agent_window, self.window)
        self.assertTrue(self.window.isVisible())

    def test_step_conclusions_are_default_and_step_evidence_opens_on_request(self):
        self.assertIs(self.window.centralWidget(), self.window.root_panel)
        self.assertIsInstance(self.window.splitter, QSplitter)
        self.assertEqual(self.window.splitter.orientation(), Qt.Horizontal)
        self.assertEqual(self.window.splitter.count(), 2)
        self.assertTrue(self.window.summary_page.isVisible())
        self.assertFalse(self.window.overview_panel.isVisible())
        self.assertFalse(self.window.evidence_panel.isVisible())
        self.assertFalse(self.window.fit_matrix.isVisible())
        self.assertFalse(self.window.plan_text.isVisible())
        self.assertFalse(self.window.summary_page.results_button.isVisible())
        for row in self.window.summary_page.rows:
            self.assertFalse(row.evidence_panel.isVisible())
        self.window.summary_page.rows[0].toggle_button.click()
        self.assertTrue(self.window.summary_page.rows[0].evidence_panel.isVisible())
        self.window._inspect_summary_step(0)
        self.assertTrue(self.window.step_evidence_page.isVisible())
        self.assertEqual(self.window.step_evidence_heading.text(), "Dataset check")
        self.assertFalse(self.window.detail_page.isVisible())
        self.assertFalse(self.window.evidence_panel.isVisible())
        self.assertFalse(self.window.dataset_summary_label.isVisible())
        self.assertFalse(hasattr(self.window, "assessment_text"))
        self.assertFalse(hasattr(self.window, "objective_combo"))
        self.assertFalse(hasattr(self.window, "start_button"))
        self.assertEqual(self.window.settings_button.text(), "Agent settings…")
        self.window.show_step_conclusions()
        self.assertTrue(self.window.summary_page.isVisible())
        self.assertFalse(self.window.evidence_panel.isVisible())

    def test_model_readiness_discloses_cloud_transfer_and_snapshot_authorization(self):
        self.window._agent_settings = AgentModelSettings()
        self.window._update_model_readiness()
        self.assertIn("No LLM model is selected", self.window.model_readiness_label.text())
        self.assertIn("offline substitute plan", self.window.model_readiness_label.text())
        self.assertFalse(hasattr(self.window, "cloud_consent_checkbox"))
        self.assertFalse(hasattr(self.window, "vision_checkbox"))
        self.assertFalse(hasattr(self.window, "fallback_checkbox"))

        self.window.start_autonomous_analysis()
        self.assertIn("LLM model is not available", self.window.status_label.text())
        self.assertIsNone(self.window._worker)
        self.assertIsNone(self.window._planning_worker)

        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.OPENAI_CLOUD,
            model_id="",
        )
        self.window._update_model_readiness()
        self.assertIn("OpenAI LLM is not configured", self.window.model_readiness_label.text())
        self.assertIn("No request or visual evidence will be sent", self.window.model_readiness_label.text())

        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.OPENAI_CLOUD,
            model_id="configured-model",
        )
        self.window._update_model_readiness()
        self.assertIn("automatically sends", self.window.model_readiness_label.text())
        self.assertIn("up to 16 bounded PNGs", self.window.model_readiness_label.text())
        self.assertIn("up to 3 image previews", self.window.model_readiness_label.text())
        self.assertIn("configured-model", self.window.model_readiness_label.text())

        snapshot = self.window._snapshot_run_configuration()
        self.assertEqual(snapshot.objective, PlanningObjective.DATASET_READINESS)
        self.assertTrue(snapshot.cloud_egress_consent)
        self.assertFalse(snapshot.deterministic_fallback)

        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.LOCAL,
            model_id="changed-after-start",
            endpoint="http://127.0.0.1:11434/v1",
        )
        self.window._update_model_readiness()
        self.assertEqual(snapshot.settings.model_id, "configured-model")
        self.assertEqual(snapshot.settings.mode, AgentModelMode.OPENAI_CLOUD)
        self.assertTrue(snapshot.cloud_egress_consent)
        self.assertIn("loopback endpoint", self.window.model_readiness_label.text())
        self.assertIn("not sent in local-model mode", self.window.model_readiness_label.text())

    def test_model_connection_is_user_initiated_and_uses_tracked_worker(self):
        created_workers = []
        saved_settings = []
        saved_credentials = []

        class MemorySettings:
            def load(self):
                return AgentModelSettings()

            def save(self, settings):
                saved_settings.append(settings)
                return settings

        class MemoryCredentials:
            def get(self):
                return None

            def set(self, _credential):
                saved_credentials.append(_credential)

            def delete(self):
                return None

        class FakeSignal:
            def __init__(self):
                self.callbacks = []

            def connect(self, callback):
                self.callbacks.append(callback)

            def emit(self, *args):
                for callback in tuple(self.callbacks):
                    callback(*args)

        class FakeConnectionWorker:
            def __init__(
                self, settings, *, pending_api_key=None,
                pending_langsmith_api_key=None, parent=None,
            ):
                self.settings = settings
                self.pending_api_key = pending_api_key
                self.pending_langsmith_api_key = pending_langsmith_api_key
                self.parent = parent
                self.completed = FakeSignal()
                self.finished = FakeSignal()
                self.running = False
                created_workers.append(self)

            def isRunning(self):
                return self.running

            def start(self):
                self.running = True

            def stop(self):
                self.running = False

        class FakeHost:
            def __init__(self):
                self.tracked = []

            def _track_shutdown_worker(self, worker, role):
                self.tracked.append((worker, role))

            def _connect_shutdown_worker_signal(self, worker, signal, callback):
                signal.connect(callback)

        fake_host = FakeHost()
        dialog = AgentModelSettingsDialog(
            self.window,
            main_window=fake_host,
            settings_repository=MemorySettings(),
            credential_store=MemoryCredentials(),
            connection_worker_class=FakeConnectionWorker,
        )
        try:
            self.assertEqual(created_workers, [])
            dialog.mode_combo.setCurrentIndex(
                dialog.mode_combo.findData(AgentModelMode.OPENAI_CLOUD.value)
            )
            dialog.model_edit.setText("explicit-model")
            dialog.key_edit.setText("unsaved-agent-key")
            self.assertTrue(dialog.test_button.isEnabled())
            dialog.test_button.click()
            self.assertEqual(len(created_workers), 1)
            self.assertEqual(
                created_workers[0].settings.mode, AgentModelMode.OPENAI_CLOUD
            )
            self.assertEqual(created_workers[0].pending_api_key, "unsaved-agent-key")
            self.assertEqual(saved_credentials, [])
            self.assertEqual(saved_settings, [])
            self.assertEqual(fake_host.tracked, [(created_workers[0], "agent_model_test")])
            self.assertFalse(dialog.test_button.isEnabled())
            created_workers[0].completed.emit(True, "Connection test succeeded.")
            created_workers[0].running = False
            created_workers[0].finished.emit()
            self.assertEqual(dialog.status_label.text(), "Connection test succeeded.")
            self.assertTrue(dialog.test_button.isEnabled())
            self.assertEqual(dialog.key_edit.text(), "unsaved-agent-key")
            self.assertEqual(saved_credentials, [])

            dialog.save_and_accept()
            self.assertEqual(saved_credentials, ["unsaved-agent-key"])
            self.assertEqual(len(saved_settings), 1)
            self.assertEqual(dialog.result(), dialog.Accepted)
        finally:
            dialog.close()

    def test_agent_model_settings_cancel_discards_pending_key(self):
        saved_credentials = []

        class MemorySettings:
            def load(self):
                return AgentModelSettings()

            def save(self, _settings):
                raise AssertionError("Cancel must not save settings")

        class MemoryCredentials:
            def get(self):
                return None

            def set(self, credential):
                saved_credentials.append(credential)

            def delete(self):
                return None

        dialog = AgentModelSettingsDialog(
            self.window,
            settings_repository=MemorySettings(),
            credential_store=MemoryCredentials(),
        )
        dialog.mode_combo.setCurrentIndex(
            dialog.mode_combo.findData(AgentModelMode.OPENAI_CLOUD.value)
        )
        dialog.model_edit.setText("explicit-model")
        dialog.key_edit.setText("cancelled-agent-key")
        dialog.reject()
        self.assertEqual(saved_credentials, [])
        self.assertEqual(dialog.key_edit.text(), "")
        self.assertEqual(dialog.result(), dialog.Rejected)

    def test_connection_worker_releases_pending_key_and_is_shutdown_tracked(self):
        import NEAT.ui.agent_model_settings_dialog as settings_module

        entered = threading.Event()
        release = threading.Event()
        observed = []

        def paused_connection_test(
            settings, resolver, *, credential_override=None, transport=None
        ):
            observed.append((settings.mode, credential_override))
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release connection worker")

        class MemorySettings:
            def load(self):
                return AgentModelSettings()

            def save(self, _settings):
                return None

        class MemoryCredentials:
            def get(self):
                return None

            def set(self, _credential):
                raise AssertionError("Test Connection must not persist a key")

            def delete(self):
                return None

        dialog = AgentModelSettingsDialog(
            self.window,
            main_window=self.host,
            settings_repository=MemorySettings(),
            credential_store=MemoryCredentials(),
        )
        dialog.mode_combo.setCurrentIndex(
            dialog.mode_combo.findData(AgentModelMode.OPENAI_CLOUD.value)
        )
        dialog.model_edit.setText("explicit-model")
        dialog.key_edit.setText("ephemeral-worker-key")
        with patch.object(
            settings_module,
            "run_model_connection_test",
            side_effect=paused_connection_test,
        ):
            dialog.test_button.click()
            worker = dialog.connection_worker
            self.assertIsInstance(worker, AgentModelConnectionWorker)
            self.assertTrue(entered.wait(2))
            self.assertIn("agent_model_test", self.host._shutdown_worker_inventory.roles)
            self.assertIsNone(worker._pending_api_key)
            self.assertEqual(
                observed,
                [(AgentModelMode.OPENAI_CLOUD, "ephemeral-worker-key")],
            )

            self.host._shutdown_worker_inventory.request_shutdown()
            self.assertTrue(worker._transport._cancelled)
            self.assertIn(worker, self.host._shutdown_worker_inventory.workers)
            release.set()
            self._wait_for(
                lambda: not self.host._shutdown_worker_inventory.workers
            )

        self.assertIsNone(self.host.agent_model_test_worker)
        self.assertEqual(dialog.key_edit.text(), "ephemeral-worker-key")
        dialog.reject()
        self.assertEqual(dialog.key_edit.text(), "")

    def test_agent_menu_survives_each_workflow_tab_rebuild(self):
        self.host.tabs = QTabWidget(self.host)
        self.host.FittingTab = QWidget(self.host.tabs)
        self.host.PostProcessingTab = QWidget(self.host.tabs)
        preprocessing = QWidget(self.host.tabs)
        for label, tab in (
            ("Fitting", self.host.FittingTab),
            ("Postprocessing", self.host.PostProcessingTab),
            ("Preprocessing", preprocessing),
        ):
            self.host.tabs.addTab(tab, label)
        self.host._menu_bar_ready = True
        self.host._build_fitting_menu_bar = lambda menu: menu.addMenu("Fitting")
        self.host._build_postprocessing_menu_bar = lambda menu: menu.addMenu("Postprocessing")
        self.host._build_preprocessing_menu_bar = lambda menu: menu.addMenu("Preprocessing")
        self.host._add_assistant_menu = lambda menu: menu.addMenu("AI Assistant")
        self.host._add_about_menu = lambda menu: menu.addMenu("About")
        for index in range(3):
            self.host.tabs.setCurrentIndex(index)
            self.host.refresh_menu_bar_for_current_tab()
            labels = [action.text() for action in self.host.menuBar().actions()]
            self.assertEqual(labels.count("AI Agent"), 1)
            self.assertEqual(labels.count("AI Assistant"), 1)

    def test_empty_and_invalid_axis_are_not_reported_ready(self):
        self.host.scientific_observation_api.invalidate_dataset(replace_dataset=True)
        self.host.images = []
        self.host.wavelengths = np.array([])
        self.window.start_preflight()
        self._wait_for(lambda: self.window.result is not None)
        self.assertEqual(self.window.result.assessment.readiness, PreflightReadiness.INVALID)
        self._wait_for(lambda: self.window._worker is None)
        self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)
        self.host.scientific_observation_api.invalidate_dataset(replace_dataset=True)
        self.host.images = [np.ones((6, 6)), np.ones((6, 6))]
        self.host.wavelengths = np.array([2.0, 2.0])
        self.window.start_preflight()
        self._wait_for(lambda: self.window.result is not None and self.window.result.dataset.mode is DatasetMode.IMAGES)
        self.assertEqual(self.window.result.assessment.readiness, PreflightReadiness.LIMITED)

    def test_loader_busy_returns_retry_without_worker(self):
        self.host.fits_image_load_worker = Mock(isRunning=Mock(return_value=True))
        self.window.start_autonomous_analysis()
        self.assertIsNone(self.window._worker)
        self.assertIsNone(self.window.result)
        self.assertIn("loader is busy", self.window.status_label.text())
        self.assertFalse(self.window._has_active_worker())
        self.assertTrue(self.window.start_autonomous_button.isEnabled())

    def test_completed_worker_releases_old_image_stack_after_reload(self):
        old_frame = weakref.ref(self.host.images[0])
        self.window.start_preflight()
        self._wait_for(lambda: self.window.result is not None)
        self._wait_for(lambda: self.window._worker is None)
        self.host.scientific_observation_api.invalidate_dataset(replace_dataset=True)
        self.host.images = [np.zeros((6, 6))]
        self.host.wavelengths = np.array([2.0])
        self.window._check_revision()
        gc.collect()
        self.assertIsNone(old_frame())
        self.assertIsNone(self.window.result)

    def test_missing_roi_and_profile_mode_have_honest_evidence(self):
        self.host.selected_area = None
        self.window.start_preflight()
        self._wait_for(lambda: self.window.result is not None)
        self.assertEqual(self.window.result.assessment.readiness, PreflightReadiness.LIMITED)
        self.assertIsNone(self.window.result.spectrum)
        self._wait_for(lambda: self.window._worker is None)
        self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)

        self.host.scientific_observation_api.invalidate_dataset(replace_dataset=True)
        self.host.fitting_data_source = "profile"
        self.host.wavelengths = np.array([1.0, 2.0, 3.0, 4.0])
        self.host.intensities = np.array([2.0, 4.0, 6.0, 8.0])
        self.window.start_preflight()
        self._wait_for(lambda: self.window.result is not None and self.window.result.dataset.mode is DatasetMode.PROFILE)
        self.assertEqual(self.window.result.frames, ())
        self.assertIsNone(self.window.result.roi)
        self.assertIsNotNone(self.window.result.spectrum)
        self.assertIn("spectrum-only", self.window.dataset_summary_label.text().lower())

    def test_dataset_change_discards_visible_evidence_and_late_result(self):
        self.window.start_preflight()
        self._wait_for(lambda: self.window.result is not None)
        self.host.scientific_observation_api.invalidate_dataset(replace_dataset=True)
        self.host.images = [np.zeros((6, 6))]
        self.host.wavelengths = np.array([2.0])
        self.window._check_revision()
        self.assertIsNone(self.window.result)
        self.assertIn("Dataset changed", self.window.status_label.text())
        self.assertIn("Dataset changed", self.window.evidence_layout.itemAt(0).widget().text())

    def test_worker_keeps_gui_responsive_and_cancellation_drops_results(self):
        import NEAT.ui.agent_preflight_worker as worker_module

        entered = threading.Event()
        release = threading.Event()
        original = worker_module.get_image_preview

        def paused_preview(*args, **kwargs):
            entered.set()
            release.wait(5)
            return original(*args, **kwargs)

        with patch.object(worker_module, "get_image_preview", side_effect=paused_preview):
            self.window.start_preflight()
            self.assertTrue(entered.wait(2))
            ticked = []
            QTimer.singleShot(0, lambda: ticked.append(True))
            self._wait_for(lambda: bool(ticked))
            self.assertFalse(self.window.start_autonomous_button.isEnabled())
            active_worker = self.window._worker
            self.window.start_preflight()
            self.assertIs(self.window._worker, active_worker)
            self.window.stop_preflight()
            release.set()
            self._wait_for(lambda: self.window._worker is None)
            self.assertIsNone(self.window.result)
            self.assertIn("cancelled", self.window.status_label.text().lower())

    def test_preflight_controls_are_visible_until_cancelled_worker_retires(self):
        import NEAT.ui.agent_preflight_worker as worker_module

        entered = threading.Event()
        release = threading.Event()
        original = worker_module.get_image_preview

        def paused_preview(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release preflight")
            return original(*args, **kwargs)

        with patch.object(worker_module, "get_image_preview", side_effect=paused_preview):
            try:
                self.window.start_autonomous_button.click()
                self.assertTrue(self.window.stop_button.isVisible())
                self.assertTrue(self.window.stop_button.isEnabled())
                self.assertTrue(self.window.progress_bar.isVisible())
                self.assertFalse(self.window.start_autonomous_button.isEnabled())
                self.assertTrue(entered.wait(2))

                self.window.stop_button.click()
                self.assertTrue(self.window.stop_button.isVisible())
                self.assertFalse(self.window.stop_button.isEnabled())
                self.assertEqual(self.window.stop_button.text(), "Stopping…")
                self.assertTrue(self.window.progress_bar.isVisible())
                self.assertFalse(self.window.start_autonomous_button.isEnabled())
            finally:
                release.set()
            self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)

        self.assertFalse(self.window.stop_button.isVisible())
        self.assertFalse(self.window.progress_bar.isVisible())
        self.assertTrue(self.window.start_autonomous_button.isEnabled())
        self.assertEqual(self.window._stage, "cancelled")

    def test_cancel_before_worker_start_returns_no_evidence(self):
        handle = self.host.scientific_observation_api.capture_dataset()
        roi = self.host.scientific_observation_api.capture_selected_roi(handle)
        worker = PreflightWorker("cancel-before-start", handle, roi)
        outcomes = []
        worker.completed.connect(outcomes.append)
        worker.stop()
        worker.start()
        self._wait_for(lambda: bool(outcomes))
        self.assertEqual(outcomes[0].state.value, "cancelled")
        self.assertIsNone(outcomes[0].result)
        self.assertIsNone(worker._handle)
        worker.wait(5000)
        worker.deleteLater()

    def test_closing_agent_during_work_hides_and_cancels_safely(self):
        import NEAT.ui.agent_preflight_worker as worker_module

        entered = threading.Event()
        release = threading.Event()
        original = worker_module.get_image_preview

        def paused_preview(*args, **kwargs):
            entered.set()
            release.wait(5)
            return original(*args, **kwargs)

        with patch.object(worker_module, "get_image_preview", side_effect=paused_preview):
            self.window.start_preflight()
            self.assertTrue(entered.wait(2))
            self.window.close()
            self.assertFalse(self.window.isVisible())
            self.assertIsNotNone(self.window._worker)
            release.set()
            self._wait_for(lambda: self.window._worker is None)
            self.assertIsNone(self.window.result)
            self.host.open_ai_agent()
            self.assertIs(self.host.agent_window, self.window)
            self.assertTrue(self.window.isVisible())

    def test_reload_during_worker_discards_late_result(self):
        import NEAT.ui.agent_preflight_worker as worker_module

        entered = threading.Event()
        release = threading.Event()
        original = worker_module.get_image_preview

        def paused_preview(*args, **kwargs):
            entered.set()
            release.wait(5)
            return original(*args, **kwargs)

        with patch.object(worker_module, "get_image_preview", side_effect=paused_preview):
            self.window.start_preflight()
            self.assertTrue(entered.wait(2))
            self.host.scientific_observation_api.invalidate_dataset(replace_dataset=True)
            self.host.images = [np.zeros((6, 6))]
            self.host.wavelengths = np.array([1.0])
            self.window._check_revision()
            release.set()
            self._wait_for(lambda: self.window._worker is None)
            self.assertIsNone(self.window.result)
            self.assertIn("Dataset changed", self.window.status_label.text())

    def test_main_close_during_preflight_is_retryable_and_retires_worker(self):
        import NEAT.ui.agent_preflight_worker as worker_module

        entered = threading.Event()
        release = threading.Event()
        original = worker_module.get_image_preview

        def paused_preview(*args, **kwargs):
            entered.set()
            release.wait(5)
            return original(*args, **kwargs)

        with patch.object(worker_module, "get_image_preview", side_effect=paused_preview):
            self.window.start_preflight()
            self.assertTrue(entered.wait(2))
            self.assertFalse(self.host.close())
            self.assertTrue(self.host.isVisible())
            release.set()
            self._wait_for(lambda: self.window._worker is None)
            self._wait_for(lambda: not self.host._shutdown_close_pending)
            self.assertTrue(self.host.close())
            self.assertFalse(self.window.isVisible())

    @unittest.skipUnless(find_spec("langgraph") is not None, "requires the optional NEAT [agent] extra")
    def test_one_click_runs_current_preflight_then_real_graph_without_second_action(self):
        from test_agent_model import FakeModel

        import NEAT.ui.agent_window as agent_window_module
        from NEAT.ui.agent_planning_worker import PlanningWorker as RealPlanningWorker

        model = FakeModel()

        def build_worker(context, revision_clock, snapshot, **kwargs):
            return RealPlanningWorker(
                context,
                revision_clock,
                snapshot,
                model=model,
                **kwargs,
            )

        with patch.object(agent_window_module, "PlanningWorker", side_effect=build_worker):
            self.window.start_autonomous_button.click()
            first_run_id = self.window._agent_run_id
            self.window.start_autonomous_button.click()
            self.assertFalse(hasattr(self.window, "start_button"))
            self._wait_for(
                lambda: self.window._planning_outcome is not None
                and self.window._planning_worker is None
                and self.window._worker is None
                and not self.host._shutdown_worker_inventory.workers
            )
        outcome = self.window._planning_outcome
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.run_id, first_run_id)
        self.assertEqual(outcome.preflight_run_id, self.window.result.run_id)
        self.assertEqual(outcome.dataset_id, self.window.result.dataset_id)
        self.assertEqual(outcome.revision, self.window.result.revision)
        self.assertIn("validate_preflight", outcome.transitions)
        self.assertIn("propose_read_only_plan", outcome.transitions)
        self.assertEqual(outcome.transitions[-1], "terminal")
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(self.window.start_autonomous_button.isEnabled())
        self.assertFalse(self.window.stop_button.isVisible())
        self.assertFalse(self.window.progress_bar.isVisible())
        self.assertEqual(self.window.splitter.count(), 2)
        self.assertTrue(self.window.summary_page.isVisible())
        self.window._inspect_summary_step(0)
        self.assertTrue(self.window.step_evidence_page.isVisible())
        self.assertEqual(self.window.step_evidence_heading.text(), "Dataset check")
        self.assertFalse(self.window.evidence_panel.isVisible())
        self.assertNotIn("Preview Next Steps", self.window.plan_text.toPlainText())
        self.assertNotIn("fitting successful", self.window.plan_text.toPlainText().lower())
        if outcome.status is PlanningStatus.READY_PLAN:
            self.assertIn(
                "Dataset preflight plan prepared. Visual checks and local exploratory fitting are the next stages.",
                self.window.plan_text.toPlainText(),
            )

    def _configure_synthetic_vision_dataset(self):
        wavelengths = np.linspace(1.0, 5.0, 32)
        self.host.images = []
        for wavelength in wavelengths:
            frame = np.zeros((64, 64), dtype=float)
            frame[18:50, 16:52] = 20 + (wavelength >= 2.4) * 3
            self.host.images.append(frame)
        self.host.wavelengths = wavelengths
        self.host.selected_area = (16, 52, 18, 50)
        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.OPENAI_CLOUD, model_id="gpt-6-luna"
        )
        self.window._update_model_readiness()

    def _fake_agent_keyring_patch(self):
        import NEAT.ui.agent_window as agent_window_module

        class FakeKeyStore:
            def get(self):
                return "test-key"

        return patch.object(
            agent_window_module,
            "AgentKeyringCredentialStore",
            return_value=FakeKeyStore(),
        )

    @staticmethod
    def _successful_fake_plan(context, **_kwargs):
        return PlanningOutcome(
            run_id=context.run_id,
            preflight_run_id=context.preflight_run_id,
            dataset_id=context.dataset_id,
            revision=context.revision,
            status=PlanningStatus.READY_PLAN,
            transitions=(
                "validate_preflight",
                "propose_read_only_plan",
                "validate_plan",
                "terminal",
            ),
            planner_source=PlannerSource.LLM_GENERATED,
        )

    def test_openai_visual_first_discovers_roi_when_no_neat_roi_is_selected(self):
        from test_agent_vision import FakeVision

        import NEAT.ui.agent_vision_worker as worker_module

        self._configure_synthetic_vision_dataset()
        self.host.selected_area = None
        model = FakeVision()
        with (
            self._fake_agent_keyring_patch(),
            patch.object(worker_module, "OpenAIVisionInspector", return_value=model),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(
                lambda: self.window._vision_outcome is not None
                and self.window._vision_worker is None
            )
            self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)

        outcome = self.window._vision_outcome
        self.assertEqual(outcome.status, VisionStatus.NEEDS_REVIEW)
        self.assertEqual(outcome.roi, RoiBounds(20, 36, 24, 40))
        self.assertEqual(outcome.phase_id, "Fe_bcc")
        self.assertIsNone(self.window._planning_outcome)
        self.assertIsNone(self.window._planning_worker)
        self.assertIn("M4 metadata planning was skipped", self.window.plan_text.toPlainText())
        self.assertIn("no planning result is claimed", self.window.plan_text.toPlainText())
        self.assertEqual(
            [call[0] for call in model.calls],
            ["sample_roi", "phase_shortlist", "phase_overlay", "sample_locations", "sample_locations", "sample_locations"],
        )
        self.assertFalse(self.window.adjust_roi_button.isVisible())
        self.window._inspect_summary_step(2)
        self.assertTrue(self.window.adjust_roi_button.isVisible())
        self.assertEqual(self.host.selected_area, None)

    def test_visual_first_reports_missing_calibration_or_invalid_image_previews(self):
        from test_agent_vision import _fixture

        _handle, result = _fixture()
        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.OPENAI_CLOUD, model_id="gpt-6-luna"
        )
        self.window._run_snapshot = self.window._snapshot_run_configuration()
        self.window._vision_run_enabled = True

        missing_axis = replace(
            result,
            dataset=replace(
                result.dataset,
                wavelength_axis_status=WavelengthAxisStatus.MISSING,
            ),
        )
        self.assertIn(
            "valid wavelength calibration",
            self.window._visual_roi_discovery_problem(missing_axis),
        )

        invalid_image = replace(result.frames[0].image, status=ObservationStatus.INVALID)
        invalid_preview = replace(
            result,
            frames=(replace(result.frames[0], image=invalid_image), *result.frames[1:]),
        )
        self.assertIn(
            "preview is invalid or incomplete",
            self.window._visual_roi_discovery_problem(invalid_preview),
        )

    def test_adjusted_roi_is_reused_for_a_new_visual_run_and_recomputed(self):
        from test_agent_vision import FakeVision

        import NEAT.ui.agent_planning_worker as planning_worker_module
        import NEAT.ui.agent_vision_worker as vision_worker_module

        self._configure_synthetic_vision_dataset()
        models = [FakeVision(), FakeVision()]
        with (
            self._fake_agent_keyring_patch(),
            patch.object(
                planning_worker_module,
                "run_planning",
                side_effect=self._successful_fake_plan,
            ),
            patch.object(
                vision_worker_module,
                "OpenAIVisionInspector",
                side_effect=models,
            ),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(
                lambda: self.window._vision_outcome is not None
                and self.window._vision_worker is None
            )
            self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)

            adjusted = RoiBounds(1, 10, 1, 10)
            for edit, value in zip(
                self.window.roi_edits,
                (adjusted.x_min, adjusted.x_max, adjusted.y_min, adjusted.y_max),
            ):
                edit.setValue(value)
            self.window.preview_adjusted_roi()
            self.assertEqual(self.window._adjusted_roi, adjusted)
            token = self.window._dataset_token
            self.assertEqual(self.window._adjusted_roi_dataset_token, token)

            self.window.start_autonomous_analysis()
            self._wait_for(
                lambda: self.window._vision_outcome is not None
                and self.window._vision_worker is None
            )
            self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)

        outcome = self.window._vision_outcome
        self.assertEqual(outcome.roi, adjusted)
        self.assertEqual(outcome.confidence, "user-selected")
        self.assertEqual(
            [call[0] for call in models[1].calls],
            ["phase_shortlist", "phase_overlay", "sample_locations", "sample_locations", "sample_locations"],
        )
        self.assertIn("ROI x[1,10), y[1,10)", models[1].calls[0][2])
        self.assertIn("Already accepted locations", models[1].calls[2][2])
        self.assertEqual(self.host.selected_area, (16, 52, 18, 50))
        self.assertEqual(self.window._adjusted_roi, adjusted)
        self.assertEqual(self.window._adjusted_roi_dataset_token, token)
        self.host.scientific_observation_api.revision_clock.invalidate()
        self.window._check_revision()
        self.assertTrue(self.window._stale)
        self.assertIsNone(self.window._adjusted_roi)
        self.assertIsNone(self.window._adjusted_roi_dataset_token)

    @unittest.skipUnless(
        find_spec("langgraph") is not None,
        "requires the optional NEAT [agent] extra",
    )
    def test_openai_uses_same_model_for_planning_and_visual_review(self):
        from test_agent_vision import FakeVision

        import NEAT.ui.agent_vision_worker as worker_module
        import NEAT.ui.agent_window as agent_window_module
        from NEAT.services.agent_model import ModelCompletion
        from NEAT.ui.agent_planning_worker import PlanningWorker as RealPlanningWorker

        class FakePlanner:
            calls = 0

            def complete(self, _payload_json, *, available_step_types):
                self.calls += 1
                return ModelCompletion(
                    '{"steps":[{"step_type":"inspect_dataset_summary"}]}'
                )

            def cancel(self):
                return None

        self._configure_synthetic_vision_dataset()
        fake_planner = FakePlanner()
        configured_for_planning = []
        configured_for_vision = []

        class FakeKeyStore:
            def get(self):
                return "test-key"

        def make_vision(settings, **_kwargs):
            configured_for_vision.append(settings.model_id)
            return FakeVision()

        def build_planner(context, revision_clock, snapshot, **kwargs):
            configured_for_planning.append(snapshot.settings.model_id)
            return RealPlanningWorker(
                context,
                revision_clock,
                snapshot,
                model=fake_planner,
                **kwargs,
            )

        source_images = self.host.images
        source_roi = self.host.selected_area
        with (
            patch.object(agent_window_module, "PlanningWorker", side_effect=build_planner),
            patch.object(
                agent_window_module,
                "AgentKeyringCredentialStore",
                return_value=FakeKeyStore(),
            ),
            patch.object(worker_module, "OpenAIVisionInspector", side_effect=make_vision),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(
                lambda: self.window._vision_outcome is not None
                and self.window._vision_worker is None
            )
            self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)
        outcome = self.window._vision_outcome
        self.assertEqual(outcome.status, VisionStatus.NEEDS_REVIEW)
        self.assertEqual(outcome.phase_id, "Fe_bcc")
        self.assertIsNotNone(self.window._planning_outcome)
        self.assertEqual(self.window._planning_outcome.status, PlanningStatus.READY_PLAN)
        self.assertEqual(self.window._planning_outcome.planner_source.value, "llm_generated_plan")
        self.assertTrue(self._planning_request_was_authorized())
        self.assertEqual(fake_planner.calls, 1)
        self.assertEqual(configured_for_planning, ["gpt-6-luna"])
        self.assertEqual(configured_for_vision, configured_for_planning)
        self.assertIs(self.host.images, source_images)
        self.assertEqual(self.host.selected_area, source_roi)
        self.assertFalse(self.window.adjust_roi_button.isVisible())
        self.window._inspect_summary_step(2)
        self.assertTrue(self.window.adjust_roi_button.isVisible())
        self.assertIn("Local fitting paused", self.window.plan_text.toPlainText())
        self.assertIn("OpenAI visual proposal", self.window.plan_text.toPlainText())
        self.assertIn("gpt-6-luna", self.window.plan_text.toPlainText())
        self.assertFalse(self.window.stop_button.isVisible())
        self.assertFalse(self.window.progress_bar.isVisible())
        self.window.roi_edits[0].setValue(19)
        self.window.adjust_roi_button.click()
        self.assertIsNone(self.window._vision_outcome)
        self.assertIn("earlier spectrum and phase hypothesis no longer describe this ROI", self.window.plan_text.toPlainText())
        self.assertEqual(self.host.selected_area, source_roi)
        self.assertTrue(self.window.summary_page.discard_adjusted_button.isVisible())
        self.window.summary_page.discard_adjusted_button.click()
        self.assertIsNone(self.window._vision_outcome)
        self.assertIsNone(self.window._adjusted_roi)
        self.assertIsNone(self.window._adjusted_roi_dataset_token)
        self.assertIn("discarded", self.window.status_label.text().lower())

    def _planning_request_was_authorized(self):
        return bool(
            self.window._run_snapshot
            and self.window._run_snapshot.cloud_egress_consent
        )

    def test_stop_after_vision_computation_before_queued_completion_discards_result(self):
        from test_agent_vision import FakeVision

        import NEAT.ui.agent_planning_worker as planning_worker_module
        import NEAT.ui.agent_vision_worker as worker_module

        self._configure_synthetic_vision_dataset()
        with (
            self._fake_agent_keyring_patch(),
            patch.object(
                planning_worker_module,
                "run_planning",
                side_effect=self._successful_fake_plan,
            ),
            patch.object(worker_module, "OpenAIVisionInspector", return_value=FakeVision()),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(lambda: self.window._vision_worker is not None)
            worker = self.window._vision_worker
            self.assertTrue(worker.wait(5000))
            self.assertIsNone(self.window._vision_outcome)
            self.window.stop_active_run()
            self._wait_for(lambda: self.window._vision_worker is None)
            self.assertIsNone(self.window._vision_outcome)
            self.assertEqual(self.window._stage, "cancelled")
            self.assertNotIn("Provisional phase", self.window.plan_text.toPlainText())
            self.assertTrue(self.window.start_autonomous_button.isEnabled())

    def test_openai_vision_stop_and_retirement_keep_controls_visible(self):
        from test_agent_vision import FakeVision

        import NEAT.ui.agent_planning_worker as planning_worker_module
        import NEAT.ui.agent_vision_worker as worker_module

        entered = threading.Event()
        release = threading.Event()

        class BlockingVision(FakeVision):
            def inspect(self, stage, images, prompt):
                if stage == "sample_roi":
                    entered.set()
                    release.wait(5)
                return super().inspect(stage, images, prompt)

            def cancel(self):
                super().cancel()
                release.set()

        self._configure_synthetic_vision_dataset()
        with (
            self._fake_agent_keyring_patch(),
            patch.object(
                planning_worker_module,
                "run_planning",
                side_effect=self._successful_fake_plan,
            ),
            patch.object(worker_module, "OpenAIVisionInspector", return_value=BlockingVision()),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(entered.is_set)
            self.assertTrue(self.window.stop_button.isVisible())
            self.assertTrue(self.window.progress_bar.isVisible())
            self.window.stop_active_run()
            self.assertIn("Stopping", self.window.stop_button.text())
            self.assertFalse(self.window.start_autonomous_button.isEnabled())
            release.set()
            self._wait_for(lambda: self.window._vision_worker is None and not self.host._shutdown_worker_inventory.workers)
            self.assertIsNone(self.window._vision_outcome)
            self.assertTrue(self.window.start_autonomous_button.isEnabled())

    def test_incomplete_visual_review_explains_manual_review_without_claiming_uncertainty(self):
        self.window._planning_display_text = "Validated read-only plan."
        outcome = VisionOutcome(
            "run",
            "dataset",
            3,
            VisionStatus.NEEDS_REVIEW,
            "Visual review is incomplete: the model's edge list must contain 1 to 5 edges. "
            "NEAT did not accept a phase or edge suggestion. Review the displayed measured spectrum and trusted overlays manually.",
            roi=RoiBounds(10, 20, 12, 22),
            roi_reason="A possible sample region; review the boundary.",
            confidence="medium",
            phase_reason="The candidate-edge count failed NEAT's deterministic validation.",
            inspected_frames=(0, 2),
            inspected_phases=("Fe_bcc", "CeO2"),
        )

        self.window._render_vision_outcome(outcome)
        displayed = self.window.plan_text.toPlainText()

        self.assertIn("manual evidence review required", displayed)
        self.assertIn("Review status: Visual review is incomplete", displayed)
        self.assertIn("No provisional phase or candidate-edge suggestion was accepted.", displayed)
        self.assertIn("candidate-edge count failed", displayed)
        self.assertNotIn("model reported uncertainty", displayed)

    def test_visual_response_failure_shows_proposed_phase_as_unaccepted(self):
        self.window._planning_display_text = "Validated read-only plan."
        diagnostic = "Visual phase explanation exceeds the 2048-character limit (received 2049 characters)."
        outcome = VisionOutcome(
            "run", "dataset", 3, VisionStatus.NEEDS_REVIEW,
            "The visual model's phase explanation failed response validation.",
            roi=RoiBounds(10, 20, 12, 22),
            inspected_phases=("Fe_bcc", "Al"),
            proposed_phase_id="Fe_bcc",
            response_validation_error=diagnostic,
        )
        self.window._render_vision_outcome(outcome)
        displayed = self.window.plan_text.toPlainText()

        self.assertIn("Model proposed phase (not accepted): Fe_bcc", displayed)
        self.assertIn("Response validation: " + diagnostic, displayed)
        self.assertIn("No provisional phase or candidate-edge suggestion was accepted.", displayed)
        self.assertNotIn("Provisional phase: Fe_bcc", displayed)
        self.assertNotIn("model marked the overlay comparison as uncertain", displayed)

    def test_visual_review_requires_a_configured_openai_model_before_preflight(self):
        self._configure_synthetic_vision_dataset()
        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.OPENAI_CLOUD, model_id=""
        )
        self.window.start_autonomous_analysis()
        self.assertIn("llm unavailable", self.window.status_label.text().lower())
        self.assertIn("exact openai model id", self.window.status_label.text().lower())
        self.assertIsNone(self.window._planning_worker)
        self.assertIsNone(self.window._vision_worker)

    def test_parent_shutdown_retires_active_visual_worker(self):
        from test_agent_vision import FakeVision

        import NEAT.ui.agent_planning_worker as planning_worker_module
        import NEAT.ui.agent_vision_worker as worker_module

        entered = threading.Event()
        release = threading.Event()

        class BlockingVision(FakeVision):
            def inspect(self, stage, images, prompt):
                if stage == "sample_roi":
                    entered.set()
                    release.wait(5)
                return super().inspect(stage, images, prompt)

            def cancel(self):
                release.set()

        self._configure_synthetic_vision_dataset()
        with (
            self._fake_agent_keyring_patch(),
            patch.object(
                planning_worker_module,
                "run_planning",
                side_effect=self._successful_fake_plan,
            ),
            patch.object(worker_module, "OpenAIVisionInspector", return_value=BlockingVision()),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(entered.is_set)
            self.assertFalse(self.host.close())
            release.set()
            self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)
            self.assertIsNone(self.window._vision_worker)

    def test_manual_preflight_and_autonomous_start_cannot_overlap(self):
        import NEAT.ui.agent_preflight_worker as worker_module

        entered = threading.Event()
        release = threading.Event()
        original = worker_module.get_image_preview

        def paused_preview(*args, **kwargs):
            entered.set()
            release.wait(5)
            return original(*args, **kwargs)

        with patch.object(worker_module, "get_image_preview", side_effect=paused_preview):
            self.window.start_preflight()
            self.assertTrue(entered.wait(2))
            active_worker = self.window._worker
            self.window.start_autonomous_button.click()
            self.assertIs(self.window._worker, active_worker)
            self.assertIsNone(self.window._planning_worker)
            self.assertIsNone(self.window._agent_run_id)
            self.window.stop_active_run()
            release.set()
            self._wait_for(lambda: self.window._worker is None)

    def test_stop_after_preflight_emits_success_before_callback_blocks_planning(self):
        self.window.start_autonomous_analysis()
        worker = self.window._worker
        self.assertIsNotNone(worker)

        # QThread.wait blocks this thread's Qt event loop. The worker can finish
        # computation and queue both completion signals, but the window slots
        # cannot run until after Stop below.
        self.assertTrue(worker.wait(5000))
        self.assertTrue(worker.isFinished())
        self.assertIsNone(self.window._planning_worker)
        self.assertFalse(self.window.start_autonomous_button.isEnabled())

        self.window.stop_active_run()
        self.assertIn("Stopping", self.window.status_label.text())
        self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)

        self.assertIsNone(self.window.result)
        self.assertIsNone(self.window._planning_worker)
        self.assertIsNone(self.window._planning_outcome)
        self.assertEqual(self.window._stage, "cancelled")
        self.assertIn("stopped", self.window.status_label.text().lower())
        self.assertNotIn("Plan prepared", self.window.plan_text.toPlainText())
        self.assertTrue(self.window.start_autonomous_button.isEnabled())

    def test_stop_after_planning_success_blocks_late_result_and_visual_stage(self):
        import NEAT.ui.agent_planning_worker as worker_module
        import NEAT.ui.agent_vision_worker as vision_worker_module

        entered = threading.Event()
        release = threading.Event()

        def successful_planning(context, **_kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release event-controlled planning")
            return PlanningOutcome(
                run_id=context.run_id,
                preflight_run_id=context.preflight_run_id,
                dataset_id=context.dataset_id,
                revision=context.revision,
                status=PlanningStatus.READY_PLAN,
                transitions=("validate_preflight", "propose_read_only_plan", "terminal"),
            )

        self._configure_synthetic_vision_dataset()
        with (
            patch.object(worker_module, "run_planning", side_effect=successful_planning),
            patch.object(
                vision_worker_module,
                "OpenAIVisionInspector",
                side_effect=AssertionError("Stop must prevent the visual stage"),
            ),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(
                lambda: entered.is_set() and self.window._planning_worker is not None
            )
            worker = self.window._planning_worker
            self.assertIsNotNone(worker)

            # Release the controlled computation, then block this thread's Qt
            # event loop until its success signal is queued and the worker exits.
            release.set()
            self.assertTrue(worker.wait(5000))
            self.assertTrue(worker.isFinished())
            self.assertNotIn("Plan prepared", self.window.plan_text.toPlainText())
            self.window.stop_active_run()
            self.assertIn("Stopping", self.window.status_label.text())
            self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)

        self.assertIsNone(self.window._planning_outcome)
        self.assertEqual(self.window._stage, "cancelled")
        self.assertEqual(
            self.window.status_label.text(),
            "Autonomous analysis stopped during planning.",
        )
        self.assertNotIn("Plan prepared", self.window.plan_text.toPlainText())
        self.assertIsNone(self.window._pending_vision_result)
        self.assertIsNone(self.window._vision_worker)
        self.assertTrue(self.window.start_autonomous_button.isEnabled())

    def test_tracked_planning_worker_stop_and_dataset_invalidation(self):
        import NEAT.ui.agent_planning_worker as worker_module
        from NEAT.services.agent_planning import PlanningCancelled

        entered = threading.Event()
        release = threading.Event()
        original = worker_module.run_planning

        def paused_planning(context, *, cancelled=None, **kwargs):
            entered.set()
            release.wait(5)
            if cancelled is not None and cancelled():
                raise PlanningCancelled("stopped")
            return original(context, **kwargs)

        with patch.object(worker_module, "run_planning", side_effect=paused_planning):
            self.window.start_autonomous_analysis()
            self._wait_for(entered.is_set, timeout=6)
            worker = self.window._planning_worker
            self.assertIsNotNone(worker)
            self.assertIn("agent_planning", self.host._shutdown_worker_inventory.roles)
            self.window.stop_active_run()
            self.assertIn("Stopping", self.window.status_label.text())
            release.set()
            self._wait_for(lambda: self.window._planning_worker is None)
            self.assertEqual(self.window._planning_outcome.status, PlanningStatus.CANCELLED)
            self.assertIsNone(self.window._main_window.agent_planning_worker)

    def test_planning_controls_are_visible_until_cancelled_worker_retires(self):
        import NEAT.ui.agent_planning_worker as worker_module
        from NEAT.services.agent_planning import PlanningCancelled

        entered = threading.Event()
        release = threading.Event()

        def paused_planning(context, *, cancelled=None, **_kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release planning")
            if cancelled is not None and cancelled():
                raise PlanningCancelled("stopped")
            raise RuntimeError("planning was expected to be cancelled")

        with patch.object(worker_module, "run_planning", side_effect=paused_planning):
            try:
                self.window.start_autonomous_button.click()
                self._wait_for(
                    lambda: entered.is_set() and self.window._planning_worker is not None
                )
                self.assertTrue(self.window.stop_button.isVisible())
                self.assertTrue(self.window.stop_button.isEnabled())
                self.assertTrue(self.window.progress_bar.isVisible())
                self.assertGreaterEqual(self.window.progress_bar.value(), 90)
                self.assertFalse(self.window.start_autonomous_button.isEnabled())

                self.window.stop_button.click()
                self.assertTrue(self.window.stop_button.isVisible())
                self.assertFalse(self.window.stop_button.isEnabled())
                self.assertEqual(self.window.stop_button.text(), "Stopping…")
                self.assertTrue(self.window.progress_bar.isVisible())
            finally:
                release.set()
            self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)

        self.assertFalse(self.window.stop_button.isVisible())
        self.assertFalse(self.window.progress_bar.isVisible())
        self.assertTrue(self.window.start_autonomous_button.isEnabled())
        self.assertEqual(self.window._stage, "cancelled")

    def test_one_click_reports_optional_planning_unavailable_after_preflight(self):
        import NEAT.ui.agent_planning_worker as worker_module
        from NEAT.services.agent_planning import PlanningDependencyUnavailable

        with patch.object(
            worker_module,
            "run_planning",
            side_effect=PlanningDependencyUnavailable("optional extra missing"),
        ):
            self.window.start_autonomous_analysis()
            self._wait_for(
            lambda: self.window._planning_outcome is not None
            and self.window._planning_worker is None
            and self.window._worker is None
            and not self.host._shutdown_worker_inventory.workers
            )
        self.assertEqual(self.window._planning_outcome.status, PlanningStatus.UNAVAILABLE)
        self.assertIsNotNone(self.window.result)
        self.assertIn("optional [agent] extra", self.window.status_label.text())
        self.assertFalse(self.window._planning_outcome.steps)

    @unittest.skipUnless(find_spec("langgraph") is not None, "requires the optional NEAT [agent] extra")
    def test_dataset_change_during_planning_discards_evidence_and_plan(self):
        import NEAT.ui.agent_planning_worker as worker_module
        from NEAT.services.agent_planning import PlanningCancelled

        entered = threading.Event()
        release = threading.Event()

        def paused_planning(context, *, cancelled=None, **kwargs):
            entered.set()
            release.wait(5)
            if cancelled is not None and cancelled():
                raise PlanningCancelled("dataset changed")

        with patch.object(worker_module, "run_planning", side_effect=paused_planning):
            self.window.start_autonomous_analysis()
            self._wait_for(entered.is_set, timeout=6)
            self.host.scientific_observation_api.invalidate_dataset(replace_dataset=True)
            self.host.images = [np.zeros((6, 6))]
            self.host.wavelengths = np.array([2.0])
            self.window._check_revision()
            self.assertIn("Dataset changed — rerun autonomous analysis", self.window.status_label.text())
            release.set()
            self._wait_for(lambda: self.window._planning_worker is None)
            self.assertIsNone(self.window.result)
            self.assertIsNone(self.window._planning_outcome)
            self.assertIn("Dataset changed", self.window.plan_text.toPlainText())

    @unittest.skipUnless(find_spec("langgraph") is not None, "requires the optional NEAT [agent] extra")
    def test_main_close_during_planning_requests_stop_and_waits_for_retirement(self):
        import NEAT.ui.agent_planning_worker as worker_module
        from NEAT.services.agent_planning import PlanningCancelled

        entered = threading.Event()
        release = threading.Event()

        def paused_planning(context, *, cancelled=None, **kwargs):
            entered.set()
            release.wait(5)
            if cancelled is not None and cancelled():
                raise PlanningCancelled("shutdown")

        with patch.object(worker_module, "run_planning", side_effect=paused_planning):
            self.window.start_autonomous_analysis()
            self._wait_for(entered.is_set, timeout=6)
            worker = self.window._planning_worker
            self.assertIsNotNone(worker)
            self.assertFalse(self.host.close())
            self.assertTrue(self.host.isVisible())
            self.assertIn(worker, self.host._shutdown_worker_inventory.workers)
            release.set()
            self._wait_for(lambda: not self.host._shutdown_worker_inventory.workers)
            self._wait_for(lambda: not self.host._shutdown_close_pending)
            self.assertTrue(self.host.close())
            self.assertFalse(self.window.isVisible())


if __name__ == "__main__":
    unittest.main()

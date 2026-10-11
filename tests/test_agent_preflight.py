"""Local agent-window preflight and Qt worker lifetime regressions."""

from __future__ import annotations

import gc
import os
import threading
import time
import unittest
import weakref
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QEventLoop, Qt, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QWidget,
)

from NEAT.domain.agent_preflight import PreflightReadiness, representative_frame_indices
from NEAT.domain.observation import DatasetMode
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


class AgentPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.host = _Host()
        self.window = AgentWindow(self.host)
        self.host.agent_window = self.window
        self.window.show()

    def tearDown(self):
        worker = self.window._worker
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
        self.assertNotIn("C:/private", self.window.assessment_text.toPlainText())
        self.assertIn("Companion metadata", self.window.assessment_text.toPlainText())
        self.assertIs(self.host.images, old_images)
        self.assertEqual(self.host.selected_area, old_roi)
        self.assertEqual(self.host.scientific_observation_api.revision_clock.current(), old_revision)
        self._wait_for(lambda: self.window._worker is None)

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

    def test_overview_and_visual_evidence_are_visible_together(self):
        self.assertIsInstance(self.window.centralWidget(), QSplitter)
        self.assertEqual(self.window.splitter.orientation(), Qt.Horizontal)
        self.assertEqual(self.window.splitter.count(), 2)
        self.assertTrue(self.window.overview_panel.isVisible())
        self.assertTrue(self.window.evidence_panel.isVisible())
        self.assertTrue(self.window.assessment_text.isVisible())
        self.assertTrue(self.window.evidence_scroll.isVisible())
        self.assertGreater(self.window.splitter.sizes()[0], 0)
        self.assertGreater(self.window.splitter.sizes()[1], 0)

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
        self.host.scientific_observation_api.invalidate_dataset(replace_dataset=True)
        self.host.images = [np.ones((6, 6)), np.ones((6, 6))]
        self.host.wavelengths = np.array([2.0, 2.0])
        self.window.start_preflight()
        self._wait_for(lambda: self.window.result is not None and self.window.result.dataset.mode is DatasetMode.IMAGES)
        self.assertEqual(self.window.result.assessment.readiness, PreflightReadiness.LIMITED)

    def test_loader_busy_returns_retry_without_worker(self):
        self.host.fits_image_load_worker = Mock(isRunning=Mock(return_value=True))
        self.window.start_preflight()
        self.assertIsNone(self.window._worker)
        self.assertIsNone(self.window.result)
        self.assertIn("loader is busy", self.window.status_label.text())

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

        self.host.scientific_observation_api.invalidate_dataset(replace_dataset=True)
        self.host.fitting_data_source = "profile"
        self.host.wavelengths = np.array([1.0, 2.0, 3.0, 4.0])
        self.host.intensities = np.array([2.0, 4.0, 6.0, 8.0])
        self.window.start_preflight()
        self._wait_for(lambda: self.window.result is not None and self.window.result.dataset.mode is DatasetMode.PROFILE)
        self.assertEqual(self.window.result.frames, ())
        self.assertIsNone(self.window.result.roi)
        self.assertIsNotNone(self.window.result.spectrum)
        self.assertIn("spectrum-only", self.window.assessment_text.toPlainText().lower())

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
            self.assertFalse(self.window.start_button.isEnabled())
            active_worker = self.window._worker
            self.window.start_preflight()
            self.assertIs(self.window._worker, active_worker)
            self.window.stop_preflight()
            release.set()
            self._wait_for(lambda: self.window._worker is None)
            self.assertIsNone(self.window.result)
            self.assertIn("cancelled", self.window.status_label.text().lower())

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


if __name__ == "__main__":
    unittest.main()

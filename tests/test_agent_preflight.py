"""Local agent-window preflight and Qt worker lifetime regressions."""

from __future__ import annotations

import gc
import os
import threading
import time
import unittest
import weakref
from importlib.util import find_spec
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Preserve the application import order: ONNX Runtime is initialized before Qt.
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

import NEAT  # noqa: F401
from NEAT.domain.agent_model import (
    AgentModelMode,
    AgentModelSettings,
    PlanningObjective,
)
from NEAT.domain.agent_planning import PlanningOutcome, PlanningStatus
from NEAT.domain.agent_preflight import PreflightReadiness, representative_frame_indices
from NEAT.domain.agent_vision import VisionStatus
from NEAT.domain.observation import DatasetMode
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


class AgentPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.host = _Host()
        self.window = AgentWindow(self.host)
        # Keep these tests independent of a developer's saved Agent provider.
        self.window._agent_settings = AgentModelSettings()
        self.window._update_model_readiness()
        self.host.agent_window = self.window
        self.window.show()

    def tearDown(self):
        for worker in (self.window._worker, self.window._planning_worker, self.window._vision_worker):
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

    def test_overview_and_visual_evidence_are_visible_together(self):
        self.assertIsInstance(self.window.centralWidget(), QSplitter)
        self.assertEqual(self.window.splitter.orientation(), Qt.Horizontal)
        self.assertEqual(self.window.splitter.count(), 2)
        self.assertTrue(self.window.overview_panel.isVisible())
        self.assertTrue(self.window.evidence_panel.isVisible())
        self.assertTrue(self.window.dataset_summary_label.isVisible())
        self.assertFalse(hasattr(self.window, "assessment_text"))
        self.assertFalse(hasattr(self.window, "objective_combo"))
        self.assertFalse(hasattr(self.window, "start_button"))
        self.assertEqual(self.window.settings_button.text(), "Agent settings…")
        self.assertTrue(self.window.evidence_scroll.isVisible())
        self.assertGreater(self.window.splitter.sizes()[0], 0)
        self.assertGreater(self.window.splitter.sizes()[1], 0)

    def test_m4_provider_readiness_and_cloud_consent_are_scoped_to_one_run(self):
        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.OPENAI_CLOUD,
            model_id="",
        )
        self.window._update_model_readiness()
        self.assertTrue(self.window.cloud_consent_checkbox.isVisible())
        self.assertTrue(self.window.vision_checkbox.isVisible())
        self.assertIn("OpenAI", self.window.cloud_consent_checkbox.text())
        self.assertFalse(self.window.cloud_consent_checkbox.isEnabled())
        self.assertIn("metadata summary", self.window.cloud_consent_checkbox.text())
        self.assertIn("bounded image and spectrum previews", self.window.vision_checkbox.text())
        self.assertNotIn("Also", self.window.vision_checkbox.text())
        self.assertFalse(self.window.cloud_consent_checkbox.isChecked())
        self.assertIn("Model not configured", self.window.model_readiness_label.text())

        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.OPENAI_CLOUD,
            model_id="configured-model",
        )
        self.window._update_model_readiness()
        self.assertTrue(self.window.cloud_consent_checkbox.isEnabled())
        self.assertTrue(self.window.vision_checkbox.isEnabled())
        self.assertIn("configured-model", self.window.model_readiness_label.text())

        self.window.cloud_consent_checkbox.setChecked(True)
        snapshot = self.window._snapshot_run_configuration()
        self.assertEqual(snapshot.objective, PlanningObjective.DATASET_READINESS)
        self.assertTrue(snapshot.cloud_egress_consent)
        self.assertFalse(self.window.cloud_consent_checkbox.isChecked())

        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.LOCAL,
            model_id="changed-after-start",
            endpoint="http://127.0.0.1:11434/v1",
        )
        self.window._update_model_readiness()
        self.assertEqual(snapshot.settings.model_id, "configured-model")
        self.assertEqual(
            snapshot.settings.mode, AgentModelMode.OPENAI_CLOUD
        )
        self.assertFalse(self.window.cloud_consent_checkbox.isVisible())
        self.assertFalse(self.window.vision_checkbox.isVisible())

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
            def __init__(self, settings, *, pending_api_key=None, parent=None):
                self.settings = settings
                self.pending_api_key = pending_api_key
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
        self.assertTrue(self.window.start_autonomous_button.isEnabled())
        self.assertFalse(self.window.stop_button.isVisible())
        self.assertFalse(self.window.progress_bar.isVisible())
        self.assertEqual(self.window.splitter.count(), 2)
        self.assertTrue(self.window.overview_panel.isVisible())
        self.assertTrue(self.window.evidence_panel.isVisible())
        self.assertNotIn("Preview Next Steps", self.window.plan_text.toPlainText())
        self.assertNotIn("fitting successful", self.window.plan_text.toPlainText().lower())
        if outcome.status is PlanningStatus.READY_PLAN:
            self.assertIn(
                "Plan prepared — scientific execution is not yet implemented",
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
        self.window.vision_checkbox.setChecked(True)

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

    @unittest.skipUnless(
        find_spec("langgraph") is not None,
        "requires the optional NEAT [agent] extra",
    )
    def test_openai_uses_same_model_for_consented_planning_and_visual_review(self):
        from test_agent_vision import FakeVision
        import NEAT.ui.agent_window as agent_window_module
        import NEAT.ui.agent_vision_worker as worker_module
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
        self.window.cloud_consent_checkbox.setChecked(True)
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
        self.assertTrue(self._planning_request_was_consented())
        self.assertEqual(fake_planner.calls, 1)
        self.assertEqual(configured_for_planning, ["gpt-6-luna"])
        self.assertEqual(configured_for_vision, configured_for_planning)
        self.assertIs(self.host.images, source_images)
        self.assertEqual(self.host.selected_area, source_roi)
        self.assertTrue(self.window.adjust_roi_button.isVisible())
        self.assertIn("No fit, mapping", self.window.plan_text.toPlainText())
        self.assertIn("OpenAI visual proposal", self.window.plan_text.toPlainText())
        self.assertIn("gpt-6-luna", self.window.plan_text.toPlainText())
        self.assertFalse(self.window.stop_button.isVisible())
        self.assertFalse(self.window.progress_bar.isVisible())
        self.window.roi_edits[0].setValue(19)
        self.window.adjust_roi_button.click()
        self.assertIsNone(self.window._vision_outcome)
        self.assertIn("earlier spectrum and phase hypothesis no longer describe this ROI", self.window.plan_text.toPlainText())
        self.assertEqual(self.host.selected_area, source_roi)
        self.window.reject_visual_proposal()
        self.assertIsNone(self.window._vision_outcome)
        self.assertIn("rejected", self.window.status_label.text().lower())

    def _planning_request_was_consented(self):
        return bool(
            self.window._run_snapshot
            and self.window._run_snapshot.cloud_egress_consent
        )

    def test_stop_after_vision_computation_before_queued_completion_discards_result(self):
        from test_agent_vision import FakeVision
        import NEAT.ui.agent_vision_worker as worker_module

        self._configure_synthetic_vision_dataset()
        with (
            self._fake_agent_keyring_patch(),
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

    def test_visual_review_requires_a_configured_openai_model_before_preflight(self):
        self._configure_synthetic_vision_dataset()
        self.window._agent_settings = AgentModelSettings(
            mode=AgentModelMode.OPENAI_CLOUD, model_id=""
        )
        self.window.start_autonomous_analysis()
        self.assertIn("configure an openai model", self.window.status_label.text().lower())
        self.assertIsNone(self.window._planning_worker)
        self.assertIsNone(self.window._vision_worker)

    def test_parent_shutdown_retires_active_visual_worker(self):
        from test_agent_vision import FakeVision
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
        self.window.cloud_consent_checkbox.setChecked(True)
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

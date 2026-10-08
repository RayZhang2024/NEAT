"""Real-Qt regressions for FitsViewer's two-phase application shutdown."""

from __future__ import annotations

import os
import threading
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QEventLoop, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QSlider, QTextEdit

from NEAT.ui import main_window as main_window_module
from NEAT.ui.assistant_panel import AssistantDockWidget
from NEAT.ui.main_window import FitsViewer
from NEAT.ui.preprocessing_worker_registry import PreprocessingWorkerRegistry
from NEAT.ui.assistant_settings_dialog import AssistantSettingsDialog
from tools.assistant_providers import AssistantSettings


class _ControlledThread(QThread):
    payload = pyqtSignal(str)

    def __init__(self, *, stop_raises=False, stop_available=True):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.stop_calls = 0
        self.stop_raises = stop_raises
        self.stop_available = stop_available

    def run(self):
        self.entered.set()
        self.release.wait(4)

    def stop(self):
        if not self.stop_available:
            raise AttributeError("stop is intentionally unavailable")
        self.stop_calls += 1
        if self.stop_raises:
            raise RuntimeError("controlled stop failure")


class _EarlyFinishedThread(_ControlledThread):
    # Matches the public result-bearing signal used by both batch fit workers.
    finished = pyqtSignal(str)

    def publish_finished_early(self):
        self.finished.emit("partial.csv")


class _NonblockingLegacyClearThread(_ControlledThread):
    def wait(self, msecs=-1):
        if msecs == 3000:
            return False
        return super().wait(msecs)


class _PreprocessingThread(QThread):
    finished = pyqtSignal()

    def __init__(self, *, emit_completion=True, delayed_start=False):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.emit_completion = emit_completion
        self.delayed_start = delayed_start
        self.stop_calls = 0

    def start(self, *args, **kwargs):
        if self.delayed_start:
            return
        return super().start(*args, **kwargs)

    def run(self):
        self.entered.set()
        self.release.wait(4)
        if self.emit_completion:
            self.finished.emit()

    def stop(self):
        self.stop_calls += 1


class _Assistant:
    def __init__(self, ready=True):
        self.ready = ready
        self.calls = 0
        self.worker = None

    def shutdown(self):
        self.calls += 1
        return self.ready


class _MemorySettingsRepository:
    def load(self):
        return AssistantSettings()

    def save(self, settings):
        return None


class _MemoryCredentialStore:
    def get(self, _provider):
        return None

    def set(self, _provider, _credential):
        return None

    def delete(self, _provider):
        return None


class _ShutdownHarness(FitsViewer):
    """Small main-window host using production shutdown helpers and closeEvent."""

    def __init__(self, assistant=None):
        QMainWindow.__init__(self)
        self._initialize_shutdown_lifecycle()
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.assistant_dock = assistant or _Assistant()
        self.preproc_message_box = QTextEdit(self)
        self.message_box = QTextEdit(self)
        self._shutdown_worker_inventory.diagnostic.connect(
            self.preproc_message_box.append
        )
        self._preprocessing_worker_registry.diagnostic.connect(
            self.preproc_message_box.append
        )
        self.image_slider = QSlider(self)
        self.images = [np.ones((2, 2))]
        self.intensities = np.array([4.0])
        self.wavelengths = np.array([1.0])
        self.tof_array = np.array([3.0])
        self.manual_wavelength_mode = True
        self._update_check_worker = None
        self._update_check_manual = False
        self.app_version = "4.8.2"
        self.check_updates_on_startup = True
        self.last_update_check_utc = ""
        self.ignored_update_version = ""
        self._startup_update_timer = QTimer(self)
        self.display_image = Mock()
        self.save_user_settings = Mock()
        self._hide_load_progress_dialog = Mock()
        self._set_fits_image_button_states = Mock()
        self.show()


class ApplicationShutdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _wait_for(self, predicate, timeout=3.5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents(QEventLoop.AllEvents, 10)
            if predicate():
                return
            QTest.qWait(3)
        self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertTrue(predicate(), "condition did not settle before timeout")

    def _start_tracked(self, window, worker, role, attribute=None):
        if attribute:
            setattr(window, attribute, worker)
        window._track_shutdown_worker(worker, role)
        worker.start()
        self.assertTrue(worker.entered.wait(1))

    def test_inactive_close_commits_cleanup_and_settings_once(self):
        window = _ShutdownHarness()
        window.images = [np.full((2, 2), 7)]
        self.assertTrue(window.close())
        self.assertFalse(window.isVisible())
        window.save_user_settings.assert_called_once_with()
        window.display_image.assert_called_once_with()
        self.assertEqual(window.images, [])
        self.assertFalse(window._shutdown_close_pending)
        window.close()
        window.save_user_settings.assert_called_once_with()

    def test_active_legacy_worker_close_is_nonblocking_idempotent_and_retryable(self):
        window = _ShutdownHarness()
        original_images = window.images
        worker = _ControlledThread()
        self._start_tracked(window, worker, "batch_fit", "batch_fit_worker")

        started_at = time.monotonic()
        self.assertFalse(window.close())
        self.assertLess(time.monotonic() - started_at, 0.5)
        self.assertTrue(window.isVisible())
        self.assertIs(window.batch_fit_worker, worker)
        self.assertEqual(worker.stop_calls, 1)
        window.save_user_settings.assert_not_called()
        window.display_image.assert_not_called()
        self.assertIs(window.images, original_images)

        self.assertFalse(window.close())
        self.assertEqual(worker.stop_calls, 1)
        self.assertEqual(
            window.preproc_message_box.toPlainText().count("Processing is stopping"),
            1,
        )
        worker.release.set()
        self._wait_for(lambda: not window._shutdown_worker_inventory.workers)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertTrue(window.isVisible(), "drain must not auto-close the window")
        self.assertTrue(window.close())
        self.assertFalse(window.isVisible())
        window.save_user_settings.assert_called_once_with()
        window.display_image.assert_called_once_with()

    def test_early_public_batch_finished_does_not_prove_native_exit(self):
        window = _ShutdownHarness()
        worker = _EarlyFinishedThread()
        self._start_tracked(window, worker, "batch_fit", "batch_fit_worker")
        worker.publish_finished_early()
        self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertTrue(worker.isRunning())
        self.assertFalse(window.close())
        self.assertIn(worker, window._shutdown_worker_inventory.workers)
        self.assertEqual(worker.stop_calls, 1)
        self.assertTrue(window.isVisible())
        worker.release.set()
        self._wait_for(lambda: not window._shutdown_worker_inventory.workers)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertTrue(window.close())

    def test_superseded_loader_instances_remain_owned_and_identity_safe(self):
        window = _ShutdownHarness()
        old = _ControlledThread()
        new = _ControlledThread()
        seen = []
        window._start if hasattr(window, "_start") else None
        window.fits_image_load_worker = old
        window._track_shutdown_worker(old, "fits_image_loader")
        window._connect_shutdown_worker_signal(
            old,
            old.payload,
            seen.append,
            current_attribute="fits_image_load_worker",
        )
        old.start()
        self.assertTrue(old.entered.wait(1))
        window._suppress_shutdown_worker_callbacks(old)

        window.fits_image_load_worker = new
        window._track_shutdown_worker(new, "fits_image_loader")
        new.start()
        self.assertTrue(new.entered.wait(1))
        old.payload.emit("stale")
        self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertFalse(window.close())
        self.assertEqual(seen, [])
        self.assertEqual(set(window._shutdown_worker_inventory.workers), {old, new})

        old.release.set()
        self._wait_for(lambda: old not in window._shutdown_worker_inventory.workers)
        self.assertIs(window.fits_image_load_worker, new)
        new.release.set()
        self._wait_for(lambda: not window._shutdown_worker_inventory.workers)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertIsNone(window.fits_image_load_worker)
        self.assertTrue(window.close())

    def test_legacy_clear_and_profile_import_keep_superseded_loader_owned(self):
        for clear_method in (
            "clear_loaded_fits_images",
            "_clear_loaded_images_for_profile_import",
        ):
            with self.subTest(clear_method=clear_method):
                window = _ShutdownHarness()
                window._clear_image_canvas = Mock()
                window._clear_fitting_canvases = Mock()
                window._fits_progress_dialog = None
                if clear_method == "clear_loaded_fits_images":
                    from PyQt5.QtWidgets import QTableWidget

                    window.bragg_table = QTableWidget()
                worker = _NonblockingLegacyClearThread()
                self._start_tracked(
                    window,
                    worker,
                    "fits_image_loader",
                    "fits_image_load_worker",
                )
                window._connect_shutdown_worker_signal(
                    worker,
                    worker.payload,
                    lambda _value: self.fail("stale loader payload delivered"),
                    current_attribute="fits_image_load_worker",
                )

                getattr(window, clear_method)()
                self.assertIsNone(window.fits_image_load_worker)
                self.assertIn(worker, window._shutdown_worker_inventory.workers)
                self.assertEqual(worker.stop_calls, 1)
                self.assertFalse(window.close())
                self.assertEqual(worker.stop_calls, 1)
                worker.payload.emit("stale")
                self.app.processEvents(QEventLoop.AllEvents, 10)
                worker.release.set()
                self._wait_for(lambda: not window._shutdown_worker_inventory.workers)
                self._wait_for(lambda: not window._shutdown_close_pending)
                self.assertTrue(window.close())

    def test_batch_fit_families_and_loader_roles_are_all_shutdown_managed(self):
        for role, attribute in (
            ("fits_image_loader", "fits_image_load_worker"),
            ("batch_fit", "batch_fit_worker"),
            ("batch_fit_edges", "batch_fit_edges_worker"),
        ):
            with self.subTest(role=role):
                window = _ShutdownHarness()
                worker = _ControlledThread()
                self._start_tracked(window, worker, role, attribute)
                self.assertFalse(window.close())
                self.assertIn(worker, window._shutdown_worker_inventory.workers)
                worker.release.set()
                self._wait_for(lambda: not window._shutdown_worker_inventory.workers)
                self._wait_for(lambda: not window._shutdown_close_pending)
                self.assertTrue(window.close())

    def test_registry_worker_stop_and_retry_uses_generation_quiescence(self):
        window = _ShutdownHarness()
        generation = window._preprocessing_worker_registry.begin("clean")
        worker = _PreprocessingThread()
        window._preprocessing_worker_registry.start_worker(worker, generation)
        self.assertTrue(worker.entered.wait(1))
        self.assertFalse(window.close())
        self.assertIn(worker, window._preprocessing_worker_registry.workers)
        self.assertIn("clean", window._preprocessing_worker_registry.active_families)
        self.assertEqual(worker.stop_calls, 1)
        self.assertTrue(window.isVisible())
        worker.release.set()
        self._wait_for(lambda: not window._preprocessing_worker_registry.workers)
        self._wait_for(lambda: not window._preprocessing_worker_registry.active_families)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertTrue(window.close())

    def test_close_cancels_every_preprocessing_generation_and_suppresses_stages(self):
        window = _ShutdownHarness()
        families = (
            "summation",
            "clean",
            "overlap",
            "normalisation",
            "filtering",
            "full_process",
            "normalisation_open_beam_12",
        )
        workers = []
        callbacks = []
        for family in families:
            generation = window._preprocessing_worker_registry.begin(family)
            worker = _PreprocessingThread()
            workers.append(worker)
            window._preprocessing_worker_registry.start_worker(
                worker,
                generation,
                handlers={"message": lambda *_args: callbacks.append("message")},
                completion=lambda: callbacks.append("completion"),
            )
        self.assertTrue(all(worker.entered.wait(1) for worker in workers))

        self.assertFalse(window.close())
        self.assertEqual(
            set(window._preprocessing_worker_registry.active_families),
            set(families),
        )
        self.assertEqual(
            set(window._preprocessing_worker_registry.workers),
            set(workers),
        )
        for worker in workers:
            worker.release.set()
        self._wait_for(lambda: not window._preprocessing_worker_registry.active_families)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertEqual(callbacks, [])
        self.assertTrue(window.isVisible())
        self.assertTrue(window.close())

    def test_public_completion_before_native_exit_keeps_preprocessing_close_veto(self):
        window = _ShutdownHarness()
        generation = window._preprocessing_worker_registry.begin("overlap")
        worker = _PreprocessingThread()
        window._preprocessing_worker_registry.start_worker(worker, generation)
        self.assertTrue(worker.entered.wait(1))
        worker.finished.emit()
        self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertTrue(worker.isRunning())
        self.assertFalse(window.close())
        self.assertIn(worker, window._preprocessing_worker_registry.workers)
        worker.release.set()
        self._wait_for(lambda: not window._preprocessing_worker_registry.workers)
        self._wait_for(lambda: not window._preprocessing_worker_registry.active_families)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertTrue(window.close())

    def test_missing_preprocessing_completion_grace_still_blocks_close(self):
        window = _ShutdownHarness()
        generation = window._preprocessing_worker_registry.begin("filtering")
        worker = _PreprocessingThread(emit_completion=False)
        window._preprocessing_worker_registry.start_worker(worker, generation)
        self.assertTrue(worker.entered.wait(1))
        self.assertFalse(window.close())
        worker.release.set()
        self._wait_for(lambda: not worker.isRunning())
        self.app.processEvents(QEventLoop.AllEvents, 20)
        self.assertFalse(window.close())
        self.assertIn("filtering", window._preprocessing_worker_registry.active_families)
        self._wait_for(lambda: not window._preprocessing_worker_registry.active_families)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertTrue(window.close())

    def test_delayed_ambiguous_start_is_retained_after_watchdog(self):
        window = _ShutdownHarness()
        generation = window._preprocessing_worker_registry.begin("normalisation")
        worker = _PreprocessingThread(delayed_start=True)
        diagnostics = []
        window._preprocessing_worker_registry.diagnostic.connect(diagnostics.append)
        window._preprocessing_worker_registry.start_worker(worker, generation)
        self._wait_for(lambda: bool(diagnostics), timeout=1.5)
        self.assertTrue(any("retaining ownership" in item for item in diagnostics))
        self.assertIn(worker, window._preprocessing_worker_registry.workers)
        self.assertIn("normalisation", window._preprocessing_worker_registry.active_families)
        self.assertFalse(window.close())
        self.assertIn(worker, window._preprocessing_worker_registry.workers)
        self.assertTrue(window.isVisible())
        worker.delayed_start = False
        worker.start()
        self.assertTrue(worker.entered.wait(1))
        worker.release.set()
        self._wait_for(lambda: not window._preprocessing_worker_registry.active_families)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertTrue(window.close())

    def test_zero_worker_and_already_closing_generations_cancel_and_drain(self):
        window = _ShutdownHarness()
        empty = window._preprocessing_worker_registry.begin("empty")
        self.assertFalse(window.close())
        self.assertNotIn(empty.family, window._preprocessing_worker_registry.active_families)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertTrue(window.isVisible())

        closing = window._preprocessing_worker_registry.begin("already_closing")
        window._preprocessing_worker_registry.complete_generation(closing)
        self.assertNotIn(closing.family, window._preprocessing_worker_registry.active_families)
        self.assertTrue(window.close())

    def test_closing_generation_with_worker_is_cancelled_not_assumed_drained(self):
        window = _ShutdownHarness()
        generation = window._preprocessing_worker_registry.begin("full_process")
        worker = _PreprocessingThread()
        window._preprocessing_worker_registry.start_worker(worker, generation)
        self.assertTrue(worker.entered.wait(1))
        window._preprocessing_worker_registry.complete_generation(generation)
        self.assertIn("full_process", window._preprocessing_worker_registry.active_families)
        self.assertFalse(window.close())
        self.assertTrue(window._preprocessing_worker_registry.is_cancelled(generation))
        self.assertIn(worker, window._preprocessing_worker_registry.workers)
        worker.release.set()
        self._wait_for(lambda: not window._preprocessing_worker_registry.active_families)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertTrue(window.close())

    def test_stop_exception_is_diagnostic_and_not_retried_or_treated_as_exit(self):
        window = _ShutdownHarness()
        worker = _ControlledThread(stop_raises=True)
        self._start_tracked(window, worker, "fits_image_loader", "fits_image_load_worker")
        self.assertFalse(window.close())
        self.assertEqual(worker.stop_calls, 1)
        self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertIn("controlled stop failure", window.preproc_message_box.toPlainText())
        self.assertIn(worker, window._shutdown_worker_inventory.workers)
        self.assertFalse(window.close())
        self.assertEqual(worker.stop_calls, 1)
        worker.release.set()
        self._wait_for(lambda: not window._shutdown_worker_inventory.workers)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertTrue(window.close())

    def test_noncooperative_open_beam_worker_is_retained_until_native_exit(self):
        window = _ShutdownHarness()
        generation = window._preprocessing_worker_registry.begin("normalisation_open_beam_7")

        class OpenBeamThread(_PreprocessingThread):
            stop = None

        worker = OpenBeamThread()
        window._preprocessing_worker_registry.start_worker(worker, generation)
        self.assertTrue(worker.entered.wait(1))
        self.assertFalse(window.close())
        self.assertIn(worker, window._preprocessing_worker_registry.workers)
        worker.release.set()
        self._wait_for(lambda: not window._preprocessing_worker_registry.active_families)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertTrue(window.close())

    def test_close_veto_precedes_assistant_and_assistant_only_veto_is_preserved(self):
        assistant = _Assistant(ready=False)
        window = _ShutdownHarness(assistant)
        generation = window._preprocessing_worker_registry.begin("summation")
        self.assertFalse(window.close())
        self.assertEqual(assistant.calls, 0)
        self.assertNotIn(generation.family, window._preprocessing_worker_registry.active_families)
        self._wait_for(lambda: not window._shutdown_close_pending)

        with patch.object(main_window_module.QMessageBox, "information") as information:
            self.assertFalse(window.close())
        self.assertEqual(assistant.calls, 1)
        information.assert_called_once()
        window.save_user_settings.assert_not_called()
        window.display_image.assert_not_called()
        assistant.ready = True
        self.assertTrue(window.close())

    def test_assistant_real_thread_keeps_1500ms_shutdown_veto_contract(self):
        dock = AssistantDockWidget(context_provider=lambda: {})

        class SlowAssistantThread(QThread):
            def __init__(self):
                super().__init__()
                self.release = threading.Event()

            def run(self):
                self.release.wait(3)

        worker = SlowAssistantThread()
        dock.worker = worker
        worker.start()
        self.assertTrue(worker.isRunning())
        self.assertFalse(dock.shutdown())
        self.assertTrue(worker.isRunning())
        self.assertTrue(worker.isInterruptionRequested())

        worker.release.set()
        self.assertTrue(worker.wait(1000))
        self.assertTrue(dock.shutdown())
        self.assertIsNone(dock.worker)
        self.assertNotIn(id(worker), dock._retiring_workers)
        dock.close()

    def test_cleanup_resources_cannot_bypass_live_worker_guard(self):
        window = _ShutdownHarness()
        original_images = window.images
        worker = _ControlledThread()
        self._start_tracked(window, worker, "batch_fit")
        self.assertFalse(window.cleanup_resources())
        self.assertIs(window.images, original_images)
        window.display_image.assert_not_called()
        window.save_user_settings.assert_not_called()
        worker.release.set()
        self._wait_for(lambda: not window._shutdown_worker_inventory.workers)
        self.assertTrue(window.cleanup_resources())
        self.assertEqual(window.images, [])
        window.close()

    def test_rejected_close_preserves_data_and_allows_fresh_work(self):
        window = _ShutdownHarness()
        original_images = window.images
        generation = window._preprocessing_worker_registry.begin("clean")
        self.assertFalse(window.close())
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertIs(window.images, original_images)
        self.assertTrue(window.isVisible())
        new_generation = window._preprocessing_worker_registry.begin("summation")
        self.assertIsNot(new_generation, generation)
        self.assertFalse(window.close())
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertIs(window.images, original_images)
        self.assertTrue(window.close())

    def test_update_completion_clearing_attribute_cannot_hide_live_thread(self):
        window = _ShutdownHarness()
        dialogs = []

        class UpdateThread(QThread):
            check_finished = pyqtSignal(dict)

            def __init__(self, current_version, api_url=None, parent=None):
                super().__init__(parent)
                self.entered = threading.Event()
                self.release = threading.Event()

            def run(self):
                self.entered.set()
                self.release.wait(4)

        with patch.object(main_window_module, "UpdateCheckWorker", UpdateThread), patch.object(
            window, "_show_update_available_dialog", lambda *args: dialogs.append(args)
        ):
            window.start_update_check(manual=False)
            worker = window._update_check_worker
            self.assertTrue(worker.entered.wait(1))
            worker.check_finished.emit(
                {"ok": True, "update_available": False, "latest_version": "4.8.2"}
            )
            self._wait_for(lambda: window._update_check_worker is None)
            save_calls_before_close = window.save_user_settings.call_count
            self.assertFalse(window.close())
            self.assertIn(worker, window._shutdown_worker_inventory.workers)
            worker.check_finished.emit(
                {
                    "ok": True,
                    "update_available": True,
                    "latest_version": "9.0",
                    "html_url": "https://example.invalid",
                }
            )
            self.app.processEvents(QEventLoop.AllEvents, 10)
            self.assertEqual(dialogs, [])
            self.assertEqual(window.save_user_settings.call_count, save_calls_before_close)
            worker.release.set()
            self._wait_for(lambda: not window._shutdown_worker_inventory.workers)
            self._wait_for(lambda: not window._shutdown_close_pending)
            self.assertTrue(window.close())
            self.assertEqual(
                window.save_user_settings.call_count,
                save_calls_before_close + 1,
            )

    def test_deferred_update_check_cannot_start_after_committed_close(self):
        window = _ShutdownHarness()
        window.check_updates_on_startup = True
        with patch.object(main_window_module, "UpdateCheckWorker") as worker_class:
            self.assertTrue(window.close())
            window.maybe_check_for_updates_on_startup()
            window.start_update_check(manual=True)
        worker_class.assert_not_called()

    def test_parent_close_is_blocked_by_child_dialog_worker_guard(self):
        window = _ShutdownHarness()

        class BusyDialog(QDialog):
            has_unsettled_workers = True

        dialog = BusyDialog(window)
        dialog.show()
        self.assertFalse(window.close())
        self.assertTrue(window.isVisible())
        self.assertTrue(dialog.isVisible())
        window.save_user_settings.assert_not_called()
        dialog.has_unsettled_workers = False
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertTrue(window.close())

    def test_real_assistant_settings_dialog_keeps_parent_close_safe(self):
        window = _ShutdownHarness()
        store = _MemoryCredentialStore()
        dialog = AssistantSettingsDialog(
            window,
            settings_repository=_MemorySettingsRepository(),
            credential_store=store,
            environment_store=store,
            auto_discover_local_models=False,
            shared_service_available=False,
        )
        dialog.show()
        worker = _ControlledThread()
        dialog.test_worker = worker
        dialog._track_dialog_worker(worker)
        worker.start()
        self.assertTrue(worker.entered.wait(1))

        self.assertFalse(dialog.close())
        self.assertFalse(window.close())
        self.assertTrue(dialog.isVisible())
        self.assertTrue(window.isVisible())
        worker.release.set()
        self._wait_for(lambda: not dialog.has_unsettled_workers)
        self._wait_for(lambda: not window._shutdown_close_pending)
        self.assertIsNone(dialog.test_worker)
        self.assertTrue(dialog.close())
        self.assertTrue(window.close())


if __name__ == "__main__":
    unittest.main()

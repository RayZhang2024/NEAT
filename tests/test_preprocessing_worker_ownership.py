"""Real-QThread regressions for GUI preprocessing ownership and retirement."""

import os
import time
import unittest
import ast
import inspect
import weakref
import gc
from threading import Event
from unittest.mock import patch
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEventLoop, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QLineEdit, QPushButton, QTextEdit, QWidget
import numpy as np

from NEAT.ui.preprocessing_worker_registry import (
    PreprocessingWorkerRegistry,
    PreprocessingWorkerStartRejected,
)
from NEAT.ui.mixins.preprocessing import PreprocessingMixin
from NEAT.ui.mixins import preprocessing as preprocessing_module


class _ControlledWorker(QThread):
    finished = pyqtSignal()
    message = pyqtSignal(str)
    run_loaded = pyqtSignal(str, dict)

    def __init__(self, *, emit_finished=True, block_after_signal=False):
        super().__init__()
        self.emit_finished = emit_finished
        self.block_after_signal = block_after_signal
        self.entered = Event()
        self.release = Event()
        self.stop_requested = False

    def run(self):
        self.entered.set()
        if self.block_after_signal:
            self.finished.emit()
            self.release.wait(2)
        elif self.emit_finished:
            self.finished.emit()

    def stop(self):
        self.stop_requested = True
        self.release.set()


class _RejectedWorker(_ControlledWorker):
    def start(self, *args, **kwargs):
        raise PreprocessingWorkerStartRejected("synthetic QThread start rejection")


class _DelayedStartWorker(_ControlledWorker):
    def start(self, *args, **kwargs):
        # Model Qt accepting a start request but not scheduling the native
        # thread until after the registry's initial startup grace has expired.
        QTimer.singleShot(650, lambda: QThread.start(self))


class _RaiseAfterStartingWorker(_ControlledWorker):
    def start(self, *args, **kwargs):
        QThread.start(self)
        raise RuntimeError("wrapper raised after delegating to QThread.start")


class _NonCooperativeWorker(QThread):
    finished = pyqtSignal()
    open_beam_loaded = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def run(self):
        self.entered.set()
        self.release.wait(2)
        self.finished.emit()


class _PayloadWorker(QThread):
    finished = pyqtSignal()
    run_loaded = pyqtSignal(str, dict)

    def __init__(self, *, emit_finished=True):
        super().__init__()
        self.emit_finished = emit_finished

    def run(self):
        self.run_loaded.emit("sample", {"frame": object()})
        if self.emit_finished:
            self.finished.emit()


class _HeldWorkflowWorker(QThread):
    finished = pyqtSignal()
    run_loaded = pyqtSignal(str, dict)
    progress_updated = pyqtSignal(int)
    message = pyqtSignal(str)

    instances = []

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.entered = Event()
        self.release = Event()
        self.stop_requested = False
        type(self).instances.append(self)

    def run(self):
        self.entered.set()
        self.release.wait(3)
        self.finished.emit()

    def stop(self):
        self.stop_requested = True
        self.release.set()


class _AutoLoader(QThread):
    finished = pyqtSignal()
    run_loaded = pyqtSignal(str, dict)

    folders = []

    def __init__(self, folder, *_args, **_kwargs):
        super().__init__()
        self.folder = folder
        type(self).folders.append(folder)

    def run(self):
        self.run_loaded.emit(self.folder, {"0001": object()})
        self.finished.emit()


class _AutoServiceWorker(QThread):
    finished = pyqtSignal()

    instances = []

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        type(self).instances.append(self)

    def run(self):
        self.finished.emit()


class _PreprocessingHarness(QWidget, PreprocessingMixin):
    def __init__(self):
        super().__init__()
        self._preprocessing_worker_registry = PreprocessingWorkerRegistry(self)
        self._preprocessing_workflow_generations = {}
        self.preproc_message_box = QTextEdit(self)
        for name in (
            "summation_sum_button", "summation_stop_button",
            "outlier_process_button", "outlier_stop_button",
            "overlap_correction_correct_button", "overlap_correction_stop_button",
            "normalisation_normalise_button", "normalisation_stop_button",
            "filtering_filter_button", "filtering_stop_button",
            "full_process_start_button", "full_process_stop_button",
        ):
            setattr(self, name, QPushButton(self))
            getattr(self, name).setEnabled(True)
        self._summation_cancelled = False
        self.outlier_output_input = QLineEdit(self)
        self.outlier_basename_input = QLineEdit("clean", self)
        self.overlap_correction_output_input = QLineEdit(self)
        self.overlap_correction_basename_input = QLineEdit("overlap", self)
        self.normalisation_output_input = QLineEdit(self)
        self.normalisation_basename_input = QLineEdit("normalised", self)
        self.normalisation_window_half_input = QLineEdit("1", self)
        self.normalisation_adjacent_input = QLineEdit("0", self)
        self.summation_output_input = QLineEdit(self)
        self.summation_basename_input = QLineEdit("sum", self)
        self.normalisation_open_beam_runs = []
        self.normalisation_image_runs = []
        self.filtering_output_input = QLineEdit(self)
        self.filtering_basename_input = QLineEdit("filtered", self)
        self.filtering_image_runs = []
        self.filtering_mask_image = object()


class TestPreprocessingWorkerRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _wait_for(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents(QEventLoop.AllEvents, 20)
        self.assertTrue(predicate(), "condition did not become true before the test timeout")

    def _registry(self):
        registry = PreprocessingWorkerRegistry()
        diagnostics = []
        registry.diagnostic.connect(diagnostics.append)
        return registry, diagnostics

    def test_public_finished_before_native_exit_keeps_owner_and_defers_completion(self):
        registry, _ = self._registry()
        worker = _ControlledWorker(block_after_signal=True)
        completed = []
        generation = registry.begin("overlap")
        registry.start_worker(
            worker,
            generation,
            completion=lambda: (completed.append(True), registry.complete_generation(generation)),
        )
        self.assertTrue(worker.entered.wait(1))
        self._wait_for(lambda: registry.workers and worker in registry.workers)
        self.app.processEvents(QEventLoop.AllEvents, 30)
        self.assertEqual(completed, [])
        self.assertIn(worker, registry.workers)
        worker.release.set()
        self._wait_for(lambda: completed == [True])
        self.assertEqual(registry.workers, ())
        self.assertEqual(registry.active_families, ())

    def test_missing_public_completion_uses_nonblocking_grace_and_retires(self):
        registry, diagnostics = self._registry()
        worker = _ControlledWorker(emit_finished=False)
        completed = []
        drained = []
        generation = registry.begin("clean", on_drained=lambda abnormal: drained.append(abnormal))
        registry.start_worker(worker, generation, completion=lambda: completed.append(True))
        self._wait_for(lambda: not worker.isRunning() and not registry.workers, timeout=2.0)
        self.assertEqual(completed, [])
        self.assertEqual(drained, [True])
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("without completion notification", diagnostics[0])
        self.assertFalse(registry._poll_timer.isActive())

        # A completion emitted after retirement is stale and cannot re-enter.
        worker.finished.emit()
        for _ in range(10):
            self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertEqual(completed, [])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(registry.workers, ())

    def test_payload_is_handed_off_before_completion_and_discarded_without_finished(self):
        registry, diagnostics = self._registry()
        delivered = []
        completed = []
        worker = _PayloadWorker()
        generation = registry.begin("overlap")

        def on_completion():
            delivered.append("completion")
            registry.complete_generation(generation)

        registry.start_worker(
            worker,
            generation,
            handlers={"run_loaded": lambda *_args: delivered.append("payload")},
            completion=on_completion,
        )
        self._wait_for(lambda: delivered == ["payload", "completion"])
        self.assertEqual(registry.workers, ())

        no_finished = _PayloadWorker(emit_finished=False)
        generation = registry.begin("clean")
        registry.start_worker(
            no_finished,
            generation,
            handlers={"run_loaded": lambda *_args: completed.append("payload")},
            completion=lambda: completed.append("completion"),
        )
        self._wait_for(lambda: not registry.workers)
        self.assertEqual(completed, [])
        self.assertEqual(len(diagnostics), 1)

    def test_structured_result_is_consumed_before_retirement_cleanup(self):
        registry, _ = self._registry()
        worker = _ControlledWorker()
        worker.result = {"status": "SUCCEEDED", "outputs": ["partial-or-complete"]}
        observed = []
        generation = registry.begin("clean")
        registry.start_worker(
            worker,
            generation,
            completion=lambda: observed.append(worker.result),
            on_retired=lambda retired: observed.append(("retired", retired is worker)),
        )
        self._wait_for(lambda: len(observed) == 2)
        self.assertEqual(observed[0], {"status": "SUCCEEDED", "outputs": ["partial-or-complete"]})
        self.assertEqual(observed[1], ("retired", True))
        registry.complete_generation(generation)

    def test_start_failure_is_not_scientific_completion(self):
        registry, diagnostics = self._registry()
        worker = _RejectedWorker()
        completed = []
        drained = []
        generation = registry.begin("filtering", on_drained=lambda abnormal: drained.append(abnormal))
        self.assertTrue(worker.wait(0))  # Qt reports true for a never-started QThread.
        registry.start_worker(worker, generation, completion=lambda: completed.append(True))
        self._wait_for(lambda: not registry.workers, timeout=1.5)
        self.assertEqual(completed, [])
        self.assertEqual(drained, [True])
        self.assertTrue(any("failed to start" in item for item in diagnostics))

    def test_delayed_start_remains_owned_past_startup_timeout_then_retires(self):
        registry, diagnostics = self._registry()
        worker = _DelayedStartWorker(block_after_signal=True)
        worker_ref = weakref.ref(worker)
        completed = []
        generation = registry.begin("overlap")
        registry.start_worker(
            worker,
            generation,
            completion=lambda: (completed.append(True), registry.complete_generation(generation)),
        )

        # Qt reports wait(0) true for a never-started thread; this is not proof
        # that a delayed start request was rejected.
        self.assertTrue(worker.wait(0))
        self._wait_for(
            lambda: any("startup is still unacknowledged" in item for item in diagnostics),
            timeout=1.2,
        )
        self.assertIn(worker, registry.workers)
        self.assertIsNotNone(worker_ref())
        self.assertEqual(completed, [])
        self.assertIn("overlap", registry.active_families)

        self._wait_for(worker.entered.is_set, timeout=1.5)
        self.assertIn(worker, registry.workers)
        worker.release.set()
        self._wait_for(lambda: completed == [True], timeout=2.0)
        self.assertEqual(registry.workers, ())
        self.assertEqual(registry.active_families, ())
        self.assertFalse(registry._poll_timer.isActive())
        self.assertFalse(any("failed to start" in item for item in diagnostics))

    def test_generic_start_exception_after_qthread_start_keeps_worker_owned(self):
        registry, diagnostics = self._registry()
        worker = _RaiseAfterStartingWorker(block_after_signal=True)
        completed = []
        generation = registry.begin("filtering")
        registry.start_worker(worker, generation, completion=lambda: completed.append(True))
        self.assertTrue(worker.entered.wait(1))
        self.assertIn(worker, registry.workers)
        worker.release.set()
        self._wait_for(lambda: completed == [True])
        self.assertEqual(registry.workers, ())
        self.assertFalse(any("failed to start" in item for item in diagnostics))

    def test_unacknowledged_start_timeout_retains_ownership_and_reports_once(self):
        registry, diagnostics = self._registry()
        worker = _DelayedStartWorker(block_after_signal=True)
        generation = registry.begin("clean")
        registry.start_worker(worker, generation)
        self.assertTrue(worker.wait(0))
        self._wait_for(
            lambda: any("startup is still unacknowledged" in item for item in diagnostics),
            timeout=1.2,
        )
        for _ in range(15):
            self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertIn(worker, registry.workers)
        self.assertEqual(
            sum("startup is still unacknowledged" in item for item in diagnostics), 1
        )
        worker.release.set()
        self._wait_for(lambda: not registry.workers, timeout=2.0)

    def test_fast_exit_is_confirmed_without_observing_running_state(self):
        registry, _ = self._registry()
        worker = _ControlledWorker()
        completed = []
        generation = registry.begin("summation")
        registry.start_worker(worker, generation, completion=lambda: completed.append(True))
        self._wait_for(lambda: completed == [True])
        self.assertFalse(worker.isRunning())
        self.assertTrue(worker.isFinished())
        self.assertEqual(registry.workers, ())

    def test_stop_cancels_payload_and_completion_but_retains_until_exit(self):
        registry, _ = self._registry()
        worker = _ControlledWorker(block_after_signal=True)
        payloads = []
        completed = []
        drained = []
        generation = registry.begin("normalisation", on_drained=lambda abnormal: drained.append(abnormal))
        registry.start_worker(
            worker,
            generation,
            handlers={"run_loaded": lambda *args: payloads.append(args)},
            completion=lambda: completed.append(True),
        )
        self.assertTrue(worker.entered.wait(1))
        self.assertTrue(registry.cancel_family("normalisation"))
        self.assertTrue(worker.stop_requested)
        worker.run_loaded.emit("sample", {"frame": object()})
        self._wait_for(lambda: not registry.workers)
        self.assertEqual(payloads, [])
        self.assertEqual(completed, [])
        self.assertEqual(drained, [False])

    def test_family_guard_spans_modes_and_other_families_remain_independent(self):
        registry, _ = self._registry()
        summation = registry.begin("summation")
        self.assertIs(registry.begin("summation"), registry.FAMILY_BUSY)
        normalisation = registry.begin("normalisation")
        self.assertIsNot(normalisation, registry.FAMILY_BUSY)
        registry.complete_generation(summation)
        registry.complete_generation(normalisation)
        later = registry.begin("summation")
        self.assertEqual(later.number, summation.number + 1)

    def test_worker_callbacks_are_dispatched_on_the_registry_gui_thread(self):
        registry, _ = self._registry()
        worker = _ControlledWorker()
        callback_threads = []
        generation = registry.begin("clean")
        registry.start_worker(
            worker,
            generation,
            handlers={"message": lambda _value: callback_threads.append(QThread.currentThread())},
        )
        worker.message.emit("thread check")
        self._wait_for(lambda: bool(callback_threads))
        self._wait_for(lambda: not registry.workers)
        self.assertTrue(all(thread == self.app.thread() for thread in callback_threads))

    def test_preprocessing_mixins_keep_run_disabled_until_stopped_worker_retires(self):
        cases = (
            ("summation", "stop_summation", "summation_sum_button", "summation_stop_button", "summation_worker"),
            ("clean", "stop_outlier_removal", "outlier_process_button", "outlier_stop_button", "outlier_worker"),
            ("overlap", "stop_overlap_correction", "overlap_correction_correct_button", "overlap_correction_stop_button", "overlap_correction_worker"),
            ("normalisation", "stop_normalisation", "normalisation_normalise_button", "normalisation_stop_button", "normalisation_worker"),
            ("filtering", "stop_filtering", "filtering_filter_button", "filtering_stop_button", "filtering_worker"),
            ("full_process", "stop_full_process", "full_process_start_button", "full_process_stop_button", "full_process_worker"),
        )
        for family, stop_method, run_name, stop_name, worker_attr in cases:
            with self.subTest(family=family):
                harness = _PreprocessingHarness()
                worker = _ControlledWorker(block_after_signal=True)
                setattr(harness, worker_attr, worker)
                harness._begin_preprocessing_workflow(family)
                getattr(harness, run_name).setEnabled(False)
                getattr(harness, stop_name).setEnabled(True)
                harness._start_preprocessing_worker(worker, family)
                self.assertTrue(worker.entered.wait(1))
                self._wait_for(lambda: worker in harness._preprocessing_worker_registry.workers)

                getattr(harness, stop_method)()
                self.assertTrue(worker.stop_requested)
                self.assertIn(worker, harness._preprocessing_worker_registry.workers)
                self.assertFalse(getattr(harness, run_name).isEnabled())
                self.assertFalse(getattr(harness, stop_name).isEnabled())
                self.assertIsNone(harness._begin_preprocessing_workflow(family))

                self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)
                self.assertFalse(harness._preprocessing_workflow_generations)
                self.assertTrue(getattr(harness, run_name).isEnabled())
                self.assertFalse(getattr(harness, stop_name).isEnabled())
                self.assertFalse(harness._preprocessing_worker_registry._poll_timer.isActive())
                harness.close()

    def test_cancelled_workers_release_exact_convenience_references_after_exit(self):
        attributes = (
            ("summation", "summation_worker"),
            ("clean", "outlier_worker"),
            ("overlap", "overlap_correction_worker"),
            ("normalisation", "normalisation_worker"),
            ("filtering", "filtering_worker"),
            ("full_process", "full_process_worker"),
            ("clean", "_outlier_data_load_worker"),
            ("clean", "outlier_image_load_worker"),
            ("overlap", "_overlap_data_load_worker"),
            ("overlap", "overlap_correction_image_load_worker"),
            ("summation", "_lazy_load_worker_2"),
            ("summation", "_lazy_load_worker_3"),
            ("summation", "summation_image_load_worker"),
            ("normalisation", "_data_load_worker"),
            ("normalisation", "normalisation_data_load_worker"),
            ("filtering", "filtering_data_load_worker"),
            ("normalisation_open_beam_17", "open_beam_load_worker"),
        )
        for family, attribute in attributes:
            with self.subTest(attribute=attribute):
                harness = _PreprocessingHarness()
                worker = _ControlledWorker(block_after_signal=True)
                worker_ref = weakref.ref(worker)
                setattr(harness, attribute, worker)
                harness._begin_preprocessing_workflow(family)
                harness._start_preprocessing_worker(worker, family)
                self.assertTrue(worker.entered.wait(1))
                self.assertTrue(harness._preprocessing_worker_registry.cancel_family(family))
                self.assertIs(getattr(harness, attribute), worker)
                self.assertIn(worker, harness._preprocessing_worker_registry.workers)

                worker.release.set()
                self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)
                self.assertIsNone(getattr(harness, attribute))
                self.assertFalse(harness._preprocessing_worker_registry._poll_timer.isActive())
                del worker
                gc.collect()
                for _ in range(5):
                    self.app.processEvents(QEventLoop.AllEvents, 10)
                self.assertIsNone(worker_ref())
                harness.close()

    def test_stale_worker_retirement_does_not_clear_newer_convenience_reference(self):
        harness = _PreprocessingHarness()
        old_worker = _ControlledWorker(block_after_signal=True)
        newer_worker = _ControlledWorker(block_after_signal=True)
        harness._begin_preprocessing_workflow("normalisation_open_beam_1")
        harness.open_beam_load_worker = old_worker
        harness._start_preprocessing_worker(old_worker, "normalisation_open_beam_1")
        self.assertTrue(old_worker.entered.wait(1))
        harness.open_beam_load_worker = newer_worker
        old_worker.release.set()
        self._wait_for(lambda: old_worker not in harness._preprocessing_worker_registry.workers)
        self.assertIs(harness.open_beam_load_worker, newer_worker)
        self.assertEqual(harness._preprocessing_worker_registry.workers, ())
        harness.close()

    def test_stop_keeps_noncooperative_worker_attribute_until_native_exit(self):
        harness = _PreprocessingHarness()
        worker = _NonCooperativeWorker()
        harness.filtering_worker = worker
        harness._begin_preprocessing_workflow("filtering")
        harness._start_preprocessing_worker(worker, "filtering")
        self.assertTrue(worker.entered.wait(1))
        harness.stop_filtering()
        self.assertTrue(worker.isRunning())
        self.assertIs(harness.filtering_worker, worker)
        self.assertIn(worker, harness._preprocessing_worker_registry.workers)
        worker.release.set()
        self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)
        self.assertIsNone(harness.filtering_worker)
        harness.close()

    def test_family_drain_preserves_disabled_run_button_when_inputs_unavailable(self):
        harness = _PreprocessingHarness()
        harness.outlier_process_button.setEnabled(False)
        generation = harness._begin_preprocessing_workflow("clean")
        harness.outlier_process_button.setEnabled(False)
        harness._preprocessing_worker_registry.complete_generation(generation)
        self.assertFalse(harness.outlier_process_button.isEnabled())
        self.assertFalse(harness.outlier_stop_button.isEnabled())
        harness.close()

    def test_missing_finished_fallback_clears_cancelled_worker_reference(self):
        harness = _PreprocessingHarness()
        worker = _ControlledWorker(emit_finished=False)
        harness.filtering_worker = worker
        harness._begin_preprocessing_workflow("filtering")
        harness._start_preprocessing_worker(worker, "filtering")
        self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)
        self.assertIsNone(harness.filtering_worker)
        self.assertTrue(harness.filtering_filter_button.isEnabled())
        self.assertFalse(harness.filtering_stop_button.isEnabled())
        harness.close()

    def test_outlier_stop_during_loading_discards_late_partial_payload(self):
        with TemporaryDirectory() as root:
            harness = _PreprocessingHarness()
            harness._outlier_batch_paths = [root, root + "-second"]
            harness.outlier_output_input.setText(root)
            loader_instances = []

            class Loader(_HeldWorkflowWorker):
                def __init__(self, folder):
                    super().__init__(folder)
                    loader_instances.append(self)

            with patch.object(preprocessing_module, "ImageLoadWorker", Loader), patch.object(
                preprocessing_module, "OutlierFilteringWorker"
            ) as outlier_worker:
                harness.remove_outliers()
                loader = harness._outlier_data_load_worker
                self.assertTrue(loader.entered.wait(1))
                harness.stop_outlier_removal()
                self.assertIs(harness._outlier_data_load_worker, loader)
                loader.run_loaded.emit(root, {"0001": object()})
                loader.release.set()
                self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)

                outlier_worker.assert_not_called()
                self.assertEqual(len(loader_instances), 1)
                self.assertIsNone(harness._outlier_data_load_worker)
                self.assertEqual(harness._current_outlier_index, 0)
                self.assertTrue(harness.outlier_process_button.isEnabled())
                self.assertFalse(harness.outlier_stop_button.isEnabled())
                self.assertNotIn("Batch outlier removal completed.", harness.preproc_message_box.toPlainText())
            harness.close()

    def test_overlap_stop_during_active_worker_prevents_next_dataset(self):
        with TemporaryDirectory() as root:
            first = os.path.join(root, "run1")
            second = os.path.join(root, "run2")
            output = os.path.join(root, "output")
            os.makedirs(first)
            os.makedirs(second)
            os.makedirs(output)
            np.savetxt(os.path.join(first, "a_Spectra.txt"), [[1, 2], [3, 4]])
            np.savetxt(os.path.join(first, "a_ShutterCount.txt"), [[1, 2], [3, 4]])
            harness = _PreprocessingHarness()
            harness._overlap_batch_paths = [first, second]
            harness.overlap_correction_output_input.setText(output)
            loader_instances = []

            class Loader(_HeldWorkflowWorker):
                def __init__(self, folder):
                    super().__init__(folder)
                    loader_instances.append(self)

            class Service(_HeldWorkflowWorker):
                instances = []

            with patch.object(preprocessing_module, "ImageLoadWorker", Loader), patch.object(
                preprocessing_module, "OverlapCorrectionWorker", Service
            ):
                harness.correct_overlap()
                loader = harness._overlap_data_load_worker
                self.assertTrue(loader.entered.wait(1))
                loader.run_loaded.emit(first, {"0001": np.ones((2, 2))})
                loader.release.set()
                self._wait_for(
                    lambda: len(Service.instances) == 1
                    and Service.instances[0].entered.is_set()
                )
                service = Service.instances[0]
                harness.stop_overlap_correction()
                self.assertIn(service, harness._preprocessing_worker_registry.workers)
                self.assertIs(harness.overlap_correction_worker, service)
                service.release.set()
                self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)

                self.assertEqual(len(loader_instances), 1)
                self.assertEqual(harness._current_overlap_index, 0)
                self.assertIsNone(harness.overlap_correction_worker)
                self.assertTrue(harness.overlap_correction_correct_button.isEnabled())
                self.assertFalse(harness.overlap_correction_stop_button.isEnabled())
                self.assertNotIn("Batch overlap correction completed.", harness.preproc_message_box.toPlainText())
            harness.close()

    def test_overlap_stop_during_loading_discards_late_payload(self):
        with TemporaryDirectory() as root:
            output = os.path.join(root, "output")
            os.makedirs(output)
            first = os.path.join(root, "run1")
            os.makedirs(first)
            harness = _PreprocessingHarness()
            harness._overlap_batch_paths = [first, os.path.join(root, "run2")]
            harness.overlap_correction_output_input.setText(output)
            loaders = []

            class Loader(_HeldWorkflowWorker):
                def __init__(self, folder):
                    super().__init__(folder)
                    loaders.append(self)

            with patch.object(preprocessing_module, "ImageLoadWorker", Loader), patch.object(
                preprocessing_module, "OverlapCorrectionWorker"
            ) as service:
                harness.correct_overlap()
                loader = harness._overlap_data_load_worker
                self.assertTrue(loader.entered.wait(1))
                harness.stop_overlap_correction()
                loader.run_loaded.emit(first, {"0001": np.ones((2, 2))})
                loader.release.set()
                self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)

                service.assert_not_called()
                self.assertEqual(len(loaders), 1)
                self.assertEqual(harness._current_overlap_index, 0)
                self.assertIsNone(harness._overlap_data_load_worker)
                self.assertTrue(harness.overlap_correction_correct_button.isEnabled())
                self.assertFalse(harness.overlap_correction_stop_button.isEnabled())
            harness.close()

    def test_normalisation_stop_during_classic_loading_discards_payload(self):
        with TemporaryDirectory() as root:
            harness = _PreprocessingHarness()
            harness.normalisation_output_input.setText(root)
            harness.normalisation_open_beam_runs = [
                {"folder_path": root, "images": {"0001": object()}, "kind": "classic_image_folder"}
            ]
            harness._normalisation_batch_paths = [
                {"folder_path": root, "kind": "classic_image_folder"},
                {"folder_path": root + "-second", "kind": "classic_image_folder"},
            ]
            loader_instances = []

            class Loader(_HeldWorkflowWorker):
                def __init__(self, folder):
                    super().__init__(folder)
                    loader_instances.append(self)

            with patch.object(preprocessing_module, "ImageLoadWorker", Loader), patch.object(
                preprocessing_module, "NormalisationWorker"
            ) as normalisation_worker:
                harness.normalise_images()
                loader = harness._data_load_worker
                self.assertTrue(loader.entered.wait(1))
                harness.stop_normalisation()
                loader.run_loaded.emit(root, {"0001": object()})
                loader.release.set()
                self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)

                normalisation_worker.assert_not_called()
                self.assertEqual(len(loader_instances), 1)
                self.assertIsNone(harness._data_load_worker)
                self.assertEqual(harness._current_dataset_index, 0)
                self.assertTrue(harness.normalisation_normalise_button.isEnabled())
                self.assertFalse(harness.normalisation_stop_button.isEnabled())
            harness.close()

    def test_normalisation_stop_during_active_worker_prevents_next_dataset(self):
        with TemporaryDirectory() as root:
            first = os.path.join(root, "sample1")
            second = os.path.join(root, "sample2")
            os.makedirs(first)
            os.makedirs(second)
            harness = _PreprocessingHarness()
            harness.normalisation_output_input.setText(root)
            harness.normalisation_open_beam_runs = [
                {"folder_path": root, "images": {"0001": np.ones((2, 2))}, "kind": "classic_image_folder"}
            ]
            harness._normalisation_batch_paths = [
                {"folder_path": first, "kind": "classic_image_folder"},
                {"folder_path": second, "kind": "classic_image_folder"},
            ]
            loader_instances = []

            class Loader(_HeldWorkflowWorker):
                def __init__(self, folder):
                    super().__init__(folder)
                    loader_instances.append(self)

            class Service(_HeldWorkflowWorker):
                instances = []

            with patch.object(preprocessing_module, "ImageLoadWorker", Loader), patch.object(
                preprocessing_module, "NormalisationWorker", Service
            ):
                harness.normalise_images()
                loader = harness._data_load_worker
                loader.run_loaded.emit(first, {"0001": np.ones((2, 2))})
                loader.release.set()
                self._wait_for(
                    lambda: len(Service.instances) == 1
                    and Service.instances[0].entered.is_set()
                )
                service = Service.instances[0]
                harness.stop_normalisation()
                self.assertIs(harness.normalisation_worker, service)
                # Switching the newly selected input format to RADEN cannot
                # bypass the still-live classic-normalisation family lock.
                harness.normalisation_open_beam_runs = [
                    {"folder_path": root, "kind": "raden_tiff_stack", "info": {}}
                ]
                harness._normalisation_batch_paths = [
                    {"folder_path": first, "kind": "raden_tiff_stack", "info": {}}
                ]
                harness.normalise_images()
                self.assertEqual(len(Service.instances), 1)
                service.release.set()
                self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)

                self.assertEqual(len(loader_instances), 1)
                self.assertEqual(harness._current_dataset_index, 0)
                self.assertIsNone(harness.normalisation_worker)
                self.assertTrue(harness.normalisation_normalise_button.isEnabled())
                self.assertFalse(harness.normalisation_stop_button.isEnabled())
            harness.close()

    def test_summation_two_and_three_level_stop_during_lazy_loading(self):
        with TemporaryDirectory() as root:
            output = os.path.join(root, "output")
            os.makedirs(output)
            two_runs = [os.path.join(root, "two1"), os.path.join(root, "two2")]
            three_runs = [os.path.join(root, "sample", "run1"), os.path.join(root, "sample", "run2")]
            for folder in two_runs + three_runs:
                os.makedirs(folder)

            for mode in ("two", "three"):
                with self.subTest(mode=mode):
                    harness = _PreprocessingHarness()
                    harness.summation_output_input.setText(output)
                    if mode == "two":
                        harness._summation_samples_2level = two_runs
                        start_method = harness._init_summation_2level
                        active_loader_name = "_lazy_load_worker_2"
                    else:
                        harness._summation_samples_3level = {os.path.join(root, "sample"): three_runs}
                        start_method = harness._init_summation_3level
                        active_loader_name = "_lazy_load_worker_3"
                    loader_instances = []

                    class Loader(_HeldWorkflowWorker):
                        def __init__(self, folder):
                            super().__init__(folder)
                            loader_instances.append(self)

                    _HeldWorkflowWorker.instances = []
                    with patch.object(preprocessing_module, "ImageLoadWorker", Loader), patch.object(
                        preprocessing_module, "SummationWorker"
                    ) as summation_worker, patch.object(
                        preprocessing_module.QMessageBox, "information"
                    ) as success_popup:
                        start_method()
                        loader = getattr(harness, active_loader_name)
                        self.assertTrue(loader.entered.wait(1))
                        harness.stop_summation()
                        self.assertTrue(loader.stop_requested)
                        loader.run_loaded.emit(two_runs[0], {"0001": np.ones((2, 2))})

                        # A cross-mode attempt enters the production summation
                        # dispatcher but cannot start another generation.
                        if mode == "two":
                            harness._summation_samples_3level = {
                                os.path.join(root, "sample"): three_runs
                            }
                            harness.sum_images()
                        else:
                            harness._summation_samples_3level = None
                            harness._summation_samples_2level = two_runs
                            harness.sum_images()
                        self.assertEqual(len(loader_instances), 1)
                        self.assertEqual(len(harness._preprocessing_worker_registry.workers), 1)
                        loader.release.set()
                        self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)

                        summation_worker.assert_not_called()
                        self.assertEqual(len(loader_instances), 1)
                        self.assertIsNone(getattr(harness, active_loader_name))
                        self.assertTrue(harness.summation_sum_button.isEnabled())
                        self.assertFalse(harness.summation_stop_button.isEnabled())
                        self.assertNotIn("completed successfully", harness.preproc_message_box.toPlainText())
                        success_popup.assert_not_called()
                    harness.close()

    def test_one_level_to_lazy_summation_mode_switch_waits_for_retirement(self):
        with TemporaryDirectory() as root:
            two_runs = [os.path.join(root, "run1"), os.path.join(root, "run2")]
            for folder in two_runs:
                os.makedirs(folder)
            harness = _PreprocessingHarness()
            harness.summation_image_runs = [{"folder_path": root, "images": {"0001": np.ones((2, 2))}}]
            harness.summation_output_input.setText(root)

            class Service(_HeldWorkflowWorker):
                instances = []

            with patch.object(preprocessing_module, "SummationWorker", Service), patch.object(
                preprocessing_module, "ImageLoadWorker"
            ) as loader:
                harness.sum_images()
                self._wait_for(lambda: len(Service.instances) == 1 and Service.instances[0].entered.is_set())
                worker = Service.instances[0]
                harness.stop_summation()
                harness._summation_samples_2level = two_runs
                harness.sum_images()
                loader.assert_not_called()
                self.assertEqual(len(Service.instances), 1)
                self.assertIn(worker, harness._preprocessing_worker_registry.workers)
                worker.release.set()
                self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)
                self.assertIsNone(harness.summation_worker)
                self.assertTrue(harness.summation_sum_button.isEnabled())
                self.assertFalse(harness.summation_stop_button.isEnabled())
            harness.close()

    def test_successful_outlier_batch_advances_each_dataset_once(self):
        with TemporaryDirectory() as root:
            first = os.path.join(root, "run1")
            second = os.path.join(root, "run2")
            output = os.path.join(root, "output")
            for folder in (first, second, output):
                os.makedirs(folder)
            harness = _PreprocessingHarness()
            harness._outlier_batch_paths = [first, second]
            harness.outlier_output_input.setText(output)
            _AutoLoader.folders = []
            _AutoServiceWorker.instances = []
            with patch.object(preprocessing_module, "ImageLoadWorker", _AutoLoader), patch.object(
                preprocessing_module, "OutlierFilteringWorker", _AutoServiceWorker
            ):
                harness.remove_outliers()
                self._wait_for(lambda: not harness._preprocessing_worker_registry.active_families)
                self.assertEqual(_AutoLoader.folders, [first, second])
                self.assertEqual(len(_AutoServiceWorker.instances), 2)
                self.assertEqual(harness._current_outlier_index, 2)
                self.assertEqual(
                    harness.preproc_message_box.toPlainText().count("Batch outlier removal completed."),
                    1,
                )
                self.assertTrue(harness.outlier_process_button.isEnabled())
                self.assertFalse(harness.outlier_stop_button.isEnabled())
            harness.close()

    def test_filtering_production_start_stop_cleans_reference_and_restores_valid_state(self):
        with TemporaryDirectory() as root:
            harness = _PreprocessingHarness()
            harness.filtering_image_runs = [{"folder_path": root, "images": {"0001": np.ones((2, 2))}}]
            harness.filtering_output_input.setText(root)
            _HeldWorkflowWorker.instances = []
            with patch.object(preprocessing_module, "FilteringWorker", _HeldWorkflowWorker):
                harness.filter_images()
                worker = harness.filtering_worker
                self.assertTrue(worker.entered.wait(1))
                harness.stop_filtering()
                self.assertIs(harness.filtering_worker, worker)
                worker.release.set()
                self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)
                self.assertIsNone(harness.filtering_worker)
                self.assertTrue(harness.filtering_filter_button.isEnabled())
                self.assertFalse(harness.filtering_stop_button.isEnabled())
                self.assertNotIn("Filtering process finished.", harness.preproc_message_box.toPlainText())
            harness.close()

    def test_open_beam_loader_without_stop_is_retained_and_late_payload_is_discarded(self):
        registry, _ = self._registry()
        worker = _NonCooperativeWorker()
        payloads = []
        drained = []
        generation = registry.begin("normalisation", on_drained=lambda abnormal: drained.append(abnormal))
        registry.start_worker(
            worker,
            generation,
            handlers={"open_beam_loaded": lambda *args: payloads.append(args)},
        )
        self.assertTrue(worker.entered.wait(1))
        self.assertTrue(registry.cancel_family("normalisation"))
        self.assertIn(worker, registry.workers)
        worker.open_beam_loaded.emit("obsolete", {"frame": {}})
        for _ in range(5):
            self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertEqual(payloads, [])
        self.assertIn(worker, registry.workers)
        worker.release.set()
        self._wait_for(lambda: not registry.workers)
        self.assertEqual(drained, [False])

    def test_superseded_open_beam_selection_stays_owned_but_cannot_replace_new_selection(self):
        harness = _PreprocessingHarness()
        harness.wavelengths = []
        harness.normalisation_open_beam_runs = []
        old_worker = _NonCooperativeWorker()
        harness._open_beam_selection_id = 1
        old_family = "normalisation_open_beam_1"
        harness._begin_preprocessing_workflow(old_family)
        harness.open_beam_load_worker = old_worker
        harness._start_preprocessing_worker(
            old_worker,
            old_family,
            handlers={
                "open_beam_loaded": lambda folder, images: harness._open_beam_loaded_for_selection(
                    1, folder, images
                )
            },
            completion=lambda: harness._open_beam_loading_finished_for_selection(
                old_worker, old_family
            ),
        )
        self.assertTrue(old_worker.entered.wait(1))

        new_worker = _NonCooperativeWorker()
        harness._open_beam_selection_id = 2
        new_family = "normalisation_open_beam_2"
        harness._begin_preprocessing_workflow(new_family)
        harness.open_beam_load_worker = new_worker
        harness._start_preprocessing_worker(
            new_worker,
            new_family,
            handlers={
                "open_beam_loaded": lambda folder, images: harness._open_beam_loaded_for_selection(
                    2, folder, images
                )
            },
            completion=lambda: harness._open_beam_loading_finished_for_selection(
                new_worker, new_family
            ),
        )
        self.assertTrue(new_worker.entered.wait(1))

        old_worker.open_beam_loaded.emit("old", {"1": object()})
        for _ in range(5):
            self.app.processEvents(QEventLoop.AllEvents, 10)
        self.assertEqual(harness.normalisation_open_beam_runs, [])
        self.assertEqual(set(harness._preprocessing_worker_registry.workers), {old_worker, new_worker})
        self.assertIs(harness.open_beam_load_worker, new_worker)

        old_worker.release.set()
        self._wait_for(lambda: old_worker not in harness._preprocessing_worker_registry.workers)
        self.assertIs(harness.open_beam_load_worker, new_worker)
        new_worker.open_beam_loaded.emit("new", {"2": object()})
        for _ in range(5):
            self.app.processEvents(QEventLoop.AllEvents, 10)
        new_worker.release.set()
        self._wait_for(lambda: not harness._preprocessing_worker_registry.workers)
        self.assertEqual(harness.normalisation_open_beam_runs[0]["folder_path"], "new")
        self.assertIsNone(harness.open_beam_load_worker)
        self.assertFalse(harness._preprocessing_workflow_generations)
        harness.close()

    def test_all_preprocessing_worker_construction_paths_use_the_registry(self):
        tree = ast.parse(inspect.getsource(PreprocessingMixin))
        worker_names = {
            "FilteringWorker",
            "FullProcessWorker",
            "NormalisationWorker",
            "OutlierFilteringWorker",
            "OverlapCorrectionWorker",
            "RadenNormalisationWorker",
            "SummationWorker",
            "ImageLoadWorker",
            "OpenBeamLoadWorker",
        }
        constructor_functions = []
        direct_starts = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls = [
                    child
                    for child in ast.walk(node)
                    if isinstance(child, ast.Call)
                ]
                constructed = any(
                    isinstance(call.func, ast.Name) and call.func.id in worker_names
                    for call in calls
                )
                if constructed:
                    constructor_functions.append(node.name)
                    self.assertTrue(
                        any(
                            isinstance(call.func, ast.Attribute)
                            and call.func.attr == "_start_preprocessing_worker"
                            for call in calls
                        ),
                        f"{node.name} constructs a worker without registry start",
                    )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "start":
                    direct_starts.append(node)
        self.assertTrue(constructor_functions)
        self.assertEqual(direct_starts, [])

        stop_methods = {
            "stop_summation",
            "stop_outlier_removal",
            "stop_overlap_correction",
            "stop_normalisation",
            "stop_filtering",
            "stop_full_process",
        }
        method_nodes = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in stop_methods
        }
        self.assertEqual(set(method_nodes), stop_methods)
        for name, node in method_nodes.items():
            self.assertTrue(
                any(
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "_stop_preprocessing_family"
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call)
                ),
                name,
            )
            self.assertFalse(
                any(
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr in {"wait", "quit", "requestInterruption", "terminate"}
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call)
                ),
                name,
            )


if __name__ == "__main__":
    unittest.main()

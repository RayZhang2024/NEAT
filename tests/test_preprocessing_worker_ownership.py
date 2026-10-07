"""Real-QThread regressions for GUI preprocessing ownership and retirement."""

import os
import time
import unittest
import ast
import inspect
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEventLoop, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QWidget

from NEAT.ui.preprocessing_worker_registry import PreprocessingWorkerRegistry
from NEAT.ui.mixins.preprocessing import PreprocessingMixin


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
        raise RuntimeError("synthetic QThread start rejection")


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

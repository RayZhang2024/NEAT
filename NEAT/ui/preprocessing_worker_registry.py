"""GUI-thread ownership and retirement for preprocessing QThreads.

The public ``finished`` signals on preprocessing workers are manually emitted
from ``run`` and therefore are not evidence that the native QThread has exited.
This registry keeps a strong reference until both public completion handling
and a nonblocking native-exit check have completed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot


Callback = Callable[..., None]


class PreprocessingWorkerStartRejected(RuntimeError):
    """Signal an explicit synchronous rejection before Qt accepted a start."""


@dataclass
class _Generation:
    family: str
    number: int
    cancelled: bool = False
    closing: bool = False
    abnormal: bool = False
    abnormal_diagnostic_emitted: bool = False
    workers: set[int] = field(default_factory=set)
    on_drained: Callback | None = None


@dataclass
class _WorkerRecord:
    worker: QThread
    generation: _Generation
    completion: Callback | None
    handlers: dict[str, Callback]
    start_invoked: bool = False
    start_acknowledged: bool = False
    finished_before_start: bool | None = None
    public_completion: bool = False
    payload_handoff_settled: bool = False
    completion_settle_scheduled: bool = False
    pending_payloads: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    completion_handled: bool = False
    exit_confirmed: bool = False
    retired: bool = False
    start_error: str | None = None
    start_rejected: bool = False
    startup_diagnostic_emitted: bool = False
    ambiguity_diagnostic_emitted: bool = False
    startup_timer: QTimer | None = None
    grace_timer: QTimer | None = None
    proxy: "_WorkerSignalProxy | None" = None
    on_retired: Callback | None = None
    shutdown_stop_attempted: bool = False


class _WorkerSignalProxy(QObject):
    """A GUI-thread QObject receiver for queued worker signals."""

    def __init__(self, registry: "PreprocessingWorkerRegistry", record: _WorkerRecord):
        super().__init__(registry)
        self._registry: PreprocessingWorkerRegistry | None = registry
        self._record: _WorkerRecord | None = record

    def _dispatch(self, method: str, *args) -> None:
        registry, record = self._registry, self._record
        if registry is not None and record is not None:
            getattr(registry, method)(record, *args)

    @pyqtSlot()
    def started(self):
        self._dispatch("_acknowledge_start")

    @pyqtSlot()
    def finished(self):
        self._dispatch("_public_finished")

    @pyqtSlot(int)
    def progress(self, value):
        self._dispatch("_forward", "progress_updated", value)

    @pyqtSlot(str)
    def message(self, value):
        self._dispatch("_forward", "message", value)

    @pyqtSlot(str, dict)
    def run_loaded(self, folder, images):
        self._dispatch("_forward", "run_loaded", folder, images)

    @pyqtSlot(str, dict)
    def open_beam_loaded(self, folder, images):
        self._dispatch("_forward", "open_beam_loaded", folder, images)

    @pyqtSlot(int)
    def load_progress(self, value):
        self._dispatch("_forward", "load_progress_updated", value)


class PreprocessingWorkerRegistry(QObject):
    """Strongly retain GUI preprocessing workers through verified QThread exit."""

    diagnostic = pyqtSignal(str)
    state_changed = pyqtSignal()
    FAMILY_BUSY = object()
    STARTUP_GRACE_MS = 500
    POLL_INTERVAL_MS = 25
    MISSING_COMPLETION_GRACE_MS = 250

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._next_generation: dict[str, int] = {}
        self._families: dict[str, _Generation] = {}
        self._records: dict[int, _WorkerRecord] = {}
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self.POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_workers)

    @property
    def workers(self) -> tuple[QThread, ...]:
        """Snapshot of every worker still owned, for later shutdown integration."""
        return tuple(record.worker for record in self._records.values())

    @property
    def active_families(self) -> tuple[str, ...]:
        return tuple(self._families)

    def begin(self, family: str, on_drained: Callback | None = None):
        """Start a family generation, or return ``FAMILY_BUSY`` if it is active."""
        if family in self._families:
            return self.FAMILY_BUSY
        number = self._next_generation.get(family, 0) + 1
        self._next_generation[family] = number
        generation = _Generation(family, number, on_drained=on_drained)
        self._families[family] = generation
        self.state_changed.emit()
        return generation

    def is_current(self, generation: _Generation) -> bool:
        return self._families.get(generation.family) is generation

    def is_cancelled(self, generation: _Generation) -> bool:
        return generation is None or generation.cancelled or generation.abnormal

    def complete_generation(self, generation: _Generation) -> None:
        if self._families.get(generation.family) is not generation:
            return
        generation.closing = True
        self._drain_generation_if_ready(generation)

    def cancel_family(self, family: str) -> bool:
        generation = self._families.get(family)
        if generation is None:
            return False
        changed = not generation.cancelled or not generation.closing
        generation.cancelled = True
        generation.closing = True
        for record in tuple(self._records.values()):
            if record.generation is not generation:
                continue
            self._request_worker_stop(record)
        self._drain_generation_if_ready(generation)
        if changed:
            self.state_changed.emit()
        return changed

    def request_shutdown(self) -> tuple[QThread, ...]:
        """Cancel every extant generation and request each worker stop once.

        The returned snapshot is useful to close preflight: its presence means
        that this close event must be rejected even if cancellation drains
        synchronously. Generations already marked closing are still cancelled
        and drained when empty; live/ambiguous records remain owned normally.
        """
        workers_at_entry = tuple(record.worker for record in self._records.values())
        for generation in tuple(self._families.values()):
            generation.cancelled = True
            generation.closing = True
            for record in tuple(self._records.values()):
                if record.generation is generation:
                    self._request_worker_stop(record)
            self._drain_generation_if_ready(generation)
        self.state_changed.emit()
        return workers_at_entry

    def _request_worker_stop(self, record: _WorkerRecord) -> None:
        if record.shutdown_stop_attempted:
            return
        record.shutdown_stop_attempted = True
        stop = getattr(record.worker, "stop", None)
        if not callable(stop):
            return
        try:
            stop()
        except Exception as exc:
            record.generation.abnormal = True
            if not record.generation.abnormal_diagnostic_emitted:
                record.generation.abnormal_diagnostic_emitted = True
                self.diagnostic.emit(
                    f"[WARN] Could not request preprocessing worker stop: {type(exc).__name__}: {exc}."
                )

    def start_worker(
        self,
        worker: QThread,
        generation: _Generation,
        handlers: dict[str, Callback] | None = None,
        completion: Callback | None = None,
        on_retired: Callback | None = None,
    ) -> bool:
        """Register, connect GUI-thread gates, then start one owned worker."""
        if (
            not self.is_current(generation)
            or generation.closing
            or self.is_cancelled(generation)
        ):
            return False
        key = id(worker)
        record = _WorkerRecord(
            worker=worker,
            generation=generation,
            completion=completion,
            handlers=dict(handlers or {}),
            on_retired=on_retired,
        )
        proxy = _WorkerSignalProxy(self, record)
        record.proxy = proxy
        self._records[key] = record
        generation.workers.add(key)
        self.state_changed.emit()
        self._connect(worker.started, proxy.started)
        self._connect(worker.finished, proxy.finished)
        for name, slot in (
            ("progress_updated", proxy.progress),
            ("message", proxy.message),
            ("load_progress_updated", proxy.load_progress),
            ("run_loaded", proxy.run_loaded),
            ("open_beam_loaded", proxy.open_beam_loaded),
        ):
            if name in record.handlers and hasattr(worker, name):
                self._connect(getattr(worker, name), slot)

        try:
            record.finished_before_start = worker.isFinished()
        except RuntimeError:
            record.finished_before_start = None
        record.start_invoked = True
        try:
            worker.start()
        except PreprocessingWorkerStartRejected as exc:
            record.start_error = str(exc)
            record.start_rejected = True
        except Exception as exc:
            # An override may raise after delegating to QThread.start(). Keep
            # ownership until Qt provides evidence either way.
            record.start_error = str(exc)
        record.startup_timer = QTimer(self)
        record.startup_timer.setSingleShot(True)
        record.startup_timer.timeout.connect(lambda rec=record: self._startup_expired(rec))
        record.startup_timer.start(self.STARTUP_GRACE_MS)
        self._ensure_polling()
        self._poll_record(record)
        return True

    @staticmethod
    def _connect(source, slot):
        source.connect(slot, Qt.QueuedConnection)

    def _acknowledge_start(self, record: _WorkerRecord) -> None:
        if record.retired:
            return
        record.start_acknowledged = True
        if record.startup_timer is not None:
            record.startup_timer.stop()

    def _forward(self, record: _WorkerRecord, name: str, *args) -> None:
        if record.retired or self._records.get(id(record.worker)) is not record:
            return
        if not self.is_current(record.generation):
            return
        if name in ("run_loaded", "open_beam_loaded"):
            if self.is_cancelled(record.generation):
                return
            if not record.payload_handoff_settled:
                record.pending_payloads.append((name, args))
                return
            self._deliver_payload(record, name, args)
            return
        callback = record.handlers.get(name)
        if callback is not None:
            try:
                callback(*args)
            except Exception as exc:
                self._mark_abnormal(record.generation, f"GUI signal handling failed: {exc}")

    def _public_finished(self, record: _WorkerRecord) -> None:
        if record.retired or self._records.get(id(record.worker)) is not record:
            return
        record.public_completion = True
        if record.grace_timer is not None:
            record.grace_timer.stop()
        if not record.completion_settle_scheduled:
            record.completion_settle_scheduled = True
            QTimer.singleShot(0, lambda rec=record: self._settle_public_completion(rec))

    def _settle_public_completion(self, record: _WorkerRecord) -> None:
        if record.retired or not record.public_completion:
            return
        pending = tuple(record.pending_payloads)
        record.pending_payloads.clear()
        if not self.is_cancelled(record.generation):
            for name, args in pending:
                self._deliver_payload(record, name, args)
                if record.generation.abnormal:
                    break
        record.payload_handoff_settled = True
        if record.exit_confirmed:
            self._handle_completion(record)

    def _deliver_payload(
        self,
        record: _WorkerRecord,
        name: str,
        args: tuple[object, ...],
    ) -> None:
        if record.retired or not self.is_current(record.generation):
            return
        if self.is_cancelled(record.generation):
            return
        callback = record.handlers.get(name)
        if callback is not None:
            try:
                callback(*args)
            except Exception as exc:
                self._mark_abnormal(record.generation, f"GUI signal handling failed: {exc}")

    def _handle_completion(self, record: _WorkerRecord) -> None:
        if (
            record.retired
            or record.completion_handled
            or not record.public_completion
            or not record.payload_handoff_settled
        ):
            return
        if not record.exit_confirmed:
            return
        record.completion_handled = True
        if record.grace_timer is not None:
            record.grace_timer.stop()
        try:
            if not self.is_cancelled(record.generation) and record.completion is not None:
                try:
                    record.completion()
                except Exception as exc:
                    self._mark_abnormal(
                        record.generation, f"GUI completion handling failed: {exc}"
                    )
        finally:
            # Public completion may need the convenience attribute to read the
            # structured result. Release that attribute only after the callback
            # has consumed it, and only if it still points at this worker.
            self._retire(record)
            self._drain_generation_if_ready(record.generation)

    def _poll_workers(self) -> None:
        for record in tuple(self._records.values()):
            self._poll_record(record)
        if not self._records:
            self._poll_timer.stop()

    def _poll_record(self, record: _WorkerRecord) -> None:
        if record.retired:
            return
        worker = record.worker
        try:
            running = worker.isRunning()
            finished = worker.isFinished()
        except RuntimeError:
            self._ambiguous(record, "worker state became unavailable before exit was confirmed")
            return
        if running:
            record.start_acknowledged = True
            if record.startup_timer is not None:
                record.startup_timer.stop()
            return
        if self._has_post_start_exit_evidence(record, finished):
            # A fast thread may finish before the GUI observes isRunning()/started.
            record.start_acknowledged = True
        if not record.start_acknowledged:
            return
        if not running and finished:
            try:
                exited = worker.wait(0)
            except RuntimeError:
                exited = False
            if exited:
                self._confirm_exit(record)

    def _confirm_exit(self, record: _WorkerRecord) -> None:
        if record.retired or record.exit_confirmed:
            return
        record.exit_confirmed = True
        if record.startup_timer is not None:
            record.startup_timer.stop()
        if record.public_completion:
            self._handle_completion(record)
            return
        # One more GUI event-loop dispatch drains signals queued before exit.
        QTimer.singleShot(0, lambda rec=record: self._begin_missing_completion_grace(rec))

    def _begin_missing_completion_grace(self, record: _WorkerRecord) -> None:
        if record.retired or record.public_completion:
            if record.public_completion:
                self._handle_completion(record)
            return
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda rec=record: self._missing_completion_expired(rec))
        record.grace_timer = timer
        timer.start(self.MISSING_COMPLETION_GRACE_MS)

    def _missing_completion_expired(self, record: _WorkerRecord) -> None:
        if record.retired or record.public_completion:
            self._handle_completion(record)
            return
        generation = record.generation
        self._mark_abnormal(
            generation,
            "Preprocessing worker exited without completion notification.",
            exclude=record,
        )
        self._retire(record)
        self._drain_generation_if_ready(generation)

    def _mark_abnormal(
        self,
        generation: _Generation,
        detail: str,
        *,
        exclude: _WorkerRecord | None = None,
    ) -> None:
        generation.abnormal = True
        generation.closing = True
        if not generation.abnormal_diagnostic_emitted:
            generation.abnormal_diagnostic_emitted = True
            self.diagnostic.emit(f"[WARN] {detail}.")
        for other in tuple(self._records.values()):
            if other.generation is generation and other is not exclude:
                stop = getattr(other.worker, "stop", None)
                if callable(stop):
                    try:
                        stop()
                    except Exception:
                        pass

    def _startup_expired(self, record: _WorkerRecord) -> None:
        if record.retired:
            return
        self._poll_record(record)
        if record.retired or record.start_acknowledged:
            return
        try:
            running = record.worker.isRunning()
            finished = record.worker.isFinished()
        except RuntimeError:
            self._ambiguous(record, "could not verify whether the worker started")
            return
        if running or self._has_post_start_exit_evidence(record, finished):
            self._poll_record(record)
            return
        if record.start_rejected and not record.start_acknowledged:
            try:
                no_thread_started = not running and record.worker.wait(0)
            except RuntimeError:
                no_thread_started = False
            # A nonblocking wait is not proof on its own. It is used only with
            # an explicit rejection contract and unchanged pre-start state.
            if no_thread_started and finished == record.finished_before_start:
                self.diagnostic.emit(
                    "[WARN] Preprocessing worker failed to start; the operation was not run."
                )
                generation = record.generation
                generation.abnormal = True
                generation.closing = True
                self._retire(record)
                self._drain_generation_if_ready(generation)
                return
        if not record.startup_diagnostic_emitted:
            record.startup_diagnostic_emitted = True
            detail = "Preprocessing worker startup is still unacknowledged; retaining ownership"
            if record.start_error:
                detail += f" ({record.start_error})"
            self.diagnostic.emit(f"[WARN] {detail}.")

    def _ambiguous(self, record: _WorkerRecord, detail: str) -> None:
        if record.retired:
            return
        if not record.ambiguity_diagnostic_emitted:
            record.ambiguity_diagnostic_emitted = True
            self.diagnostic.emit(f"[WARN] Preprocessing worker lifetime is uncertain: {detail}.")
        # Keep the strong reference. An ambiguous QThread must never be deleted.

    @staticmethod
    def _has_post_start_exit_evidence(record: _WorkerRecord, finished: bool) -> bool:
        return (
            record.start_invoked
            and finished
            and record.finished_before_start is False
        )

    def _retire(self, record: _WorkerRecord) -> None:
        if record.retired:
            return
        record.retired = True
        for timer in (record.startup_timer, record.grace_timer):
            if timer is not None:
                timer.stop()
                try:
                    timer.timeout.disconnect()
                except (RuntimeError, TypeError):
                    pass
                timer.deleteLater()
        record.startup_timer = None
        record.grace_timer = None
        key = id(record.worker)
        self._records.pop(key, None)
        record.generation.workers.discard(key)
        record.pending_payloads.clear()
        if record.on_retired is not None:
            try:
                record.on_retired(record.worker)
            except Exception as exc:
                self.diagnostic.emit(
                    f"[WARN] Preprocessing worker reference cleanup failed: {exc}."
                )
            record.on_retired = None
        record.completion = None
        record.handlers.clear()
        if record.proxy is not None:
            proxy = record.proxy
            record.proxy = None
            proxy._record = None
            proxy._registry = None
            proxy.deleteLater()
        if not self._records:
            self._poll_timer.stop()
        self.state_changed.emit()

    def _drain_generation_if_ready(self, generation: _Generation) -> None:
        if not generation.closing or generation.workers:
            return
        if self._families.get(generation.family) is not generation:
            return
        self._families.pop(generation.family, None)
        callback = generation.on_drained
        generation.on_drained = None
        if callback is not None:
            callback(generation.abnormal)
        self.state_changed.emit()

    def _ensure_polling(self) -> None:
        if not self._poll_timer.isActive():
            self._poll_timer.start()


__all__ = ["PreprocessingWorkerRegistry", "PreprocessingWorkerStartRejected"]

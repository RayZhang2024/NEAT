"""Lifetime accounting for the legacy QThreads owned directly by FitsViewer.

Preprocessing workers use :mod:`preprocessing_worker_registry`.  This small
inventory covers only the existing fitting loaders, batch-fit workers and
application update-check thread; it is not a workflow or task manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Callable, cast

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot


_QUEUED_CONNECTION = cast(Any, Qt).QueuedConnection
_DIRECT_CONNECTION = cast(Any, Qt).DirectConnection


@dataclass
class _ThreadRecord:
    worker: QThread
    role: str
    finished_before_start: bool | None
    started: bool = False
    public_finished: bool = False
    exit_confirmed: bool = False
    callbacks_suppressed: bool = False
    stop_attempted: bool = False
    stop_error: str | None = None
    grace_timer: QTimer | None = None
    settle_scheduled: bool = False
    retirement_ready: bool = False
    pending_callbacks: int = 0
    pending_callbacks_lock: Any = field(default_factory=threading.Lock, repr=False)


class _ThreadSignalProxy(QObject):
    def __init__(self, inventory: "WindowThreadInventory", record: _ThreadRecord):
        super().__init__(inventory)
        self._inventory: WindowThreadInventory | None = inventory
        self._record: _ThreadRecord | None = record

    @pyqtSlot()
    def started(self) -> None:
        inventory, record = self._inventory, self._record
        if inventory is not None and record is not None:
            inventory._started(record)

    @pyqtSlot()
    @pyqtSlot(str)
    def finished(self, *_args) -> None:
        inventory, record = self._inventory, self._record
        if inventory is not None and record is not None:
            inventory._public_finished(record)


class WindowThreadInventory(QObject):
    """Retain FitsViewer's non-preprocessing workers through safe retirement."""

    changed = pyqtSignal()
    diagnostic = pyqtSignal(str)
    POLL_INTERVAL_MS = 25
    MISSING_FINISHED_GRACE_MS = 250

    def __init__(self, parent: QObject):
        super().__init__(parent)
        self._records: dict[int, _ThreadRecord] = {}
        self._proxies: dict[int, _ThreadSignalProxy] = {}
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self.POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll)

    @property
    def workers(self) -> tuple[QThread, ...]:
        return tuple(record.worker for record in self._records.values())

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(record.role for record in self._records.values())

    @property
    def has_unsettled_workers(self) -> bool:
        return bool(self._records)

    def has_role(self, role: str) -> bool:
        return any(record.role == role for record in self._records.values())

    def track(self, worker: QThread, role: str) -> None:
        """Record a worker before ``start()`` and observe its QThread lifetime."""
        key = id(worker)
        if key in self._records:
            return
        try:
            finished_before_start = worker.isFinished()
        except RuntimeError:
            finished_before_start = None
        record = _ThreadRecord(worker, role, finished_before_start)
        proxy = _ThreadSignalProxy(self, record)
        self._records[key] = record
        self._proxies[key] = proxy
        cast(Any, worker.started).connect(proxy.started, _QUEUED_CONNECTION)
        cast(Any, worker.finished).connect(proxy.finished, _QUEUED_CONNECTION)
        self._poll_timer.start()
        self.changed.emit()
        self._poll_record(record)

    def connect_guarded(self, worker: QThread, signal, callback: Callable, *, current_attribute: str | None = None) -> None:
        """Queue a legacy UI callback and suppress it after cancellation/supersession."""
        key = id(worker)
        record = self._records.get(key)
        if record is None:
            raise RuntimeError("Worker must be tracked before connecting UI callbacks")

        def mark_callback_pending(*_args) -> None:
            with record.pending_callbacks_lock:
                record.pending_callbacks += 1

        def dispatch(*args):
            try:
                if self._records.get(key) is not record or record.callbacks_suppressed:
                    return
                owner = self.parent()
                if (
                    current_attribute is not None
                    and getattr(owner, current_attribute, None) is not worker
                ):
                    return
                try:
                    callback(*args)
                except RuntimeError as exc:
                    self.diagnostic.emit(
                        f"[WARN] {worker.__class__.__name__} UI callback failed: {exc}."
                    )
            finally:
                with record.pending_callbacks_lock:
                    record.pending_callbacks -= 1
                    callbacks_pending = record.pending_callbacks
                if record.retirement_ready and callbacks_pending == 0:
                    self._schedule_retirement(record)

        cast(Any, signal).connect(mark_callback_pending, _DIRECT_CONNECTION)
        cast(Any, signal).connect(dispatch, _QUEUED_CONNECTION)

    def suppress_callbacks(self, worker: QThread) -> None:
        record = self._records.get(id(worker))
        if record is not None:
            record.callbacks_suppressed = True

    def request_stop(self, worker: QThread, *, suppress_callbacks: bool = False) -> bool:
        record = self._records.get(id(worker))
        if record is None:
            return False
        if suppress_callbacks:
            record.callbacks_suppressed = True
        self._request_stop(record)
        return True

    def request_shutdown(self) -> tuple[QThread, ...]:
        """Suppress callbacks and request each available cooperative stop once."""
        workers = tuple(record.worker for record in self._records.values())
        for record in tuple(self._records.values()):
            record.callbacks_suppressed = True
            self._request_stop(record)
        return workers

    def _request_stop(self, record: _ThreadRecord) -> None:
        if record.stop_attempted:
            return
        record.stop_attempted = True
        stop = getattr(record.worker, "stop", None)
        if not callable(stop):
            if record.role == "update_check":
                self.diagnostic.emit(
                    "[INFO] Update check cannot be interrupted; keeping it owned until it exits."
                )
            return
        try:
            stop()
        except Exception as exc:
            record.stop_error = str(exc)
            self.diagnostic.emit(
                f"[WARN] Could not request stop for {record.role}: {type(exc).__name__}: {exc}."
            )

    def _started(self, record: _ThreadRecord) -> None:
        if id(record.worker) not in self._records:
            return
        record.started = True

    def _public_finished(self, record: _ThreadRecord) -> None:
        if id(record.worker) not in self._records:
            return
        record.public_finished = True
        if record.exit_confirmed:
            self._schedule_retirement(record)

    def _poll(self) -> None:
        for record in tuple(self._records.values()):
            self._poll_record(record)
        if not self._records:
            self._poll_timer.stop()

    def _poll_record(self, record: _ThreadRecord) -> None:
        try:
            running = record.worker.isRunning()
            finished = record.worker.isFinished()
        except RuntimeError:
            return
        if running:
            record.started = True
            return
        post_start_finished = (
            record.finished_before_start is False and finished
        )
        if not (record.started or post_start_finished):
            return
        if not finished:
            return
        try:
            native_exit_verified = record.worker.wait(0)
        except RuntimeError:
            native_exit_verified = False
        if not native_exit_verified:
            return
        if not record.exit_confirmed:
            record.exit_confirmed = True
            if record.public_finished:
                self._schedule_retirement(record)
            else:
                QTimer.singleShot(0, lambda rec=record: self._start_missing_signal_grace(rec))

    def _start_missing_signal_grace(self, record: _ThreadRecord) -> None:
        if id(record.worker) not in self._records or record.public_finished:
            self._schedule_retirement(record)
            return
        if record.grace_timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda rec=record: self._missing_signal_expired(rec))
            record.grace_timer = timer
            timer.start(self.MISSING_FINISHED_GRACE_MS)

    def _missing_signal_expired(self, record: _ThreadRecord) -> None:
        if id(record.worker) not in self._records:
            return
        self.diagnostic.emit(
            f"[WARN] {record.role} exited without its public completion signal; retiring after grace."
        )
        self._schedule_retirement(record)

    def _schedule_retirement(self, record: _ThreadRecord) -> None:
        if id(record.worker) not in self._records:
            return
        record.retirement_ready = True
        with record.pending_callbacks_lock:
            if record.pending_callbacks:
                return
        if record.settle_scheduled:
            return
        record.settle_scheduled = True
        self._retire(record)

    def _retire(self, record: _ThreadRecord) -> None:
        key = id(record.worker)
        if key not in self._records or not record.exit_confirmed:
            return
        with record.pending_callbacks_lock:
            if record.pending_callbacks:
                record.settle_scheduled = False
                return
        if record.grace_timer is not None:
            record.grace_timer.stop()
            record.grace_timer.deleteLater()
            record.grace_timer = None
        self._records.pop(key, None)
        proxy = self._proxies.pop(key, None)
        if proxy is not None:
            proxy._inventory = None
            proxy._record = None
            proxy.deleteLater()
        owner = self.parent()
        callback = getattr(owner, "_window_thread_retired", None)
        if callable(callback):
            try:
                callback(record.worker, record.role, record.callbacks_suppressed)
            except Exception as exc:
                self.diagnostic.emit(f"[WARN] Window worker retirement callback failed: {exc}.")
        try:
            record.worker.deleteLater()
        except RuntimeError:
            pass
        self.changed.emit()
        if not self._records:
            self._poll_timer.stop()


__all__ = ["WindowThreadInventory"]

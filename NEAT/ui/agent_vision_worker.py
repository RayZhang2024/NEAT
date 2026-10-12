"""Tracked, cancellable Qt worker for optional local visual review."""

from __future__ import annotations

from threading import Event, Lock

from PyQt5.QtCore import QThread, pyqtSignal

from ..domain.agent_preflight import PreflightResult
from ..domain.agent_vision import VisionOutcome, VisionSettings, VisionStatus
from ..services.agent_vision import (
    LocalVisionInspector,
    VisionCancelled,
    VisionError,
    VisionInspector,
    VisionStale,
    VisionUnavailable,
    run_visual_rehearsal,
)
from ..services.scientific_observation import ScientificDatasetHandle


class VisionWorker(QThread):
    """Owns a captured handle until retirement; never reads live QWidget state."""

    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)

    def __init__(
        self,
        run_id: str,
        handle: ScientificDatasetHandle,
        preflight: PreflightResult,
        settings: VisionSettings,
        *,
        inspector: VisionInspector | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.run_id = run_id
        self.dataset_id = handle.dataset_id
        self.revision = handle.revision
        self._handle: ScientificDatasetHandle | None = handle
        self._preflight: PreflightResult | None = preflight
        self._settings = settings
        self._inspector: VisionInspector | None = inspector
        self._cancel = Event()
        self._lock = Lock()

    def stop(self) -> None:
        self._cancel.set()
        with self._lock:
            inspector = self._inspector
        if inspector is not None:
            inspector.cancel()

    def run(self) -> None:
        handle = self._handle
        preflight = self._preflight
        if handle is None or preflight is None:
            return
        try:
            if self._cancel.is_set():
                raise VisionCancelled("Visual review stopped.")
            with self._lock:
                if self._inspector is None:
                    self._inspector = LocalVisionInspector(self._settings)
                inspector = self._inspector
            if self._cancel.is_set():
                inspector.cancel()
                raise VisionCancelled("Visual review stopped.")
            outcome = run_visual_rehearsal(
                self.run_id,
                handle,
                preflight,
                inspector,
                cancelled=self._cancel.is_set,
                progress=self.progress.emit,
            )
            if self._cancel.is_set():
                raise VisionCancelled("Visual review stopped.")
        except VisionCancelled:
            outcome = VisionOutcome(
                self.run_id,
                self.dataset_id,
                self.revision,
                VisionStatus.CANCELLED,
                "Local visual review stopped.",
            )
        except VisionStale:
            outcome = VisionOutcome(
                self.run_id,
                self.dataset_id,
                self.revision,
                VisionStatus.STALE,
                "Dataset changed during local visual review.",
            )
        except VisionUnavailable:
            outcome = VisionOutcome(
                self.run_id,
                self.dataset_id,
                self.revision,
                VisionStatus.UNAVAILABLE,
                "Local vision service is unavailable or incompatible.",
            )
        except VisionError:
            outcome = VisionOutcome(
                self.run_id,
                self.dataset_id,
                self.revision,
                VisionStatus.BLOCKED,
                "Local visual proposal or evidence failed validation.",
            )
        except Exception:  # noqa: BLE001 - contain worker failure without leaking model payloads
            outcome = VisionOutcome(
                self.run_id,
                self.dataset_id,
                self.revision,
                VisionStatus.BLOCKED,
                "Local visual review failed safely.",
            )
        finally:
            self._handle = None
            self._preflight = None
            with self._lock:
                self._inspector = None
        self.completed.emit(outcome)


__all__ = ["VisionWorker"]

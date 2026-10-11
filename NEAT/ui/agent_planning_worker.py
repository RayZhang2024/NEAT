"""Tracked, cooperative worker for the optional read-only planning graph."""

from __future__ import annotations

from threading import Event

from PyQt5.QtCore import QThread, pyqtSignal

from ..domain.agent_planning import (
    PlanningContext,
    PlanningErrorCode,
    PlanningOutcome,
    PlanningStatus,
)
from ..domain.observation import DatasetRevisionClock
from ..services.agent_planning import (
    PlanningCancelled,
    PlanningDependencyUnavailable,
    PlanningStale,
    run_planning,
)


class PlanningWorker(QThread):
    """Run only a safe scalar summary; never retain handles, frames, or plots."""

    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)

    def __init__(
        self,
        context: PlanningContext,
        revision_clock: DatasetRevisionClock,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.run_id = context.run_id
        self.dataset_id = context.dataset_id
        self.revision = context.revision
        self._context: PlanningContext | None = context
        self._revision_clock: DatasetRevisionClock | None = revision_clock
        self._cancel = Event()

    def stop(self) -> None:
        """Request cooperative cancellation between graph nodes."""
        self._cancel.set()

    def run(self) -> None:
        context = self._context
        revision_clock = self._revision_clock
        if context is None or revision_clock is None:
            return
        self.progress.emit(10, "Starting the local deterministic planning graph")
        try:
            outcome = run_planning(
                context,
                cancelled=self._cancel.is_set,
                current_dataset_token=revision_clock.current,
            )
            self.progress.emit(100, "Read-only planning stage complete")
        except PlanningDependencyUnavailable:
            outcome = PlanningOutcome(
                context.run_id,
                context.preflight_run_id,
                context.dataset_id,
                context.revision,
                PlanningStatus.UNAVAILABLE,
                error_code=PlanningErrorCode.DEPENDENCY_MISSING,
                message="Planning requires NEAT's optional [agent] extra.",
            )
        except PlanningCancelled:
            outcome = PlanningOutcome(
                context.run_id,
                context.preflight_run_id,
                context.dataset_id,
                context.revision,
                PlanningStatus.CANCELLED,
                error_code=PlanningErrorCode.CANCELLED,
                message="Autonomous analysis stopped during planning.",
            )
        except PlanningStale:
            outcome = PlanningOutcome(
                context.run_id,
                context.preflight_run_id,
                context.dataset_id,
                context.revision,
                PlanningStatus.STALE,
                error_code=PlanningErrorCode.STALE,
                message="Dataset changed — rerun autonomous analysis.",
            )
        except Exception:  # noqa: BLE001 - contain unexpected worker failures at the Qt boundary
            outcome = PlanningOutcome(
                context.run_id,
                context.preflight_run_id,
                context.dataset_id,
                context.revision,
                PlanningStatus.FAILED,
                error_code=PlanningErrorCode.UNEXPECTED,
                message="Planning could not complete safely. Retry the autonomous analysis.",
            )
        finally:
            # The graph state is scalar-only; release the worker's context and
            # revision-clock reference before publishing the bounded outcome.
            self._context = None
            self._revision_clock = None
        self.completed.emit(outcome)


__all__ = ["PlanningWorker"]

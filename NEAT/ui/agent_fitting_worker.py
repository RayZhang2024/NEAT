"""Tracked Qt worker for bounded local seven-ROI edge fitting."""

from __future__ import annotations

from threading import Event

from PyQt5.QtCore import QThread, pyqtSignal

from ..domain.agent_fitting import (
    AgentFitOutcome,
    AgentFitPlan,
    AgentFitRunStatus,
)
from ..domain.agent_model import AgentModelSettings
from ..services.agent_fitting import AgentFitBlocked, run_agent_fitting
from ..services.agent_model import AgentCredentialResolver, AgentModelError
from ..services.scientific_observation import ScientificDatasetHandle, StaleDatasetError


class AgentFitWorker(QThread):
    """Perform only local numerical work; never read a QWidget from this thread."""

    progress = pyqtSignal(object)
    completed = pyqtSignal(object)

    def __init__(
        self,
        handle: ScientificDatasetHandle,
        plan: AgentFitPlan,
        *,
        agent_settings: AgentModelSettings | None = None,
        langsmith_credential_resolver: AgentCredentialResolver | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.run_id = plan.run_id
        self.dataset_id = plan.dataset_id
        self.revision = plan.revision
        self._handle: ScientificDatasetHandle | None = handle
        self._plan: AgentFitPlan | None = plan
        self._agent_settings = agent_settings or AgentModelSettings()
        self._langsmith_credential_resolver = langsmith_credential_resolver
        self._cancel = Event()

    def stop(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        handle = self._handle
        plan = self._plan
        if handle is None or plan is None:
            return
        try:
            langsmith_api_key = None
            if (
                self._agent_settings.langsmith_tracing_enabled
                and self._langsmith_credential_resolver is not None
            ):
                try:
                    langsmith_api_key = self._langsmith_credential_resolver.get()
                except AgentModelError:
                    langsmith_api_key = None
            outcome = run_agent_fitting(
                handle,
                plan,
                cancelled=self._cancel.is_set,
                progress=self.progress.emit,
                langsmith_enabled=self._agent_settings.langsmith_tracing_enabled,
                langsmith_api_key=langsmith_api_key,
                langsmith_agent_id=self._agent_settings.langsmith_agent_id,
                langsmith_environment=self._agent_settings.langsmith_environment,
                langsmith_region=self._agent_settings.langsmith_region,
                langsmith_workspace_id=self._agent_settings.langsmith_workspace_id,
            )
        except StaleDatasetError:
            outcome = AgentFitOutcome(
                plan.run_id,
                plan.dataset_id,
                plan.revision,
                AgentFitRunStatus.STALE,
                plan.phase_id,
                plan.edges,
                (),
                len(plan.spatial_rois) * len(plan.edges),
                0,
                0,
                0,
                0,
                0,
                0,
                "Dataset changed during fitting; all displayed fit evidence was withdrawn.",
            )
        except AgentFitBlocked as exc:
            total = len(plan.spatial_rois) * len(plan.edges)
            outcome = AgentFitOutcome(
                plan.run_id,
                plan.dataset_id,
                plan.revision,
                AgentFitRunStatus.BLOCKED,
                plan.phase_id,
                plan.edges,
                (),
                total,
                0,
                0,
                0,
                0,
                total,
                0,
                str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - contain unexpected worker errors at Qt boundary
            total = len(plan.spatial_rois) * len(plan.edges)
            outcome = AgentFitOutcome(
                plan.run_id,
                plan.dataset_id,
                plan.revision,
                AgentFitRunStatus.FAILED,
                plan.phase_id,
                plan.edges,
                (),
                total,
                0,
                0,
                0,
                0,
                total,
                0,
                f"Local fitting failed safely ({type(exc).__name__}).",
            )
        finally:
            self._handle = None
            self._plan = None
        if self._cancel.is_set() and outcome.status is not AgentFitRunStatus.STALE:
            outcome = AgentFitOutcome(
                outcome.run_id,
                outcome.dataset_id,
                outcome.revision,
                AgentFitRunStatus.STOPPED,
                outcome.phase_id,
                outcome.edges,
                outcome.cells,
                outcome.total_fits,
                outcome.attempted,
                outcome.succeeded,
                outcome.failed,
                outcome.needs_review,
                outcome.skipped,
                outcome.cancelled,
                "Stop was requested as fitting completed; completed cell evidence is retained as stopped-run evidence.",
            )
        self.completed.emit(outcome)


__all__ = ["AgentFitWorker"]

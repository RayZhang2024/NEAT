"""Tracked Qt worker for bounded local seven-ROI edge fitting."""

from __future__ import annotations

import logging
from threading import Event

from PyQt5.QtCore import QThread, pyqtSignal

from ..domain.agent_fitting import (
    AgentFitCellResult,
    AgentFitCellState,
    AgentFitOutcome,
    AgentFitPlan,
    AgentFitRunStatus,
)
from ..domain.agent_model import AgentModelSettings
from ..services.agent_fitting import (
    AgentFitBlocked,
    roi_for_macro_pixel_size,
    run_agent_fitting,
)
from ..services.agent_model import AgentCredentialResolver, AgentModelError
from ..services.scientific_observation import ScientificDatasetHandle, StaleDatasetError

_LOGGER = logging.getLogger(__name__)


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
        mapping_calibration: bool = False,
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
        self._mapping_calibration = bool(mapping_calibration)
        self._cancel = Event()

    def stop(self) -> None:
        self._cancel.set()

    @staticmethod
    def _terminal_outcome(
        plan: AgentFitPlan,
        *,
        status: AgentFitRunStatus,
        cell_state: AgentFitCellState,
        message: str,
    ) -> AgentFitOutcome:
        cells = tuple(
            AgentFitCellResult(
                roi_index=roi_index,
                roi_label=candidate.label,
                roi=(
                    roi_for_macro_pixel_size(candidate.roi, size)
                    if candidate.roi is not None else None
                ),
                hkl=edge.hkl,
                state=cell_state,
                message=message,
                size_index=size_index,
                macro_pixel_width=size.width,
                macro_pixel_height=size.height,
            )
            for roi_index, candidate in enumerate(plan.spatial_rois)
            for size_index, size in enumerate(plan.macro_pixel_sizes)
            for edge in plan.edges
        )
        total = len(plan.spatial_rois) * len(plan.macro_pixel_sizes) * len(plan.edges)
        failed = total if cell_state is AgentFitCellState.FAILED else 0
        skipped = total if cell_state is AgentFitCellState.SKIPPED else 0
        return AgentFitOutcome(
            plan.run_id,
            plan.dataset_id,
            plan.revision,
            status,
            plan.phase_id,
            plan.edges,
            cells,
            total,
            0,
            0,
            failed,
            0,
            skipped,
            0,
            message,
            plan.macro_pixel_sizes,
        )

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
                mapping_calibration=self._mapping_calibration,
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
                len(plan.spatial_rois) * len(plan.macro_pixel_sizes) * len(plan.edges),
                0,
                0,
                0,
                0,
                0,
                0,
                "Dataset changed during fitting; all displayed fit evidence was withdrawn.",
                plan.macro_pixel_sizes,
            )
        except AgentFitBlocked as exc:
            outcome = self._terminal_outcome(
                plan,
                status=AgentFitRunStatus.BLOCKED,
                cell_state=AgentFitCellState.SKIPPED,
                message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - contain unexpected worker errors at Qt boundary
            _LOGGER.exception("Unexpected error in the local fitting workflow")
            try:
                detail = str(exc).strip()
            except Exception:  # noqa: BLE001 - an unusual exception must not escape Qt boundary
                detail = "Error details unavailable."
            if not detail:
                detail = "No additional error details were provided."
            outcome = self._terminal_outcome(
                plan,
                status=AgentFitRunStatus.FAILED,
                cell_state=AgentFitCellState.FAILED,
                message=(
                    f"Local fitting failed safely ({type(exc).__name__}: "
                    f"{detail[:300]}). See the launch console for the traceback."
                ),
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
                outcome.macro_pixel_sizes,
            )
        self.completed.emit(outcome)


__all__ = ["AgentFitWorker"]

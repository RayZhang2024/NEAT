"""Tracked, cooperative worker for the optional read-only planning graph."""

from __future__ import annotations

from dataclasses import replace
from threading import Event, Lock

from PyQt5.QtCore import QThread, pyqtSignal

from ..domain.agent_model import AgentModelMode, AgentRunSnapshot
from ..domain.agent_planning import (
    PlannerSource,
    PlanningContext,
    PlanningErrorCode,
    PlanningOutcome,
    PlanningStatus,
)
from ..domain.agent_preflight import PreflightReadiness
from ..domain.observation import DatasetRevisionClock
from ..services.agent_model import (
    AgentCredentialResolver,
    AgentModelError,
    AgentModelOutputRejected,
    LLMPlanProposer,
    MissingAgentCredential,
    StructuredModel,
)
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
        run_snapshot: AgentRunSnapshot | None = None,
        *,
        credential_resolver: AgentCredentialResolver | None = None,
        model_factory=None,
        model: StructuredModel | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.run_id = context.run_id
        self.dataset_id = context.dataset_id
        self.revision = context.revision
        self._context: PlanningContext | None = context
        self._revision_clock: DatasetRevisionClock | None = revision_clock
        self._run_snapshot = run_snapshot
        self._credential_resolver = credential_resolver
        self._model_factory = model_factory
        self._injected_model = model
        self._cancel = Event()
        self._proposer_lock = Lock()
        self._active_proposer = None

    def stop(self) -> None:
        """Request cooperative cancellation between graph nodes."""
        self._cancel.set()
        with self._proposer_lock:
            proposer = self._active_proposer
        if proposer is not None:
            proposer.cancel()

    def _needs_permission_outcome(self, context: PlanningContext) -> PlanningOutcome:
        return PlanningOutcome(
            context.run_id,
            context.preflight_run_id,
            context.dataset_id,
            context.revision,
            PlanningStatus.NEEDS_PERMISSION,
            error_code=PlanningErrorCode.PERMISSION_REQUIRED,
            message=(
                "Cloud planning was not run. Allow transmission of the disclosed "
                "planning summary for this run and start again."
            ),
            planner_source=PlannerSource.NONE,
            objective=context.objective,
        )

    def _error_outcome(
        self,
        context: PlanningContext,
        exc: AgentModelError,
        snapshot: AgentRunSnapshot,
    ) -> PlanningOutcome:
        if isinstance(exc, MissingAgentCredential):
            error_code = PlanningErrorCode.MISSING_CREDENTIAL
        elif isinstance(exc, AgentModelOutputRejected):
            error_code = PlanningErrorCode.MODEL_OUTPUT_REJECTED
        else:
            error_code = PlanningErrorCode.MODEL_UNAVAILABLE
        provider = "OpenAI" if snapshot.settings.mode is AgentModelMode.OPENAI_CLOUD else "Local model"
        return PlanningOutcome(
            context.run_id,
            context.preflight_run_id,
            context.dataset_id,
            context.revision,
            PlanningStatus.FAILED,
            error_code=error_code,
            message=str(exc),
            planner_source=PlannerSource.NONE,
            provider=provider,
            model_id=snapshot.settings.model_id,
            objective=context.objective,
        )

    def run(self) -> None:
        context = self._context
        revision_clock = self._revision_clock
        if context is None or revision_clock is None:
            return
        snapshot = self._run_snapshot
        provider = ""
        model_id = ""
        planner_source = PlannerSource.LOCAL_DETERMINISTIC
        proposer = None
        if snapshot is not None:
            model_id = snapshot.settings.model_id
            if snapshot.settings.mode is AgentModelMode.OPENAI_CLOUD:
                provider = "OpenAI"
            elif snapshot.settings.mode is AgentModelMode.LOCAL:
                provider = "Local model"

        self.progress.emit(10, "Validating the preflight before planning")
        try:
            if (
                snapshot is not None
                and snapshot.settings.mode is AgentModelMode.OPENAI_CLOUD
                and not snapshot.cloud_egress_consent
                and context.readiness is not PreflightReadiness.INVALID
            ):
                outcome = self._needs_permission_outcome(context)
            elif (
                snapshot is not None
                and snapshot.settings.mode is not AgentModelMode.OFFLINE
                and context.readiness is PreflightReadiness.READY
                and context.preflight_complete
            ):
                planner_source = PlannerSource.LLM_GENERATED

                class LazyProposer:
                    def __init__(inner_self) -> None:
                        inner_self.proposer: LLMPlanProposer | None = None

                    def __call__(inner_self, state):
                        if self._cancel.is_set():
                            raise PlanningCancelled("Planning was stopped.")

                        def check_live_before_request() -> None:
                            if self._cancel.is_set():
                                raise PlanningCancelled("Planning was stopped.")
                            if revision_clock.current() != (context.dataset_id, context.revision):
                                raise PlanningStale("The dataset changed before model planning.")

                        if inner_self.proposer is None:
                            inner_self.proposer = LLMPlanProposer(
                                context,
                                snapshot,
                                model=self._injected_model,
                                credential_resolver=self._credential_resolver,
                                model_factory=self._model_factory,
                                check_before_request=check_live_before_request,
                            )
                            with self._proposer_lock:
                                self._active_proposer = inner_self.proposer
                            if self._cancel.is_set():
                                inner_self.proposer.cancel()
                                raise PlanningCancelled("Planning was stopped.")
                        check_live_before_request()
                        return inner_self.proposer(state)

                    def cancel(inner_self) -> None:
                        if inner_self.proposer is not None:
                            inner_self.proposer.cancel()

                proposer = LazyProposer()
                with self._proposer_lock:
                    self._active_proposer = proposer
                try:
                    outcome = run_planning(
                        context,
                        cancelled=self._cancel.is_set,
                        current_dataset_token=revision_clock.current,
                        proposer=proposer,
                        planner_source=planner_source,
                        provider=provider,
                        model_id=model_id,
                    )
                    accepted = getattr(proposer, "proposer", None)
                    if accepted is not None:
                        outcome = replace(
                            outcome,
                            input_tokens=accepted.input_tokens,
                            output_tokens=accepted.output_tokens,
                        )
                except AgentModelError as exc:
                    if self._cancel.is_set():
                        raise PlanningCancelled("Planning was stopped.") from exc
                    if snapshot.deterministic_fallback:
                        self.progress.emit(80, "Using the selected deterministic fallback")
                        outcome = run_planning(
                            context,
                            cancelled=self._cancel.is_set,
                            current_dataset_token=revision_clock.current,
                            planner_source=PlannerSource.DETERMINISTIC_FALLBACK,
                            provider=provider,
                            model_id=model_id,
                        )
                        outcome = replace(
                            outcome,
                            message=(
                                "LLM unavailable / output rejected — deterministic fallback. "
                                + outcome.message
                            ),
                            planner_source=PlannerSource.DETERMINISTIC_FALLBACK,
                            provider=provider,
                            model_id=model_id,
                        )
                    else:
                        outcome = self._error_outcome(context, exc, snapshot)
            else:
                # A limited or invalid preflight remains deterministic. It is
                # never sent to a model, even when local/cloud was selected.
                outcome = run_planning(
                    context,
                    cancelled=self._cancel.is_set,
                    current_dataset_token=revision_clock.current,
                    planner_source=PlannerSource.LOCAL_DETERMINISTIC,
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
                objective=context.objective,
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
                planner_source=PlannerSource.NONE,
                objective=context.objective,
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
                planner_source=PlannerSource.NONE,
                objective=context.objective,
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
                objective=context.objective,
            )
        finally:
            with self._proposer_lock:
                self._active_proposer = None
            # The graph state is scalar-only; release the worker's context and
            # revision-clock reference before publishing the bounded outcome.
            self._context = None
            self._revision_clock = None
            self._run_snapshot = None
            self._credential_resolver = None
            self._model_factory = None
            self._injected_model = None
        self.completed.emit(outcome)


__all__ = ["PlanningWorker"]

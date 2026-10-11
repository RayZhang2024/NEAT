"""Independent, non-modal window for local read-only dataset preflight."""

from __future__ import annotations

from uuid import uuid4

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..domain.agent_preflight import (
    PreflightOutcome,
    PreflightResult,
    PreflightRunState,
)
from ..domain.agent_planning import (
    PlanStep,
    PlanningContext,
    PlanningOutcome,
    PlanningStatus,
    context_from_preflight,
)
from ..domain.observation import DatasetMode, RoiBounds
from ..services.scientific_observation import (
    DatasetUnavailableError,
    ObservationErrorBase,
)
from .agent_preflight_worker import PreflightWorker
from .agent_planning_worker import PlanningWorker


class AgentWindow(QMainWindow):
    """One top-level companion window per NEAT main window."""

    def __init__(self, main_window) -> None:
        super().__init__(main_window, Qt.Window)
        self._main_window = main_window
        self._worker: PreflightWorker | None = None
        self._planning_worker: PlanningWorker | None = None
        self._active_run_id: str | None = None
        self._agent_run_id: str | None = None
        self._stage = "idle"
        self._dataset_token: tuple[str, int] | None = None
        self._result: PreflightResult | None = None
        self._planning_outcome: PlanningOutcome | None = None
        self._stale = False
        self._parent_closing = False
        self._preflight_completion_received = False
        self._awaiting_preflight_completion = False
        self._pending_planning_context: PlanningContext | None = None
        self._cancellation_requested = False

        self.setWindowTitle("NEAT AI Agent — Autonomous Analysis (M3)")
        self.setMinimumSize(820, 520)
        self.resize(1000, 720)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setObjectName("preflightOverviewEvidenceSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

        self.overview_panel = QWidget(self.splitter)
        self.overview_panel.setObjectName("preflightOverviewPanel")
        overview_layout = QVBoxLayout(self.overview_panel)
        overview_heading = QLabel("Overview", self.overview_panel)
        overview_heading.setStyleSheet("font-size: 15px; font-weight: bold;")
        overview_layout.addWidget(overview_heading)
        title = QLabel("One-click Autonomous Analysis — M3", self.overview_panel)
        overview_layout.addWidget(title)
        self.scope_label = QLabel(
            "Local preflight + deterministic plan only; no LLM or scientific execution.",
            self.overview_panel,
        )
        self.scope_label.setWordWrap(True)
        self.scope_label.setObjectName("m3ScopeLimitLabel")
        overview_layout.addWidget(self.scope_label)
        self.status_label = QLabel("No preflight has been run.", self.overview_panel)
        self.status_label.setAccessibleName("Preflight status")
        self.status_label.setWordWrap(True)
        overview_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar(self.overview_panel)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setAccessibleName("Preflight progress")
        overview_layout.addWidget(self.progress_bar)
        controls = QHBoxLayout()
        self.start_autonomous_button = QPushButton(
            "Start Autonomous Analysis", self.overview_panel
        )
        self.start_autonomous_button.setAccessibleName("Start Autonomous Analysis")
        self.start_autonomous_button.setObjectName("startAutonomousAnalysisButton")
        self.start_autonomous_button.clicked.connect(self.start_autonomous_analysis)
        controls.addWidget(self.start_autonomous_button)
        self.start_button = QPushButton("Start Preflight", self.overview_panel)
        self.start_button.setAccessibleName("Start local read-only preflight")
        self.start_button.clicked.connect(self.start_preflight)
        controls.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop", self.overview_panel)
        self.stop_button.setAccessibleName("Cancel current preflight")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_preflight)
        controls.addWidget(self.stop_button)
        controls.addStretch()
        overview_layout.addLayout(controls)
        self.assessment_text = QPlainTextEdit(self.overview_panel)
        self.assessment_text.setReadOnly(True)
        self.assessment_text.setAccessibleName("Dataset assessment")
        self.assessment_text.setPlainText(
            "Start Autonomous Analysis to run preflight and then prepare a "
            "deterministic read-only plan in one action. Start Preflight remains "
            "available as a diagnostic action."
        )
        overview_layout.addWidget(self.assessment_text)
        plan_heading = QLabel("Read-only plan proposal", self.overview_panel)
        plan_heading.setStyleSheet("font-weight: bold;")
        overview_layout.addWidget(plan_heading)
        self.plan_text = QPlainTextEdit(self.overview_panel)
        self.plan_text.setReadOnly(True)
        self.plan_text.setMaximumHeight(170)
        self.plan_text.setAccessibleName("Deterministic plan proposal")
        self.plan_text.setPlainText(
            "The plan will appear automatically after a successful current preflight."
        )
        overview_layout.addWidget(self.plan_text)

        self.evidence_panel = QWidget(self.splitter)
        self.evidence_panel.setObjectName("preflightEvidencePanel")
        evidence_layout = QVBoxLayout(self.evidence_panel)
        evidence_heading = QLabel("Visual Evidence", self.evidence_panel)
        evidence_heading.setStyleSheet("font-size: 15px; font-weight: bold;")
        evidence_layout.addWidget(evidence_heading)
        self.evidence_scroll = QScrollArea(self.evidence_panel)
        self.evidence_scroll.setWidgetResizable(True)
        self.evidence_scroll.setAccessibleName("Visual evidence")
        self.evidence_content = QWidget(self.evidence_scroll)
        self.evidence_layout = QVBoxLayout(self.evidence_content)
        self.evidence_layout.setAlignment(Qt.AlignTop)
        self.evidence_scroll.setWidget(self.evidence_content)
        evidence_layout.addWidget(self.evidence_scroll)
        self.splitter.addWidget(self.overview_panel)
        self.splitter.addWidget(self.evidence_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([390, 610])
        self._show_evidence_message(
            "No visual evidence yet. Start Autonomous Analysis or Start Preflight to inspect the dataset."
        )

        self._revision_timer = QTimer(self)
        self._revision_timer.setInterval(200)
        self._revision_timer.timeout.connect(self._check_revision)

    @property
    def result(self) -> PreflightResult | None:
        return self._result

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _clear_evidence(self) -> None:
        while self.evidence_layout.count():
            item = self.evidence_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _show_evidence_message(self, message: str) -> None:
        self._clear_evidence()
        label = QLabel(message, self.evidence_content)
        label.setWordWrap(True)
        self.evidence_layout.addWidget(label)

    def _check_revision(self) -> None:
        if self._dataset_token is None:
            self._revision_timer.stop()
            return
        adapter = getattr(self._main_window, "scientific_observation_api", None)
        if adapter is None or adapter.revision_clock.current() != self._dataset_token:
            self._mark_stale()

    def _mark_stale(self) -> None:
        if self._stale:
            return
        self._stale = True
        self._result = None
        self._planning_outcome = None
        if self._worker is not None:
            self._worker.stop()
        if self._planning_worker is not None:
            self._planning_worker.stop()
        self._stage = "stale"
        stale_message = (
            "Dataset changed — rerun autonomous analysis."
            if self._agent_run_id is not None
            else "Dataset changed — rerun preflight."
        )
        self._set_status(stale_message)
        self.plan_text.setPlainText(stale_message)
        self.assessment_text.setPlainText("Previous evidence was discarded because the dataset changed.")
        self._show_evidence_message(
            "Dataset changed — rerun autonomous analysis to inspect current evidence."
            if self._agent_run_id is not None
            else "Dataset changed — rerun preflight to see current evidence."
        )
        if self._worker is None and self._planning_worker is None:
            self._revision_timer.stop()
        self._update_controls()

    def _has_active_worker(self) -> bool:
        inventory = getattr(self._main_window, "_shutdown_worker_inventory", None)
        tracked_worker = bool(
            inventory is not None
            and any(
                inventory.has_role(role)
                for role in ("agent_preflight", "agent_planning")
            )
        )
        return bool(
            self._worker is not None
            or self._planning_worker is not None
            or self._awaiting_preflight_completion
            or self._pending_planning_context is not None
            or tracked_worker
        )

    def _can_stop_active_run(self) -> bool:
        if self._cancellation_requested:
            return False
        return bool(
            self._awaiting_preflight_completion
            or self._pending_planning_context is not None
            or (
                self._worker is not None
                and not self._preflight_completion_received
            )
            or (
                self._planning_worker is not None
                and self._planning_outcome is None
            )
        )

    def _update_controls(self) -> None:
        active = self._has_active_worker()
        host_closing = bool(
            getattr(self._main_window, "_shutdown_close_pending", False)
            or getattr(self._main_window, "_shutdown_committed", False)
        )
        can_start = not active and not self._parent_closing and not host_closing
        self.start_button.setEnabled(can_start)
        self.start_autonomous_button.setEnabled(can_start)
        self.stop_button.setEnabled(
            self._can_stop_active_run() and not self._parent_closing
        )
        if not self._can_stop_active_run():
            self.stop_button.setText("Stop")

    def start_autonomous_analysis(self) -> None:
        """Run existing preflight and automatically continue to read-only planning."""
        if self._has_active_worker() or self._parent_closing:
            return
        run_id = uuid4().hex
        self._agent_run_id = run_id
        self._planning_outcome = None
        self._stage = "preflight_running"
        self.plan_text.clear()
        self.plan_text.setPlainText("Waiting for the current dataset preflight.")
        self.start_preflight(_agent_run_id=run_id)

    def start_preflight(self, *, _agent_run_id: str | None = None) -> None:
        """Capture live state only on the Qt owner thread, then start bounded work."""
        if self._has_active_worker() or self._parent_closing:
            return
        if getattr(self._main_window, "_shutdown_close_pending", False) or getattr(self._main_window, "_shutdown_committed", False):
            self._set_status("NEAT is closing; retry after the close attempt finishes.")
            return
        self._result = None
        self._stale = False
        self._cancellation_requested = False
        self._dataset_token = None
        self._preflight_completion_received = False
        self._awaiting_preflight_completion = True
        self._pending_planning_context = None
        self._planning_outcome = None
        if _agent_run_id is None:
            self._agent_run_id = None
            self._stage = "preflight_running"
        self._revision_timer.stop()
        self.progress_bar.setValue(0)
        self._show_evidence_message("Preflight is preparing the observation snapshot.")
        self.assessment_text.clear()
        if _agent_run_id is None:
            self.plan_text.setPlainText(
                "Diagnostic preflight only. Use Start Autonomous Analysis to run planning afterward."
            )
        run_id = uuid4().hex
        self._active_run_id = run_id
        adapter = getattr(self._main_window, "scientific_observation_api", None)
        if adapter is None:
            self._awaiting_preflight_completion = False
            self._set_status("Scientific observation API is unavailable.")
            self._stage = "failed"
            self._update_controls()
            return
        try:
            handle = adapter.capture_dataset()
            roi = adapter.capture_selected_roi(handle)
        except DatasetUnavailableError:
            self._awaiting_preflight_completion = False
            self._set_status("Dataset loader is busy — retry preflight after loading finishes.")
            self._stage = "busy_retry"
            self._update_controls()
            return
        except (ObservationErrorBase, ValueError):
            self._awaiting_preflight_completion = False
            self._set_status("Dataset capture is unavailable — check loaded data and retry.")
            self._stage = "blocked_invalid_data"
            self._update_controls()
            return

        self._dataset_token = (handle.dataset_id, handle.revision)
        worker = PreflightWorker(run_id, handle, roi, parent=self._main_window)
        self._worker = worker
        self._main_window.agent_preflight_worker = worker
        self._main_window._track_shutdown_worker(worker, "agent_preflight")
        self._main_window._connect_shutdown_worker_signal(
            worker, worker.progress,
            lambda percent, message, rid=run_id: self._on_progress(rid, percent, message),
            current_attribute="agent_preflight_worker",
        )
        self._main_window._connect_shutdown_worker_signal(
            worker, worker.completed, self._on_completed,
            current_attribute="agent_preflight_worker",
        )
        self._main_window._connect_shutdown_worker_signal(
            worker, worker.finished,
            lambda rid=run_id, current=worker: self._on_worker_finished(rid, current),
            current_attribute="agent_preflight_worker",
        )
        self.start_autonomous_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_button.setText("Stop")
        self._set_status(
            "Preflight running locally. Planning will start automatically after current evidence is validated."
            if _agent_run_id is not None
            else "Preflight running locally. No model or network request is used."
        )
        self._revision_timer.start()
        worker.start()

    def stop_preflight(self) -> None:
        """Keep the historical shutdown method name for existing callers."""
        self.stop_active_run()

    def stop_active_run(self) -> None:
        if not self._can_stop_active_run():
            return
        # Record cancellation for the whole run before asking a worker to stop.
        # The worker may already have emitted a successful result whose queued
        # Qt callback has not reached this window yet.
        self._cancellation_requested = True
        self._pending_planning_context = None
        for worker in (self._worker, self._planning_worker):
            if worker is not None:
                try:
                    worker.stop()
                except RuntimeError:
                    # Keep the reference until the shutdown inventory confirms
                    # retirement; clearing it here could allow overlapping runs.
                    pass
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stopping…")
        self._stage = "cancellation_requested"
        if not self._stale:
            self._set_status("Stopping — waiting for the current bounded operation to finish safely.")
        self._update_controls()

    def _on_progress(self, run_id: str, percent: int, message: str) -> None:
        if run_id != self._active_run_id or self._parent_closing or self._stale:
            return
        self._check_revision()
        if self._stale:
            return
        self.progress_bar.setValue(percent)
        if self.stop_button.isEnabled():
            self._set_status(message)

    def _on_completed(self, outcome: PreflightOutcome) -> None:
        if (
            outcome.run_id != self._active_run_id
            or self._parent_closing
            or getattr(self._main_window, "_shutdown_close_pending", False)
        ):
            return
        self._preflight_completion_received = True
        self._awaiting_preflight_completion = False
        if self._cancellation_requested:
            self._pending_planning_context = None
            self._result = None
            self._planning_outcome = None
            self._stage = "cancelled"
            message = (
                "Autonomous analysis stopped during preflight."
                if self._agent_run_id is not None
                else "Preflight cancelled."
            )
            self._set_status(message)
            self.assessment_text.setPlainText(
                "The preflight result was discarded because Stop was requested."
            )
            self.plan_text.setPlainText(message)
            self._show_evidence_message(message)
            self._update_controls()
            return
        self._check_revision()
        if self._stale or self._dataset_token != (outcome.dataset_id, outcome.revision):
            self._mark_stale()
            return
        if outcome.state is PreflightRunState.COMPLETE and outcome.result is not None:
            self._result = outcome.result
            self.progress_bar.setValue(100)
            self._render_result(outcome.result)
            if self._agent_run_id is not None:
                try:
                    context = context_from_preflight(
                        outcome.result, run_id=self._agent_run_id
                    )
                except (ValueError, TypeError):
                    self._stage = "failed"
                    self._set_status(
                        "Preflight completed, but its safe planning summary was invalid."
                    )
                    self.plan_text.setPlainText(
                        "Planning stopped because the preflight summary failed validation."
                    )
                else:
                    self._pending_planning_context = context
                    if self._worker is None:
                        self._start_pending_planning()
        else:
            self._result = None
            self._show_evidence_message(outcome.message or "No evidence was produced.")
            self.assessment_text.setPlainText(outcome.message or "Preflight did not complete.")
            self._set_status(outcome.message or "Preflight did not complete.")
            if outcome.state is PreflightRunState.STALE:
                self._mark_stale()
            elif outcome.state is PreflightRunState.CANCELLED:
                self._stage = "cancelled"
                self.plan_text.setPlainText("Autonomous analysis stopped during preflight.")
            elif outcome.state is PreflightRunState.BUSY:
                self._stage = "busy_retry"
            else:
                self._stage = "failed"

    def _start_pending_planning(self) -> None:
        context = self._pending_planning_context
        if (
            context is None
            or self._cancellation_requested
            or self._stale
            or self._parent_closing
            or getattr(self._main_window, "_shutdown_close_pending", False)
            or getattr(self._main_window, "_shutdown_committed", False)
        ):
            return
        self._pending_planning_context = None
        self._start_planning(context)

    def _start_planning(self, context: PlanningContext) -> None:
        if (
            self._agent_run_id != context.run_id
            or self._worker is not None
            or self._stale
            or self._cancellation_requested
            or self._parent_closing
        ):
            return
        adapter = getattr(self._main_window, "scientific_observation_api", None)
        if adapter is None or adapter.revision_clock.current() != (
            context.dataset_id,
            context.revision,
        ):
            self._mark_stale()
            return
        worker = PlanningWorker(
            context, adapter.revision_clock, parent=self._main_window
        )
        self._planning_worker = worker
        self._main_window.agent_planning_worker = worker
        self._main_window._track_shutdown_worker(worker, "agent_planning")
        self._main_window._connect_shutdown_worker_signal(
            worker,
            worker.progress,
            lambda percent, message, rid=context.run_id, current=worker: self._on_planning_progress(
                rid, current, percent, message
            ),
            current_attribute="agent_planning_worker",
        )
        self._main_window._connect_shutdown_worker_signal(
            worker,
            worker.completed,
            lambda outcome, rid=context.run_id, current=worker: self._on_planning_completed(
                rid, current, outcome
            ),
            current_attribute="agent_planning_worker",
        )
        self._main_window._connect_shutdown_worker_signal(
            worker,
            worker.finished,
            lambda rid=context.run_id, current=worker: self._on_planning_finished(
                rid, current
            ),
            current_attribute="agent_planning_worker",
        )
        self._stage = "planning_running"
        self.progress_bar.setValue(90)
        self.plan_text.setPlainText(
            "Preflight is current. The deterministic local LangGraph planner is preparing a read-only proposal."
        )
        self._set_status("Preflight complete — planning is advancing automatically.")
        worker.start()

    def _on_planning_progress(
        self, run_id: str, worker: PlanningWorker, percent: int, message: str
    ) -> None:
        if (
            run_id != self._agent_run_id
            or worker is not self._planning_worker
            or self._parent_closing
            or self._stale
            or getattr(self._main_window, "_shutdown_close_pending", False)
        ):
            return
        self._check_revision()
        if self._stale:
            return
        self.progress_bar.setValue(max(90, percent))
        if self.stop_button.isEnabled():
            self._set_status(message)

    @staticmethod
    def _format_plan_step(step: PlanStep, index: int) -> tuple[str, ...]:
        evidence = ", ".join(step.evidence_ids) if step.evidence_ids else "none"
        input_note = "Human input required" if step.requires_human_input else "Read-only proposal"
        return (
            f"{index}. {step.step_type.value}",
            f"   {step.rationale}",
            f"   Evidence: {evidence}",
            f"   {input_note}; supported by current preflight: {step.supported}.",
        )

    def _on_planning_completed(
        self, run_id: str, worker: PlanningWorker, outcome: PlanningOutcome
    ) -> None:
        if (
            run_id != self._agent_run_id
            or worker is not self._planning_worker
            or outcome.run_id != run_id
            or self._parent_closing
            or getattr(self._main_window, "_shutdown_close_pending", False)
        ):
            return
        if self._cancellation_requested:
            self._planning_outcome = (
                outcome if outcome.status is PlanningStatus.CANCELLED else None
            )
            self._stage = "cancelled"
            message = "Autonomous analysis stopped during planning."
            self._set_status(message)
            self.plan_text.setPlainText(message)
            self._update_controls()
            return
        self._check_revision()
        if self._stale or self._dataset_token != (outcome.dataset_id, outcome.revision):
            self._mark_stale()
            return
        if self._result is None or self._result.run_id != outcome.preflight_run_id:
            self._mark_stale()
            return
        self._planning_outcome = outcome
        self._stage = outcome.status.value
        lines = [
            f"Status: {outcome.status.value.replace('_', ' ')}",
            f"Agent run: {outcome.run_id}",
            f"Preflight run: {outcome.preflight_run_id}",
            f"Dataset: {outcome.dataset_id} revision {outcome.revision}",
            f"Graph transitions: {' → '.join(outcome.transitions) or 'none'}",
        ]
        if outcome.error_code is not None:
            lines.append(f"Reason code: {outcome.error_code.value}")
        if outcome.steps:
            lines.extend(("", "Proposed steps (not executed):"))
            for index, step in enumerate(outcome.steps, start=1):
                lines.extend(self._format_plan_step(step, index))
        if outcome.status is PlanningStatus.READY_PLAN:
            lines.extend(
                (
                    "",
                    "Plan prepared — scientific execution is not yet implemented.",
                    "No fitting, mapping, phase identification, or source-data changes occurred.",
                )
            )
            self._set_status(
                "Plan prepared — scientific execution is not yet implemented."
            )
            self.progress_bar.setValue(100)
        elif outcome.status is PlanningStatus.NEEDS_INPUT:
            self._set_status("Plan needs user input before it can be considered ready.")
        elif outcome.status is PlanningStatus.BLOCKED_INVALID_DATA:
            self._set_status("Autonomous analysis stopped — preflight data or evidence is invalid.")
        elif outcome.status is PlanningStatus.UNAVAILABLE:
            self._set_status(
                "Preflight complete — planning is unavailable; install the optional [agent] extra."
            )
        elif outcome.status is PlanningStatus.CANCELLED:
            self._set_status("Autonomous analysis stopped during planning.")
        elif outcome.status is PlanningStatus.STALE:
            self._mark_stale()
            return
        else:
            self._set_status("Planning failed safely; no scientific action was run.")
        if outcome.message:
            lines.extend(("", outcome.message))
        self.plan_text.setPlainText("\n".join(lines))

    def _on_planning_finished(self, run_id: str, worker: PlanningWorker) -> None:
        if self._planning_worker is worker:
            self._planning_worker = None
        if getattr(self._main_window, "agent_planning_worker", None) is worker:
            self._main_window.agent_planning_worker = None
        if (
            run_id == self._agent_run_id
            and not self._parent_closing
            and not self._cancellation_requested
        ):
            if self._planning_outcome is None and not self._stale:
                self._stage = "failed"
                self._set_status("Planning stopped before a result was received.")
                self.plan_text.setPlainText("No plan was produced.")
        self._update_controls()
        if not self._has_active_worker() and self._result is None:
            self._revision_timer.stop()

    def _on_worker_finished(self, run_id: str, worker: PreflightWorker) -> None:
        if self._worker is worker:
            self._worker = None
        if getattr(self._main_window, "agent_preflight_worker", None) is worker:
            self._main_window.agent_preflight_worker = None
        if (
            run_id == self._active_run_id
            and not self._cancellation_requested
            and self._preflight_completion_received
            and self._stage == "preflight_running"
            and self._agent_run_id is None
        ):
            self._stage = "complete"
        self._update_controls()
        if self._result is None and self._planning_worker is None:
            self._revision_timer.stop()

    def worker_retired(self, worker: PreflightWorker, callbacks_suppressed: bool) -> None:
        """Clear the window reference even when shutdown suppressed signal slots."""
        if self._worker is worker:
            self._worker = None
        if getattr(worker, "run_id", None) == self._active_run_id:
            self._awaiting_preflight_completion = False
        if not self._parent_closing:
            if self._cancellation_requested and not self._stale:
                self._pending_planning_context = None
                self._result = None
                self._planning_outcome = None
                if self._stage == "cancellation_requested":
                    self._set_status(
                        "Autonomous analysis stopped during preflight."
                        if self._agent_run_id is not None
                        else "Preflight cancelled."
                    )
                self._stage = "cancelled"
            elif callbacks_suppressed and not self._stale:
                self._pending_planning_context = None
                self._result = None
                self._revision_timer.stop()
                self._set_status("Preflight stopped while NEAT was closing. Start again if NEAT remains open.")
            elif self._pending_planning_context is not None:
                self._start_pending_planning()
            self._update_controls()

    def planning_worker_retired(
        self, worker: PlanningWorker, callbacks_suppressed: bool
    ) -> None:
        """Clear window ownership when shutdown retires the planning QThread."""
        if self._planning_worker is worker:
            self._planning_worker = None
        if not self._parent_closing:
            if self._cancellation_requested and not self._stale:
                if (
                    self._planning_outcome is None
                    or self._planning_outcome.status is not PlanningStatus.CANCELLED
                ):
                    self._planning_outcome = None
                if self._stage == "cancellation_requested":
                    message = "Autonomous analysis stopped during planning."
                    self._set_status(message)
                    self.plan_text.setPlainText(message)
                self._stage = "cancelled"
            elif callbacks_suppressed and not self._stale:
                self._planning_outcome = None
                self.plan_text.clear()
                self._set_status(
                    "Planning stopped while NEAT was closing. Start again if NEAT remains open."
                )
            self._update_controls()

    @staticmethod
    def _preview_pixmap(image, roi: RoiBounds | None) -> QPixmap:
        pixmap = QPixmap()
        if not pixmap.loadFromData(image.png_bytes or b"", "PNG"):
            return pixmap
        mapping = image.coordinate_mapping
        if roi is None or mapping is None:
            return pixmap
        painted = pixmap.copy()
        painter = QPainter(painted)
        pen = QPen(QColor("yellow"))
        pen.setWidth(2)
        painter.setPen(pen)
        x = round(roi.x_min * mapping.preview_width / mapping.original_width)
        y = round(roi.y_min * mapping.preview_height / mapping.original_height)
        right = round(roi.x_max * mapping.preview_width / mapping.original_width)
        bottom = round(roi.y_max * mapping.preview_height / mapping.original_height)
        painter.drawRect(x, y, max(1, right - x) - 1, max(1, bottom - y) - 1)
        painter.end()
        return painted

    def _render_result(self, result: PreflightResult) -> None:
        dataset = result.dataset
        assessment = result.assessment
        self._set_status(
            f"Preflight complete — {assessment.readiness.value.replace('_', ' ')}."
        )
        lines = [
            f"Run: {result.run_id}",
            f"Dataset: {result.dataset_id}  revision {result.revision}",
            f"Assessment: {assessment.readiness.value.replace('_', ' ')}",
            f"Mode: {dataset.mode.value}",
            f"Frames: {dataset.frame_count if dataset.frame_count is not None else 'not applicable'}",
            f"Image shape (rows, columns): {dataset.image_shape if dataset.image_shape else 'not applicable'}",
            f"Wavelength points: {dataset.wavelength_count}",
            f"Wavelength range (Å): {dataset.wavelength_min} to {dataset.wavelength_max}",
            f"Wavelength axis: {dataset.wavelength_axis_status.value}",
            f"Input format: {dataset.provenance.input_format}",
            f"Wavelength source: {dataset.provenance.wavelength_source}",
            f"Flight path source: {dataset.provenance.flight_path_source}",
            f"Flight path (m): {dataset.provenance.flight_path_m}",
            f"Delay (ms): {dataset.provenance.delay_ms}",
            (
                f"Capabilities: images={dataset.capabilities.images_available}, "
                f"spectrum={dataset.capabilities.spectrum_available}, "
                f"mapping applicable={dataset.capabilities.mapping_applicable}"
            ),
            "",
            "Observed facts:",
            *assessment.observations,
            "",
            "Evidence references:",
            *(assessment.evidence_references or ("None",)),
            "",
            "Warnings:",
            *(assessment.warnings or ("None",)),
            "",
            "Unsupported operations:",
            *(assessment.unsupported_operations or ("None",)),
        ]
        self.assessment_text.setPlainText("\n".join(lines))
        self._clear_evidence()
        if dataset.mode is DatasetMode.PROFILE:
            self.evidence_layout.addWidget(QLabel("Spectrum-only input: image controls do not apply."))
        elif result.frames:
            self.evidence_layout.addWidget(QLabel("Measured image previews; image row zero is at the top."))
            for frame in result.frames:
                caption = f"Frame {frame.frame_index}"
                if frame.wavelength is not None:
                    caption += f" — {frame.wavelength:.6g} Å"
                self.evidence_layout.addWidget(QLabel(caption))
                label = QLabel(self.evidence_content)
                label.setAccessibleName(caption + " image preview")
                label.setPixmap(self._preview_pixmap(frame.image, result.roi))
                self.evidence_layout.addWidget(label)
        else:
            self.evidence_layout.addWidget(QLabel("Image previews are unavailable."))

        if result.roi is not None:
            roi = result.roi
            self.evidence_layout.addWidget(QLabel(
                f"Selected NEAT ROI overlay and spectrum source: "
                f"x[{roi.x_min},{roi.x_max}), y[{roi.y_min},{roi.y_max}) (half-open pixels)."
            ))
        if result.spectrum is None:
            self.evidence_layout.addWidget(QLabel(
                "ROI spectrum not yet available." if dataset.mode is DatasetMode.IMAGES
                else "Measured spectrum unavailable."
            ))
        else:
            spectrum = result.spectrum
            self.evidence_layout.addWidget(QLabel(
                f"Measured intensity ({spectrum.intensity_unit}); full source: "
                f"{spectrum.sample_count} points, {spectrum.finite_pair_count} finite pairs, "
                f"{spectrum.nonfinite_intensity_count} non-finite intensities. "
                f"Axis: {spectrum.wavelength_axis_status.value}. "
                f"Plot window (Å): {result.plot.display_window if result.plot else 'unavailable'}."
            ))
            if result.plot is not None and result.plot.png_bytes is not None:
                plot = QLabel(self.evidence_content)
                plot.setAccessibleName("Measured spectrum plot")
                pixmap = QPixmap()
                if pixmap.loadFromData(result.plot.png_bytes, "PNG"):
                    plot.setPixmap(pixmap)
                    self.evidence_layout.addWidget(plot)
        self.evidence_layout.addStretch()

    def shutdown_for_parent_close(self) -> None:
        """Called only after the main window's worker inventory has drained."""
        self._parent_closing = True
        self._active_run_id = None
        self._agent_run_id = None
        self._pending_planning_context = None
        self._awaiting_preflight_completion = False
        self._revision_timer.stop()
        self._result = None
        self._planning_outcome = None
        self._dataset_token = None
        self.plan_text.clear()
        self._show_evidence_message("NEAT is closing.")
        self.hide()

    def closeEvent(self, event) -> None:
        self.stop_active_run()
        self._active_run_id = None
        self._agent_run_id = None
        self._pending_planning_context = None
        self._awaiting_preflight_completion = False
        self._result = None
        self._planning_outcome = None
        self._dataset_token = None
        self.plan_text.clear()
        self._revision_timer.stop()
        self.assessment_text.setPlainText(
            "The Agent window was closed. Any active stage is stopping safely."
        )
        self._show_evidence_message("No current evidence is displayed.")
        self.hide()
        event.accept()

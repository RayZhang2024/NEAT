"""Independent, non-modal window for local read-only dataset preflight."""

from __future__ import annotations

import json
from uuid import uuid4

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..domain.agent_model import (
    AgentModelMode,
    AgentModelSettings,
    AgentRunSnapshot,
    canonical_planning_schema,
)
from ..domain.agent_planning import (
    PlanningContext,
    PlanningErrorCode,
    PlanningObjective,
    PlanningOutcome,
    PlanningStatus,
    PlanStep,
    PlanStepType,
    context_from_preflight,
)
from ..domain.agent_preflight import (
    PreflightOutcome,
    PreflightResult,
    PreflightRunState,
)
from ..domain.agent_vision import VisionOutcome, VisionStatus
from ..domain.observation import (
    DatasetMode,
    ObservationStatus,
    RoiBounds,
    WavelengthAxisStatus,
)
from ..services.agent_model import (
    AgentCredentialResolver,
    AgentKeyringCredentialStore,
    AgentModelSettingsRepository,
    canonical_payload_for_context,
)
from ..services.agent_vision import render_roi_overlay
from ..services.scientific_observation import (
    DatasetUnavailableError,
    ObservationErrorBase,
)
from .agent_model_settings_dialog import AgentModelSettingsDialog
from .agent_planning_worker import PlanningWorker
from .agent_preflight_worker import PreflightWorker
from .agent_vision_worker import VisionWorker


class AgentWindow(QMainWindow):
    """One top-level companion window per NEAT main window."""

    def __init__(self, main_window) -> None:
        super().__init__(main_window, Qt.Window)
        self._main_window = main_window
        self._worker: PreflightWorker | None = None
        self._planning_worker: PlanningWorker | None = None
        self._vision_worker: VisionWorker | None = None
        self._active_run_id: str | None = None
        self._agent_run_id: str | None = None
        self._stage = "idle"
        self._dataset_token: tuple[str, int] | None = None
        self._result: PreflightResult | None = None
        self._planning_outcome: PlanningOutcome | None = None
        self._planning_display_text = ""
        self._vision_outcome: VisionOutcome | None = None
        self._pending_vision_result: PreflightResult | None = None
        self._vision_run_enabled = False
        self._adjusted_roi: RoiBounds | None = None
        self._adjusted_roi_dataset_token: tuple[str, int] | None = None
        self._stale = False
        self._parent_closing = False
        self._preflight_completion_received = False
        self._awaiting_preflight_completion = False
        self._pending_planning_context: PlanningContext | None = None
        self._cancellation_requested = False
        self._agent_settings_repository = AgentModelSettingsRepository()
        self._agent_settings: AgentModelSettings = self._agent_settings_repository.load()
        self._run_snapshot: AgentRunSnapshot | None = None
        self._run_credential_resolver: AgentCredentialResolver | None = None

        self.setWindowTitle("NEAT AI Agent — Visual Review (M5A-1)")
        self.setMinimumSize(820, 520)
        self.resize(1280, 820)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setObjectName("preflightOverviewEvidenceSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(10)
        self.splitter.setOpaqueResize(True)
        self.splitter.setAccessibleName("Adjustable divider between Overview and Visual Evidence")
        self.splitter.setToolTip("Drag the divider to resize Overview and Visual Evidence.")
        self.splitter.setStyleSheet(
            "QSplitter::handle:horizontal { background: #c5cbd3; border: 1px solid #aab2bc; }"
            "QSplitter::handle:horizontal:hover { background: #7b9fc8; }"
        )
        self.setCentralWidget(self.splitter)

        self.overview_scroll = QScrollArea(self.splitter)
        self.overview_scroll.setWidgetResizable(True)
        self.overview_scroll.setFrameShape(QFrame.NoFrame)
        self.overview_scroll.setMinimumWidth(0)
        self.overview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.overview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.overview_scroll.setAccessibleName("Overview")
        self.overview_panel = QWidget(self.overview_scroll)
        self.overview_panel.setObjectName("preflightOverviewPanel")
        self.overview_panel.setMinimumWidth(0)
        self.overview_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        overview_layout = QVBoxLayout(self.overview_panel)
        overview_layout.setSpacing(6)
        overview_heading = QLabel("Overview", self.overview_panel)
        overview_heading.setStyleSheet("font-size: 15px; font-weight: bold;")
        overview_layout.addWidget(overview_heading)
        title = QLabel("Prepare a read-only analysis", self.overview_panel)
        overview_layout.addWidget(title)
        self.scope_label = QLabel(
            "NEAT checks the dataset, prepares a read-only plan when the required evidence is available, and reviews bounded visual evidence with the configured model. In OpenAI mode, pressing Start authorizes a metadata-only request when eligible and up to 7 bounded PNGs in 3 visual requests (up to 3 representative image previews, 1 spectrum plot, up to 3 phase overlays). Saving settings sends nothing. Model failures stop with a reason; no offline plan is substituted. No fitting or mapping is run.",
            self.overview_panel,
        )
        self.scope_label.setWordWrap(True)
        self.scope_label.setObjectName("m3ScopeLimitLabel")
        overview_layout.addWidget(self.scope_label)

        self.settings_button = QPushButton("Agent settings…", self.overview_panel)
        self.settings_button.clicked.connect(self.open_model_settings)
        self.model_readiness_label = QLabel(self.overview_panel)
        self.model_readiness_label.setWordWrap(True)
        self.model_readiness_label.setAccessibleName("Agent model readiness")
        overview_layout.addWidget(self.model_readiness_label)
        self.status_label = QLabel("No analysis has been started.", self.overview_panel)
        self.status_label.setAccessibleName("Preflight status")
        self.status_label.setWordWrap(True)
        controls = QHBoxLayout()
        self.start_autonomous_button = QPushButton(
            "Start Autonomous Analysis", self.overview_panel
        )
        self.start_autonomous_button.setAccessibleName("Start Autonomous Analysis")
        self.start_autonomous_button.setObjectName("startAutonomousAnalysisButton")
        self.start_autonomous_button.setMinimumHeight(36)
        self.start_autonomous_button.setStyleSheet("font-weight: 600; padding: 6px 12px;")
        self.start_autonomous_button.setToolTip(
            "Check the dataset and prepare a read-only plan in one action."
        )
        self.start_autonomous_button.clicked.connect(self.start_autonomous_analysis)
        controls.addWidget(self.start_autonomous_button, 1)
        self.stop_button = QPushButton("Stop", self.overview_panel)
        self.stop_button.setAccessibleName("Cancel current preflight")
        self.stop_button.setEnabled(False)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_preflight)
        controls.addWidget(self.stop_button)
        overview_layout.addLayout(controls)

        overview_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar(self.overview_panel)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setAccessibleName("Preflight progress")
        self.progress_bar.setVisible(False)
        overview_layout.addWidget(self.progress_bar)

        secondary_controls = QHBoxLayout()
        secondary_controls.addWidget(self.settings_button)
        self.preview_data_button = QPushButton("Preview model data…", self.overview_panel)
        self.preview_data_button.setToolTip(
            "Inspect the exact allowed metadata request. Opening this preview sends nothing."
        )
        self.preview_data_button.clicked.connect(self.show_data_preview)
        secondary_controls.addStretch()
        overview_layout.addLayout(secondary_controls)
        overview_layout.addWidget(self.preview_data_button, 0, Qt.AlignLeft)

        self.dataset_summary_label = QLabel("No dataset check yet.", self.overview_panel)
        self.dataset_summary_label.setWordWrap(True)
        self.dataset_summary_label.setAccessibleName("Dataset check summary")
        overview_layout.addWidget(self.dataset_summary_label)

        plan_heading = QLabel("Plan and visual review", self.overview_panel)
        plan_heading.setStyleSheet("font-weight: bold;")
        overview_layout.addWidget(plan_heading)
        self.plan_text = QPlainTextEdit(self.overview_panel)
        self.plan_text.setReadOnly(True)
        self.plan_text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.plan_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plan_text.setMinimumHeight(140)
        self.plan_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plan_text.setAccessibleName("Read-only plan proposal")
        self.plan_text.setPlainText(
            "Start Autonomous Analysis to check the dataset and prepare a plan."
        )
        overview_layout.addWidget(self.plan_text)
        roi_review = QGridLayout()
        roi_review.setColumnStretch(0, 1)
        roi_review.setColumnStretch(1, 1)
        self.roi_review_label = QLabel("Review ROI (original pixels):", self.overview_panel)
        roi_review.addWidget(self.roi_review_label, 0, 0, 1, 2)
        self.roi_edits = tuple(QSpinBox(self.overview_panel) for _ in range(4))
        for index, (edit, name) in enumerate(
            zip(self.roi_edits, ("x min", "x max", "y min", "y max"))
        ):
            edit.setPrefix(name + " ")
            edit.setRange(0, 100000)
            roi_review.addWidget(edit, 1 + index // 2, index % 2)
        self.adjust_roi_button = QPushButton("Preview adjusted ROI", self.overview_panel)
        self.adjust_roi_button.clicked.connect(self.preview_adjusted_roi)
        self.reject_vision_button = QPushButton("Reject visual proposal", self.overview_panel)
        self.reject_vision_button.clicked.connect(self.reject_visual_proposal)
        roi_review.addWidget(self.adjust_roi_button, 3, 0)
        roi_review.addWidget(self.reject_vision_button, 3, 1)
        overview_layout.addLayout(roi_review)
        self._set_vision_review_visible(False)

        self.overview_scroll.setWidget(self.overview_panel)

        self.evidence_panel = QWidget(self.splitter)
        self.evidence_panel.setObjectName("preflightEvidencePanel")
        self.evidence_panel.setMinimumWidth(0)
        self.evidence_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        evidence_layout = QVBoxLayout(self.evidence_panel)
        evidence_heading = QLabel("Visual Evidence", self.evidence_panel)
        evidence_heading.setStyleSheet("font-size: 15px; font-weight: bold;")
        evidence_layout.addWidget(evidence_heading)
        evidence_explanation = QLabel(
            "Measured images and spectrum from the current dataset.", self.evidence_panel
        )
        evidence_explanation.setWordWrap(True)
        evidence_layout.addWidget(evidence_explanation)
        self.evidence_scroll = QScrollArea(self.evidence_panel)
        self.evidence_scroll.setWidgetResizable(True)
        self.evidence_scroll.setMinimumWidth(0)
        self.evidence_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.evidence_scroll.setAccessibleName("Visual evidence")
        self.evidence_content = QWidget(self.evidence_scroll)
        self.evidence_layout = QVBoxLayout(self.evidence_content)
        self.evidence_layout.setAlignment(Qt.AlignTop)
        self._evidence_pixmaps: list[tuple[QLabel, QPixmap]] = []
        self.evidence_scroll.setWidget(self.evidence_content)
        evidence_layout.addWidget(self.evidence_scroll)
        self.splitter.addWidget(self.overview_scroll)
        self.splitter.addWidget(self.evidence_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([560, 720])
        self.splitter.splitterMoved.connect(self._rescale_evidence_images)
        self.splitter.handle(1).setCursor(Qt.SplitHCursor)
        self.splitter.handle(1).setToolTip(
            "Drag to give more space to Overview or Visual Evidence."
        )
        self._show_evidence_message(
            "No measured evidence yet. Start Autonomous Analysis to review the dataset."
        )

        self._revision_timer = QTimer(self)
        self._revision_timer.setInterval(200)
        self._revision_timer.timeout.connect(self._check_revision)
        self._update_model_readiness()
        self._update_controls()

    @property
    def result(self) -> PreflightResult | None:
        return self._result

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _clear_evidence(self) -> None:
        self._evidence_pixmaps.clear()
        while self.evidence_layout.count():
            item = self.evidence_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _show_evidence_message(self, message: str) -> None:
        self._clear_evidence()
        label = QLabel(message, self.evidence_content)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.evidence_layout.addWidget(label)

    def _add_evidence_text(self, text: str) -> QLabel:
        label = QLabel(text, self.evidence_content)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.evidence_layout.addWidget(label)
        return label

    def _set_vision_review_visible(self, visible: bool) -> None:
        self.roi_review_label.setVisible(visible)
        for edit in self.roi_edits:
            edit.setVisible(visible)
        self.adjust_roi_button.setVisible(visible)
        self.reject_vision_button.setVisible(visible)

    def _add_vision_png(self, caption: str, png: bytes | None) -> None:
        caption_label = QLabel(caption, self.evidence_content)
        caption_label.setWordWrap(True)
        caption_label.setMinimumWidth(0)
        self.evidence_layout.addWidget(caption_label)
        if png is None:
            return
        pixmap = QPixmap()
        if pixmap.loadFromData(png, "PNG"):
            self._add_evidence_pixmap(caption, pixmap)

    def _add_evidence_pixmap(self, accessible_name: str, pixmap: QPixmap) -> None:
        label = QLabel(self.evidence_content)
        label.setAccessibleName(accessible_name)
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.evidence_layout.addWidget(label)
        self._evidence_pixmaps.append((label, pixmap))
        self._rescale_evidence_images()
        QTimer.singleShot(0, self._rescale_evidence_images)

    def _rescale_evidence_images(self, *_args) -> None:
        if not hasattr(self, "_evidence_pixmaps"):
            return
        available_width = max(1, self.evidence_scroll.viewport().width() - 8)
        for label, source in self._evidence_pixmaps:
            width = min(available_width, source.width())
            scaled = source.scaled(
                width,
                source.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            label.setPixmap(scaled)
            label.setFixedSize(scaled.size())

    def _render_vision_outcome(self, outcome: VisionOutcome) -> None:
        roi = outcome.roi
        if roi is None:
            self.plan_text.setPlainText(
                self._planning_display_text
                + "\n\nOpenAI visual review\n"
                + outcome.message
                + "\nNo ROI was proposed. Review the available evidence manually."
            )
            return
        self._set_vision_review_visible(True)
        self.reject_vision_button.setText("Reject visual proposal")
        if self._result is not None and self._result.roi is None:
            self.dataset_summary_label.setText(
                self.dataset_summary_label.text()
                + "\nOpenAI visual review extracted the actual spectrum from a proposed ROI; "
                "NEAT's selected fitting ROI remains unchanged."
            )
        for edit, value in zip(self.roi_edits, (roi.x_min, roi.x_max, roi.y_min, roi.y_max)):
            edit.setValue(value)
        review_heading = (
            "OpenAI visual proposal — user review required; no settings applied"
            if outcome.phase_id
            else "OpenAI visual review — manual evidence review required; no settings applied"
        )
        lines = [
            self._planning_display_text,
            "",
            review_heading,
            f"Review status: {outcome.message}",
            "Provider/model: OpenAI / "
            + (
                self._run_snapshot.settings.model_id
                if self._run_snapshot is not None
                else "configured Agent model"
            ),
            f"Dataset: {outcome.dataset_id} revision {outcome.revision}",
            f"Original-pixel ROI: x[{roi.x_min},{roi.x_max}), y[{roi.y_min},{roi.y_max})",
        ]
        if outcome.confidence == "user-selected":
            lines.extend((
                "ROI source: user-adjusted by the user.",
                "No image frames were sent for ROI selection in this run.",
                outcome.roi_reason,
            ))
        else:
            lines.extend((
                f"Inspected frames: {', '.join(map(str, outcome.inspected_frames))}",
                f"ROI confidence: {outcome.confidence}; visual reason: {outcome.roi_reason}",
            ))
        if outcome.ambiguity_flags:
            lines.append("Image ambiguities: " + "; ".join(outcome.ambiguity_flags))
        lines.append("Compared trusted overlays: " + (", ".join(outcome.inspected_phases) or "none"))
        if outcome.phase_id:
            lines.append(f"Provisional phase: {outcome.phase_id}")
            lines.append(f"Visual reason: {outcome.phase_reason}")
            for hkl, reason in zip(outcome.hkls, outcome.edge_reasons):
                lines.append(f"Candidate edge {hkl}: {reason}")
            if outcome.display_window is not None:
                lines.append(
                    "Review display interval (includes NEAT's default edge-fit flanks; not applied): "
                    f"{outcome.display_window[0]:.6g}–{outcome.display_window[1]:.6g} Å"
                )
        else:
            lines.append("No provisional phase or candidate-edge suggestion was accepted.")
            if outcome.phase_reason:
                lines.append(f"Validation/review note: {outcome.phase_reason}")
        if outcome.alternatives:
            lines.append("Other plausible candidates: " + ", ".join(outcome.alternatives))
        if outcome.unresolved:
            lines.append("Unresolved: " + "; ".join(outcome.unresolved))
        if outcome.display_interval_diagnostic:
            lines.append("Display interval check: " + outcome.display_interval_diagnostic)
        lines.extend((
            "Visual agreement is not crystallographic identification or proof of fit quality.",
            "Physical d/strain interpretation still needs separate validation, including the legacy d_fit/factor-of-two caveat.",
            "No fit, mapping, phase-table edit or fitting-tab change occurred.",
        ))
        self.plan_text.setPlainText("\n".join(lines))
        self._clear_evidence()
        self._add_vision_png("Actual frame with proposed ROI (yellow); image row zero at top.", outcome.roi_preview_png)
        self._add_vision_png("Actual measured spectrum from proposed ROI; full source wavelength axis.", outcome.spectrum_png)
        for phase, png in outcome.phase_overlays:
            self._add_vision_png(f"Measured spectrum and NEAT theoretical dashed markers: {phase} (provisional).", png)
        self.evidence_layout.addStretch()

    def preview_adjusted_roi(self) -> None:
        if self._result is None or not self._result.frames or self._vision_outcome is None or self._has_active_worker():
            return
        self._check_revision()
        if self._stale:
            return
        roi = RoiBounds(
            x_min=self.roi_edits[0].value(),
            x_max=self.roi_edits[1].value(),
            y_min=self.roi_edits[2].value(),
            y_max=self.roi_edits[3].value(),
        )
        shape = self._result.dataset.image_shape
        if shape is None or not (0 <= roi.x_min < roi.x_max <= shape[1] and 0 <= roi.y_min < roi.y_max <= shape[0]):
            self._set_status("Adjusted ROI is outside the current image or empty.")
            return
        frame = self._result.frames[0].image
        if frame.coordinate_mapping is None or frame.png_bytes is None:
            self._set_status("Original image preview is unavailable.")
            return
        try:
            png = render_roi_overlay(frame.png_bytes, frame.coordinate_mapping, roi)
        except ValueError:
            self._set_status("Adjusted ROI could not be previewed.")
            return
        self._adjusted_roi = roi
        self._adjusted_roi_dataset_token = self._dataset_token
        self._vision_outcome = None
        self._clear_evidence()
        self._add_vision_png("User-adjusted ROI on actual frame; the next run will extract a fresh spectrum.", png)
        self.plan_text.setPlainText(
            f"User-adjusted ROI x[{roi.x_min},{roi.x_max}), y[{roi.y_min},{roi.y_max}). "
            "The earlier spectrum and phase hypothesis no longer describe this ROI. "
            "No fitting setting was applied. Start Autonomous Analysis to reuse this ROI, "
            "extract its spectrum again, and regenerate phase overlays and recommendations."
        )
        self._set_status("Adjusted ROI saved for the next visual review; earlier spectral findings were withdrawn.")
        self._set_vision_review_visible(False)
        self.reject_vision_button.setText("Discard adjusted ROI")
        self.reject_vision_button.setVisible(True)

    def _discard_adjusted_roi(self) -> None:
        self._adjusted_roi = None
        self._adjusted_roi_dataset_token = None

    def reject_visual_proposal(self) -> None:
        if self._has_active_worker():
            return
        had_adjusted_roi = self._adjusted_roi is not None
        self._vision_outcome = None
        self._discard_adjusted_roi()
        self.reject_vision_button.setText("Reject visual proposal")
        self._set_vision_review_visible(False)
        message = (
            "Adjusted ROI discarded. No fitting configuration was changed."
            if had_adjusted_roi
            else "Visual proposal rejected. No fitting configuration was changed."
        )
        self.plan_text.setPlainText(message)
        self._show_evidence_message(message + " Start another run to review current evidence.")
        self._set_status(message)

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
        self._planning_display_text = ""
        self._vision_outcome = None
        self._pending_vision_result = None
        self._discard_adjusted_roi()
        self._set_vision_review_visible(False)
        if self._worker is not None:
            self._worker.stop()
        if self._planning_worker is not None:
            self._planning_worker.stop()
        if self._vision_worker is not None:
            self._vision_worker.stop()
        self._stage = "stale"
        stale_message = (
            "Dataset changed — rerun autonomous analysis."
            if self._agent_run_id is not None
            else "Dataset changed — rerun preflight."
        )
        self._set_status(stale_message)
        self.plan_text.setPlainText(stale_message)
        self.dataset_summary_label.setText("Dataset changed. Run the analysis again.")
        self._show_evidence_message(
            "Dataset changed — rerun autonomous analysis to inspect current evidence."
            if self._agent_run_id is not None
            else "Dataset changed — rerun preflight to see current evidence."
        )
        if self._worker is None and self._planning_worker is None and self._vision_worker is None:
            self._revision_timer.stop()
        self._update_controls()

    def _has_active_worker(self) -> bool:
        inventory = getattr(self._main_window, "_shutdown_worker_inventory", None)
        tracked_worker = bool(
            inventory is not None
            and any(
                inventory.has_role(role)
                for role in ("agent_preflight", "agent_planning", "agent_vision")
            )
        )
        return bool(
            self._worker is not None
            or self._planning_worker is not None
            or self._vision_worker is not None
            or self._awaiting_preflight_completion
            or self._pending_planning_context is not None
            or self._pending_vision_result is not None
            or tracked_worker
        )

    def _can_stop_active_run(self) -> bool:
        if self._cancellation_requested:
            return False
        return bool(
            self._awaiting_preflight_completion
            or self._pending_planning_context is not None
            or self._pending_vision_result is not None
            or (
                self._worker is not None
                and not self._preflight_completion_received
            )
            or (
                self._planning_worker is not None
                and self._planning_outcome is None
            )
            or (self._vision_worker is not None and self._vision_outcome is None)
        )

    def _update_controls(self) -> None:
        active = self._has_active_worker()
        host_closing = bool(
            getattr(self._main_window, "_shutdown_close_pending", False)
            or getattr(self._main_window, "_shutdown_committed", False)
        )
        can_start = not active and not self._parent_closing and not host_closing
        self.start_autonomous_button.setEnabled(can_start)
        self.stop_button.setVisible(active)
        self.progress_bar.setVisible(active)
        self.stop_button.setEnabled(
            self._can_stop_active_run() and not self._parent_closing
        )
        self.stop_button.setText(
            "Stopping…" if active and self._cancellation_requested else "Stop"
        )

    def _update_model_readiness(self) -> None:
        settings = self._agent_settings
        if settings.mode is AgentModelMode.OFFLINE:
            readiness = (
                "No LLM model is selected. Choose OpenAI cloud or a local model in "
                "Agent settings; autonomous analysis will stop with an error instead "
                "of preparing an offline substitute plan."
            )
        elif settings.mode is AgentModelMode.LOCAL:
            configured = bool(settings.model_id and settings.endpoint)
            readiness = (
                f"Local LLM: {settings.model_id} at {settings.endpoint}. "
                "Requests stay on this computer's loopback endpoint. OpenAI visual "
                "review is not sent in local-model mode. If the model is unavailable "
                "or returns an invalid response, the run stops with the reason."
                if configured
                else "Local LLM is not configured. Enter a loopback endpoint and exact "
                "model ID in Agent settings. No offline plan will be substituted."
            )
        else:
            configured = bool(settings.model_id)
            readiness = (
                f"OpenAI LLM: {settings.model_id}. Each run automatically sends the "
                "metadata-only dataset summary. Eligible image-stack reviews may send "
                "up to 7 bounded PNGs in 3 visual requests: up to 3 image previews, 1 measured "
                "spectrum plot, and up to 3 phase-overlay plots. No full image stack or source "
                "files are sent. If the model is unavailable, the run stops with its "
                "reason; no offline plan is substituted."
                if configured
                else "OpenAI LLM is not configured. Enter an exact model ID in Agent "
                "settings. No request or visual evidence will be sent until configured."
            )
        self.model_readiness_label.setText("Planning method: " + readiness)

    def open_model_settings(self) -> None:
        dialog = AgentModelSettingsDialog(
            self,
            main_window=self._main_window,
            settings_repository=self._agent_settings_repository,
        )
        if dialog.exec_() == QDialog.Accepted:
            self._agent_settings = self._agent_settings_repository.load()
            self._update_model_readiness()

    def _snapshot_run_configuration(self) -> AgentRunSnapshot:
        settings = self._agent_settings
        consent = settings.mode is AgentModelMode.OPENAI_CLOUD
        snapshot = AgentRunSnapshot(
            settings=settings,
            objective=PlanningObjective.DATASET_READINESS,
            cloud_egress_consent=consent,
            deterministic_fallback=False,
        )
        resolver = None
        if settings.mode is AgentModelMode.OPENAI_CLOUD:
            resolver = AgentCredentialResolver(
                AgentKeyringCredentialStore(),
                environment_allowed=settings.allow_environment_credential,
            )
        self._run_credential_resolver = resolver
        return snapshot

    def show_data_preview(self) -> None:
        self._check_revision()
        settings = self._agent_settings
        if settings.mode is AgentModelMode.OFFLINE:
            destination = "No LLM model selected — autonomous analysis will report that no model is configured."
        elif settings.mode is AgentModelMode.LOCAL:
            destination = (
                f"Local model {settings.model_id or '(model not configured)'} at "
                f"{settings.endpoint or '(loopback endpoint not configured)'}; "
                "loopback only, no proxy or remote redirects."
            )
        else:
            destination = (
                "OpenAI cloud model "
                + (settings.model_id or "(model not configured)")
                + "; Start authorizes one canonical metadata-only planning request when eligible, plus up to 7 bounded PNGs across 3 visual requests: up to 3 image previews, 1 measured-spectrum plot and up to 3 phase-overlay plots. A no-ROI run may skip metadata planning and start visual ROI discovery directly."
            )
        content = [
            "Configured destination:",
            destination,
            "",
            "Canonical planning request schema (the same schema used by the serializer):",
            "",
            json.dumps(canonical_planning_schema(), indent=2),
            "",
            "Planning limits: at most 1 request per run; complete serialized body ≤ 4,096 ASCII bytes; output ≤ 384 tokens; response ≤ 4,096 bytes; HTTP timeout 10 seconds and total deadline 15 seconds.",
            "For OpenAI mode, visual review runs automatically using the same model and key: at most 3 vision requests with at most 7 PNGs total (up to 3 measured frames + 1 ROI spectrum plot + up to 3 trusted phase-overlay plots). Each request is ≤ 4 MB, each PNG ≤ 1 MB; responses ≤ 12 KB; 45-second per-request and 150-second total visual-stage limits. A separate metadata-only planning request may run when its preflight requirements are met.",
            "The connection test is a separate user-initiated operation and sends only its fixed test payload.",
            "",
                "This metadata planning request never contains image bytes, thumbnails, spectra or arrays, actual wavelength/intensity values, file paths or names, sample titles, free-form observations/warnings, dataset/run IDs, credentials, or local files.",
            "Opening this preview and saving settings send nothing. Pressing Start in OpenAI mode authorizes only the disclosed metadata and bounded PNG transfers for that run. Local mode uses loopback and sends no visual evidence to OpenAI; model failure does not trigger an offline fallback.",
        ]
        if self._result is not None and not self._stale:
            try:
                run_id = self._agent_run_id or uuid4().hex
                context = context_from_preflight(
                    self._result,
                    run_id=run_id,
                    objective=(
                        self._run_snapshot.objective
                        if self._run_snapshot is not None
                        else PlanningObjective.DATASET_READINESS
                    ),
                )
                payload = canonical_payload_for_context(context)
            except (ValueError, TypeError):
                payload = None
            if payload is not None:
                content.extend(
                    (
                        "",
                        "Current payload after preflight (serializer output only; not transmitted):",
                        "",
                        payload,
                    )
                )
        dialog = QDialog(self)
        dialog.setWindowTitle("Review model request")
        dialog.resize(680, 600)
        layout = QVBoxLayout(dialog)
        preview = QPlainTextEdit(dialog)
        preview.setReadOnly(True)
        preview.setPlainText("\n".join(content))
        layout.addWidget(preview)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=dialog)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec_()

    def start_autonomous_analysis(self) -> None:
        """Run existing preflight and automatically continue to read-only planning."""
        if self._has_active_worker() or self._parent_closing:
            return
        if self._adjusted_roi is not None:
            adapter = getattr(self._main_window, "scientific_observation_api", None)
            if (
                adapter is None
                or adapter.revision_clock.current() != self._adjusted_roi_dataset_token
            ):
                self._discard_adjusted_roi()
        if self._agent_settings.mode is AgentModelMode.OFFLINE:
            message = (
                "LLM model is not available: no LLM is selected. Choose OpenAI cloud "
                "or a local model in Agent settings. No offline plan was substituted."
            )
            self._stage = "failed"
            self._set_status(message)
            self.plan_text.setPlainText(message)
            self._update_controls()
            return
        vision_enabled = self._agent_settings.mode is AgentModelMode.OPENAI_CLOUD
        if vision_enabled and (
            self._agent_settings.mode is not AgentModelMode.OPENAI_CLOUD
            or not self._agent_settings.model_id
        ):
            self._set_status(
                "LLM unavailable: enter an exact OpenAI model ID in Agent settings before starting an analysis run."
            )
            return
        try:
            self._run_snapshot = self._snapshot_run_configuration()
        except ValueError as exc:
            self._set_status(str(exc))
            return
        run_id = uuid4().hex
        self._agent_run_id = run_id
        self._planning_outcome = None
        self._planning_display_text = ""
        self._vision_outcome = None
        self._set_vision_review_visible(False)
        self._vision_run_enabled = vision_enabled
        self._stage = "preflight_running"
        self.plan_text.clear()
        self.plan_text.setPlainText("Waiting for the current dataset preflight.")
        self.start_preflight(
            _agent_run_id=run_id,
            _run_snapshot=self._run_snapshot,
        )

    def start_preflight(
        self,
        *,
        _agent_run_id: str | None = None,
        _run_snapshot: AgentRunSnapshot | None = None,
    ) -> None:
        """Capture live state only on the Qt owner thread, then start bounded work."""
        if self._has_active_worker() or self._parent_closing:
            return
        if _run_snapshot is not None:
            self._run_snapshot = _run_snapshot
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
        self._pending_vision_result = None
        self._planning_outcome = None
        self._vision_outcome = None
        if _agent_run_id is None:
            self._agent_run_id = None
            self._run_snapshot = None
            self._run_credential_resolver = None
            self._vision_run_enabled = False
            self._stage = "preflight_running"
        self._revision_timer.stop()
        self.progress_bar.setValue(0)
        self._show_evidence_message("Preflight is preparing the observation snapshot.")
        self.dataset_summary_label.setText("Checking the current dataset…")
        if _agent_run_id is None:
            self.plan_text.setPlainText(
                "Dataset check only. Start Autonomous Analysis to prepare a plan."
            )
        run_id = uuid4().hex
        self._active_run_id = run_id
        adapter = getattr(self._main_window, "scientific_observation_api", None)
        if adapter is None:
            self._awaiting_preflight_completion = False
            self._set_status("Scientific observation API is unavailable.")
            self.dataset_summary_label.setText("Dataset check could not start.")
            self._stage = "failed"
            self._update_controls()
            return
        try:
            handle = adapter.capture_dataset()
            roi = adapter.capture_selected_roi(handle)
        except DatasetUnavailableError:
            self._awaiting_preflight_completion = False
            self._set_status("Dataset loader is busy — retry preflight after loading finishes.")
            self.dataset_summary_label.setText("Dataset check could not start while loading.")
            self._stage = "busy_retry"
            self._update_controls()
            return
        except (ObservationErrorBase, ValueError):
            self._awaiting_preflight_completion = False
            self._set_status("Dataset capture is unavailable — check loaded data and retry.")
            self.dataset_summary_label.setText("Dataset check could not start.")
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
        self._set_status(
            "Checking the dataset. Planning will follow after the configured model and dataset evidence are checked."
            if _agent_run_id is not None
            else "Checking the dataset locally. No model or network request is used."
        )
        self._update_controls()
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
        self._pending_vision_result = None
        for worker in (self._worker, self._planning_worker, self._vision_worker):
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
            self._pending_vision_result = None
            self._result = None
            self._planning_outcome = None
            self._stage = "cancelled"
            message = (
                "Autonomous analysis stopped during preflight."
                if self._agent_run_id is not None
                else "Preflight cancelled."
            )
            self._set_status(message)
            self.dataset_summary_label.setText("Dataset check stopped.")
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
            if self._adjusted_roi is not None and self._adjusted_roi_dataset_token != (
                outcome.dataset_id,
                outcome.revision,
            ):
                self._discard_adjusted_roi()
            self.progress_bar.setValue(100)
            self._render_result(outcome.result)
            if self._agent_run_id is not None:
                if outcome.result.roi is None:
                    problem = self._visual_roi_discovery_problem(outcome.result)
                    if problem is not None:
                        message = (
                            "No selected NEAT ROI; M4 metadata planning was not run, "
                            "and no plan is claimed. "
                            + problem
                        )
                        self._planning_display_text = message
                        self._stage = "blocked_no_roi"
                        self.plan_text.setPlainText(message)
                        self._set_status(message)
                        self._update_controls()
                        return
                    if self._adjusted_roi is not None:
                        self._planning_display_text = (
                            "No selected NEAT fitting ROI. Reusing the validated "
                            "user-adjusted Agent ROI. M4 metadata planning was skipped; "
                            "no planning result is claimed. NEAT will re-extract the "
                            "spectrum and regenerate the phase comparison."
                        )
                    else:
                        self._planning_display_text = (
                            "No selected NEAT fitting ROI. M4 metadata planning was "
                            "skipped; no planning result is claimed. The configured "
                            "OpenAI model will inspect the bounded image previews, "
                            "propose an ROI, and NEAT will extract its measured spectrum "
                            "for the visual phase comparison."
                        )
                    self._pending_vision_result = outcome.result
                    self._stage = "visual_roi_discovery_pending"
                    self.plan_text.setPlainText(self._planning_display_text)
                    self._set_status(
                        "No fitting ROI is selected. Starting visual-first ROI discovery after preflight retires."
                    )
                    if self._worker is None:
                        self._start_pending_vision()
                    self._update_controls()
                    return
                try:
                    context = context_from_preflight(
                        outcome.result,
                        run_id=self._agent_run_id,
                        objective=(
                            self._run_snapshot.objective
                            if self._run_snapshot is not None
                            else PlanningObjective.DATASET_READINESS
                        ),
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
            self.dataset_summary_label.setText("Dataset check did not complete.")
            self._show_evidence_message(outcome.message or "No evidence was produced.")
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

    def _start_pending_vision(self) -> None:
        result = self._pending_vision_result
        if (
            result is None or self._worker is not None or self._vision_worker is not None
            or self._planning_worker is not None
            or self._cancellation_requested or self._stale or self._parent_closing
            or getattr(self._main_window, "_shutdown_close_pending", False)
            or getattr(self._main_window, "_shutdown_committed", False)
        ):
            return
        self._pending_vision_result = None
        if (
            self._run_snapshot is None
            or self._run_snapshot.settings.mode is not AgentModelMode.OPENAI_CLOUD
            or not self._vision_run_enabled
            or self._run_credential_resolver is None
        ):
            self._stage = VisionStatus.UNAVAILABLE.value
            self._set_status("OpenAI visual review is unavailable. Check the configured model and Agent key.")
            self.plan_text.setPlainText("No visual finding was generated. The dataset preflight remains available for manual review.")
            self._update_controls()
            return
        adapter = getattr(self._main_window, "scientific_observation_api", None)
        if adapter is None or adapter.revision_clock.current() != self._dataset_token:
            self._mark_stale()
            return
        try:
            handle = adapter.capture_dataset()
        except (ObservationErrorBase, ValueError):
            self._stage = VisionStatus.UNAVAILABLE.value
            self._set_status("OpenAI visual review could not capture the current dataset.")
            self._update_controls()
            return
        if (handle.dataset_id, handle.revision) != self._dataset_token:
            self._mark_stale()
            return
        run_id = self._agent_run_id
        if run_id is None:
            return
        roi_override = None
        if self._adjusted_roi is not None:
            if self._adjusted_roi_dataset_token == self._dataset_token:
                roi_override = self._adjusted_roi
            else:
                self._discard_adjusted_roi()
        worker = VisionWorker(
            run_id,
            handle,
            result,
            self._run_snapshot.settings,
            roi_override=roi_override,
            credential_resolver=self._run_credential_resolver,
            cloud_egress_authorized=self._vision_run_enabled,
            parent=self._main_window,
        )
        self._vision_worker = worker
        self._main_window.agent_vision_worker = worker
        self._main_window._track_shutdown_worker(worker, "agent_vision")
        self._main_window._connect_shutdown_worker_signal(
            worker, worker.progress,
            lambda percent, message, rid=run_id, current=worker: self._on_vision_progress(rid, current, percent, message),
            current_attribute="agent_vision_worker",
        )
        self._main_window._connect_shutdown_worker_signal(
            worker, worker.completed,
            lambda outcome, rid=run_id, current=worker: self._on_vision_completed(rid, current, outcome),
            current_attribute="agent_vision_worker",
        )
        self._main_window._connect_shutdown_worker_signal(
            worker, worker.finished,
            lambda rid=run_id, current=worker: self._on_vision_finished(rid, current),
            current_attribute="agent_vision_worker",
        )
        self._stage = "vision_running"
        visual_status = (
            "The configured OpenAI model is inspecting bounded previews. This may propose settings for your review; nothing will be applied."
        )
        self.plan_text.setPlainText(
            self._planning_display_text + "\n\n" + visual_status
            if self._planning_display_text
            else visual_status
        )
        self._set_status(
            "Planning is complete. OpenAI visual review is now inspecting bounded previews."
            if self._planning_outcome is not None
            else "No plan was claimed. OpenAI visual review is starting with the available bounded evidence."
        )
        self._update_controls()
        worker.start()

    def _visual_roi_discovery_problem(self, result: PreflightResult) -> str | None:
        """Return why this run cannot safely start image-based ROI discovery."""
        if not self._vision_run_enabled or self._run_snapshot is None or (
            self._run_snapshot.settings.mode is not AgentModelMode.OPENAI_CLOUD
        ):
            return (
                "Visual ROI discovery requires OpenAI mode. Select OpenAI in Agent "
                "settings and start again; no offline plan will be substituted."
            )
        dataset = result.dataset
        if dataset.mode is not DatasetMode.IMAGES or not dataset.capabilities.images_available:
            return "The current dataset does not provide an image stack for ROI discovery."
        if dataset.image_shape is None or not dataset.frame_count:
            return "The current image stack has no usable frame dimensions."
        if dataset.wavelength_axis_status is not WavelengthAxisStatus.VALID:
            return "A valid wavelength calibration is required before spectrum-based visual review."
        if not result.frames:
            return "NEAT could not create any bounded image preview for visual review."
        for frame in result.frames[:3]:
            image = frame.image
            if (
                image.status is not ObservationStatus.OK
                or not image.png_bytes
                or image.coordinate_mapping is None
                or image.preview_shape is None
            ):
                return "A representative image preview is invalid or incomplete; no model request was made."
        return None

    def _on_vision_progress(self, run_id: str, worker: VisionWorker, percent: int, message: str) -> None:
        if run_id != self._agent_run_id or worker is not self._vision_worker or self._parent_closing or self._stale:
            return
        self._check_revision()
        if not self._stale and not self._cancellation_requested:
            self.progress_bar.setValue(min(99, 80 + percent // 5))
            self._set_status(message)

    def _on_vision_completed(self, run_id: str, worker: VisionWorker, outcome: VisionOutcome) -> None:
        if (
            run_id != self._agent_run_id or worker is not self._vision_worker
            or outcome.run_id != run_id or self._parent_closing
            or getattr(self._main_window, "_shutdown_close_pending", False)
        ):
            return
        if self._cancellation_requested:
            self._vision_outcome = None
            self._stage = "cancelled"
            self._set_status("Autonomous analysis stopped during OpenAI visual review.")
            self.plan_text.setPlainText(
                self._planning_display_text
                + "\n\nOpenAI visual review stopped. No proposal was accepted."
            )
            self._update_controls()
            return
        self._check_revision()
        if self._stale or self._dataset_token != (outcome.dataset_id, outcome.revision):
            self._mark_stale()
            return
        self._vision_outcome = outcome
        self._stage = outcome.status.value
        if outcome.status is VisionStatus.STALE:
            self._mark_stale()
            return
        if outcome.status is not VisionStatus.NEEDS_REVIEW:
            self.plan_text.setPlainText(
                self._planning_display_text
                + "\n\nOpenAI visual review: "
                + outcome.message
                + "\nNo visual settings were applied."
            )
            self._set_status(outcome.message)
        else:
            self._render_vision_outcome(outcome)
            if outcome.phase_id:
                self._set_status("Provisional visual suggestions are ready for your review. No fitting or configuration change occurred.")
            else:
                self._set_status("Visual evidence is ready for manual review; no phase or edge suggestion was accepted.")
            self.progress_bar.setValue(100)
        self._update_controls()

    def _on_vision_finished(self, run_id: str, worker: VisionWorker) -> None:
        if self._vision_worker is worker:
            self._vision_worker = None
        if getattr(self._main_window, "agent_vision_worker", None) is worker:
            self._main_window.agent_vision_worker = None
        if run_id == self._agent_run_id and self._cancellation_requested and not self._stale:
            self._stage = "cancelled"
            self._set_status("Autonomous analysis stopped during OpenAI visual review.")
        self._update_controls()

    def vision_worker_retired(self, worker: VisionWorker, callbacks_suppressed: bool) -> None:
        if self._vision_worker is worker:
            self._vision_worker = None
        if not self._parent_closing:
            if self._cancellation_requested and not self._stale:
                self._vision_outcome = None
                self._stage = "cancelled"
                self._set_status("Autonomous analysis stopped during OpenAI visual review.")
            elif callbacks_suppressed and not self._stale:
                self._vision_outcome = None
                self._set_status("OpenAI visual review stopped while NEAT was closing.")
            self._update_controls()

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
            context,
            adapter.revision_clock,
            self._run_snapshot,
            credential_resolver=self._run_credential_resolver,
            parent=self._main_window,
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
        mode = (
            self._run_snapshot.settings.mode
            if self._run_snapshot is not None
            else AgentModelMode.OFFLINE
        )
        planner = {
            AgentModelMode.OFFLINE: "local deterministic planner",
            AgentModelMode.LOCAL: "configured loopback structured model",
            AgentModelMode.OPENAI_CLOUD: "OpenAI structured model",
        }[mode]
        self.plan_text.setPlainText(
            f"Preflight is current. The {planner} is preparing a read-only proposal."
        )
        self._set_status("Preflight complete — planning is advancing automatically.")
        self._update_controls()
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
        titles = {
            PlanStepType.INSPECT_DATASET_SUMMARY: "Check the dataset summary",
            PlanStepType.REVIEW_IMAGE_PREVIEWS: "Review the image previews",
            PlanStepType.REVIEW_MEASURED_SPECTRUM: "Review the measured spectrum",
            PlanStepType.REQUEST_ROI: "Select an ROI in NEAT",
            PlanStepType.REQUEST_CALIBRATION_REVIEW: "Confirm the calibration metadata",
            PlanStepType.STOP_NEEDS_MORE_EVIDENCE: "Pause until more evidence is available",
        }
        lines = [f"{index}. {titles[step.step_type]}", f"   {step.rationale}"]
        if step.requires_human_input:
            lines.append("   Your review is needed before this step can proceed.")
        return tuple(lines)

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
            f"Review focus: {outcome.objective.value.replace('_', ' ')}",
        ]
        if outcome.steps:
            lines.extend(("", "Suggested steps (not executed):"))
            for index, step in enumerate(outcome.steps, start=1):
                lines.extend(self._format_plan_step(step, index))
        else:
            lines.extend(("", "No steps were proposed."))
        lines.extend((
            "",
            "Run details:",
            f"Status: {outcome.status.value.replace('_', ' ')}",
            f"Agent run: {outcome.run_id}",
            f"Preflight run: {outcome.preflight_run_id}",
            f"Dataset: {outcome.dataset_id} revision {outcome.revision}",
            f"Graph transitions: {' → '.join(outcome.transitions) or 'none'}",
            f"Objective: {outcome.objective.value}",
            f"Planner source: {outcome.planner_source.value.replace('_', ' ')}",
        ))
        if outcome.provider:
            lines.append(f"Provider/model: {outcome.provider} / {outcome.model_id}")
        if self._run_snapshot is not None:
            if self._run_snapshot.settings.mode is AgentModelMode.OPENAI_CLOUD:
                lines.append(
                    "Dataset summary transfer: enabled for this OpenAI run"
                    if self._run_snapshot.cloud_egress_consent
                    else "Dataset summary transfer: disabled; no OpenAI request was made"
                )
                visual_stage_available = (
                    self._vision_run_enabled
                    and self._result is not None
                    and self._result.dataset.mode is DatasetMode.IMAGES
                )
                lines.append(
                    "Bounded visual review is configured; it starts after successful planning"
                    if visual_stage_available
                    else "No preview images are sent for this dataset or model mode"
                )
            elif self._run_snapshot.settings.mode is AgentModelMode.LOCAL:
                lines.append("Network scope: configured loopback endpoint only")
            else:
                lines.append("Network scope: no model request")
        if outcome.input_tokens is not None or outcome.output_tokens is not None:
            lines.append(
                f"Provider-reported tokens: input={outcome.input_tokens}, output={outcome.output_tokens}"
            )
        if outcome.error_code is not None:
            lines.append(f"Reason code: {outcome.error_code.value}")
        if outcome.steps:
            lines.extend(("", "Evidence behind the suggested steps:"))
            for index, step in enumerate(outcome.steps, start=1):
                evidence = ", ".join(step.evidence_ids) if step.evidence_ids else "none"
                lines.append(f"{index}. {step.step_type.value}: {evidence}")
        if outcome.status is PlanningStatus.READY_PLAN:
            lines.extend(
                (
                    "",
                    "Plan prepared — scientific execution is not yet implemented.",
                    "No fitting, mapping, phase identification, or source-data changes occurred.",
                )
            )
            self._set_status(
                "Plan ready to review. No fitting or mapping has run."
            )
            self.progress_bar.setValue(100)
        elif outcome.status is PlanningStatus.NEEDS_INPUT:
            self._set_status("Plan needs user input before it can be considered ready.")
        elif outcome.status is PlanningStatus.BLOCKED_INVALID_DATA:
            self._set_status("Autonomous analysis stopped — preflight data or evidence is invalid.")
        elif outcome.status is PlanningStatus.UNAVAILABLE:
            self._set_status("LLM model is not available: " + outcome.message)
        elif outcome.status is PlanningStatus.NEEDS_PERMISSION:
            self._set_status(
                "Planning stopped — cloud permission was not granted for this run."
            )
        elif outcome.status is PlanningStatus.CANCELLED:
            self._set_status("Autonomous analysis stopped during planning.")
        elif outcome.status is PlanningStatus.STALE:
            self._mark_stale()
            return
        elif outcome.status is PlanningStatus.FAILED:
            if outcome.error_code in {
                PlanningErrorCode.MODEL_UNAVAILABLE,
                PlanningErrorCode.MISSING_CREDENTIAL,
            }:
                self._set_status("LLM model is not available: " + outcome.message)
            elif outcome.error_code is PlanningErrorCode.MODEL_OUTPUT_REJECTED:
                self._set_status("LLM returned an unusable response: " + outcome.message)
            else:
                self._set_status("LLM planning failed: " + outcome.message)
        else:
            self._set_status("Planning failed safely; no scientific action was run.")
        if outcome.message:
            lines.extend(("", outcome.message))
        self._planning_display_text = "\n".join(lines)
        self.plan_text.setPlainText(self._planning_display_text)
        if (
            self._vision_run_enabled
            and self._result.dataset.mode is DatasetMode.IMAGES
            and outcome.status
            not in {
                PlanningStatus.CANCELLED,
                PlanningStatus.STALE,
                PlanningStatus.BLOCKED_INVALID_DATA,
                PlanningStatus.UNAVAILABLE,
                PlanningStatus.FAILED,
                PlanningStatus.NEEDS_PERMISSION,
            }
        ):
            self._pending_vision_result = self._result
            self._set_status(
                "Planning finished. OpenAI visual review will start when the planning worker retires."
            )

    def _on_planning_finished(self, run_id: str, worker: PlanningWorker) -> None:
        if self._planning_worker is worker:
            self._planning_worker = None
        if getattr(self._main_window, "agent_planning_worker", None) is worker:
            self._main_window.agent_planning_worker = None
        if (
            run_id == self._agent_run_id
            and not self._parent_closing
            and not self._cancellation_requested
            and self._planning_outcome is None
            and not self._stale
        ):
            self._stage = "failed"
            self._set_status("Planning stopped before a result was received.")
            self.plan_text.setPlainText("No plan was produced.")
        if self._pending_vision_result is not None and not self._cancellation_requested:
            self._start_pending_vision()
        self._update_controls()
        if not self._has_active_worker() and self._result is None:
            self._revision_timer.stop()

    def _on_worker_finished(self, run_id: str, worker: PreflightWorker) -> None:
        if self._worker is worker:
            self._worker = None
        if getattr(self._main_window, "agent_preflight_worker", None) is worker:
            self._main_window.agent_preflight_worker = None
        if self._pending_vision_result is not None and not self._cancellation_requested:
            self._start_pending_vision()
        if (
            run_id == self._active_run_id
            and not self._cancellation_requested
            and self._preflight_completion_received
            and self._stage == "preflight_running"
            and self._agent_run_id is None
        ):
            self._stage = "complete"
        self._update_controls()
        if self._result is None and self._planning_worker is None and self._vision_worker is None:
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
                self._pending_vision_result = None
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
                self._pending_vision_result = None
                self._result = None
                self._revision_timer.stop()
                self._set_status("Preflight stopped while NEAT was closing. Start again if NEAT remains open.")
            elif self._pending_planning_context is not None:
                self._start_pending_planning()
            elif self._pending_vision_result is not None:
                self._start_pending_vision()
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
            elif self._pending_vision_result is not None:
                self._start_pending_vision()
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
            f"Dataset check complete — {assessment.readiness.value.replace('_', ' ')}."
        )
        summary_parts = []
        if dataset.mode is DatasetMode.PROFILE:
            summary_parts.append("spectrum-only dataset")
        else:
            if dataset.frame_count is not None:
                summary_parts.append(f"{dataset.frame_count:,} image frames")
            if dataset.image_shape is not None:
                rows, columns = dataset.image_shape
                summary_parts.append(f"{columns} × {rows} pixels")
        summary_parts.append(f"{dataset.wavelength_count:,} wavelength points")
        summary_parts.append(
            "wavelength axis: " + dataset.wavelength_axis_status.value.replace("_", " ")
        )
        if result.roi is not None:
            summary_parts.append(
                f"ROI: {result.roi.x_max - result.roi.x_min} × "
                f"{result.roi.y_max - result.roi.y_min} pixels"
            )
        summary = "Dataset check: " + " · ".join(summary_parts) + "."
        if assessment.warnings:
            summary += "\nNeeds review: " + assessment.warnings[0]
        self.dataset_summary_label.setText(summary)
        self._clear_evidence()
        if dataset.mode is DatasetMode.PROFILE:
            self._add_evidence_text("Spectrum-only input: image controls do not apply.")
        elif result.frames:
            self._add_evidence_text("Measured image previews; image row zero is at the top.")
            for frame in result.frames:
                caption = f"Frame {frame.frame_index}"
                if frame.wavelength is not None:
                    caption += f" — {frame.wavelength:.6g} Å"
                self._add_evidence_text(caption)
                self._add_evidence_pixmap(
                    caption + " image preview",
                    self._preview_pixmap(frame.image, result.roi),
                )
        else:
            self._add_evidence_text("Image previews are unavailable.")

        if result.roi is not None:
            roi = result.roi
            self._add_evidence_text(
                f"Selected NEAT ROI overlay and spectrum source: "
                f"x[{roi.x_min},{roi.x_max}), y[{roi.y_min},{roi.y_max}) (half-open pixels)."
            )
        if result.spectrum is None:
            self._add_evidence_text(
                "ROI spectrum not yet available." if dataset.mode is DatasetMode.IMAGES
                else "Measured spectrum unavailable."
            )
        else:
            spectrum = result.spectrum
            self._add_evidence_text(
                f"Measured intensity ({spectrum.intensity_unit}); full source: "
                f"{spectrum.sample_count} points, {spectrum.finite_pair_count} finite pairs, "
                f"{spectrum.nonfinite_intensity_count} non-finite intensities. "
                f"Axis: {spectrum.wavelength_axis_status.value}. "
                f"Plot window (Å): {result.plot.display_window if result.plot else 'unavailable'}."
            )
            if result.plot is not None and result.plot.png_bytes is not None:
                pixmap = QPixmap()
                if pixmap.loadFromData(result.plot.png_bytes, "PNG"):
                    self._add_evidence_pixmap("Measured spectrum plot", pixmap)
        self.evidence_layout.addStretch()

    def shutdown_for_parent_close(self) -> None:
        """Called only after the main window's worker inventory has drained."""
        self._parent_closing = True
        self._active_run_id = None
        self._agent_run_id = None
        self._pending_planning_context = None
        self._pending_vision_result = None
        self._awaiting_preflight_completion = False
        self._revision_timer.stop()
        self._result = None
        self._planning_outcome = None
        self._vision_outcome = None
        self._vision_worker = None
        self._run_snapshot = None
        self._run_credential_resolver = None
        self._dataset_token = None
        self._discard_adjusted_roi()
        self.plan_text.clear()
        self._show_evidence_message("NEAT is closing.")
        self.hide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale_evidence_images()

    def closeEvent(self, event) -> None:
        self.stop_active_run()
        self._active_run_id = None
        self._agent_run_id = None
        self._pending_planning_context = None
        self._pending_vision_result = None
        self._awaiting_preflight_completion = False
        self._result = None
        self._planning_outcome = None
        self._vision_outcome = None
        self._dataset_token = None
        self._discard_adjusted_roi()
        self._run_snapshot = None
        self._run_credential_resolver = None
        self.plan_text.clear()
        self._revision_timer.stop()
        self._show_evidence_message("No current evidence is displayed.")
        self.hide()
        event.accept()

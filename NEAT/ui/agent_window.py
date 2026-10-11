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
from ..domain.observation import DatasetMode, RoiBounds
from ..services.scientific_observation import (
    DatasetUnavailableError,
    ObservationErrorBase,
)
from .agent_preflight_worker import PreflightWorker


class AgentWindow(QMainWindow):
    """One top-level companion window per NEAT main window."""

    def __init__(self, main_window) -> None:
        super().__init__(main_window, Qt.Window)
        self._main_window = main_window
        self._worker: PreflightWorker | None = None
        self._active_run_id: str | None = None
        self._dataset_token: tuple[str, int] | None = None
        self._result: PreflightResult | None = None
        self._stale = False
        self._parent_closing = False

        self.setWindowTitle("NEAT AI Agent — Local Read-only Preflight")
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
        title = QLabel("Local Read-only Preflight (no LLM yet)", self.overview_panel)
        overview_layout.addWidget(title)
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
            "Start Preflight to inspect the currently loaded dataset. "
            "This local step does not identify a phase, fit an edge, or run mapping."
        )
        overview_layout.addWidget(self.assessment_text)

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
        self._show_evidence_message("No visual evidence yet. Start Preflight to inspect the dataset.")

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
        if self._worker is not None:
            self._worker.stop()
        self._set_status("Dataset changed — rerun preflight.")
        self.assessment_text.setPlainText("Previous evidence was discarded because the dataset changed.")
        self._show_evidence_message("Dataset changed — rerun preflight to see current evidence.")
        if self._worker is None:
            self._revision_timer.stop()

    def start_preflight(self) -> None:
        """Capture live state only on the Qt owner thread, then start bounded work."""
        if self._worker is not None or self._parent_closing:
            return
        if getattr(self._main_window, "_shutdown_close_pending", False) or getattr(
            self._main_window, "_shutdown_committed", False
        ):
            self._set_status("NEAT is closing; retry after the close attempt finishes.")
            return
        self._result = None
        self._stale = False
        self._dataset_token = None
        self._revision_timer.stop()
        self.progress_bar.setValue(0)
        self._show_evidence_message("Preflight is preparing the observation snapshot.")
        self.assessment_text.clear()
        run_id = uuid4().hex
        self._active_run_id = run_id
        adapter = getattr(self._main_window, "scientific_observation_api", None)
        if adapter is None:
            self._set_status("Scientific observation API is unavailable.")
            return
        try:
            handle = adapter.capture_dataset()
            roi = adapter.capture_selected_roi(handle)
        except DatasetUnavailableError:
            self._set_status("Dataset loader is busy — retry preflight after loading finishes.")
            return
        except (ObservationErrorBase, ValueError):
            self._set_status("Dataset capture is unavailable — check loaded data and retry.")
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
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_status("Preflight running locally. No model or network request is used.")
        self._revision_timer.start()
        worker.start()

    def stop_preflight(self) -> None:
        if self._worker is None:
            return
        try:
            self._worker.stop()
        except RuntimeError:
            # Qt can delete a thread after the shutdown inventory retires it.
            self._worker = None
            return
        self.stop_button.setEnabled(False)
        if not self._stale:
            self._set_status("Cancelling — waiting for the current observation step to finish.")

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
        if outcome.run_id != self._active_run_id or self._parent_closing:
            return
        self._check_revision()
        if self._stale or self._dataset_token != (outcome.dataset_id, outcome.revision):
            self._mark_stale()
            return
        if outcome.state is PreflightRunState.COMPLETE and outcome.result is not None:
            self._result = outcome.result
            self.progress_bar.setValue(100)
            self._render_result(outcome.result)
        else:
            self._result = None
            self._show_evidence_message(outcome.message or "No evidence was produced.")
            self.assessment_text.setPlainText(outcome.message or "Preflight did not complete.")
            self._set_status(outcome.message or "Preflight did not complete.")

    def _on_worker_finished(self, run_id: str, worker: PreflightWorker) -> None:
        if self._worker is worker:
            self._worker = None
        if getattr(self._main_window, "agent_preflight_worker", None) is worker:
            self._main_window.agent_preflight_worker = None
        if run_id != self._active_run_id or self._parent_closing:
            return
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self._result is None:
            self._revision_timer.stop()

    def worker_retired(self, worker: PreflightWorker, callbacks_suppressed: bool) -> None:
        """Clear the window reference even when shutdown suppressed signal slots."""
        if self._worker is not worker:
            return
        self._worker = None
        self.stop_button.setEnabled(False)
        if not self._parent_closing:
            self.start_button.setEnabled(True)
            if callbacks_suppressed and not self._stale:
                self._result = None
                self._revision_timer.stop()
                self._set_status("Preflight stopped while NEAT was closing. Start again if NEAT remains open.")

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
        self._revision_timer.stop()
        self._result = None
        self._dataset_token = None
        self._show_evidence_message("NEAT is closing.")
        self.hide()

    def closeEvent(self, event) -> None:
        self.stop_preflight()
        self.hide()
        event.accept()

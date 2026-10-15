"""Concise, progressively disclosed conclusions for the Agent workflow."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ConclusionRow(QFrame):
    """One conclusion, with optional evidence and a link to the full view."""

    inspectRequested = pyqtSignal()

    def __init__(self, title: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        header = QHBoxLayout()
        self.toggle_button = QPushButton(title, self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setMinimumWidth(170)
        self.toggle_button.setAccessibleName(f"Show evidence for {title}")
        self.conclusion_label = QLabel("Not started", self)
        self.conclusion_label.setTextFormat(Qt.PlainText)
        self.conclusion_label.setWordWrap(True)
        self.conclusion_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.conclusion_label.setStyleSheet("font-weight: 600;")
        header.addWidget(self.toggle_button)
        header.addWidget(self.conclusion_label, 1)
        layout.addLayout(header)
        self.evidence_panel = QWidget(self)
        evidence_layout = QVBoxLayout(self.evidence_panel)
        evidence_layout.setContentsMargins(4, 6, 4, 0)
        self.evidence_label = QLabel(self.evidence_panel)
        self.evidence_label.setTextFormat(Qt.PlainText)
        self.evidence_label.setWordWrap(True)
        evidence_layout.addWidget(self.evidence_label)
        self.inspect_button = QPushButton("Inspect evidence…", self.evidence_panel)
        self.inspect_button.clicked.connect(self.inspectRequested)
        evidence_layout.addWidget(self.inspect_button, 0, Qt.AlignLeft)
        layout.addWidget(self.evidence_panel)
        self.toggle_button.toggled.connect(self.evidence_panel.setVisible)
        self.evidence_panel.hide()

    def set_conclusion(self, conclusion: str, evidence: str, *, inspectable: bool) -> None:
        self.conclusion_label.setText(conclusion)
        self.evidence_label.setText(evidence)
        self.inspect_button.setEnabled(inspectable)


class AgentSummaryWidget(QScrollArea):
    """Default Agent screen; no plots, matrices, or planning trace on load."""

    inspectRequested = pyqtSignal(int)
    resultsRequested = pyqtSignal()
    reviewRequested = pyqtSignal()
    settingsRequested = pyqtSignal()
    dataPreviewRequested = pyqtSignal()
    adjustedRoiDiscardRequested = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAccessibleName("Analysis step conclusions")
        outer = QWidget(self)
        outer_layout = QHBoxLayout(outer)
        outer_layout.setContentsMargins(16, 16, 16, 16)
        content = QWidget(outer)
        content.setMaximumWidth(900)
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        self.heading = QLabel("Explore your dataset", content)
        self.heading.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(self.heading)
        self.subtitle = QLabel("Each step will show a short conclusion.", content)
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)
        layout.addSpacing(10)
        self.rows = tuple(
            ConclusionRow(title, content)
            for title in (
                "1 · Dataset check", "2 · Phase comparison",
                "3 · Sample locations", "4 · Local fitting",
            )
        )
        for index, row in enumerate(self.rows):
            row.inspectRequested.connect(lambda current=index: self.inspectRequested.emit(current))
            layout.addWidget(row)
        self.notice = QLabel(content)
        self.notice.setTextFormat(Qt.PlainText)
        self.notice.setWordWrap(True)
        self.notice.setAccessibleName("Analysis action required")
        layout.addWidget(self.notice)
        actions = QHBoxLayout()
        actions.addStretch()
        self.results_button = QPushButton("View all results", content)
        self.results_button.clicked.connect(self.resultsRequested)
        self.review_button = QPushButton("Review flagged fits", content)
        self.review_button.clicked.connect(self.reviewRequested)
        self.discard_adjusted_button = QPushButton("Discard adjusted ROI", content)
        self.discard_adjusted_button.clicked.connect(self.adjustedRoiDiscardRequested)
        self.discard_adjusted_button.hide()
        actions.addWidget(self.discard_adjusted_button)
        actions.addWidget(self.results_button)
        actions.addWidget(self.review_button)
        layout.addLayout(actions)
        self.limitation_label = QLabel(
            "Exploratory analysis · phase, calibration and physical interpretation need separate validation.",
            content,
        )
        self.limitation_label.setWordWrap(True)
        layout.addWidget(self.limitation_label)
        layout.addSpacing(8)
        model_actions = QHBoxLayout()
        self.model_label = QLabel(content)
        self.model_label.setTextFormat(Qt.PlainText)
        self.model_label.setWordWrap(True)
        model_actions.addWidget(self.model_label, 1)
        settings = QPushButton("Agent settings…", content)
        settings.clicked.connect(self.settingsRequested)
        preview = QPushButton("Data sent…", content)
        preview.clicked.connect(self.dataPreviewRequested)
        model_actions.addWidget(settings)
        model_actions.addWidget(preview)
        layout.addLayout(model_actions)
        layout.addStretch()
        outer_layout.addStretch()
        outer_layout.addWidget(content, 1)
        outer_layout.addStretch()
        self.setWidget(outer)

    def reset_disclosures(self) -> None:
        for row in self.rows:
            row.toggle_button.setChecked(False)
        self.verticalScrollBar().setValue(0)

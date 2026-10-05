"""Dockable, non-blocking NEAT AI support panel."""

from __future__ import annotations

import html
from typing import Callable, Mapping, Optional, Sequence

from NEAT.package_resources import assistant_knowledge_root
from tools.assistant_feedback import record_assistant_feedback
from tools.assistant_runtime import (
    default_assistant_index_directory,
    prepare_optional_semantic_runtime,
)

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QApplication,
)


INDEX_DIRECTORY = default_assistant_index_directory()


def _widget_text(window, name: str) -> Optional[str]:
    widget = getattr(window, name, None)
    text_method = getattr(widget, "text", None)
    if not callable(text_method):
        return None
    value = str(text_method()).strip()
    return value or None


def collect_neat_context(window) -> dict[str, object]:
    """Collect a small allow-listed context without file paths or raw data."""

    context: dict[str, object] = {
        "neat_version": str(getattr(window, "app_version", "unknown")),
    }
    tabs = getattr(window, "tabs", None)
    if tabs is not None:
        index = tabs.currentIndex()
        context["current_module"] = str(tabs.tabText(index))

    fitting_tab = getattr(window, "FittingTab", None)
    if tabs is not None and tabs.currentWidget() is fitting_tab:
        data_source = str(getattr(window, "fitting_data_source", "images"))
        if data_source in {"images", "profile"}:
            context["fitting_data_source"] = data_source
        context["images_loaded"] = bool(getattr(window, "images", []))

        fitting_mode = str(getattr(window, "fitting_mode", ""))
        if fitting_mode in {"individual", "pattern"}:
            context["fitting_mode"] = fitting_mode

        batch_map_button = getattr(window, "batch_map_button", None)
        enabled_method = getattr(batch_map_button, "isEnabled", None)
        if callable(enabled_method):
            context["mapping_available"] = bool(enabled_method())

        phase_dropdown = getattr(window, "phase_dropdown", None)
        if phase_dropdown is not None:
            context["phase"] = str(phase_dropdown.currentText())

        fields = {
            "wavelength_min": "min_wavelength_input",
            "wavelength_max": "max_wavelength_input",
            "macro_pixel_width": "box_width_input",
            "macro_pixel_height": "box_height_input",
        }
        for context_name, widget_name in fields.items():
            value = _widget_text(window, widget_name)
            if value is not None:
                context[context_name] = value

        for parameter in ("s", "t", "eta"):
            method = getattr(window, f"fix_{parameter}_enabled", None)
            if callable(method):
                context[f"{parameter}_fixed"] = bool(method())

        current_d = getattr(window, "current_d", None)
        if isinstance(current_d, (int, float)):
            context["selected_edge_d"] = float(current_d)

    return context


def prepare_semantic_runtime_for_gui() -> bool:
    """Prepare semantic search or allow the worker to use lexical fallback."""

    return prepare_optional_semantic_runtime()


class QuestionInput(QPlainTextEdit):
    """Small editor that submits on Ctrl+Enter."""

    submit_requested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (
            event.key() in (Qt.Key_Return, Qt.Key_Enter)
            and event.modifiers() & Qt.ControlModifier
        ):
            self.submit_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class AssistantRequestWorker(QThread):
    """Run local retrieval and the selected supplier away from the GUI thread."""

    answer_ready = pyqtSignal(object)
    request_failed = pyqtSignal(str)

    def __init__(
        self,
        question: str,
        context: Mapping[str, object],
        history: Sequence[Mapping[str, str]],
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.question = question
        self.context = dict(context)
        self.history = [dict(turn) for turn in history]

    def run(self) -> None:
        try:
            from tools.assistant_answering import ConversationTurn
            from tools.assistant_model_factory import (
                create_personal_chat_model,
                describe_provider_error,
            )
            from tools.assistant_providers import (
                AssistantAccessMode,
                ChainedCredentialStore,
                EnvironmentCredentialStore,
                KeyringCredentialStore,
                MissingProviderCredential,
            )
            from tools.assistant_retrieval import load_knowledge_base
            from tools.assistant_runtime import create_release_safe_retriever
            from tools.assistant_service import NEATAssistantService
            from tools.assistant_settings_store import AssistantSettingsRepository
            from tools.assistant_shared_client import (
                MissingSharedServiceConfiguration,
                SharedAssistantClient,
                SharedServiceError,
                load_shared_service_settings,
            )
        except ImportError as exc:
            self.request_failed.emit(str(exc))
            return

        try:
            settings = AssistantSettingsRepository().load()
            turns = [
                ConversationTurn(
                    role=str(turn.get("role", "")),
                    content=str(turn.get("content", "")),
                )
                for turn in self.history
            ]
            if settings.access_mode == AssistantAccessMode.NEAT_SHARED:
                result = SharedAssistantClient(
                    load_shared_service_settings()
                ).ask(
                    self.question,
                    context=self.context,
                    history=turns,
                )
                self.answer_ready.emit(result)
                return

            credential_store = ChainedCredentialStore(
                [
                    KeyringCredentialStore(),
                    EnvironmentCredentialStore(),
                ]
            )
            sections = load_knowledge_base(assistant_knowledge_root())
            retriever = create_release_safe_retriever(
                sections,
                persist_directory=INDEX_DIRECTORY,
            )
            result = NEATAssistantService(
                retriever,
                create_personal_chat_model(
                    settings,
                    credential_store,
                ),
            ).ask(
                self.question,
                context=self.context,
                history=turns,
                source_limit=3,
                config={
                    "tags": [
                        "neat-gui-assistant",
                        settings.provider.value,
                    ]
                },
            )
        except (
            MissingProviderCredential,
            MissingSharedServiceConfiguration,
            SharedServiceError,
        ) as exc:
            self.request_failed.emit(str(exc))
            return
        except Exception as exc:
            try:
                provider = getattr(settings, "provider", None)
                if provider is None:
                    raise RuntimeError("No provider was selected")
                message = describe_provider_error(provider, exc)
            except Exception:
                message = f"Assistant request failed: {type(exc).__name__}"
            self.request_failed.emit(message)
            return

        self.answer_ready.emit(result)


class AssistantDockWidget(QDockWidget):
    """Chat UI that exposes grounded NEAT support inside the main window."""

    def __init__(
        self,
        *,
        context_provider: Callable[[], Mapping[str, object]],
        parent=None,
        worker_class=AssistantRequestWorker,
        runtime_preparer=prepare_semantic_runtime_for_gui,
        feedback_logger=record_assistant_feedback,
    ) -> None:
        super().__init__("NEAT AI Assistant", parent)
        self.setObjectName("neat_ai_assistant_dock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(330)
        self.resize(420, 650)
        self.context_provider = context_provider
        self.worker_class = worker_class
        self.runtime_preparer = runtime_preparer
        self.feedback_logger = feedback_logger
        self._semantic_runtime_ready = False
        self.worker: Optional[QThread] = None
        self.history: list[dict[str, str]] = []
        self._pending_feedback: Optional[dict[str, object]] = None
        self._request_neat_version = "unknown"

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Ask how to use NEAT")
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title)

        self.privacy_label = QLabel(
            "Uses approved NEAT guidance and selected on-screen settings. "
            "Raw image data and file paths are not sent. Optional answer "
            "feedback is stored locally."
        )
        self.privacy_label.setWordWrap(True)
        self.privacy_label.setStyleSheet("color: #59636e;")
        layout.addWidget(self.privacy_label)

        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(False)
        self.transcript.setPlaceholderText("Your NEAT support conversation appears here.")
        layout.addWidget(self.transcript, 1)

        quick_action_layout = QHBoxLayout()
        self.explain_screen_button = QPushButton("Explain current screen")
        self.explain_screen_button.setToolTip(
            "Explain the active NEAT workflow and its main controls."
        )
        self.explain_screen_button.clicked.connect(self.explain_current_screen)
        quick_action_layout.addWidget(self.explain_screen_button)
        quick_action_layout.addStretch(1)
        self.settings_button = QPushButton("AI settings...")
        self.settings_button.setToolTip(
            "Choose the model supplier and securely configure a personal API key."
        )
        self.settings_button.clicked.connect(self.open_settings_dialog)
        quick_action_layout.addWidget(self.settings_button)
        layout.addLayout(quick_action_layout)

        self.question_input = QuestionInput()
        self.question_input.setPlaceholderText(
            "For example: Why is my Bragg-edge fit unstable?\n"
            "Press Ctrl+Enter to send."
        )
        self.question_input.setMaximumHeight(100)
        self.question_input.submit_requested.connect(self.submit_question)
        layout.addWidget(self.question_input)

        button_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_conversation)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch(1)
        self.ask_button = QPushButton("Ask")
        self.ask_button.setDefault(True)
        self.ask_button.clicked.connect(self.submit_question)
        button_layout.addWidget(self.ask_button)
        layout.addLayout(button_layout)

        self.feedback_widget = QWidget()
        feedback_layout = QHBoxLayout(self.feedback_widget)
        feedback_layout.setContentsMargins(0, 0, 0, 0)
        feedback_layout.addWidget(QLabel("Was this answer helpful?"))
        feedback_layout.addStretch(1)
        self.helpful_button = QPushButton("Helpful")
        self.helpful_button.setToolTip("Store a helpful rating locally.")
        self.helpful_button.clicked.connect(
            lambda: self._record_feedback("helpful")
        )
        feedback_layout.addWidget(self.helpful_button)
        self.not_helpful_button = QPushButton("Not helpful")
        self.not_helpful_button.setToolTip("Store a not-helpful rating locally.")
        self.not_helpful_button.clicked.connect(
            lambda: self._record_feedback("not_helpful")
        )
        feedback_layout.addWidget(self.not_helpful_button)
        self.feedback_widget.setVisible(False)
        layout.addWidget(self.feedback_widget)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.setWidget(container)
        self._set_ready_status()

        toggle_action = self.toggleViewAction()
        toggle_action.setText("AI Assistant")
        toggle_action.setShortcut("Ctrl+Shift+A")

    def _set_ready_status(self) -> None:
        try:
            from tools.assistant_providers import (
                AssistantAccessMode,
                ChainedCredentialStore,
                EnvironmentCredentialStore,
                KeyringCredentialStore,
                PROVIDER_ENVIRONMENT_VARIABLES,
                PROVIDER_LABELS,
                provider_requires_api_key,
            )
            from tools.assistant_settings_store import AssistantSettingsRepository

            settings = AssistantSettingsRepository().load()
            if settings.access_mode == AssistantAccessMode.NEAT_SHARED:
                from tools.assistant_shared_client import (
                    load_shared_service_settings,
                )

                load_shared_service_settings()
                self.privacy_label.setText(
                    "Uses approved NEAT guidance and selected safe screen settings. "
                    "Questions are sent through the hosted NEAT service to OpenAI. "
                    "Raw images and automatically collected file paths are not sent."
                )
                self.status_label.setText(
                    "Configured — NEAT shared access (availability is checked "
                    "when you ask; limited request allowance per day)"
                )
                self.status_label.setStyleSheet("color: #26734d;")
                return

            label = PROVIDER_LABELS[settings.provider]
            requires_key = provider_requires_api_key(settings)
            credential = None
            if requires_key:
                credential = ChainedCredentialStore(
                    [
                        KeyringCredentialStore(),
                        EnvironmentCredentialStore(),
                    ]
                ).get(settings.provider)
        except Exception:
            self.status_label.setText(
                "AI settings could not be loaded. Open AI settings to repair them."
            )
            self.status_label.setStyleSheet("color: #b42318;")
            return

        self.privacy_label.setText(
            f"Uses approved NEAT guidance and selected safe screen settings. "
            f"Questions are sent directly to {label}. Raw images and automatically "
            "collected file paths are not sent. Optional feedback is stored locally."
        )
        if credential or not requires_key:
            self.status_label.setText(
                f"Ready — {label} / {settings.model}"
            )
            self.status_label.setStyleSheet("color: #26734d;")
        else:
            variable = PROVIDER_ENVIRONMENT_VARIABLES[settings.provider]
            self.status_label.setText(
                f"No {label} key is configured. Open AI settings or set {variable}."
            )
            self.status_label.setStyleSheet("color: #9a6700;")

    def open_settings_dialog(self) -> None:
        """Open supplier/model settings and refresh the assistant status."""

        if self.worker is not None and self.worker.isRunning():
            self.status_label.setText(
                "Wait for the current answer before changing AI settings."
            )
            self.status_label.setStyleSheet("color: #9a6700;")
            return
        try:
            from NEAT.ui.assistant_settings_dialog import AssistantSettingsDialog

            dialog = AssistantSettingsDialog(self)
            accepted = bool(dialog.exec_())
        except Exception as exc:
            self.status_label.setText(
                f"AI settings could not open: {type(exc).__name__}."
            )
            self.status_label.setStyleSheet("color: #b42318;")
            return
        if accepted:
            self.history.clear()
            self._pending_feedback = None
            self.feedback_widget.setVisible(False)
            self._set_ready_status()

    def submit_question(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        question = self.question_input.toPlainText().strip()
        if not question:
            self.status_label.setText("Enter a question first.")
            return

        try:
            context = dict(self.context_provider())
        except Exception:
            context = {}
        self._request_neat_version = str(context.get("neat_version", "unknown"))

        if not self._semantic_runtime_ready:
            self.status_label.setText("Starting the local NEAT search engine…")
            self.status_label.setStyleSheet("color: #59636e;")
            QApplication.processEvents()
            try:
                self.runtime_preparer()
            except Exception:
                # The request worker will use dependency-free BM25 retrieval.
                pass
            self._semantic_runtime_ready = True

        self._append_message("You", question, "#1f5a94")
        self._pending_feedback = None
        self.feedback_widget.setVisible(False)
        self.question_input.clear()
        self.ask_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.explain_screen_button.setEnabled(False)
        self.settings_button.setEnabled(False)
        self.status_label.setText("Searching NEAT guidance and preparing an answer…")
        self.status_label.setStyleSheet("color: #59636e;")

        self.worker = self.worker_class(
            question,
            context,
            self.history[-12:],
            parent=self,
        )
        self.worker.answer_ready.connect(
            lambda result, sent_question=question: self._on_answer_ready(
                sent_question, result
            )
        )
        self.worker.request_failed.connect(self._on_request_failed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def explain_current_screen(self) -> None:
        """Ask for a grounded explanation of the active NEAT workflow."""

        if self.worker is not None and self.worker.isRunning():
            return
        try:
            context = dict(self.context_provider())
        except Exception:
            context = {}

        current_module = str(context.get("current_module", "")).strip()
        if current_module:
            question = (
                f"Explain the current NEAT screen: {current_module}. "
                "Describe its main areas and controls, what I can do here, "
                "important prerequisites, and relevant cautions."
            )
        else:
            question = (
                "Explain the current NEAT screen. Describe its main areas and "
                "controls, what I can do here, important prerequisites, and "
                "relevant cautions."
            )

        draft = self.question_input.toPlainText()
        self.question_input.setPlainText(question)
        self.submit_question()
        self.question_input.setPlainText(draft)

    def _append_message(self, speaker: str, text: str, color: str) -> None:
        safe_text = html.escape(str(text)).replace("\n", "<br>")
        self.transcript.append(
            f'<p><b style="color:{color}">{html.escape(speaker)}</b><br>'
            f"{safe_text}</p>"
        )

    def _on_answer_ready(self, question: str, result) -> None:
        self._append_message("NEAT Assistant", result.answer, "#26734d")
        sources = "<br>".join(
            f"[{citation.number}] {html.escape(citation.heading_path)}"
            for citation in result.citations
        )
        self.transcript.append(
            '<p style="color:#59636e"><b>Verified NEAT sources</b><br>'
            f"{sources}</p>"
        )
        if result.requires_human_review:
            self.transcript.append(
                '<p style="color:#9a6700"><b>Human scientific review is required.</b></p>'
            )
        self.history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": result.answer},
            ]
        )
        self.history = self.history[-12:]
        self._pending_feedback = {
            "question": question,
            "citations": [
                {
                    "source_id": str(getattr(citation, "source_id", "")),
                    "heading_path": citation.heading_path,
                }
                for citation in result.citations
            ],
            "route": str(getattr(result, "route", "unknown")),
            "requires_human_review": bool(result.requires_human_review),
            "neat_version": self._request_neat_version,
        }
        self.helpful_button.setEnabled(True)
        self.not_helpful_button.setEnabled(True)
        self.feedback_widget.setVisible(True)
        shared_remaining = getattr(result, "shared_remaining", None)
        shared_limit = getattr(result, "shared_daily_limit", None)
        if shared_remaining is not None and shared_limit is not None:
            self.status_label.setText(
                f"Answer complete — {shared_remaining} of {shared_limit} "
                "shared requests remain today"
            )
        else:
            self.status_label.setText("Answer complete")
        self.status_label.setStyleSheet("color: #26734d;")

    def _record_feedback(self, rating: str) -> None:
        if self._pending_feedback is None:
            return
        try:
            saved_path = self.feedback_logger(
                rating=rating,
                **self._pending_feedback,
            )
        except Exception:
            self.status_label.setText(
                "The feedback could not be saved locally. The answer is unaffected."
            )
            self.status_label.setStyleSheet("color: #9a6700;")
            return

        self._pending_feedback = None
        self.helpful_button.setEnabled(False)
        self.not_helpful_button.setEnabled(False)
        if saved_path:
            self.status_label.setText(
                f"Thank you. Feedback saved locally: {saved_path}"
            )
            self.status_label.setToolTip(str(saved_path))
        else:
            self.status_label.setText("Thank you. Feedback saved locally.")
        self.status_label.setStyleSheet("color: #26734d;")

    def _on_request_failed(self, message: str) -> None:
        self._append_message("Assistant error", message, "#b42318")
        self.status_label.setText("The assistant could not answer this question.")
        self.status_label.setStyleSheet("color: #b42318;")

    def _on_worker_finished(self) -> None:
        self.ask_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.explain_screen_button.setEnabled(True)
        self.settings_button.setEnabled(True)
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()

    def clear_conversation(self) -> None:
        self.history.clear()
        self._pending_feedback = None
        self.feedback_widget.setVisible(False)
        self.transcript.clear()
        self._set_ready_status()

    def shutdown(self) -> bool:
        worker = self.worker
        if worker is None or not worker.isRunning():
            return True
        worker.requestInterruption()
        return bool(worker.wait(1500))


__all__ = [
    "AssistantDockWidget",
    "AssistantRequestWorker",
    "QuestionInput",
    "collect_neat_context",
    "prepare_semantic_runtime_for_gui",
]

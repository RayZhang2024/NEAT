"""Explicit configuration dialog for the independent AI Agent model."""

from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ..domain.agent_model import AgentModelMode, AgentModelSettings
from ..services.agent_model import (
    AgentCredentialResolver,
    AgentKeyringCredentialStore,
    AgentModelSettingsRepository,
    AgentModelUnavailable,
    HttpxJsonTransport,
    run_model_connection_test,
    validate_agent_settings,
)
from ..services.agent_tracing import (
    LangSmithConnectionError,
    agent_langsmith_context,
    test_langsmith_trace_connection,
)


class AgentModelConnectionWorker(QThread):
    """One explicit, bounded settings request containing no NEAT data."""

    completed = pyqtSignal(bool, str)

    def __init__(
        self,
        settings: AgentModelSettings,
        *,
        pending_api_key: str | None = None,
        pending_langsmith_api_key: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._pending_api_key = pending_api_key
        self._pending_langsmith_api_key = pending_langsmith_api_key
        self._transport = HttpxJsonTransport()

    def stop(self) -> None:
        self._transport.cancel()

    def run(self) -> None:
        pending_api_key = self._pending_api_key
        pending_langsmith_api_key = self._pending_langsmith_api_key
        # The worker owns the temporary credential only while its request runs.
        # Clear the instance reference immediately so shutdown tracking cannot
        # keep the key alive after the worker has finished.
        self._pending_api_key = None
        self._pending_langsmith_api_key = None
        succeeded = False
        messages: list[str] = []
        outcomes: list[bool] = []
        langsmith_key = pending_langsmith_api_key
        if self._settings.langsmith_tracing_enabled and not langsmith_key:
            try:
                langsmith_key = AgentKeyringCredentialStore(
                    account_name="langsmith"
                ).get()
            except Exception:
                langsmith_key = None
        if self._settings.mode is not AgentModelMode.OFFLINE:
            try:
                resolver = AgentCredentialResolver(
                    AgentKeyringCredentialStore(),
                    environment_allowed=self._settings.allow_environment_credential,
                )

                def test_model() -> None:
                    run_model_connection_test(
                        self._settings,
                        resolver,
                        credential_override=pending_api_key,
                        transport=self._transport,
                    )

                if self._settings.langsmith_tracing_enabled:
                    with agent_langsmith_context(
                        enabled=True,
                        api_key=langsmith_key,
                        agent_id=self._settings.langsmith_agent_id,
                        environment=self._settings.langsmith_environment,
                        region=self._settings.langsmith_region,
                        workspace_id=self._settings.langsmith_workspace_id,
                        workflow="neat_connection_test",
                    ):
                        test_model()
                else:
                    test_model()
            except AgentModelUnavailable as exc:
                outcomes.append(False)
                messages.append(f"Model: {exc}")
            except Exception as exc:  # noqa: BLE001 - do not display exception contents
                outcomes.append(False)
                messages.append(f"Model test failed ({type(exc).__name__}).")
            else:
                outcomes.append(True)
                messages.append("Model connection succeeded.")

        if self._settings.langsmith_tracing_enabled:
            try:
                if not langsmith_key:
                    raise LangSmithConnectionError("Enter or save a LangSmith API key.")
                test_langsmith_trace_connection(
                    api_key=langsmith_key,
                    agent_id=self._settings.langsmith_agent_id,
                    environment=self._settings.langsmith_environment,
                    region=self._settings.langsmith_region,
                    workspace_id=self._settings.langsmith_workspace_id,
                )
            except LangSmithConnectionError as exc:
                outcomes.append(False)
                messages.append(f"LangSmith: {exc}")
            except Exception as exc:  # noqa: BLE001 - do not display exception contents
                outcomes.append(False)
                messages.append(f"LangSmith test failed ({type(exc).__name__}).")
            else:
                outcomes.append(True)
                messages.append("LangSmith trace upload succeeded.")
        # Drop local references before notifying the UI or allowing the tracked
        # QThread to retire.
        pending_api_key = None
        pending_langsmith_api_key = None
        succeeded = bool(outcomes) and all(outcomes)
        message = " ".join(messages) or "Select a model or enable LangSmith before testing."
        self.completed.emit(succeeded, message)


class AgentModelSettingsDialog(QDialog):
    """Save Agent-only provider settings and credentials after user action."""

    def __init__(
        self,
        parent=None,
        *,
        main_window=None,
        settings_repository: AgentModelSettingsRepository | None = None,
        credential_store=None,
        langsmith_credential_store=None,
        connection_worker_class=AgentModelConnectionWorker,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("NEAT AI Agent Settings")
        self.setMinimumWidth(520)
        self.settings_repository = settings_repository or AgentModelSettingsRepository()
        self.credential_store = credential_store or AgentKeyringCredentialStore()
        self.langsmith_credential_store = (
            langsmith_credential_store
            or AgentKeyringCredentialStore(account_name="langsmith")
        )
        self.connection_worker_class = connection_worker_class
        self.main_window = main_window
        self.connection_worker: AgentModelConnectionWorker | None = None

        loaded = self.settings_repository.load()
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "These settings are separate from NEAT AI Assistant. In OpenAI cloud "
            "mode, pressing Start authorizes a metadata-only planning request when "
            "eligible and up to 16 bounded PNGs across 6 visual requests for image-stack "
            "review: reference image previews, 1 measured-spectrum plot, up to 3 phase-overlay "
            "plots, and bounded fitting-location retries. Saving settings alone sends nothing.",
            self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem("No LLM (analysis unavailable)", AgentModelMode.OFFLINE.value)
        self.mode_combo.addItem("Local model (loopback only)", AgentModelMode.LOCAL.value)
        self.mode_combo.addItem("OpenAI cloud", AgentModelMode.OPENAI_CLOUD.value)
        self.mode_combo.setCurrentIndex(
            max(0, self.mode_combo.findData(loaded.mode.value))
        )
        form.addRow("Agent planning mode", self.mode_combo)

        self.model_edit = QLineEdit(loaded.model_id, self)
        self.model_edit.setPlaceholderText("Enter the exact model ID")
        form.addRow("Model ID", self.model_edit)
        self.endpoint_edit = QLineEdit(loaded.endpoint, self)
        self.endpoint_edit.setPlaceholderText("http://127.0.0.1:11434/v1")
        form.addRow("Local endpoint", self.endpoint_edit)

        self.key_edit = QLineEdit(self)
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("Optional; stored only in OS keyring")
        form.addRow("OpenAI API key", self.key_edit)
        self.environment_checkbox = QCheckBox(
            "Allow OPENAI_API_KEY environment fallback (keyring takes precedence)", self
        )
        self.environment_checkbox.setChecked(loaded.allow_environment_credential)
        form.addRow("Credential lookup", self.environment_checkbox)
        self.langsmith_tracing_checkbox = QCheckBox(
            "Enable LangSmith traces (LLM calls and graph stages)", self
        )
        self.langsmith_tracing_checkbox.setChecked(loaded.langsmith_tracing_enabled)
        form.addRow("LangSmith", self.langsmith_tracing_checkbox)
        self.langsmith_agent_id_edit = QLineEdit(loaded.langsmith_agent_id, self)
        self.langsmith_agent_id_edit.setMaxLength(63)
        self.langsmith_agent_id_edit.setPlaceholderText("neat-agentic-roi")
        form.addRow("LangSmith agent ID", self.langsmith_agent_id_edit)
        self.langsmith_environment_combo = QComboBox(self)
        for environment in ("local", "development", "staging", "production"):
            self.langsmith_environment_combo.addItem(environment.title(), environment)
        self.langsmith_environment_combo.setCurrentIndex(
            max(0, self.langsmith_environment_combo.findData(loaded.langsmith_environment))
        )
        form.addRow("Agent environment", self.langsmith_environment_combo)
        self.langsmith_region_combo = QComboBox(self)
        for label, region in (
            ("EU (GCP)", "eu"),
            ("US (GCP)", "us"),
            ("APAC (GCP)", "apac"),
            ("US (AWS)", "aws-us"),
        ):
            self.langsmith_region_combo.addItem(label, region)
        self.langsmith_region_combo.setCurrentIndex(
            max(0, self.langsmith_region_combo.findData(loaded.langsmith_region))
        )
        form.addRow("LangSmith region", self.langsmith_region_combo)
        self.langsmith_workspace_edit = QLineEdit(loaded.langsmith_workspace_id, self)
        self.langsmith_workspace_edit.setMaxLength(128)
        self.langsmith_workspace_edit.setPlaceholderText("Optional")
        form.addRow("Workspace ID", self.langsmith_workspace_edit)
        self.langsmith_key_edit = QLineEdit(self)
        self.langsmith_key_edit.setEchoMode(QLineEdit.Password)
        self.langsmith_key_edit.setPlaceholderText("Stored only in OS keyring")
        form.addRow("LangSmith API key", self.langsmith_key_edit)
        layout.addLayout(form)

        self.privacy_label = QLabel(
            "OpenAI analysis uses at most one bounded planning request. For image stacks, "
            "visual review may send up to 16 bounded PNGs in 6 requests: up to 3 reference "
            "frames, 1 ROI-spectrum plot, up to 3 trusted phase-overlay plots, and up to "
            "9 repeated frames for fitting-location selection and two retries. These "
            "requests use the same model ID and Agent API key. Local mode stays on the "
            "configured loopback endpoint and sends no visual evidence to OpenAI; model "
            "failure does not fall back to an offline plan. A run without a selected ROI "
            "may start visual ROI discovery directly and skip metadata planning. LangSmith "
            "is separately opt-in; when enabled, graph state and each model request and "
            "response are uploaded. Visual LLM traces include the prompts and bounded PNG "
            "previews sent for review. Traces may include dataset-derived values, ROI "
            "coordinates, spectra or rendered previews. The agent is created "
            "automatically on its first trace. Copy the exact Agent ID from its Overview; it can differ from the "
            "display name. Select the region that matches your LangSmith workspace. "
            "Enter a workspace ID only when your API key spans multiple workspaces.",
            self,
        )
        self.privacy_label.setWordWrap(True)
        layout.addWidget(self.privacy_label)
        self.status_label = QLabel("No connection is tested automatically.", self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        actions = QHBoxLayout()
        self.test_button = QPushButton("Test Connections…", self)
        self.test_button.clicked.connect(self.test_connection)
        actions.addWidget(self.test_button)
        self.remove_key_button = QPushButton("Remove saved Agent key", self)
        self.remove_key_button.clicked.connect(self.remove_saved_key)
        actions.addWidget(self.remove_key_button)
        self.remove_langsmith_key_button = QPushButton("Remove LangSmith key", self)
        self.remove_langsmith_key_button.clicked.connect(self.remove_saved_langsmith_key)
        actions.addWidget(self.remove_langsmith_key_button)
        actions.addStretch()
        layout.addLayout(actions)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self
        )
        self.button_box.accepted.connect(self.save_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self.mode_combo.currentIndexChanged.connect(self._update_fields)
        self.langsmith_tracing_checkbox.stateChanged.connect(self._update_fields)
        self._update_fields()

    @property
    def has_unsettled_workers(self) -> bool:
        worker = self.connection_worker
        if worker is None:
            return False
        try:
            return worker.isRunning()
        except RuntimeError:
            # The main-window inventory may retire and delete the QThread before
            # this dialog processes its queued finished notification.
            self.connection_worker = None
            return False

    def selected_settings(self) -> AgentModelSettings:
        mode = AgentModelMode(self.mode_combo.currentData())
        return validate_agent_settings(
            AgentModelSettings(
                mode=mode,
                model_id=self.model_edit.text().strip(),
                endpoint=self.endpoint_edit.text().strip(),
                allow_environment_credential=self.environment_checkbox.isChecked(),
                langsmith_tracing_enabled=self.langsmith_tracing_checkbox.isChecked(),
                langsmith_agent_id=self.langsmith_agent_id_edit.text().strip(),
                langsmith_environment=str(self.langsmith_environment_combo.currentData()),
                langsmith_region=str(self.langsmith_region_combo.currentData()),
                langsmith_workspace_id=self.langsmith_workspace_edit.text().strip(),
            )
        )

    def _update_fields(self) -> None:
        mode = AgentModelMode(self.mode_combo.currentData())
        self.model_edit.setEnabled(mode is not AgentModelMode.OFFLINE)
        self.endpoint_edit.setEnabled(mode is AgentModelMode.LOCAL)
        self.key_edit.setEnabled(mode is AgentModelMode.OPENAI_CLOUD)
        self.environment_checkbox.setEnabled(mode is AgentModelMode.OPENAI_CLOUD)
        tracing_enabled = self.langsmith_tracing_checkbox.isChecked()
        self.langsmith_agent_id_edit.setEnabled(tracing_enabled)
        self.langsmith_environment_combo.setEnabled(tracing_enabled)
        self.langsmith_region_combo.setEnabled(tracing_enabled)
        self.langsmith_workspace_edit.setEnabled(tracing_enabled)
        self.langsmith_key_edit.setEnabled(tracing_enabled)
        self.test_button.setEnabled(
            (mode is not AgentModelMode.OFFLINE or tracing_enabled)
            and not self.has_unsettled_workers
        )

    def save_and_accept(self) -> None:
        try:
            settings = self.selected_settings()
            pending_key = self.key_edit.text().strip()
            pending_langsmith_key = self.langsmith_key_edit.text().strip()
            if settings.langsmith_tracing_enabled and not (
                pending_langsmith_key or self.langsmith_credential_store.get()
            ):
                self.status_label.setText(
                    "Enter a LangSmith API key to enable tracing."
                )
                return
            if pending_key:
                if settings.mode is not AgentModelMode.OPENAI_CLOUD:
                    self.status_label.setText(
                        "Select OpenAI cloud before saving an OpenAI key."
                    )
                    return
                self.credential_store.set(pending_key)
            if pending_langsmith_key:
                self.langsmith_credential_store.set(pending_langsmith_key)
            self.settings_repository.save(settings)
        except (ValueError, RuntimeError) as exc:
            self.status_label.setText(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - never include secret-bearing details
            self.status_label.setText(f"Settings could not be saved ({type(exc).__name__}).")
            return
        self.key_edit.clear()
        self.langsmith_key_edit.clear()
        self.accept()

    def test_connection(self) -> None:
        if self.has_unsettled_workers:
            return
        try:
            settings = self.selected_settings()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        pending_key = self.key_edit.text().strip()
        pending_langsmith_key = self.langsmith_key_edit.text().strip()
        self.status_label.setText(
            "Checking each enabled service. This sends one small model request when "
            "configured and one synthetic LangSmith trace when tracing is "
            "enabled. The model request is also recorded as an LLM run when tracing "
            "is enabled. No NEAT dataset or planning summary is included."
        )
        worker = self.connection_worker_class(
            settings,
            pending_api_key=(
                pending_key or None
                if settings.mode is AgentModelMode.OPENAI_CLOUD
                else None
            ),
            pending_langsmith_api_key=pending_langsmith_key or None,
            parent=self,
        )
        self.connection_worker = worker
        main_window = self.main_window
        if main_window is not None and callable(
            getattr(main_window, "_track_shutdown_worker", None)
        ):
            main_window.agent_model_test_worker = worker
            main_window._track_shutdown_worker(worker, "agent_model_test")
            main_window._connect_shutdown_worker_signal(
                worker, worker.completed, self._on_test_completed
            )
        else:
            worker.completed.connect(self._on_test_completed)
        worker.finished.connect(self._on_test_finished)
        self.test_button.setEnabled(False)
        self.button_box.setEnabled(False)
        worker.start()

    def _on_test_completed(self, succeeded: bool, message: str) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            "color: #26734d;" if succeeded else "color: #b42318;"
        )

    def _on_test_finished(self) -> None:
        # The shutdown inventory keeps ownership through native thread exit.
        # Drop the dialog's convenience reference so it cannot outlive a
        # retired/deleted QThread wrapper.
        self.connection_worker = None
        self.test_button.setEnabled(True)
        self.button_box.setEnabled(True)
        self._update_fields()

    def remove_saved_key(self) -> None:
        try:
            self.credential_store.delete()
        except Exception as exc:  # noqa: BLE001 - safe diagnostic only
            self.status_label.setText(
                f"Saved Agent key could not be removed ({type(exc).__name__})."
            )
            return
        self.key_edit.clear()
        self.status_label.setText("Saved Agent key removed from the OS keyring.")

    def remove_saved_langsmith_key(self) -> None:
        try:
            self.langsmith_credential_store.delete()
        except Exception as exc:  # noqa: BLE001 - safe diagnostic only
            self.status_label.setText(
                f"Saved LangSmith key could not be removed ({type(exc).__name__})."
            )
            return
        self.langsmith_key_edit.clear()
        self.status_label.setText("Saved LangSmith key removed from the OS keyring.")

    def reject(self) -> None:
        if self.has_unsettled_workers:
            self.status_label.setText(
                "Wait for the bounded connection test to finish before closing settings."
            )
            return
        self.key_edit.clear()
        self.langsmith_key_edit.clear()
        super().reject()

    def closeEvent(self, event) -> None:
        if self.has_unsettled_workers:
            self.status_label.setText(
                "Wait for the bounded connection test to finish before closing settings."
            )
            event.ignore()
            return
        self.key_edit.clear()
        self.langsmith_key_edit.clear()
        super().closeEvent(event)


__all__ = ["AgentModelConnectionWorker", "AgentModelSettingsDialog"]

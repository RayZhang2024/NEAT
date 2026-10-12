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


class AgentModelConnectionWorker(QThread):
    """One explicit, bounded settings request containing no NEAT data."""

    completed = pyqtSignal(bool, str)

    def __init__(
        self,
        settings: AgentModelSettings,
        *,
        pending_api_key: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._pending_api_key = pending_api_key
        self._transport = HttpxJsonTransport()

    def stop(self) -> None:
        self._transport.cancel()

    def run(self) -> None:
        pending_api_key = self._pending_api_key
        # The worker owns the temporary credential only while its request runs.
        # Clear the instance reference immediately so shutdown tracking cannot
        # keep the key alive after the worker has finished.
        self._pending_api_key = None
        succeeded = False
        message = "Connection test succeeded."
        try:
            resolver = AgentCredentialResolver(
                AgentKeyringCredentialStore(),
                environment_allowed=self._settings.allow_environment_credential,
            )
            run_model_connection_test(
                self._settings,
                resolver,
                credential_override=pending_api_key,
                transport=self._transport,
            )
        except AgentModelUnavailable as exc:
            message = str(exc)
        except Exception as exc:  # noqa: BLE001 - do not display exception contents
            message = f"Connection test failed ({type(exc).__name__})."
        else:
            succeeded = True
        finally:
            # Drop the local reference before notifying the UI or allowing the
            # tracked QThread to retire.
            pending_api_key = None
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
        connection_worker_class=AgentModelConnectionWorker,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("NEAT AI Agent Model Settings")
        self.setMinimumWidth(520)
        self.settings_repository = settings_repository or AgentModelSettingsRepository()
        self.credential_store = credential_store or AgentKeyringCredentialStore()
        self.connection_worker_class = connection_worker_class
        self.main_window = main_window
        self.connection_worker: AgentModelConnectionWorker | None = None

        loaded = self.settings_repository.load()
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "These settings are separate from NEAT AI Assistant. Saving a cloud "
            "provider does not permit data transmission. Cloud planning also needs "
            "a fresh permission for each analysis run.",
            self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem("Offline deterministic", AgentModelMode.OFFLINE.value)
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
        layout.addLayout(form)

        self.privacy_label = QLabel(
            "Model requests are one-shot, metadata-only, and limited to one request "
            "per analysis run. Local mode accepts only loopback endpoints and does "
            "not use proxy settings.",
            self,
        )
        self.privacy_label.setWordWrap(True)
        layout.addWidget(self.privacy_label)
        self.status_label = QLabel("No connection is tested automatically.", self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        actions = QHBoxLayout()
        self.test_button = QPushButton("Test Connection…", self)
        self.test_button.clicked.connect(self.test_connection)
        actions.addWidget(self.test_button)
        self.remove_key_button = QPushButton("Remove saved Agent key", self)
        self.remove_key_button.clicked.connect(self.remove_saved_key)
        actions.addWidget(self.remove_key_button)
        actions.addStretch()
        layout.addLayout(actions)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self
        )
        self.button_box.accepted.connect(self.save_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self.mode_combo.currentIndexChanged.connect(self._update_fields)
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
            )
        )

    def _update_fields(self) -> None:
        mode = AgentModelMode(self.mode_combo.currentData())
        self.model_edit.setEnabled(mode is not AgentModelMode.OFFLINE)
        self.endpoint_edit.setEnabled(mode is AgentModelMode.LOCAL)
        self.key_edit.setEnabled(mode is AgentModelMode.OPENAI_CLOUD)
        self.environment_checkbox.setEnabled(mode is AgentModelMode.OPENAI_CLOUD)
        self.test_button.setEnabled(
            mode is not AgentModelMode.OFFLINE and not self.has_unsettled_workers
        )

    def save_and_accept(self) -> None:
        try:
            settings = self.selected_settings()
            pending_key = self.key_edit.text().strip()
            if pending_key:
                if settings.mode is not AgentModelMode.OPENAI_CLOUD:
                    self.status_label.setText(
                        "Select OpenAI cloud before saving an OpenAI key."
                    )
                    return
                self.credential_store.set(pending_key)
            self.settings_repository.save(settings)
        except (ValueError, RuntimeError) as exc:
            self.status_label.setText(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - never include secret-bearing details
            self.status_label.setText(f"Settings could not be saved ({type(exc).__name__}).")
            return
        self.key_edit.clear()
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
        self.status_label.setText(
            "Sending one small connection-test request to the configured model. "
            "No NEAT dataset or planning summary is included."
        )
        worker = self.connection_worker_class(
            settings,
            pending_api_key=(
                pending_key or None
                if settings.mode is AgentModelMode.OPENAI_CLOUD
                else None
            ),
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

    def reject(self) -> None:
        if self.has_unsettled_workers:
            self.status_label.setText(
                "Wait for the bounded connection test to finish before closing settings."
            )
            return
        self.key_edit.clear()
        super().reject()

    def closeEvent(self, event) -> None:
        if self.has_unsettled_workers:
            self.status_label.setText(
                "Wait for the bounded connection test to finish before closing settings."
            )
            event.ignore()
            return
        self.key_edit.clear()
        super().closeEvent(event)


__all__ = ["AgentModelConnectionWorker", "AgentModelSettingsDialog"]

"""User-facing supplier, model, and secure API-key settings."""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from tools.assistant_model_factory import (
    create_personal_chat_model,
    describe_provider_error,
)
from tools.assistant_local_models import (
    LOCAL_MODEL_ENGINES,
    discover_local_models,
)
from tools.assistant_providers import (
    AssistantAccessMode,
    AssistantProvider,
    AssistantSettings,
    ChainedCredentialStore,
    CredentialStore,
    EnvironmentCredentialStore,
    KeyringCredentialStore,
    MutableCredentialStore,
    PROVIDER_ENVIRONMENT_VARIABLES,
    PROVIDER_LABELS,
    models_for_provider,
    normalize_compatible_base_url,
    normalize_local_base_url,
    provider_requires_api_key,
)
from tools.assistant_settings_store import AssistantSettingsRepository


class _SingleCredentialStore:
    def __init__(self, provider: AssistantProvider, credential: str) -> None:
        self.provider = provider
        self.credential = credential

    def get(self, provider: AssistantProvider) -> Optional[str]:
        if provider == self.provider:
            return self.credential
        return None


CUSTOM_MODEL_DATA = "__custom_model__"


class ProviderConnectionTestWorker(QThread):
    """Make a minimal provider request without blocking the settings window."""

    succeeded = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        settings: AssistantSettings,
        credential: str,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.credential = credential

    def run(self) -> None:
        try:
            from langchain_core.messages import HumanMessage

            model = create_personal_chat_model(
                self.settings,
                _SingleCredentialStore(
                    self.settings.provider,
                    self.credential,
                ),
            )
            model.invoke(
                [HumanMessage(content="Reply with only the word OK.")],
                config={
                    "tags": [
                        "neat-assistant",
                        "connection-test",
                        self.settings.provider.value,
                    ]
                },
            )
        except Exception as exc:
            self.failed.emit(
                describe_provider_error(self.settings.provider, exc)
            )
            return
        self.succeeded.emit()


class LocalModelDiscoveryWorker(QThread):
    """Probe only the known Ollama and LM Studio localhost endpoints."""

    discovered = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            result = discover_local_models()
        except Exception:
            self.failed.emit("Local model detection failed unexpectedly.")
            return
        self.discovered.emit(result)


class AssistantSettingsDialog(QDialog):
    """Configure assistant access without persisting secrets in NEAT settings."""

    settings_saved = pyqtSignal(object)

    def __init__(
        self,
        parent=None,
        *,
        settings_repository: Optional[AssistantSettingsRepository] = None,
        credential_store: Optional[MutableCredentialStore] = None,
        environment_store: Optional[CredentialStore] = None,
        test_worker_class=ProviderConnectionTestWorker,
        local_discovery_worker_class=LocalModelDiscoveryWorker,
        auto_discover_local_models: bool = True,
        shared_service_available: Optional[bool] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("NEAT AI Assistant Settings")
        self.setMinimumWidth(500)
        self.settings_repository = (
            settings_repository or AssistantSettingsRepository()
        )
        self.credential_store = credential_store or KeyringCredentialStore()
        self.environment_store = (
            environment_store or EnvironmentCredentialStore()
        )
        self.test_worker_class = test_worker_class
        self.local_discovery_worker_class = local_discovery_worker_class
        self.auto_discover_local_models = bool(auto_discover_local_models)
        self.test_worker: Optional[QThread] = None
        self.local_discovery_worker: Optional[QThread] = None
        self.local_models_by_url: dict[str, tuple[str, ...]] = {}
        self.initial_settings = self.settings_repository.load()
        if shared_service_available is None:
            from tools.assistant_shared_client import (
                is_shared_service_configured,
            )

            shared_service_available = is_shared_service_configured()
        self.shared_service_available = bool(shared_service_available)

        layout = QVBoxLayout(self)

        access_group = QGroupBox("Access")
        access_layout = QVBoxLayout(access_group)
        self.personal_radio = QRadioButton("Use my own API key")
        access_layout.addWidget(self.personal_radio)
        self.shared_radio = QRadioButton(
            "Use NEAT shared access (20 requests per day)"
        )
        self.shared_radio.setEnabled(self.shared_service_available)
        self.shared_radio.setToolTip(
            "Uses the hosted NEAT service without exposing its OpenAI API key."
        )
        access_layout.addWidget(self.shared_radio)
        self.shared_note = QLabel()
        self.shared_note.setWordWrap(True)
        self.shared_note.setStyleSheet("color: #59636e;")
        access_layout.addWidget(self.shared_note)
        layout.addWidget(access_group)

        self.provider_group = QGroupBox("Personal API")
        form = QFormLayout(self.provider_group)
        self.provider_combo = QComboBox()
        for provider in AssistantProvider:
            self.provider_combo.addItem(PROVIDER_LABELS[provider], provider.value)
        form.addRow("Supplier:", self.provider_combo)

        self.local_engine_combo = QComboBox()
        for engine in LOCAL_MODEL_ENGINES:
            self.local_engine_combo.addItem(engine.label, engine.base_url)
        form.addRow("Local engine:", self.local_engine_combo)

        self.detect_local_button = QPushButton("Detect local models")
        self.detect_local_button.setToolTip(
            "Checks only Ollama and LM Studio on this computer."
        )
        form.addRow("", self.detect_local_button)

        self.model_combo = QComboBox()
        form.addRow("Model:", self.model_combo)

        self.custom_model_edit = QLineEdit()
        self.custom_model_edit.setPlaceholderText(
            "Enter the exact model ID supplied by the provider"
        )
        form.addRow("Custom model ID:", self.custom_model_edit)

        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText(
            "https://provider.example/v1 or http://localhost:11434/v1"
        )
        form.addRow("API base URL:", self.base_url_edit)

        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText(
            "Leave blank to keep the existing secure key"
        )
        form.addRow("API key:", self.key_edit)

        key_buttons = QHBoxLayout()
        self.test_button = QPushButton("Test connection")
        self.test_button.setToolTip(
            "Makes one small billable request to the selected supplier."
        )
        key_buttons.addWidget(self.test_button)
        self.remove_button = QPushButton("Remove saved key")
        key_buttons.addWidget(self.remove_button)
        key_buttons.addStretch(1)
        form.addRow("", key_buttons)

        self.key_status = QLabel()
        self.key_status.setWordWrap(True)
        form.addRow("Key status:", self.key_status)
        layout.addWidget(self.provider_group)

        self.privacy_label = QLabel()
        self.privacy_label.setWordWrap(True)
        self.privacy_label.setStyleSheet("color: #59636e;")
        layout.addWidget(self.privacy_label)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        layout.addWidget(self.button_box)

        self.provider_combo.currentIndexChanged.connect(
            self._on_provider_changed
        )
        self.local_engine_combo.currentIndexChanged.connect(
            self._on_local_engine_changed
        )
        self.detect_local_button.clicked.connect(self.detect_local_models)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.custom_model_edit.textChanged.connect(self._refresh_privacy_text)
        self.base_url_edit.textChanged.connect(self._refresh_privacy_text)
        self.personal_radio.toggled.connect(self._on_access_mode_changed)
        self.shared_radio.toggled.connect(self._on_access_mode_changed)
        self.key_edit.textChanged.connect(self._refresh_key_status)
        self.test_button.clicked.connect(self.test_connection)
        self.remove_button.clicked.connect(self.remove_saved_key)
        self.button_box.accepted.connect(self.save_and_accept)
        self.button_box.rejected.connect(self.reject)

        provider_index = self.provider_combo.findData(
            self.initial_settings.provider.value
        )
        self.provider_combo.setCurrentIndex(max(0, provider_index))
        local_engine_index = self.local_engine_combo.findData(
            self.initial_settings.base_url
        )
        if local_engine_index >= 0:
            self.local_engine_combo.setCurrentIndex(local_engine_index)
        self.base_url_edit.setText(self.initial_settings.base_url)
        self._populate_models(self.initial_settings.model)
        if (
            self.initial_settings.access_mode == AssistantAccessMode.NEAT_SHARED
            and self.shared_service_available
        ):
            self.shared_radio.setChecked(True)
        else:
            self.personal_radio.setChecked(True)
        self._on_access_mode_changed()
        self._refresh_key_status()
        self._refresh_privacy_text()
        if (
            self.auto_discover_local_models
            and self.selected_provider() == AssistantProvider.LOCAL
        ):
            self.detect_local_models()
        # Reserve enough room for the tallest provider form before Windows
        # creates the native dialog. Otherwise, changing between providers can
        # briefly request a height below the form's minimum and Qt logs a
        # QWindowsWindow::setGeometry warning while Windows corrects it.
        self.resize(
            max(self.width(), 560),
            max(self.height(), self.sizeHint().height(), 620),
        )

    def selected_provider(self) -> AssistantProvider:
        return AssistantProvider(self.provider_combo.currentData())

    def selected_settings(self) -> AssistantSettings:
        if self.model_combo.currentData() == CUSTOM_MODEL_DATA:
            model = self.custom_model_edit.text().strip()
        else:
            model = str(self.model_combo.currentData() or "").strip()
        if not model:
            raise ValueError("Enter a model ID.")
        provider = self.selected_provider()
        base_url = ""
        if provider == AssistantProvider.LOCAL:
            base_url = normalize_local_base_url(self.base_url_edit.text())
        elif provider == AssistantProvider.OPENAI_COMPATIBLE:
            base_url = normalize_compatible_base_url(self.base_url_edit.text())
        return AssistantSettings(
            access_mode=(
                AssistantAccessMode.NEAT_SHARED
                if self.shared_radio.isChecked()
                else AssistantAccessMode.PERSONAL_KEY
            ),
            provider=provider,
            model=model,
            base_url=base_url,
        )

    def _populate_models(self, preferred_model: str = "") -> None:
        provider = self.selected_provider()
        self.model_combo.clear()
        if provider == AssistantProvider.LOCAL:
            for model_id in self.local_models_by_url.get(
                self.base_url_edit.text().strip(),
                (),
            ):
                self.model_combo.addItem(f"Local — {model_id}", model_id)
        for descriptor in models_for_provider(provider):
            self.model_combo.addItem(
                f"{descriptor.label} — {descriptor.model_id}",
                descriptor.model_id,
            )
            index = self.model_combo.count() - 1
            self.model_combo.setItemData(index, descriptor.usage_class, 3)
        self.model_combo.addItem("Enter model ID manually…", CUSTOM_MODEL_DATA)
        preferred_index = self.model_combo.findData(preferred_model)
        if preferred_index >= 0:
            self.model_combo.setCurrentIndex(preferred_index)
        elif preferred_model:
            self.model_combo.setCurrentIndex(
                self.model_combo.findData(CUSTOM_MODEL_DATA)
            )
            self.custom_model_edit.setText(preferred_model)
        elif (
            provider == AssistantProvider.LOCAL
            and not self.local_models_by_url.get(
                self.base_url_edit.text().strip(),
                (),
            )
        ) or provider == AssistantProvider.OPENAI_COMPATIBLE:
            self.model_combo.setCurrentIndex(
                self.model_combo.findData(CUSTOM_MODEL_DATA)
            )
        self._on_model_changed()

    def _on_provider_changed(self) -> None:
        self.key_edit.clear()
        if self.selected_provider() == AssistantProvider.LOCAL:
            self.base_url_edit.setText(
                str(self.local_engine_combo.currentData() or "")
            )
        self._populate_models()
        self._refresh_key_status()
        self._on_access_mode_changed()
        if (
            self.auto_discover_local_models
            and self.selected_provider() == AssistantProvider.LOCAL
            and self.isVisible()
        ):
            self.detect_local_models()

    def _on_local_engine_changed(self) -> None:
        if self.selected_provider() != AssistantProvider.LOCAL:
            return
        preferred_model = ""
        if self.model_combo.currentData() == CUSTOM_MODEL_DATA:
            preferred_model = self.custom_model_edit.text().strip()
        else:
            preferred_model = str(self.model_combo.currentData() or "").strip()
        self.base_url_edit.setText(
            str(self.local_engine_combo.currentData() or "")
        )
        self._populate_models(preferred_model)
        self._refresh_key_status()

    def _on_model_changed(self) -> None:
        custom = self.model_combo.currentData() == CUSTOM_MODEL_DATA
        self.custom_model_edit.setVisible(custom)
        label = self.provider_group.layout().labelForField(self.custom_model_edit)
        if label is not None:
            label.setVisible(custom)
        self._refresh_privacy_text()

    def _on_access_mode_changed(self) -> None:
        personal = self.personal_radio.isChecked()
        self.provider_group.setEnabled(personal)
        self.test_button.setEnabled(personal)
        local_provider = self.selected_provider() == AssistantProvider.LOCAL
        custom_endpoint = (
            self.selected_provider() == AssistantProvider.OPENAI_COMPATIBLE
        )
        self.local_engine_combo.setVisible(local_provider)
        local_engine_label = self.provider_group.layout().labelForField(
            self.local_engine_combo
        )
        if local_engine_label is not None:
            local_engine_label.setVisible(local_provider)
        self.detect_local_button.setVisible(local_provider)
        detect_label = self.provider_group.layout().labelForField(
            self.detect_local_button
        )
        if detect_label is not None:
            detect_label.setVisible(local_provider)
        self.base_url_edit.setVisible(local_provider or custom_endpoint)
        self.base_url_edit.setReadOnly(local_provider)
        label = self.provider_group.layout().labelForField(self.base_url_edit)
        if label is not None:
            label.setVisible(local_provider or custom_endpoint)
        self.key_edit.setVisible(not local_provider)
        key_label = self.provider_group.layout().labelForField(self.key_edit)
        if key_label is not None:
            key_label.setVisible(not local_provider)
        self.remove_button.setVisible(not local_provider)
        self.test_button.setToolTip(
            "Makes one local test request."
            if local_provider
            else "Makes one small, potentially billable supplier request."
        )
        if self.shared_service_available:
            self.shared_note.setText(
                "The 20-request allowance is shared globally across all NEAT "
                "users and resets at 00:00 UTC."
            )
        else:
            self.shared_note.setText(
                "Shared access is not configured for this installation yet."
            )
        self._refresh_privacy_text()

    def _available_credential(self) -> Optional[str]:
        pending = self.key_edit.text().strip()
        if pending:
            return pending
        return ChainedCredentialStore(
            [self.credential_store, self.environment_store]
        ).get(self.selected_provider())

    def _refresh_key_status(self) -> None:
        provider = self.selected_provider()
        pending = self.key_edit.text().strip()
        if provider == AssistantProvider.LOCAL:
            text = "No API key is required for the local model server."
            color = "#26734d"
        elif pending:
            text = "A new key is entered. It will be saved securely when you save."
            color = "#26734d"
        else:
            try:
                saved = self.credential_store.get(provider)
            except RuntimeError as exc:
                saved = None
                text = str(exc)
                color = "#b42318"
            else:
                environment = self.environment_store.get(provider)
                if saved:
                    text = "A key is saved securely in the operating system."
                    color = "#26734d"
                elif environment:
                    variable = PROVIDER_ENVIRONMENT_VARIABLES[provider]
                    text = f"A key is available from {variable}."
                    color = "#26734d"
                elif provider == AssistantProvider.OPENAI_COMPATIBLE:
                    try:
                        settings = self.selected_settings()
                        optional = not provider_requires_api_key(settings)
                    except ValueError:
                        optional = False
                    if optional:
                        text = "No key is required for this local endpoint."
                        color = "#26734d"
                    else:
                        text = "No key is configured for this supplier."
                        color = "#9a6700"
                else:
                    text = "No key is configured for this supplier."
                    color = "#9a6700"
        self.key_status.setText(text)
        self.key_status.setStyleSheet(f"color: {color};")

    def _refresh_privacy_text(self) -> None:
        if self.shared_radio.isChecked():
            self.privacy_label.setText(
                "Questions, approved NEAT excerpts and selected safe screen context "
                "will be sent through the hosted NEAT service to OpenAI. The "
                "server's OpenAI API key is never sent to this computer."
            )
            return
        if self.selected_provider() == AssistantProvider.LOCAL:
            self.privacy_label.setText(
                "Questions, approved NEAT excerpts and selected safe screen "
                "context will be sent only to the local model server on this "
                "computer. Raw images and automatically collected file paths "
                "are not sent."
            )
            return
        label = PROVIDER_LABELS[self.selected_provider()]
        destination = label
        if self.selected_provider() == AssistantProvider.OPENAI_COMPATIBLE:
            parsed = urlparse(self.base_url_edit.text().strip())
            destination = parsed.hostname or "the custom endpoint shown above"
        self.privacy_label.setText(
            f"Questions, approved NEAT excerpts and selected safe screen context "
            f"will be sent directly from this computer to {destination}. Raw "
            "images and automatically collected file paths are not sent. Confirm "
            "that you trust this destination before saving."
        )

    def detect_local_models(self) -> None:
        if self.selected_provider() != AssistantProvider.LOCAL:
            return
        if self.local_discovery_worker is not None:
            if self.local_discovery_worker.isRunning():
                return
        self.result_label.setText(
            "Checking Ollama and LM Studio on this computer…"
        )
        self.result_label.setStyleSheet("color: #59636e;")
        self.detect_local_button.setEnabled(False)
        self.local_discovery_worker = self.local_discovery_worker_class(
            parent=self
        )
        self.local_discovery_worker.discovered.connect(
            self._on_local_models_discovered
        )
        self.local_discovery_worker.failed.connect(
            self._on_local_discovery_failed
        )
        self.local_discovery_worker.finished.connect(
            self._on_local_discovery_finished
        )
        self.local_discovery_worker.start()

    def _on_local_models_discovered(self, discovered: object) -> None:
        result = dict(discovered or {})
        engine_by_id = {
            engine.engine_id: engine for engine in LOCAL_MODEL_ENGINES
        }
        self.local_models_by_url = {
            engine_by_id[engine_id].base_url: tuple(models)
            for engine_id, models in result.items()
            if engine_id in engine_by_id
        }
        detected_with_models = [
            engine
            for engine in LOCAL_MODEL_ENGINES
            if self.local_models_by_url.get(engine.base_url)
        ]
        current_url = str(self.local_engine_combo.currentData() or "")
        selected_engine = next(
            (
                engine
                for engine in detected_with_models
                if engine.base_url == current_url
            ),
            detected_with_models[0] if detected_with_models else None,
        )
        if selected_engine is not None:
            index = self.local_engine_combo.findData(selected_engine.base_url)
            self.local_engine_combo.setCurrentIndex(index)
            self.base_url_edit.setText(selected_engine.base_url)
            self._populate_models()
            model_count = len(self.local_models_by_url[selected_engine.base_url])
            self.result_label.setText(
                f"Detected {selected_engine.label} with {model_count} model(s)."
            )
            self.result_label.setStyleSheet("color: #26734d;")
            return
        if result:
            self._populate_models()
            self.result_label.setText(
                "A local model server is running, but it reported no models."
            )
        else:
            self.result_label.setText(
                "No Ollama or LM Studio server was detected. Start a local "
                "server, then select Detect local models."
            )
        self.result_label.setStyleSheet("color: #9a6700;")

    def _on_local_discovery_failed(self, message: str) -> None:
        self.result_label.setText(message)
        self.result_label.setStyleSheet("color: #b42318;")

    def _on_local_discovery_finished(self) -> None:
        self.detect_local_button.setEnabled(True)
        worker = self.local_discovery_worker
        self.local_discovery_worker = None
        if worker is not None:
            worker.deleteLater()

    def save_and_accept(self) -> None:
        pending_key = self.key_edit.text().strip()
        try:
            settings = self.selected_settings()
            if (
                settings.access_mode == AssistantAccessMode.PERSONAL_KEY
                and pending_key
            ):
                self.credential_store.set(settings.provider, pending_key)
            self.settings_repository.save(settings)
        except ValueError as exc:
            self.result_label.setText(str(exc))
            self.result_label.setStyleSheet("color: #b42318;")
            return
        except Exception as exc:
            self.result_label.setText(
                f"The settings could not be saved securely: {type(exc).__name__}."
            )
            self.result_label.setStyleSheet("color: #b42318;")
            return
        self.key_edit.clear()
        self.settings_saved.emit(settings)
        self.accept()

    def remove_saved_key(self) -> None:
        provider = self.selected_provider()
        try:
            self.credential_store.delete(provider)
        except Exception as exc:
            self.result_label.setText(
                f"The saved key could not be removed: {type(exc).__name__}."
            )
            self.result_label.setStyleSheet("color: #b42318;")
            return
        self.key_edit.clear()
        self.result_label.setText(
            "The operating-system copy of this supplier's key was removed."
        )
        self.result_label.setStyleSheet("color: #26734d;")
        self._refresh_key_status()

    def test_connection(self) -> None:
        if self.test_worker is not None and self.test_worker.isRunning():
            return
        try:
            settings = self.selected_settings()
            credential = self._available_credential()
        except (RuntimeError, ValueError) as exc:
            self.result_label.setText(str(exc))
            self.result_label.setStyleSheet("color: #b42318;")
            return
        if not credential and provider_requires_api_key(settings):
            self.result_label.setText("Enter or configure an API key first.")
            self.result_label.setStyleSheet("color: #9a6700;")
            return

        self.result_label.setText(
            "Testing the supplier connection with one small API request..."
        )
        self.result_label.setStyleSheet("color: #59636e;")
        self.test_button.setEnabled(False)
        self.button_box.setEnabled(False)
        self.test_worker = self.test_worker_class(
            settings,
            credential or "",
            parent=self,
        )
        self.test_worker.succeeded.connect(self._on_test_succeeded)
        self.test_worker.failed.connect(self._on_test_failed)
        self.test_worker.finished.connect(self._on_test_finished)
        self.test_worker.start()

    def _on_test_succeeded(self) -> None:
        self.result_label.setText("Connection successful.")
        self.result_label.setStyleSheet("color: #26734d;")

    def _on_test_failed(self, message: str) -> None:
        self.result_label.setText(message)
        self.result_label.setStyleSheet("color: #b42318;")

    def _on_test_finished(self) -> None:
        self.test_button.setEnabled(True)
        self.button_box.setEnabled(True)
        worker = self.test_worker
        self.test_worker = None
        if worker is not None:
            worker.deleteLater()

    def closeEvent(self, event) -> None:
        if self.test_worker is not None and self.test_worker.isRunning():
            self.result_label.setText(
                "Please wait for the connection test to finish before closing."
            )
            self.result_label.setStyleSheet("color: #9a6700;")
            event.ignore()
            return
        if (
            self.local_discovery_worker is not None
            and self.local_discovery_worker.isRunning()
        ):
            self.result_label.setText(
                "Please wait for local model detection to finish before closing."
            )
            self.result_label.setStyleSheet("color: #9a6700;")
            event.ignore()
            return
        super().closeEvent(event)


__all__ = [
    "AssistantSettingsDialog",
    "LocalModelDiscoveryWorker",
    "ProviderConnectionTestWorker",
]

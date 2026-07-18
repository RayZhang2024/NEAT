"""Headless tests for the NEAT AI settings dialog."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from NEAT.ui.assistant_settings_dialog import AssistantSettingsDialog
from tools.assistant_providers import (
    AssistantProvider,
    AssistantSettings,
)


class _MemoryCredentialStore:
    def __init__(self, values=None) -> None:
        self.values = dict(values or {})

    def get(self, provider):
        return self.values.get(provider)

    def set(self, provider, credential) -> None:
        self.values[provider] = credential

    def delete(self, provider) -> None:
        self.values.pop(provider, None)


class _MemorySettingsRepository:
    def __init__(self, settings=None) -> None:
        self.settings = settings or AssistantSettings()
        self.saved = None

    def load(self):
        return self.settings

    def save(self, settings):
        self.settings = settings
        self.saved = settings


class AssistantSettingsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_shared_access_displays_global_daily_request_limit(self) -> None:
        dialog = AssistantSettingsDialog(
            settings_repository=_MemorySettingsRepository(),
            credential_store=_MemoryCredentialStore(),
            environment_store=_MemoryCredentialStore(),
            shared_service_available=False,
        )
        self.assertTrue(dialog.shared_radio.isVisible() or not dialog.isVisible())
        self.assertFalse(dialog.shared_radio.isEnabled())
        self.assertIn("20 requests per day", dialog.shared_radio.text().lower())
        dialog.close()

    def test_dialog_reserves_space_for_dynamic_provider_fields(self) -> None:
        dialog = AssistantSettingsDialog(
            settings_repository=_MemorySettingsRepository(),
            credential_store=_MemoryCredentialStore(),
            environment_store=_MemoryCredentialStore(),
            auto_discover_local_models=False,
            shared_service_available=False,
        )
        self.assertGreaterEqual(dialog.width(), 560)
        self.assertGreaterEqual(dialog.height(), 620)
        dialog.close()

    def test_supplier_change_populates_curated_models(self) -> None:
        dialog = AssistantSettingsDialog(
            settings_repository=_MemorySettingsRepository(),
            credential_store=_MemoryCredentialStore(),
            environment_store=_MemoryCredentialStore(),
            shared_service_available=False,
        )
        index = dialog.provider_combo.findData(AssistantProvider.GOOGLE.value)
        dialog.provider_combo.setCurrentIndex(index)
        model_ids = [
            dialog.model_combo.itemData(item)
            for item in range(dialog.model_combo.count())
        ]
        self.assertIn("gemini-3.5-flash", model_ids)
        dialog.close()

    def test_new_compatible_suppliers_are_available(self) -> None:
        dialog = AssistantSettingsDialog(
            settings_repository=_MemorySettingsRepository(),
            credential_store=_MemoryCredentialStore(),
            environment_store=_MemoryCredentialStore(),
            shared_service_available=False,
        )
        providers = {
            dialog.provider_combo.itemData(item)
            for item in range(dialog.provider_combo.count())
        }
        self.assertIn(AssistantProvider.DEEPSEEK.value, providers)
        self.assertIn(AssistantProvider.MOONSHOT.value, providers)
        self.assertIn(AssistantProvider.LOCAL.value, providers)
        self.assertIn(AssistantProvider.OPENAI_COMPATIBLE.value, providers)
        dialog.close()

    def test_local_provider_detects_models_without_api_key(self) -> None:
        dialog = AssistantSettingsDialog(
            settings_repository=_MemorySettingsRepository(),
            credential_store=_MemoryCredentialStore(),
            environment_store=_MemoryCredentialStore(),
            auto_discover_local_models=False,
            shared_service_available=False,
        )
        provider_index = dialog.provider_combo.findData(
            AssistantProvider.LOCAL.value
        )
        dialog.provider_combo.setCurrentIndex(provider_index)
        dialog._on_local_models_discovered(
            {"ollama": ("qwen3:8b", "gemma3:4b")}
        )

        model_ids = {
            dialog.model_combo.itemData(item)
            for item in range(dialog.model_combo.count())
        }
        self.assertIn("qwen3:8b", model_ids)
        self.assertIn("gemma3:4b", model_ids)
        self.assertEqual(
            dialog.selected_settings().provider,
            AssistantProvider.LOCAL,
        )
        self.assertEqual(
            dialog.selected_settings().base_url,
            "http://localhost:11434/v1",
        )
        self.assertFalse(dialog.key_edit.isVisible())
        self.assertIn("only to the local model server", dialog.privacy_label.text())
        dialog.close()

    def test_manual_model_id_can_be_selected_for_known_provider(self) -> None:
        dialog = AssistantSettingsDialog(
            settings_repository=_MemorySettingsRepository(),
            credential_store=_MemoryCredentialStore(),
            environment_store=_MemoryCredentialStore(),
            shared_service_available=False,
        )
        custom_index = dialog.model_combo.findData("__custom_model__")
        dialog.model_combo.setCurrentIndex(custom_index)
        dialog.custom_model_edit.setText("future-openai-model")

        self.assertEqual(dialog.selected_settings().model, "future-openai-model")
        dialog.close()

    def test_custom_endpoint_and_key_are_saved_separately(self) -> None:
        credentials = _MemoryCredentialStore()
        repository = _MemorySettingsRepository()
        dialog = AssistantSettingsDialog(
            settings_repository=repository,
            credential_store=credentials,
            environment_store=_MemoryCredentialStore(),
            shared_service_available=False,
        )
        provider_index = dialog.provider_combo.findData(
            AssistantProvider.OPENAI_COMPATIBLE.value
        )
        dialog.provider_combo.setCurrentIndex(provider_index)
        dialog.custom_model_edit.setText("institutional-model")
        dialog.base_url_edit.setText("https://models.example.org/v1")
        dialog.key_edit.setText("separate-compatible-key")
        dialog.save_and_accept()

        self.assertEqual(
            repository.saved.provider,
            AssistantProvider.OPENAI_COMPATIBLE,
        )
        self.assertEqual(
            repository.saved.base_url,
            "https://models.example.org/v1",
        )
        self.assertEqual(
            credentials.get(AssistantProvider.OPENAI_COMPATIBLE),
            "separate-compatible-key",
        )
        self.assertFalse(hasattr(repository.saved, "api_key"))
        dialog.close()

    def test_save_places_key_only_in_credential_store(self) -> None:
        credentials = _MemoryCredentialStore()
        repository = _MemorySettingsRepository()
        dialog = AssistantSettingsDialog(
            settings_repository=repository,
            credential_store=credentials,
            environment_store=_MemoryCredentialStore(),
            shared_service_available=False,
        )
        index = dialog.provider_combo.findData(AssistantProvider.ANTHROPIC.value)
        dialog.provider_combo.setCurrentIndex(index)
        dialog.key_edit.setText("private-anthropic-key")
        dialog.save_and_accept()

        self.assertEqual(repository.saved.provider, AssistantProvider.ANTHROPIC)
        self.assertEqual(
            credentials.get(AssistantProvider.ANTHROPIC),
            "private-anthropic-key",
        )
        self.assertFalse(hasattr(repository.saved, "api_key"))
        dialog.close()

    def test_configured_shared_service_can_be_selected(self) -> None:
        repository = _MemorySettingsRepository()
        dialog = AssistantSettingsDialog(
            settings_repository=repository,
            credential_store=_MemoryCredentialStore(),
            environment_store=_MemoryCredentialStore(),
            shared_service_available=True,
        )
        self.assertTrue(dialog.shared_radio.isEnabled())
        dialog.shared_radio.setChecked(True)
        dialog.save_and_accept()
        self.assertEqual(
            repository.saved.access_mode.value,
            "neat_shared",
        )
        dialog.close()


if __name__ == "__main__":
    unittest.main()

"""Tests for non-secret assistant settings persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.assistant_providers import (
    AssistantAccessMode,
    AssistantProvider,
    AssistantSettings,
)
from tools.assistant_settings_store import AssistantSettingsRepository


class AssistantSettingsRepositoryTests(unittest.TestCase):
    def test_round_trip_contains_no_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assistant_settings.json"
            repository = AssistantSettingsRepository(path)
            expected = AssistantSettings(
                access_mode=AssistantAccessMode.PERSONAL_KEY,
                provider=AssistantProvider.ANTHROPIC,
                model="claude-sonnet-5",
            )
            repository.save(expected)

            self.assertEqual(repository.load(), expected)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("api_key", payload)
            self.assertNotIn("credential", payload)

    def test_invalid_file_falls_back_to_safe_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assistant_settings.json"
            path.write_text("{invalid", encoding="utf-8")
            settings = AssistantSettingsRepository(path).load()
            self.assertEqual(settings.provider, AssistantProvider.OPENAI)

    def test_custom_endpoint_round_trip_excludes_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assistant_settings.json"
            repository = AssistantSettingsRepository(path)
            expected = AssistantSettings(
                provider=AssistantProvider.OPENAI_COMPATIBLE,
                model="institutional-model",
                base_url="https://models.example.org/v1",
            )
            repository.save(expected)

            self.assertEqual(repository.load(), expected)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["base_url"], "https://models.example.org/v1")
            self.assertNotIn("api_key", payload)

    def test_schema_one_settings_still_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assistant_settings.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "access_mode": "personal_key",
                        "provider": "anthropic",
                        "model": "claude-test",
                    }
                ),
                encoding="utf-8",
            )
            settings = AssistantSettingsRepository(path).load()
            self.assertEqual(settings.provider, AssistantProvider.ANTHROPIC)
            self.assertEqual(settings.base_url, "")

    def test_local_model_round_trip_excludes_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assistant_settings.json"
            repository = AssistantSettingsRepository(path)
            expected = AssistantSettings(
                provider=AssistantProvider.LOCAL,
                model="qwen3:8b",
                base_url="http://localhost:11434/v1",
            )
            repository.save(expected)

            self.assertEqual(repository.load(), expected)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("api_key", payload)
            self.assertNotIn("credential", payload)


if __name__ == "__main__":
    unittest.main()

"""Tests for provider-neutral assistant settings and credentials."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tools.assistant_model_factory import create_personal_chat_model
from tools.assistant_providers import (
    AssistantAccessMode,
    AssistantProvider,
    AssistantSettings,
    ChainedCredentialStore,
    EnvironmentCredentialStore,
    MissingProviderCredential,
    UnsupportedAssistantProvider,
    load_assistant_settings,
    models_for_provider,
    normalize_compatible_base_url,
    normalize_local_base_url,
    provider_requires_api_key,
    require_provider_credential,
)


class _MemoryCredentialStore:
    def __init__(self, values=None) -> None:
        self.values = dict(values or {})

    def get(self, provider):
        return self.values.get(provider)


class AssistantProviderTests(unittest.TestCase):
    def test_settings_never_contain_a_credential(self) -> None:
        settings = AssistantSettings()
        self.assertFalse(hasattr(settings, "api_key"))
        self.assertFalse(hasattr(settings, "credential"))

    def test_current_environment_defaults_remain_compatible(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "secret",
                "NEAT_ASSISTANT_MODEL": "test-model",
            },
            clear=True,
        ):
            settings = load_assistant_settings()
            credential = EnvironmentCredentialStore().get(settings.provider)
        self.assertEqual(settings.provider, AssistantProvider.OPENAI)
        self.assertEqual(settings.model, "test-model")
        self.assertEqual(credential, "secret")

    def test_unknown_provider_is_rejected_safely(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown assistant provider"):
            load_assistant_settings(provider="unknown")

    def test_missing_key_names_selected_provider_without_a_secret(self) -> None:
        settings = AssistantSettings(provider=AssistantProvider.ANTHROPIC)
        with self.assertRaises(MissingProviderCredential) as raised:
            require_provider_credential(settings, _MemoryCredentialStore())
        self.assertIn("Anthropic", str(raised.exception))
        self.assertIn("ANTHROPIC_API_KEY", str(raised.exception))

    def test_chained_store_uses_first_available_credential(self) -> None:
        store = ChainedCredentialStore(
            [
                _MemoryCredentialStore(),
                _MemoryCredentialStore({AssistantProvider.GOOGLE: "second"}),
            ]
        )
        self.assertEqual(store.get(AssistantProvider.GOOGLE), "second")

    def test_catalogue_only_advertises_implemented_models(self) -> None:
        openai_models = models_for_provider(AssistantProvider.OPENAI)
        anthropic_models = models_for_provider(AssistantProvider.ANTHROPIC)
        google_models = models_for_provider(AssistantProvider.GOOGLE)
        deepseek_models = models_for_provider(AssistantProvider.DEEPSEEK)
        moonshot_models = models_for_provider(AssistantProvider.MOONSHOT)
        self.assertTrue(openai_models)
        self.assertTrue(anthropic_models)
        self.assertTrue(google_models)
        self.assertTrue(deepseek_models)
        self.assertTrue(moonshot_models)
        self.assertTrue(
            all(
                item.implemented
                for item in (
                    openai_models
                    + anthropic_models
                    + google_models
                    + deepseek_models
                    + moonshot_models
                )
            )
        )

    def test_custom_remote_endpoint_requires_https(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use HTTPS"):
            normalize_compatible_base_url("http://models.example.org/v1")

    def test_custom_endpoint_rejects_embedded_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain credentials"):
            normalize_compatible_base_url(
                "https://user:secret@models.example.org/v1"
            )

    def test_local_provider_rejects_remote_endpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use localhost"):
            normalize_local_base_url("https://models.example.org/v1")

    def test_local_provider_does_not_require_api_key(self) -> None:
        settings = AssistantSettings(
            provider=AssistantProvider.LOCAL,
            model="qwen3:8b",
            base_url="http://localhost:11434/v1",
        )
        self.assertFalse(provider_requires_api_key(settings))

    @patch("tools.assistant_openai.ChatOpenAI")
    def test_factory_passes_personal_key_directly_to_openai(self, model_class) -> None:
        settings = AssistantSettings(
            access_mode=AssistantAccessMode.PERSONAL_KEY,
            provider=AssistantProvider.OPENAI,
            model="test-model",
        )
        create_personal_chat_model(
            settings,
            _MemoryCredentialStore({AssistantProvider.OPENAI: "personal-key"}),
        )
        self.assertEqual(model_class.call_args.kwargs["api_key"], "personal-key")
        self.assertEqual(model_class.call_args.kwargs["model"], "test-model")

    @patch("tools.assistant_anthropic.ChatAnthropic")
    def test_factory_creates_anthropic_model(self, model_class) -> None:
        settings = AssistantSettings(
            provider=AssistantProvider.ANTHROPIC,
            model="claude-test",
        )
        create_personal_chat_model(
            settings,
            _MemoryCredentialStore(
                {AssistantProvider.ANTHROPIC: "personal-key"}
            ),
        )
        self.assertEqual(model_class.call_args.kwargs["api_key"], "personal-key")
        self.assertEqual(model_class.call_args.kwargs["model_name"], "claude-test")

    @patch("tools.assistant_google.ChatGoogleGenerativeAI")
    def test_factory_creates_google_model(self, model_class) -> None:
        settings = AssistantSettings(
            provider=AssistantProvider.GOOGLE,
            model="gemini-test",
        )
        create_personal_chat_model(
            settings,
            _MemoryCredentialStore(
                {AssistantProvider.GOOGLE: "personal-key"}
            ),
        )
        self.assertEqual(model_class.call_args.kwargs["api_key"], "personal-key")
        self.assertEqual(model_class.call_args.kwargs["model"], "gemini-test")

    @patch("tools.assistant_openai_compatible.ChatOpenAI")
    def test_factory_creates_deepseek_compatible_model(self, model_class) -> None:
        settings = AssistantSettings(
            provider=AssistantProvider.DEEPSEEK,
            model="deepseek-v4-flash",
        )
        create_personal_chat_model(
            settings,
            _MemoryCredentialStore(
                {AssistantProvider.DEEPSEEK: "deepseek-personal-key"}
            ),
        )
        arguments = model_class.call_args.kwargs
        self.assertEqual(arguments["model"], "deepseek-v4-flash")
        self.assertEqual(arguments["base_url"], "https://api.deepseek.com")
        self.assertFalse(arguments["use_responses_api"])

    @patch("tools.assistant_openai_compatible.ChatOpenAI")
    def test_factory_creates_kimi_compatible_model(self, model_class) -> None:
        settings = AssistantSettings(
            provider=AssistantProvider.MOONSHOT,
            model="kimi-k3",
        )
        create_personal_chat_model(
            settings,
            _MemoryCredentialStore(
                {AssistantProvider.MOONSHOT: "moonshot-personal-key"}
            ),
        )
        arguments = model_class.call_args.kwargs
        self.assertEqual(arguments["model"], "kimi-k3")
        self.assertEqual(arguments["base_url"], "https://api.moonshot.ai/v1")

    @patch("tools.assistant_openai_compatible.ChatOpenAI")
    def test_local_compatible_endpoint_can_omit_key(self, model_class) -> None:
        settings = AssistantSettings(
            provider=AssistantProvider.OPENAI_COMPATIBLE,
            model="qwen3:8b",
            base_url="http://localhost:11434/v1",
        )
        create_personal_chat_model(settings, _MemoryCredentialStore())
        arguments = model_class.call_args.kwargs
        self.assertEqual(arguments["model"], "qwen3:8b")
        self.assertEqual(arguments["base_url"], "http://localhost:11434/v1")
        self.assertEqual(arguments["api_key"], "local-endpoint-no-key")

    @patch("tools.assistant_openai_compatible.ChatOpenAI")
    def test_factory_creates_first_class_local_model(self, model_class) -> None:
        settings = AssistantSettings(
            provider=AssistantProvider.LOCAL,
            model="qwen3:8b",
            base_url="http://localhost:11434/v1",
        )
        create_personal_chat_model(settings, _MemoryCredentialStore())
        arguments = model_class.call_args.kwargs
        self.assertEqual(arguments["model"], "qwen3:8b")
        self.assertEqual(arguments["base_url"], "http://localhost:11434/v1")
        self.assertEqual(arguments["api_key"], "local-endpoint-no-key")


if __name__ == "__main__":
    unittest.main()

"""Tests for secure OpenAI configuration without making API requests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from openai import RateLimitError

from tools.assistant_openai import (
    DEFAULT_OPENAI_MODEL,
    MissingOpenAIAPIKey,
    create_openai_chat_model,
    describe_openai_error,
    load_openai_settings,
)


class AssistantOpenAIConfigurationTests(unittest.TestCase):
    def test_missing_key_is_rejected_before_model_creation(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingOpenAIAPIKey):
                load_openai_settings()

    def test_default_model_is_selected_without_storing_key(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            settings = load_openai_settings()
        self.assertEqual(settings.model, DEFAULT_OPENAI_MODEL)
        self.assertFalse(hasattr(settings, "api_key"))

    def test_model_environment_override(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "NEAT_ASSISTANT_MODEL": "test-model",
            },
            clear=True,
        ):
            settings = load_openai_settings()
        self.assertEqual(settings.model, "test-model")

    @patch("tools.assistant_openai.ChatOpenAI")
    def test_chat_model_uses_responses_api_and_low_reasoning(self, model_class) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            settings = load_openai_settings()
            create_openai_chat_model(settings)
        model_class.assert_called_once_with(
            model=DEFAULT_OPENAI_MODEL,
            use_responses_api=True,
            reasoning_effort="low",
            timeout=60.0,
            max_retries=2,
            store=False,
        )

    def test_insufficient_quota_error_has_actionable_safe_message(self) -> None:
        response = type(
            "Response",
            (),
            {"request": None, "status_code": 429, "headers": {}},
        )()
        error = RateLimitError(
            "sensitive provider message",
            response=response,
            body={"code": "insufficient_quota"},
        )
        message = describe_openai_error(error)
        self.assertIn("insufficient_quota", message)
        self.assertIn("billing quota", message)
        self.assertNotIn("sensitive provider message", message)

    def test_generic_rate_limit_recommends_waiting(self) -> None:
        response = type(
            "Response",
            (),
            {"request": None, "status_code": 429, "headers": {}},
        )()
        error = RateLimitError(
            "rate limit",
            response=response,
            body={"code": "rate_limit_exceeded"},
        )
        message = describe_openai_error(error)
        self.assertIn("rate_limit_exceeded", message)
        self.assertIn("Wait briefly", message)


if __name__ == "__main__":
    unittest.main()

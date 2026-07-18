"""Tests for safe Ollama and LM Studio model discovery."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.assistant_local_models import (
    LocalModelDiscoveryError,
    discover_local_models,
    list_local_models,
)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class AssistantLocalModelTests(unittest.TestCase):
    @patch("tools.assistant_local_models.urlopen")
    def test_model_list_is_sanitised_and_sorted(self, urlopen) -> None:
        urlopen.return_value = _Response(
            b'{"data": ['
            b'{"id": "qwen3:8b"}, {"id": " Gemma3:4b "}, '
            b'{"id": "qwen3:8b"}, {"missing": true}]}'
        )

        models = list_local_models("http://localhost:11434/v1")

        self.assertEqual(models, ("Gemma3:4b", "qwen3:8b"))
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/v1/models")

    @patch("tools.assistant_local_models.urlopen")
    def test_invalid_model_list_has_safe_error(self, urlopen) -> None:
        urlopen.return_value = _Response(b"not-json")
        with self.assertRaisesRegex(
            LocalModelDiscoveryError,
            "invalid model list",
        ):
            list_local_models("http://localhost:1234/v1")

    @patch("tools.assistant_local_models.list_local_models")
    def test_discovery_checks_both_known_local_engines(self, list_models) -> None:
        def result_for_url(base_url, *, timeout_seconds):
            if "11434" in base_url:
                return ("qwen3:8b",)
            raise LocalModelDiscoveryError("not reachable")

        list_models.side_effect = result_for_url
        discovered = discover_local_models()

        self.assertEqual(discovered, {"ollama": ("qwen3:8b",)})
        self.assertEqual(list_models.call_count, 2)


if __name__ == "__main__":
    unittest.main()

"""Tests for the desktop hosted-assistant client."""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from tools.assistant_shared_client import (
    SharedAssistantClient,
    SharedServiceError,
    SharedServiceSettings,
    is_shared_service_configured,
    load_shared_service_settings,
)


class _Response:
    def __init__(self, payload) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class SharedAssistantClientTests(unittest.TestCase):
    def test_persisted_windows_settings_are_used_when_process_is_stale(self) -> None:
        values = {
            "NEAT_SHARED_SERVICE_URL": "https://neat.example.org",
            "NEAT_SHARED_ACCESS_TOKEN": "persisted-limited-token",
        }
        with patch.dict("os.environ", {}, clear=True), patch(
            "tools.assistant_shared_client._persisted_windows_user_setting",
            side_effect=values.get,
        ):
            self.assertTrue(is_shared_service_configured())
            settings = load_shared_service_settings()

        self.assertEqual(settings.service_url, "https://neat.example.org")
        self.assertEqual(settings.access_token, "persisted-limited-token")

    def test_remote_service_requires_https(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use HTTPS"):
            SharedServiceSettings(
                service_url="http://example.org",
                access_token="token",
            )

    @patch("tools.assistant_shared_client.urlopen")
    def test_answer_and_quota_metadata_are_normalized(self, open_url) -> None:
        open_url.return_value = _Response(
            {
                "answer": "Use the fitting tab.",
                "route": "how_to",
                "route_confidence": 0.9,
                "requires_human_review": False,
                "citations": [
                    {
                        "number": 1,
                        "source_id": "faq#fit",
                        "filename": "faq.md",
                        "heading": "Fit",
                        "heading_path": "FAQ > Fit",
                        "anchor": "fit",
                    }
                ],
                "used": 4,
                "remaining": 16,
                "daily_limit": 20,
                "reset_at_utc": "2026-07-19T00:00:00+00:00",
            }
        )
        settings = SharedServiceSettings(
            service_url="https://neat.example.org",
            access_token="limited-service-token",
        )
        result = SharedAssistantClient(settings).ask("How do I fit?")

        self.assertEqual(result.answer, "Use the fitting tab.")
        self.assertEqual(result.shared_remaining, 16)
        request = open_url.call_args.args[0]
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer limited-service-token",
        )
        self.assertNotIn("limited-service-token", repr(settings))

    @patch("tools.assistant_shared_client.urlopen")
    def test_daily_limit_has_actionable_message(self, open_url) -> None:
        payload = json.dumps(
            {
                "detail": {
                    "code": "daily_limit_reached",
                    "reset_at_utc": "2026-07-19T00:00:00+00:00",
                }
            }
        ).encode("utf-8")
        open_url.side_effect = HTTPError(
            "https://neat.example.org/v1/assistant/ask",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(payload),
        )
        client = SharedAssistantClient(
            SharedServiceSettings(
                service_url="https://neat.example.org",
                access_token="token",
            )
        )
        with self.assertRaises(SharedServiceError) as raised:
            client.ask("Question")
        self.assertIn("20 shared NEAT requests", str(raised.exception))
        self.assertIn("use your own API key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

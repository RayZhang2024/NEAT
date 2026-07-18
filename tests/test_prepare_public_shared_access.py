"""Tests for release-time public shared-access configuration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.prepare_public_shared_access import prepare_public_shared_access


class PreparePublicSharedAccessTests(unittest.TestCase):
    def test_valid_public_configuration_is_written_without_openai_key(self) -> None:
        environment = {
            "NEAT_PUBLIC_SHARED_SERVICE_URL": "https://neat.example.org",
            "NEAT_PUBLIC_SHARED_ACCESS_TOKEN": "public-client-token",
            "OPENAI_API_KEY": "must-not-be-bundled",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            environment,
            clear=True,
        ):
            output = prepare_public_shared_access(
                Path(directory) / "shared_access.json"
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["service_url"], "https://neat.example.org")
        self.assertEqual(payload["access_token"], "public-client-token")
        self.assertNotIn("OPENAI", payload)
        self.assertNotIn("must-not-be-bundled", json.dumps(payload))

    def test_missing_public_configuration_stops_the_release_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {},
            clear=True,
        ), patch(
            "tools.prepare_public_shared_access._persisted_windows_user_setting",
            return_value="",
        ):
            with self.assertRaisesRegex(RuntimeError, "Public shared access"):
                prepare_public_shared_access(
                    Path(directory) / "shared_access.json"
                )


if __name__ == "__main__":
    unittest.main()

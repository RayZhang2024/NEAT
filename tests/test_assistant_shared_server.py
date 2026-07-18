"""Endpoint tests for the restricted hosted NEAT assistant."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from tools.assistant_answering import GroundedAnswer, SourceCitation
from tools.assistant_shared_quota import SQLiteDailyQuota
from tools.assistant_shared_server import SharedServerConfig, create_app


class _FakeBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def ask(self, question, *, context, history):
        self.calls += 1
        if self.fail:
            raise RuntimeError("sensitive upstream detail")
        return GroundedAnswer(
            answer=f"Grounded answer for: {question}",
            route="how_to",
            route_confidence=0.95,
            requires_human_review=False,
            citations=[
                SourceCitation(
                    number=1,
                    source_id="faq#test",
                    filename="faq.md",
                    heading="Test",
                    heading_path="FAQ > Test",
                    anchor="test",
                )
            ],
        )


class SharedAssistantServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        directory = Path(self.temporary_directory.name)
        self.config = SharedServerConfig(
            openai_api_key="server-only-openai-key",
            service_token="desktop-access-token",
            model="test-model",
            daily_limit=2,
            database_path=directory / "quota.sqlite3",
            index_directory=directory / "chroma",
            knowledge_directory=directory / "knowledge",
        )
        self.backend = _FakeBackend()
        self.quota = SQLiteDailyQuota(
            self.config.database_path,
            daily_limit=2,
        )
        self.client = TestClient(
            create_app(
                self.config,
                backend=self.backend,
                quota=self.quota,
            )
        )
        self.headers = {
            "Authorization": "Bearer desktop-access-token",
        }

    def tearDown(self) -> None:
        self.client.close()
        self.temporary_directory.cleanup()

    @staticmethod
    def payload(request_id: str, question: str = "How do I use NEAT?"):
        return {
            "request_id": request_id,
            "question": question,
            "context": {"current_module": "Fitting"},
            "history": [],
        }

    def test_requires_service_authorization(self) -> None:
        response = self.client.post(
            "/v1/assistant/ask",
            json=self.payload("21c0d4c8-8283-4cb7-939d-aa6bf599d2ee"),
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.quota.status().used, 0)

    def test_global_daily_limit_rejects_request_after_limit(self) -> None:
        request_ids = [
            "21c0d4c8-8283-4cb7-939d-aa6bf599d2e1",
            "21c0d4c8-8283-4cb7-939d-aa6bf599d2e2",
            "21c0d4c8-8283-4cb7-939d-aa6bf599d2e3",
        ]
        first = self.client.post(
            "/v1/assistant/ask",
            json=self.payload(request_ids[0]),
            headers=self.headers,
        )
        second = self.client.post(
            "/v1/assistant/ask",
            json=self.payload(request_ids[1]),
            headers=self.headers,
        )
        rejected = self.client.post(
            "/v1/assistant/ask",
            json=self.payload(request_ids[2]),
            headers=self.headers,
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["remaining"], 1)
        self.assertEqual(second.json()["remaining"], 0)
        self.assertEqual(rejected.status_code, 429)
        self.assertEqual(
            rejected.json()["detail"]["code"],
            "daily_limit_reached",
        )
        self.assertEqual(self.backend.calls, 2)

    def test_duplicate_completed_request_is_idempotent(self) -> None:
        request_id = "21c0d4c8-8283-4cb7-939d-aa6bf599d2e4"
        first = self.client.post(
            "/v1/assistant/ask",
            json=self.payload(request_id),
            headers=self.headers,
        )
        repeated = self.client.post(
            "/v1/assistant/ask",
            json=self.payload(request_id),
            headers=self.headers,
        )
        self.assertEqual(first.json(), repeated.json())
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(self.quota.status().used, 1)

    def test_invalid_question_is_rejected_before_quota(self) -> None:
        response = self.client.post(
            "/v1/assistant/ask",
            json=self.payload(
                "21c0d4c8-8283-4cb7-939d-aa6bf599d2e5",
                question=" ",
            ),
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.quota.status().used, 0)

    def test_provider_failure_does_not_expose_raw_error(self) -> None:
        failing_backend = _FakeBackend(fail=True)
        failing_client = TestClient(
            create_app(
                self.config,
                backend=failing_backend,
                quota=self.quota,
            )
        )
        response = failing_client.post(
            "/v1/assistant/ask",
            json=self.payload("21c0d4c8-8283-4cb7-939d-aa6bf599d2e6"),
            headers=self.headers,
        )
        failing_client.close()
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("sensitive upstream detail", response.text)
        self.assertEqual(self.quota.status().used, 1)


class SharedServerCredentialTests(unittest.TestCase):
    def test_environment_credentials_take_precedence(self) -> None:
        environment = {
            "OPENAI_API_KEY": "environment-openai-key",
            "NEAT_SHARED_SERVICE_TOKEN": "environment-service-token",
        }
        with patch.dict("os.environ", environment, clear=True), patch(
            "tools.assistant_shared_server.load_server_credential"
        ) as load_credential:
            config = SharedServerConfig.from_environment()

        self.assertEqual(config.openai_api_key, "environment-openai-key")
        self.assertEqual(config.service_token, "environment-service-token")
        load_credential.assert_not_called()

    def test_windows_credential_manager_is_fallback(self) -> None:
        credentials = {
            "openai_api_key": "saved-openai-key",
            "service_token": "saved-service-token",
        }
        with patch.dict("os.environ", {}, clear=True), patch(
            "tools.assistant_shared_server.load_server_credential",
            side_effect=credentials.get,
        ):
            config = SharedServerConfig.from_environment()

        self.assertEqual(config.openai_api_key, "saved-openai-key")
        self.assertEqual(config.service_token, "saved-service-token")

    def test_main_runs_uvicorn_with_local_defaults(self) -> None:
        with patch(
            "tools.assistant_shared_server.create_app",
            return_value="test-app",
        ), patch("uvicorn.run") as run:
            from tools.assistant_shared_server import main

            main()

        run.assert_called_once_with(
            "test-app",
            host="127.0.0.1",
            port=8765,
        )


if __name__ == "__main__":
    unittest.main()

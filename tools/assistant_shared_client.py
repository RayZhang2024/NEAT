"""Restricted desktop client for the hosted NEAT assistant service."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from tools.assistant_answering import (
    ConversationTurn,
    GroundedAnswer,
    SourceCitation,
)


SHARED_SERVICE_URL_ENV = "NEAT_SHARED_SERVICE_URL"
SHARED_ACCESS_TOKEN_ENV = "NEAT_SHARED_ACCESS_TOKEN"


class MissingSharedServiceConfiguration(RuntimeError):
    """Raised when the desktop has no hosted service URL or access token."""


class SharedServiceError(RuntimeError):
    """Safe, user-facing hosted service failure."""


def _persisted_windows_user_setting(name: str) -> str:
    """Read a user environment setting even if this process inherited stale values."""

    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except (ImportError, FileNotFoundError, OSError):
        return ""
    return str(value or "").strip()


def _shared_setting(name: str) -> str:
    """Prefer the process environment, then the persisted Windows user value."""

    return (
        os.environ.get(name, "").strip()
        or _persisted_windows_user_setting(name)
    )


@dataclass(frozen=True)
class SharedServiceSettings:
    service_url: str
    access_token: str = field(repr=False)
    timeout_seconds: float = 90.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.service_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("The NEAT shared service URL is invalid.")
        if parsed.scheme != "https" and parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError(
                "A remote NEAT shared service must use HTTPS."
            )
        if not self.access_token.strip():
            raise ValueError("The NEAT shared service access token is empty.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


def is_shared_service_configured() -> bool:
    return bool(
        _shared_setting(SHARED_SERVICE_URL_ENV)
        and _shared_setting(SHARED_ACCESS_TOKEN_ENV)
    )


def load_shared_service_settings() -> SharedServiceSettings:
    service_url = _shared_setting(SHARED_SERVICE_URL_ENV)
    access_token = _shared_setting(SHARED_ACCESS_TOKEN_ENV)
    if not service_url or not access_token:
        raise MissingSharedServiceConfiguration(
            "NEAT shared access is not configured for this installation."
        )
    return SharedServiceSettings(
        service_url=service_url,
        access_token=access_token,
    )


def _safe_context(context: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): value
        for key, value in context.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


class SharedAssistantClient:
    """Call the NEAT-only hosted endpoint without exposing its OpenAI key."""

    def __init__(self, settings: SharedServiceSettings) -> None:
        self.settings = settings

    def ask(
        self,
        question: str,
        *,
        context: Optional[Mapping[str, object]] = None,
        history: Optional[Sequence[ConversationTurn]] = None,
    ) -> GroundedAnswer:
        payload = {
            "request_id": str(uuid4()),
            "question": str(question or "").strip(),
            "context": _safe_context(context or {}),
            "history": [
                {
                    "role": str(turn.role),
                    "content": str(turn.content),
                }
                for turn in list(history or [])[-12:]
            ],
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        endpoint = urljoin(
            self.settings.service_url.rstrip("/") + "/",
            "v1/assistant/ask",
        )
        request = Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.access_token}",
                "Content-Type": "application/json",
                "User-Agent": "NEAT-AI-Assistant",
            },
        )
        try:
            with urlopen(
                request,
                timeout=self.settings.timeout_seconds,
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            self._raise_http_error(exc)
        except (URLError, TimeoutError):
            raise SharedServiceError(
                "NEAT could not reach the shared assistant service. "
                "Check the network connection and retry. After the host laptop "
                "restarts, allow up to one minute for the service to become ready."
            ) from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SharedServiceError(
                "The shared assistant returned an invalid response."
            ) from None

        citations = [
            SourceCitation(
                number=int(item["number"]),
                source_id=str(item["source_id"]),
                filename=str(item["filename"]),
                heading=str(item["heading"]),
                heading_path=str(item["heading_path"]),
                anchor=str(item["anchor"]),
            )
            for item in response_payload.get("citations", [])
        ]
        return GroundedAnswer(
            answer=str(response_payload["answer"]),
            route=str(response_payload["route"]),
            route_confidence=float(response_payload["route_confidence"]),
            requires_human_review=bool(
                response_payload["requires_human_review"]
            ),
            citations=citations,
            shared_remaining=int(response_payload["remaining"]),
            shared_daily_limit=int(response_payload["daily_limit"]),
            shared_reset_at_utc=str(response_payload["reset_at_utc"]),
        )

    @staticmethod
    def _raise_http_error(exc: HTTPError) -> None:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            detail = payload.get("detail", payload)
        except Exception:
            detail = {}
        if not isinstance(detail, Mapping):
            detail = {}
        code = str(detail.get("code", "shared_service_error"))
        if exc.code == 429 and code == "daily_limit_reached":
            reset = str(detail.get("reset_at_utc", "the next UTC day"))
            raise SharedServiceError(
                "The 20 shared NEAT requests have been used for today. "
                f"The allowance resets at {reset}. You can use your own API key "
                "in AI settings."
            ) from None
        if exc.code == 401:
            raise SharedServiceError(
                "This NEAT installation is not authorised for shared access."
            ) from None
        message = str(detail.get("message", "")).strip()
        if message:
            raise SharedServiceError(message) from None
        raise SharedServiceError(
            f"The shared assistant service returned HTTP {exc.code}."
        ) from None


__all__ = [
    "MissingSharedServiceConfiguration",
    "SHARED_ACCESS_TOKEN_ENV",
    "SHARED_SERVICE_URL_ENV",
    "SharedAssistantClient",
    "SharedServiceError",
    "SharedServiceSettings",
    "is_shared_service_configured",
    "load_shared_service_settings",
]

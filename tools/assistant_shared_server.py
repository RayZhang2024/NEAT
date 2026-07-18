"""Restricted hosted API for globally shared NEAT assistant access."""

from __future__ import annotations

import os
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Optional, Protocol
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from tools.assistant_answering import ConversationTurn, GroundedAnswer
from tools.assistant_openai import (
    DEFAULT_OPENAI_MODEL,
    OpenAISettings,
    create_openai_chat_model,
    describe_openai_error,
)
from tools.assistant_retrieval import load_knowledge_base
from tools.assistant_semantic_retrieval import ChromaSemanticRetriever
from tools.assistant_service import NEATAssistantService
from tools.assistant_shared_quota import (
    DEFAULT_DAILY_LIMIT,
    DailyQuotaExceeded,
    DuplicateRequestInProgress,
    SQLiteDailyQuota,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_STATE_DIRECTORY = PROJECT_ROOT / ".assistant_server"
SERVER_KEYRING_SERVICE_NAME = "NEAT Shared Assistant Server"
SERVER_OPENAI_KEY_NAME = "openai_api_key"
SERVER_ACCESS_TOKEN_NAME = "service_token"


def load_server_credential(name: str) -> str:
    """Read a laptop-hosted server secret from the OS credential manager."""

    try:
        import keyring
    except ImportError:
        return ""
    try:
        value = keyring.get_password(SERVER_KEYRING_SERVICE_NAME, name)
    except Exception:
        return ""
    return str(value or "").strip()


@dataclass(frozen=True)
class SharedServerConfig:
    """Hosted configuration; secret fields must never be returned or logged."""

    openai_api_key: str = field(repr=False)
    service_token: str = field(repr=False)
    model: str = DEFAULT_OPENAI_MODEL
    daily_limit: int = DEFAULT_DAILY_LIMIT
    database_path: Path = DEFAULT_SERVER_STATE_DIRECTORY / "quota.sqlite3"
    index_directory: Path = DEFAULT_SERVER_STATE_DIRECTORY / "chroma"
    knowledge_directory: Path = PROJECT_ROOT / "docs" / "assistant"

    @classmethod
    def from_environment(cls) -> "SharedServerConfig":
        openai_key = (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or load_server_credential(SERVER_OPENAI_KEY_NAME)
        )
        service_token = os.environ.get(
            "NEAT_SHARED_SERVICE_TOKEN",
            "",
        ).strip() or load_server_credential(SERVER_ACCESS_TOKEN_NAME)
        if not openai_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be configured on the hosted server or saved "
                "with python -m tools.configure_assistant_laptop."
            )
        if not service_token:
            raise RuntimeError(
                "NEAT_SHARED_SERVICE_TOKEN must be configured on the hosted server "
                "or saved with python -m tools.configure_assistant_laptop."
            )
        return cls(
            openai_api_key=openai_key,
            service_token=service_token,
            model=os.environ.get(
                "NEAT_SHARED_OPENAI_MODEL",
                DEFAULT_OPENAI_MODEL,
            ).strip()
            or DEFAULT_OPENAI_MODEL,
            daily_limit=int(
                os.environ.get(
                    "NEAT_SHARED_DAILY_LIMIT",
                    str(DEFAULT_DAILY_LIMIT),
                )
            ),
            database_path=Path(
                os.environ.get(
                    "NEAT_SHARED_DATABASE_PATH",
                    str(DEFAULT_SERVER_STATE_DIRECTORY / "quota.sqlite3"),
                )
            ),
        )


class ConversationPayload(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class SharedAssistantRequest(BaseModel):
    request_id: UUID
    question: str = Field(min_length=1, max_length=2000)
    context: dict[str, object] = Field(default_factory=dict)
    history: list[ConversationPayload] = Field(
        default_factory=list,
        max_length=12,
    )

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be blank")
        return value

    @field_validator("context")
    @classmethod
    def validate_context(cls, value: Mapping[str, object]) -> dict[str, object]:
        if len(value) > 32:
            raise ValueError("too many context fields")
        safe: dict[str, object] = {}
        for key, item in value.items():
            key = str(key)
            if len(key) > 80:
                raise ValueError("context key is too long")
            if item is None or isinstance(item, (str, int, float, bool)):
                if isinstance(item, str) and len(item) > 500:
                    raise ValueError("context value is too long")
                safe[key] = item
            else:
                raise ValueError("context values must be scalar")
        return safe


class CitationPayload(BaseModel):
    number: int
    source_id: str
    filename: str
    heading: str
    heading_path: str
    anchor: str


class SharedAssistantResponse(BaseModel):
    answer: str
    route: str
    route_confidence: float
    requires_human_review: bool
    citations: list[CitationPayload]
    used: int
    remaining: int
    daily_limit: int
    reset_at_utc: str


class SharedUsageResponse(BaseModel):
    used: int
    remaining: int
    daily_limit: int
    reset_at_utc: str


class HostedAssistantBackend(Protocol):
    def ask(
        self,
        question: str,
        *,
        context: Mapping[str, object],
        history: list[ConversationTurn],
    ) -> GroundedAnswer:
        ...


class OpenAIHostedAssistant:
    """Lazily initialize server-side retrieval and the fixed OpenAI model."""

    def __init__(self, config: SharedServerConfig) -> None:
        self.config = config
        self._service: Optional[NEATAssistantService] = None
        self._initialization_lock = threading.Lock()

    def _get_service(self) -> NEATAssistantService:
        if self._service is not None:
            return self._service
        with self._initialization_lock:
            if self._service is None:
                sections = load_knowledge_base(self.config.knowledge_directory)
                retriever = ChromaSemanticRetriever(
                    sections,
                    persist_directory=self.config.index_directory,
                )
                model = create_openai_chat_model(
                    OpenAISettings(
                        model=self.config.model,
                        max_retries=1,
                    ),
                    api_key=self.config.openai_api_key,
                )
                self._service = NEATAssistantService(retriever, model)
        return self._service

    def ask(
        self,
        question: str,
        *,
        context: Mapping[str, object],
        history: list[ConversationTurn],
    ) -> GroundedAnswer:
        return self._get_service().ask(
            question,
            context=context,
            history=history,
            source_limit=3,
            config={"tags": ["neat-shared-service", "openai"]},
        )


def _response_payload(
    result: GroundedAnswer,
    *,
    used: int,
    remaining: int,
    daily_limit: int,
    reset_at_utc: str,
) -> dict[str, object]:
    return {
        "answer": result.answer,
        "route": result.route,
        "route_confidence": result.route_confidence,
        "requires_human_review": result.requires_human_review,
        "citations": [
            {
                "number": citation.number,
                "source_id": citation.source_id,
                "filename": citation.filename,
                "heading": citation.heading,
                "heading_path": citation.heading_path,
                "anchor": citation.anchor,
            }
            for citation in result.citations
        ],
        "used": used,
        "remaining": remaining,
        "daily_limit": daily_limit,
        "reset_at_utc": reset_at_utc,
    }


def create_app(
    config: Optional[SharedServerConfig] = None,
    *,
    backend: Optional[HostedAssistantBackend] = None,
    quota: Optional[SQLiteDailyQuota] = None,
) -> FastAPI:
    """Create the hosted service with injectable dependencies for tests."""

    config = config or SharedServerConfig.from_environment()
    quota = quota or SQLiteDailyQuota(
        config.database_path,
        daily_limit=config.daily_limit,
    )
    backend = backend or OpenAIHostedAssistant(config)
    app = FastAPI(
        title="NEAT Shared Assistant",
        version="1.0",
        docs_url=None,
        redoc_url=None,
    )

    def authorize(authorization: Optional[str] = Header(default=None)) -> None:
        expected = f"Bearer {config.service_token}"
        if authorization is None or not secrets.compare_digest(
            authorization,
            expected,
        ):
            raise HTTPException(
                status_code=401,
                detail={"code": "unauthorized"},
            )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "daily_limit": config.daily_limit,
            "model": config.model,
        }

    @app.get(
        "/v1/assistant/usage",
        response_model=SharedUsageResponse,
        dependencies=[Depends(authorize)],
    )
    def usage() -> dict[str, object]:
        status = quota.status()
        return {
            "used": status.used,
            "remaining": status.remaining,
            "daily_limit": status.limit,
            "reset_at_utc": status.reset_at_utc,
        }

    @app.post(
        "/v1/assistant/ask",
        response_model=SharedAssistantResponse,
        dependencies=[Depends(authorize)],
    )
    def ask(request_payload: SharedAssistantRequest) -> dict[str, object]:
        request_id = str(request_payload.request_id)
        try:
            reservation = quota.reserve(request_id)
        except DailyQuotaExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "daily_limit_reached",
                    "reset_at_utc": exc.reset_at_utc,
                },
            ) from None
        except DuplicateRequestInProgress:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "duplicate_request",
                    "message": "This shared request has already been submitted.",
                },
            ) from None

        if reservation.cached_response is not None:
            return reservation.cached_response

        history = [
            ConversationTurn(role=turn.role, content=turn.content)
            for turn in request_payload.history
        ]
        try:
            result = backend.ask(
                request_payload.question,
                context=request_payload.context,
                history=history,
            )
        except Exception as exc:
            quota.fail(request_id, error_code=type(exc).__name__)
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "model_request_failed",
                    "message": describe_openai_error(exc),
                },
            ) from None

        response = _response_payload(
            result,
            used=reservation.status.used,
            remaining=reservation.status.remaining,
            daily_limit=reservation.status.limit,
            reset_at_utc=reservation.status.reset_at_utc,
        )
        quota.complete(request_id, response)
        return response

    return app


def main() -> None:
    """Run one local/service instance; production hosting supplies HTTPS."""

    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("NEAT_SHARED_HOST", "127.0.0.1"),
        port=int(os.environ.get("NEAT_SHARED_PORT", "8765")),
    )


__all__ = [
    "ConversationPayload",
    "HostedAssistantBackend",
    "OpenAIHostedAssistant",
    "SharedAssistantRequest",
    "SharedAssistantResponse",
    "SharedServerConfig",
    "SharedUsageResponse",
    "SERVER_ACCESS_TOKEN_NAME",
    "SERVER_KEYRING_SERVICE_NAME",
    "SERVER_OPENAI_KEY_NAME",
    "create_app",
    "load_server_credential",
    "main",
]


if __name__ == "__main__":
    main()

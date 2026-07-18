"""Secure OpenAI model configuration for the NEAT assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from langchain_openai import ChatOpenAI
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        NotFoundError,
        PermissionDeniedError,
        RateLimitError,
    )
except ImportError as exc:  # pragma: no cover - optional assistant installation
    raise ImportError(
        "OpenAI support requires the optional assistant dependencies. "
        "Install them with: python -m pip install -e \".[assistant]\""
    ) from exc


DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
API_KEY_ENVIRONMENT_VARIABLE = "OPENAI_API_KEY"
MODEL_ENVIRONMENT_VARIABLE = "NEAT_ASSISTANT_MODEL"


class MissingOpenAIAPIKey(RuntimeError):
    """Raised before a model is created when no API key is configured."""


def _openai_error_code(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if code:
        return str(code)

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        nested_error = body.get("error")
        if isinstance(nested_error, dict) and nested_error.get("code"):
            return str(nested_error["code"])
        if body.get("code"):
            return str(body["code"])
    return "unknown"


def describe_openai_error(exc: Exception) -> str:
    """Return safe, actionable diagnostics without echoing credentials."""

    request_id = getattr(exc, "request_id", None)
    suffix = f"\nOpenAI request ID: {request_id}" if request_id else ""

    if isinstance(exc, RateLimitError):
        code = _openai_error_code(exc)
        if code == "insufficient_quota":
            detail = (
                "OpenAI accepted the API key, but this API project has no "
                "available billing quota. Add API credit or increase the project "
                "monthly budget, then allow a few minutes for the balance to update."
            )
        else:
            detail = (
                "OpenAI returned a request/token rate limit. Wait briefly and retry, "
                "then check the project's Limits page if it continues."
            )
        return f"OpenAI rate-limit code: {code}\n{detail}{suffix}"

    if isinstance(exc, AuthenticationError):
        return (
            "OpenAI rejected the API key. Confirm that OPENAI_API_KEY contains an "
            f"active API Platform key for the intended project.{suffix}"
        )
    if isinstance(exc, PermissionDeniedError):
        return (
            "The API project or key is not permitted to use this model. Select a "
            f"model available to that project or update its permissions.{suffix}"
        )
    if isinstance(exc, NotFoundError):
        return (
            "The selected model was not found or is not available to this project. "
            f"Try an accessible model using NEAT_ASSISTANT_MODEL.{suffix}"
        )
    if isinstance(exc, APITimeoutError):
        return f"The OpenAI request timed out. Check the network and retry.{suffix}"
    if isinstance(exc, APIConnectionError):
        return (
            "NEAT could not connect to the OpenAI API. Check the internet connection, "
            f"proxy and firewall settings.{suffix}"
        )
    return f"Unexpected OpenAI error type: {type(exc).__name__}.{suffix}"


@dataclass(frozen=True)
class OpenAISettings:
    """Non-secret settings used to create the NEAT support model."""

    model: str
    timeout_seconds: float = 60.0
    max_retries: int = 2
    reasoning_effort: str = "low"


def load_openai_settings(
    *,
    model: Optional[str] = None,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
) -> OpenAISettings:
    """Load model settings while requiring the key to remain in the environment."""

    if not os.environ.get(API_KEY_ENVIRONMENT_VARIABLE, "").strip():
        raise MissingOpenAIAPIKey(
            "OPENAI_API_KEY is not set. Configure it as an environment variable; "
            "do not save the key in the NEAT repository."
        )

    selected_model = (
        str(model or "").strip()
        or os.environ.get(MODEL_ENVIRONMENT_VARIABLE, "").strip()
        or DEFAULT_OPENAI_MODEL
    )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")

    return OpenAISettings(
        model=selected_model,
        timeout_seconds=float(timeout_seconds),
        max_retries=int(max_retries),
    )


def create_openai_chat_model(
    settings: OpenAISettings,
    *,
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    """Create the LangChain model using an explicit or environment credential."""

    arguments = dict(
        model=settings.model,
        use_responses_api=True,
        reasoning_effort=settings.reasoning_effort,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        store=False,
    )
    if api_key:
        arguments["api_key"] = api_key
    return ChatOpenAI(**arguments)

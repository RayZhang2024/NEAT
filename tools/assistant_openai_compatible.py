"""Adapter for vetted and user-configured OpenAI-compatible chat endpoints."""

from __future__ import annotations

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
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "OpenAI-compatible model support requires the optional assistant "
        "dependencies. Install them with: python -m pip install -e \".[assistant]\""
    ) from exc


def create_compatible_chat_model(
    *,
    model: str,
    api_key: str,
    base_url: str,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
) -> ChatOpenAI:
    """Use Chat Completions without sending OpenAI-only request parameters."""

    model = str(model or "").strip()
    if not model:
        raise ValueError("Enter a model ID for this supplier.")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        use_responses_api=False,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def describe_compatible_error(label: str, exc: Exception) -> str:
    """Return safe diagnostics without echoing keys, prompts, or endpoint URLs."""

    if isinstance(exc, AuthenticationError):
        return f"{label} rejected the API key. Check that it is active."
    if isinstance(exc, PermissionDeniedError):
        return f"This API key is not permitted to use the selected {label} model."
    if isinstance(exc, NotFoundError):
        return f"The selected {label} model was not found or is unavailable."
    if isinstance(exc, RateLimitError):
        return (
            f"{label} returned a rate or account limit. Wait briefly, then "
            "check the account's usage and billing."
        )
    if isinstance(exc, APITimeoutError):
        return f"The {label} request timed out. Check the network and retry."
    if isinstance(exc, APIConnectionError):
        return f"NEAT could not connect to {label}. Check the endpoint and network."
    return f"Unexpected {label} error type: {type(exc).__name__}."


__all__ = ["create_compatible_chat_model", "describe_compatible_error"]

"""Anthropic chat-model adapter and safe diagnostics for NEAT."""

from __future__ import annotations

try:
    import anthropic
    from langchain_anthropic import ChatAnthropic
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "Anthropic support requires the optional assistant dependencies. "
        "Install them with: python -m pip install -e \".[assistant]\""
    ) from exc


def create_anthropic_chat_model(
    *,
    model: str,
    api_key: str,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
) -> ChatAnthropic:
    """Create a direct Anthropic model using the user's personal key."""

    return ChatAnthropic(
        model_name=model,
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def describe_anthropic_error(exc: Exception) -> str:
    """Return credential-safe Anthropic diagnostics."""

    if isinstance(exc, anthropic.AuthenticationError):
        return "Anthropic rejected the API key. Check that it is active."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "This Anthropic key is not permitted to use the selected model."
    if isinstance(exc, anthropic.NotFoundError):
        return "The selected Anthropic model was not found or is unavailable."
    if isinstance(exc, anthropic.RateLimitError):
        return (
            "Anthropic returned a rate or account limit. Wait briefly, then check "
            "the account's usage and billing limits."
        )
    if isinstance(exc, anthropic.APITimeoutError):
        return "The Anthropic request timed out. Check the network and retry."
    if isinstance(exc, anthropic.APIConnectionError):
        return "NEAT could not connect to Anthropic. Check the network and firewall."
    return f"Unexpected Anthropic error type: {type(exc).__name__}."


__all__ = ["create_anthropic_chat_model", "describe_anthropic_error"]

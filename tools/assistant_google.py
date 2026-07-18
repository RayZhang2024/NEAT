"""Google Gemini chat-model adapter and safe diagnostics for NEAT."""

from __future__ import annotations

try:
    from google.genai import errors as google_errors
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "Google Gemini support requires the optional assistant dependencies. "
        "Install them with: python -m pip install -e \".[assistant]\""
    ) from exc


def create_google_chat_model(
    *,
    model: str,
    api_key: str,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
) -> ChatGoogleGenerativeAI:
    """Create a direct Gemini model using the user's personal key."""

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        request_timeout=timeout_seconds,
        retries=max_retries,
        temperature=0.2,
    )


def describe_google_error(exc: Exception) -> str:
    """Return credential-safe Gemini diagnostics."""

    if isinstance(exc, google_errors.APIError):
        status_code = int(getattr(exc, "code", 0) or 0)
        if status_code == 401:
            return "Google Gemini rejected the API key. Check that it is active."
        if status_code == 403:
            return "This Gemini key is not permitted to use the selected model."
        if status_code == 404:
            return "The selected Gemini model was not found or is unavailable."
        if status_code == 429:
            return (
                "Google Gemini returned a rate or account limit. Wait briefly, "
                "then check the project's quota and billing."
            )
        if status_code >= 500:
            return "Google Gemini is temporarily unavailable. Retry shortly."
        return f"Google Gemini returned API error {status_code or 'unknown'}."
    return f"Unexpected Google Gemini error type: {type(exc).__name__}."


__all__ = ["create_google_chat_model", "describe_google_error"]

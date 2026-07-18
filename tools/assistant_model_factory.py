"""Create NEAT chat models without coupling the GUI to one supplier."""

from __future__ import annotations

from tools.assistant_answering import ChatModel
from tools.assistant_providers import (
    AssistantAccessMode,
    AssistantProvider,
    AssistantSettings,
    CredentialStore,
    PROVIDER_LABELS,
    UnsupportedAssistantProvider,
    provider_base_url,
    provider_requires_api_key,
    require_provider_credential,
)


def create_personal_chat_model(
    settings: AssistantSettings,
    credential_store: CredentialStore,
) -> ChatModel:
    """Create a direct-to-provider model using the user's own credential."""

    if settings.access_mode != AssistantAccessMode.PERSONAL_KEY:
        raise ValueError("A personal chat model requires personal-key access mode")

    credential = credential_store.get(settings.provider)
    if not credential:
        if provider_requires_api_key(settings):
            credential = require_provider_credential(settings, credential_store)
        else:
            credential = "local-endpoint-no-key"
    if settings.provider == AssistantProvider.OPENAI:
        from tools.assistant_openai import OpenAISettings, create_openai_chat_model

        return create_openai_chat_model(
            OpenAISettings(model=settings.model),
            api_key=credential,
        )
    if settings.provider == AssistantProvider.ANTHROPIC:
        from tools.assistant_anthropic import create_anthropic_chat_model

        return create_anthropic_chat_model(
            model=settings.model,
            api_key=credential,
        )
    if settings.provider == AssistantProvider.GOOGLE:
        from tools.assistant_google import create_google_chat_model

        return create_google_chat_model(
            model=settings.model,
            api_key=credential,
        )
    if settings.provider in {
        AssistantProvider.DEEPSEEK,
        AssistantProvider.MOONSHOT,
        AssistantProvider.LOCAL,
        AssistantProvider.OPENAI_COMPATIBLE,
    }:
        from tools.assistant_openai_compatible import create_compatible_chat_model

        return create_compatible_chat_model(
            model=settings.model,
            api_key=credential,
            base_url=provider_base_url(settings),
        )

    label = PROVIDER_LABELS[settings.provider]
    raise UnsupportedAssistantProvider(
        f"{label} is recognised by NEAT, but its model adapter is not installed yet."
    )


def describe_provider_error(
    provider: AssistantProvider,
    exc: Exception,
) -> str:
    """Return provider-specific safe diagnostics without leaking credentials."""

    if provider == AssistantProvider.OPENAI:
        from tools.assistant_openai import describe_openai_error

        return describe_openai_error(exc)
    if provider == AssistantProvider.ANTHROPIC:
        from tools.assistant_anthropic import describe_anthropic_error

        return describe_anthropic_error(exc)
    if provider == AssistantProvider.GOOGLE:
        from tools.assistant_google import describe_google_error

        return describe_google_error(exc)
    if provider in {
        AssistantProvider.DEEPSEEK,
        AssistantProvider.MOONSHOT,
        AssistantProvider.LOCAL,
        AssistantProvider.OPENAI_COMPATIBLE,
    }:
        from tools.assistant_openai_compatible import describe_compatible_error

        return describe_compatible_error(PROVIDER_LABELS[provider], exc)
    label = PROVIDER_LABELS[provider]
    return f"Unexpected {label} error type: {type(exc).__name__}."


__all__ = ["create_personal_chat_model", "describe_provider_error"]

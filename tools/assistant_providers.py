"""Provider-neutral settings and credential storage for the NEAT assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Sequence
from urllib.parse import urlparse


class AssistantAccessMode(str, Enum):
    """How an assistant request is funded and transported."""

    PERSONAL_KEY = "personal_key"
    NEAT_SHARED = "neat_shared"


class AssistantProvider(str, Enum):
    """Model suppliers supported by the provider-neutral configuration."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"
    LOCAL = "local"
    OPENAI_COMPATIBLE = "openai_compatible"


PROVIDER_LABELS = {
    AssistantProvider.OPENAI: "OpenAI",
    AssistantProvider.ANTHROPIC: "Anthropic",
    AssistantProvider.GOOGLE: "Google Gemini",
    AssistantProvider.DEEPSEEK: "DeepSeek",
    AssistantProvider.MOONSHOT: "Kimi / Moonshot",
    AssistantProvider.LOCAL: "Local model (Ollama / LM Studio)",
    AssistantProvider.OPENAI_COMPATIBLE: "OpenAI-compatible endpoint",
}

PROVIDER_ENVIRONMENT_VARIABLES = {
    AssistantProvider.OPENAI: "OPENAI_API_KEY",
    AssistantProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    AssistantProvider.GOOGLE: "GEMINI_API_KEY",
    AssistantProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
    AssistantProvider.MOONSHOT: "MOONSHOT_API_KEY",
    AssistantProvider.LOCAL: "LOCAL_LLM_API_KEY",
    AssistantProvider.OPENAI_COMPATIBLE: "OPENAI_COMPATIBLE_API_KEY",
}

PROVIDER_BASE_URLS = {
    AssistantProvider.DEEPSEEK: "https://api.deepseek.com",
    AssistantProvider.MOONSHOT: "https://api.moonshot.ai/v1",
}

DEFAULT_PROVIDER = AssistantProvider.OPENAI
DEFAULT_MODEL = "gpt-5.6-luna"
PROVIDER_ENVIRONMENT_VARIABLE = "NEAT_ASSISTANT_PROVIDER"
MODEL_ENVIRONMENT_VARIABLE = "NEAT_ASSISTANT_MODEL"
KEYRING_SERVICE_NAME = "NEAT AI Assistant"


@dataclass(frozen=True)
class ModelDescriptor:
    """One model that NEAT has explicitly classified for support use."""

    provider: AssistantProvider
    model_id: str
    label: str
    usage_class: str
    implemented: bool


MODEL_CATALOGUE: tuple[ModelDescriptor, ...] = (
    ModelDescriptor(
        AssistantProvider.OPENAI,
        "gpt-5.6-luna",
        "Economy",
        "Routine NEAT help and interface questions",
        True,
    ),
    ModelDescriptor(
        AssistantProvider.OPENAI,
        "gpt-5.6-terra",
        "Balanced",
        "More involved technical explanations",
        True,
    ),
    ModelDescriptor(
        AssistantProvider.OPENAI,
        "gpt-5.6-sol",
        "Advanced",
        "Difficult technical synthesis",
        True,
    ),
    ModelDescriptor(
        AssistantProvider.ANTHROPIC,
        "claude-haiku-4-5-20251001",
        "Economy",
        "Routine NEAT help with low latency",
        True,
    ),
    ModelDescriptor(
        AssistantProvider.ANTHROPIC,
        "claude-sonnet-5",
        "Balanced",
        "Technical explanations with stronger reasoning",
        True,
    ),
    ModelDescriptor(
        AssistantProvider.ANTHROPIC,
        "claude-opus-4-8",
        "Advanced",
        "Difficult technical synthesis",
        True,
    ),
    ModelDescriptor(
        AssistantProvider.GOOGLE,
        "gemini-3.1-flash-lite",
        "Economy",
        "Fast, cost-sensitive NEAT help",
        True,
    ),
    ModelDescriptor(
        AssistantProvider.GOOGLE,
        "gemini-3.5-flash",
        "Balanced",
        "Technical explanations with stronger reasoning",
        True,
    ),
    ModelDescriptor(
        AssistantProvider.DEEPSEEK,
        "deepseek-v4-flash",
        "Economy",
        "Routine NEAT help with lower cost and latency",
        True,
    ),
    ModelDescriptor(
        AssistantProvider.DEEPSEEK,
        "deepseek-v4-pro",
        "Advanced",
        "More difficult technical synthesis and reasoning",
        True,
    ),
    ModelDescriptor(
        AssistantProvider.MOONSHOT,
        "kimi-k2.6",
        "Balanced",
        "Technical NEAT explanations with long context",
        True,
    ),
    ModelDescriptor(
        AssistantProvider.MOONSHOT,
        "kimi-k3",
        "Advanced",
        "Difficult technical synthesis with stronger reasoning",
        True,
    ),
)


@dataclass(frozen=True)
class AssistantSettings:
    """Non-secret assistant selection persisted by NEAT."""

    access_mode: AssistantAccessMode = AssistantAccessMode.PERSONAL_KEY
    provider: AssistantProvider = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    base_url: str = ""


class MissingProviderCredential(RuntimeError):
    """Raised when the selected provider has no configured API credential."""


class UnsupportedAssistantProvider(RuntimeError):
    """Raised when a configured supplier does not yet have a NEAT adapter."""


class CredentialStore(Protocol):
    """Secret-storage contract; assistant settings never contain API keys."""

    def get(self, provider: AssistantProvider) -> Optional[str]:
        """Return a provider credential without logging or persisting it elsewhere."""


class MutableCredentialStore(CredentialStore, Protocol):
    """Credential store used by the future settings dialog."""

    def set(self, provider: AssistantProvider, credential: str) -> None:
        """Securely save a provider credential."""

    def delete(self, provider: AssistantProvider) -> None:
        """Remove a saved provider credential."""


class EnvironmentCredentialStore:
    """Read provider keys from environment variables for existing installations."""

    def get(self, provider: AssistantProvider) -> Optional[str]:
        variable = PROVIDER_ENVIRONMENT_VARIABLES[provider]
        value = os.environ.get(variable, "").strip()
        return value or None


class KeyringCredentialStore:
    """Store personal API keys in the operating system credential manager."""

    def __init__(self, *, service_name: str = KEYRING_SERVICE_NAME) -> None:
        self.service_name = service_name

    @staticmethod
    def _keyring():
        try:
            import keyring
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Secure API-key storage requires the optional 'keyring' package."
            ) from exc
        return keyring

    def get(self, provider: AssistantProvider) -> Optional[str]:
        value = self._keyring().get_password(self.service_name, provider.value)
        value = str(value or "").strip()
        return value or None

    def set(self, provider: AssistantProvider, credential: str) -> None:
        value = str(credential or "").strip()
        if not value:
            raise ValueError("An API key cannot be empty")
        self._keyring().set_password(self.service_name, provider.value, value)

    def delete(self, provider: AssistantProvider) -> None:
        keyring = self._keyring()
        try:
            keyring.delete_password(self.service_name, provider.value)
        except keyring.errors.PasswordDeleteError:
            return


class ChainedCredentialStore:
    """Use secure saved credentials first, then environment variables."""

    def __init__(self, stores: Sequence[CredentialStore]) -> None:
        self.stores = tuple(stores)

    def get(self, provider: AssistantProvider) -> Optional[str]:
        last_error: Optional[RuntimeError] = None
        for store in self.stores:
            try:
                credential = store.get(provider)
            except RuntimeError as exc:
                last_error = exc
                continue
            if credential:
                return credential
        if last_error is not None:
            raise last_error
        return None


def models_for_provider(
    provider: AssistantProvider,
    *,
    implemented_only: bool = True,
) -> tuple[ModelDescriptor, ...]:
    """Return the curated models NEAT may show for one provider."""

    return tuple(
        descriptor
        for descriptor in MODEL_CATALOGUE
        if descriptor.provider == provider
        and (descriptor.implemented or not implemented_only)
    )


def default_model_for_provider(provider: AssistantProvider) -> str:
    """Return the preferred economical model for a supplier."""

    models = models_for_provider(provider)
    if not models:
        raise UnsupportedAssistantProvider(
            f"No supported models are registered for {PROVIDER_LABELS[provider]}."
        )
    return models[0].model_id


def normalize_compatible_base_url(value: str) -> str:
    """Validate a user-selected OpenAI-compatible endpoint without credentials."""

    base_url = str(value or "").strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a complete OpenAI-compatible API base URL.")
    if parsed.username or parsed.password:
        raise ValueError("The API base URL must not contain credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("The API base URL must not contain a query or fragment.")
    if parsed.scheme != "https" and parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("A remote custom model endpoint must use HTTPS.")
    return base_url


def normalize_local_base_url(value: str) -> str:
    """Accept only a loopback OpenAI-compatible local model server."""

    base_url = normalize_compatible_base_url(value)
    parsed = urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("A Local model endpoint must use localhost.")
    return base_url


def provider_base_url(settings: AssistantSettings) -> str:
    """Return a validated endpoint for compatible providers."""

    if settings.provider in PROVIDER_BASE_URLS:
        return PROVIDER_BASE_URLS[settings.provider]
    if settings.provider == AssistantProvider.LOCAL:
        return normalize_local_base_url(settings.base_url)
    if settings.provider == AssistantProvider.OPENAI_COMPATIBLE:
        return normalize_compatible_base_url(settings.base_url)
    return ""


def provider_requires_api_key(settings: AssistantSettings) -> bool:
    """Local compatible servers may operate without an API credential."""

    if settings.provider == AssistantProvider.LOCAL:
        provider_base_url(settings)
        return False
    if settings.provider != AssistantProvider.OPENAI_COMPATIBLE:
        return True
    parsed = urlparse(provider_base_url(settings))
    return parsed.hostname not in {"127.0.0.1", "localhost", "::1"}


def load_assistant_settings(
    *,
    access_mode: AssistantAccessMode = AssistantAccessMode.PERSONAL_KEY,
    provider: Optional[str | AssistantProvider] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> AssistantSettings:
    """Load non-secret defaults while retaining current environment overrides."""

    provider_value = (
        provider
        or os.environ.get(PROVIDER_ENVIRONMENT_VARIABLE, "").strip()
        or DEFAULT_PROVIDER
    )
    try:
        selected_provider = AssistantProvider(provider_value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in AssistantProvider)
        raise ValueError(
            f"Unknown assistant provider '{provider_value}'. Supported: {supported}."
        ) from exc

    selected_model = (
        str(model or "").strip()
        or os.environ.get(MODEL_ENVIRONMENT_VARIABLE, "").strip()
        or (
            default_model_for_provider(selected_provider)
            if selected_provider
            not in {
                AssistantProvider.LOCAL,
                AssistantProvider.OPENAI_COMPATIBLE,
            }
            else ""
        )
    )
    selected_base_url = ""
    if selected_provider in {
        AssistantProvider.LOCAL,
        AssistantProvider.OPENAI_COMPATIBLE,
    }:
        raw_base_url = (
            str(base_url or "").strip()
            or os.environ.get("NEAT_ASSISTANT_BASE_URL", "").strip()
        )
        selected_base_url = (
            normalize_local_base_url(raw_base_url)
            if selected_provider == AssistantProvider.LOCAL
            else normalize_compatible_base_url(raw_base_url)
        )
    return AssistantSettings(
        access_mode=AssistantAccessMode(access_mode),
        provider=selected_provider,
        model=selected_model,
        base_url=selected_base_url,
    )


def require_provider_credential(
    settings: AssistantSettings,
    credential_store: CredentialStore,
) -> str:
    """Return the selected key or raise a provider-specific safe message."""

    credential = credential_store.get(settings.provider)
    if credential:
        return credential
    variable = PROVIDER_ENVIRONMENT_VARIABLES[settings.provider]
    label = PROVIDER_LABELS[settings.provider]
    raise MissingProviderCredential(
        f"No {label} API key is configured. Add one in NEAT AI settings or "
        f"configure {variable} for this NEAT process."
    )


__all__ = [
    "AssistantAccessMode",
    "AssistantProvider",
    "AssistantSettings",
    "ChainedCredentialStore",
    "CredentialStore",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "EnvironmentCredentialStore",
    "KEYRING_SERVICE_NAME",
    "KeyringCredentialStore",
    "MODEL_CATALOGUE",
    "MissingProviderCredential",
    "ModelDescriptor",
    "MutableCredentialStore",
    "PROVIDER_ENVIRONMENT_VARIABLES",
    "PROVIDER_BASE_URLS",
    "PROVIDER_LABELS",
    "UnsupportedAssistantProvider",
    "default_model_for_provider",
    "load_assistant_settings",
    "models_for_provider",
    "normalize_compatible_base_url",
    "normalize_local_base_url",
    "provider_base_url",
    "provider_requires_api_key",
    "require_provider_credential",
]

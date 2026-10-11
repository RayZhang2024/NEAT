"""M4 structured planning adapters, settings, and Agent-only credentials.

This module does not depend on LangChain or import the RAG Assistant provider
configuration. The only network transport is a single bounded OpenAI-compatible
JSON request with proxy discovery and redirects disabled.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse

from ..domain.agent_model import (
    MAX_CANONICAL_REQUEST_BYTES,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MAX_PLANNING_DURATION_SECONDS,
    MAX_STRUCTURED_RESPONSE_BYTES,
    REQUEST_TIMEOUT_SECONDS,
    SCHEMA_VERSION,
    AgentModelMode,
    AgentModelSettings,
    AgentRunSnapshot,
    canonical_planning_payload,
)
from ..domain.agent_planning import (
    MAX_PLAN_STEPS,
    PlanningContext,
    PlanStep,
    PlanStepType,
    context_to_state,
)

AGENT_KEYRING_SERVICE = "NEAT AI Agent"
OPENAI_ENVIRONMENT_VARIABLE = "OPENAI_API_KEY"
MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
SYSTEM_PROMPT = (
    "Create a read-only NEAT planning proposal from metadata availability only. "
    "Do not infer phases, Bragg edges, strain, or measurements. Select only the "
    "provided step types. Return only the requested JSON schema; no instructions, "
    "rationales, evidence IDs, or scientific claims."
)


class AgentModelError(RuntimeError):
    """A sanitized provider/configuration failure safe to present in the UI."""


class AgentModelUnavailable(AgentModelError):
    pass


class MissingAgentCredential(AgentModelError):
    pass


class AgentModelOutputRejected(AgentModelError):
    pass


class AgentCredentialStore(Protocol):
    def get(self) -> str | None: ...

    def set(self, credential: str) -> None: ...

    def delete(self) -> None: ...


class AgentKeyringCredentialStore:
    """OS keyring access under a namespace that never reads Assistant keys."""

    def __init__(self, *, service_name: str = AGENT_KEYRING_SERVICE) -> None:
        self.service_name = service_name

    @staticmethod
    def _keyring():
        try:
            import keyring
        except ImportError as exc:  # pragma: no cover - optional install guard
            raise AgentModelUnavailable(
                "Secure Agent key storage requires the optional [agent-llm] extra."
            ) from exc
        return keyring

    def get(self) -> str | None:
        value = self._keyring().get_password(self.service_name, "openai")
        credential = str(value or "").strip()
        return credential or None

    def set(self, credential: str) -> None:
        value = str(credential or "").strip()
        if not value:
            raise ValueError("An API key cannot be empty.")
        self._keyring().set_password(self.service_name, "openai", value)

    def delete(self) -> None:
        keyring = self._keyring()
        try:
            keyring.delete_password(self.service_name, "openai")
        except keyring.errors.PasswordDeleteError:
            return


class AgentCredentialResolver:
    """Resolve only the Agent keyring key, with opt-in environment fallback."""

    def __init__(
        self,
        store: AgentCredentialStore,
        *,
        environment_allowed: bool,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._store = store
        self._environment_allowed = bool(environment_allowed)
        self._environ = environ

    def get(self) -> str | None:
        try:
            credential = self._store.get()
        except AgentModelUnavailable:
            if not self._environment_allowed:
                raise
            credential = None
        except Exception as exc:
            if not self._environment_allowed:
                raise AgentModelUnavailable("Agent keyring access failed safely.") from exc
            credential = None
        if credential:
            return credential
        if not self._environment_allowed:
            return None
        environment = os.environ if self._environ is None else self._environ
        value = str(environment.get(OPENAI_ENVIRONMENT_VARIABLE, "")).strip()
        return value or None


def normalize_local_endpoint(value: str) -> str:
    """Accept an explicit loopback HTTP endpoint, without credentials or redirects."""
    endpoint = str(value or "").strip().rstrip("/")
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Local model endpoint must be an explicit HTTP loopback URL.")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Local model endpoint port is invalid.") from exc
    if parsed.path not in {"", "/v1"}:
        raise ValueError("Local model endpoint path must be empty or start with /v1.")
    return endpoint


def validate_agent_settings(settings: AgentModelSettings) -> AgentModelSettings:
    if not isinstance(settings.mode, AgentModelMode):
        raise TypeError("Agent model mode is not supported.")
    if not isinstance(settings.allow_environment_credential, bool):
        raise TypeError("Environment credential setting must be boolean.")
    if settings.mode is AgentModelMode.OFFLINE:
        return AgentModelSettings(mode=AgentModelMode.OFFLINE)
    if not MODEL_ID_PATTERN.fullmatch(settings.model_id or ""):
        raise ValueError("Enter an exact supported model ID before saving.")
    if settings.mode is AgentModelMode.LOCAL:
        return AgentModelSettings(
            mode=settings.mode,
            model_id=settings.model_id,
            endpoint=normalize_local_endpoint(settings.endpoint),
            allow_environment_credential=False,
        )
    if settings.mode is AgentModelMode.OPENAI_CLOUD:
        return AgentModelSettings(
            mode=settings.mode,
            model_id=settings.model_id,
            endpoint="",
            allow_environment_credential=settings.allow_environment_credential,
        )
    raise ValueError("Agent model mode is not supported.")


def default_agent_settings_path() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    base_directory = (
        Path(local_app_data) if local_app_data else Path.home() / ".local" / "share"
    )
    return base_directory / "NEAT" / "agent_model_settings.json"


class AgentModelSettingsRepository:
    """Persist only non-secret Agent provider settings when explicitly saved."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or default_agent_settings_path()).expanduser()

    def load(self) -> AgentModelSettings:
        if not self.path.exists():
            return AgentModelSettings()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if set(payload) != {
                "schema_version",
                "mode",
                "model_id",
                "endpoint",
                "allow_environment_credential",
            } or payload["schema_version"] != 1:
                return AgentModelSettings()
            settings = AgentModelSettings(
                mode=AgentModelMode(payload["mode"]),
                model_id=str(payload["model_id"]),
                endpoint=str(payload["endpoint"]),
                allow_environment_credential=payload["allow_environment_credential"],
            )
            return validate_agent_settings(settings)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return AgentModelSettings()

    def save(self, settings: AgentModelSettings) -> Path:
        selected = validate_agent_settings(settings)
        payload = {
            "schema_version": 1,
            "mode": selected.mode.value,
            "model_id": selected.model_id,
            "endpoint": selected.endpoint,
            "allow_environment_credential": selected.allow_environment_credential,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        temporary_path.replace(self.path)
        return self.path


@dataclass(frozen=True)
class TransportResponse:
    status_code: int
    body: bytes


class JsonTransport(Protocol):
    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, object],
        timeout: float,
    ) -> TransportResponse: ...

    def cancel(self) -> None: ...


class HttpxJsonTransport:
    """One-shot HTTP transport with bounded streaming and no ambient proxy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel_request: Callable[[], None] | None = None
        self._cancelled = False

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, object],
        timeout: float,
    ) -> TransportResponse:
        started = time.monotonic()
        url = _ensure_local_endpoint_if_needed(url)
        try:
            import asyncio

            import httpx
        except ImportError as exc:  # pragma: no cover - optional install guard
            raise AgentModelUnavailable(
                "Model requests require the optional [agent-llm] extra."
            ) from exc

        async def send_request() -> TransportResponse:
            client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
                transport=httpx.AsyncHTTPTransport(retries=0),
            )
            loop = asyncio.get_running_loop()
            task = asyncio.current_task()
            if task is None:  # pragma: no cover - asyncio.run always creates a task
                await client.aclose()
                raise AgentModelUnavailable("The model request could not be started.")

            def schedule_cancellation() -> None:
                loop.call_soon_threadsafe(task.cancel)

            with self._lock:
                already_cancelled = self._cancelled
                if not already_cancelled:
                    self._cancel_request = schedule_cancellation
            if already_cancelled:
                await client.aclose()
                raise AgentModelUnavailable("The model request was cancelled.")
            try:
                async with client.stream(
                    "POST", url, headers=dict(headers), json=dict(body)
                ) as response:
                    length = response.headers.get("content-length", "")
                    if length.isdecimal() and int(length) > MAX_STRUCTURED_RESPONSE_BYTES:
                        raise AgentModelOutputRejected(
                            "Model response exceeded its size limit."
                        )
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        if time.monotonic() - started > MAX_PLANNING_DURATION_SECONDS:
                            raise AgentModelUnavailable(
                                "The model planning time limit was exceeded."
                            )
                        size += len(chunk)
                        if size > MAX_STRUCTURED_RESPONSE_BYTES:
                            raise AgentModelOutputRejected(
                                "Model response exceeded its size limit."
                            )
                        chunks.append(chunk)
                    return TransportResponse(response.status_code, b"".join(chunks))
            finally:
                with self._lock:
                    self._cancel_request = None
                await client.aclose()

        with self._lock:
            if self._cancelled:
                raise AgentModelUnavailable("The model request was cancelled.")
        remaining = MAX_PLANNING_DURATION_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            raise AgentModelUnavailable("The model planning time limit was exceeded.")
        try:
            return asyncio.run(
                asyncio.wait_for(send_request(), timeout=remaining)
            )
        except asyncio.TimeoutError as exc:
            raise AgentModelUnavailable(
                "The model planning time limit was exceeded."
            ) from exc
        except asyncio.CancelledError as exc:
            raise AgentModelUnavailable("The model request was cancelled.") from exc
        except AgentModelError:
            raise
        except Exception as exc:
            raise AgentModelUnavailable(
                "The configured model endpoint did not return a usable response."
            ) from exc

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            cancel_request = self._cancel_request
        if cancel_request is not None:
            try:
                cancel_request()
            except RuntimeError:
                return


class StructuredModel(Protocol):
    def complete(
        self,
        payload_json: str,
        *,
        available_step_types: tuple[PlanStepType, ...],
    ) -> ModelCompletion: ...

    def cancel(self) -> None: ...


@dataclass(frozen=True)
class ModelCompletion:
    content: str
    refusal: bool = False
    truncated: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None


class OpenAICompatibleStructuredModel:
    """OpenAI JSON-schema adapter for cloud OpenAI or one loopback endpoint."""

    def __init__(
        self,
        settings: AgentModelSettings,
        *,
        api_key: str | None,
        cloud_egress_authorized: bool = False,
        transport: JsonTransport | None = None,
    ) -> None:
        try:
            selected = validate_agent_settings(settings)
        except ValueError as exc:
            raise AgentModelUnavailable("The selected model is not configured.") from exc
        if selected.mode is AgentModelMode.OFFLINE:
            raise ValueError("Offline mode does not create a model client.")
        if selected.mode is AgentModelMode.OPENAI_CLOUD and not api_key:
            raise MissingAgentCredential("No Agent-specific OpenAI key is configured.")
        if selected.mode is AgentModelMode.OPENAI_CLOUD and not cloud_egress_authorized:
            raise AgentModelUnavailable("Cloud transmission was not approved for this run.")
        self.settings = selected
        self._api_key = api_key
        self._transport = transport or HttpxJsonTransport()
        self._cancelled = False
        self._started = time.monotonic()
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    @property
    def endpoint(self) -> str:
        if self.settings.mode is AgentModelMode.OPENAI_CLOUD:
            return "https://api.openai.com/v1"
        return self.settings.endpoint

    def _request(self, user_content: str, schema: Mapping[str, object]) -> ModelCompletion:
        if self._cancelled:
            raise AgentModelUnavailable("The model request was cancelled.")
        if time.monotonic() - self._started > MAX_PLANNING_DURATION_SECONDS:
            raise AgentModelUnavailable("The model planning time limit was exceeded.")
        request_body: dict[str, object] = {
            "model": self.settings.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "neat_agent_plan",
                    "strict": True,
                    "schema": dict(schema),
                },
            },
            "temperature": 0,
        }
        if self.settings.mode is AgentModelMode.OPENAI_CLOUD:
            request_body["max_completion_tokens"] = MAX_OUTPUT_TOKENS
            request_body["store"] = False
        else:
            request_body["max_tokens"] = MAX_OUTPUT_TOKENS
        encoded = json.dumps(request_body, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        if len(encoded) > MAX_CANONICAL_REQUEST_BYTES or len(encoded) > MAX_INPUT_TOKENS:
            raise AgentModelOutputRejected("Model request exceeds the approved byte limit.")
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.settings.mode is AgentModelMode.OPENAI_CLOUD:
                headers["Authorization"] = f"Bearer {self._api_key}"
            response = self._transport.post_json(
                self.endpoint.rstrip("/")
                + ("/chat/completions" if self.endpoint.endswith("/v1") else "/v1/chat/completions"),
                headers=headers,
                body=request_body,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except AgentModelError:
            raise
        except Exception as exc:
            raise AgentModelUnavailable("The model request failed safely.") from exc
        if len(response.body) > MAX_STRUCTURED_RESPONSE_BYTES:
            raise AgentModelOutputRejected("Model response exceeded its size limit.")
        if response.status_code == 401:
            raise AgentModelUnavailable("The configured model credential was rejected.")
        if response.status_code == 429:
            raise AgentModelUnavailable("The model provider rate-limited this request.")
        if response.status_code != 200:
            raise AgentModelUnavailable(
                f"The model provider returned HTTP {response.status_code}."
            )
        try:
            decoded = json.loads(response.body.decode("utf-8"))
            choice = decoded["choices"][0]
            message = choice["message"]
            content = message.get("content")
            usage = decoded.get("usage", {})
            if not isinstance(content, str):
                raise TypeError
            completion = ModelCompletion(
                content=content,
                refusal=bool(message.get("refusal")),
                truncated=choice.get("finish_reason") != "stop",
                input_tokens=_optional_nonnegative_int(usage.get("prompt_tokens")),
                output_tokens=_optional_nonnegative_int(usage.get("completion_tokens")),
            )
        except (KeyError, IndexError, TypeError, UnicodeDecodeError, ValueError) as exc:
            raise AgentModelOutputRejected("The model returned a malformed response.") from exc
        if len(completion.content.encode("utf-8")) > MAX_STRUCTURED_RESPONSE_BYTES:
            raise AgentModelOutputRejected("Model response exceeded its size limit.")
        self.input_tokens = completion.input_tokens
        self.output_tokens = completion.output_tokens
        return completion

    def complete(self, payload_json: str, *, available_step_types: tuple[PlanStepType, ...]) -> ModelCompletion:
        schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_PLAN_STEPS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_type": {
                                "type": "string",
                                "enum": [item.value for item in available_step_types],
                            }
                        },
                        "required": ["step_type"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
        }
        return self._request(payload_json, schema)

    def test_connection(self) -> None:
        completion = self._request(
            '{"purpose":"connection_test","schema_version":"neat-agent-test-v1"}',
            {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
        try:
            response = json.loads(completion.content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AgentModelOutputRejected("The test response did not match its schema.") from exc
        if completion.refusal or completion.truncated or response != {"ok": True}:
            raise AgentModelOutputRejected("The test response did not match its schema.")

    def cancel(self) -> None:
        self._cancelled = True
        self._transport.cancel()


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _ensure_local_endpoint_if_needed(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.hostname == "api.openai.com":
        return url
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise AgentModelUnavailable("The configured model endpoint is not an approved destination.")
    if parsed.hostname == "localhost":
        # Pin the conventional loopback name without consulting DNS. This
        # avoids rebinding and keeps name resolution outside the request's
        # bounded transport deadline. IPv6-only services can use [::1].
        authority = "127.0.0.1" + (f":{parsed.port}" if parsed.port is not None else "")
        return parsed._replace(netloc=authority).geturl()
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_loopback:
            raise AgentModelUnavailable("Local endpoint locality could not be verified.")
        return url
    raise AgentModelUnavailable("Local endpoint locality could not be verified.")


def _resolve_credential(
    snapshot: AgentRunSnapshot,
    resolver: AgentCredentialResolver | None,
) -> str | None:
    if snapshot.settings.mode is not AgentModelMode.OPENAI_CLOUD:
        return None
    if not snapshot.cloud_egress_consent:
        raise AgentModelUnavailable("Cloud transmission was not approved for this run.")
    if resolver is None:
        return None
    return resolver.get()


class LLMPlanProposer:
    """Call one injected structured model and return trusted catalogue steps."""

    def __init__(
        self,
        context: PlanningContext,
        snapshot: AgentRunSnapshot,
        *,
        model: StructuredModel | None = None,
        credential_resolver: AgentCredentialResolver | None = None,
        model_factory: Callable[[AgentModelSettings, str | None], StructuredModel] | None = None,
        check_before_request: Callable[[], None] | None = None,
    ) -> None:
        self.context = context
        self.snapshot = snapshot
        if snapshot.schema_version != SCHEMA_VERSION:
            raise AgentModelOutputRejected(
                "The run-authorized planning schema does not match this application."
            )
        if (
            snapshot.settings.mode is AgentModelMode.OPENAI_CLOUD
            and not snapshot.cloud_egress_consent
        ):
            raise AgentModelUnavailable("Cloud transmission was not approved for this run.")
        if model is None:
            credential = _resolve_credential(snapshot, credential_resolver)
            if snapshot.settings.mode is AgentModelMode.OPENAI_CLOUD and not credential:
                raise MissingAgentCredential("No Agent-specific OpenAI key is configured.")
            if model_factory is None:
                model_factory = lambda settings, api_key: OpenAICompatibleStructuredModel(
                    settings,
                    api_key=api_key,
                    cloud_egress_authorized=snapshot.cloud_egress_consent,
                )
            model = model_factory(snapshot.settings, credential)
        self.model = model
        self._check_before_request = check_before_request
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def __call__(self, state: Mapping[str, object]) -> tuple[PlanStep, ...]:
        from .agent_planning import _default_proposal

        if state.get("objective") != self.snapshot.objective.value:
            raise AgentModelOutputRejected("The planning objective changed during the run.")
        catalogue = _default_proposal(state)  # type: ignore[arg-type]
        available = tuple(dict.fromkeys(step.step_type for step in catalogue))
        payload = canonical_planning_payload(self.context, available)
        encoded = payload.to_json()
        if len(encoded.encode("ascii")) > MAX_CANONICAL_REQUEST_BYTES:
            raise AgentModelOutputRejected("Canonical planning request exceeds its byte limit.")
        # Preflight and request construction may take time. Recheck cancellation
        # and the live revision at the final boundary before any model egress.
        if self._check_before_request is not None:
            self._check_before_request()
        completion = self.model.complete(encoded, available_step_types=available)  # type: ignore[call-arg]
        self.input_tokens = _optional_nonnegative_int(completion.input_tokens)
        self.output_tokens = _optional_nonnegative_int(completion.output_tokens)
        if completion.refusal:
            raise AgentModelOutputRejected("The model refused to propose a plan.")
        if completion.truncated:
            raise AgentModelOutputRejected("The model response was truncated.")
        raw_bytes = completion.content.encode("utf-8")
        if len(raw_bytes) > MAX_STRUCTURED_RESPONSE_BYTES:
            raise AgentModelOutputRejected("Model response exceeded its size limit.")
        try:
            decoded = json.loads(completion.content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AgentModelOutputRejected("The model response was not valid JSON.") from exc
        if not isinstance(decoded, dict) or set(decoded) != {"steps"}:
            raise AgentModelOutputRejected("The model response contained unknown fields.")
        raw_steps = decoded["steps"]
        if (
            not isinstance(raw_steps, list)
            or not raw_steps
            or len(raw_steps) > MAX_PLAN_STEPS
        ):
            raise AgentModelOutputRejected("The model returned an invalid number of steps.")
        safe_catalogue = {step.step_type: step for step in catalogue}
        selected: list[PlanStep] = []
        selected_types: set[PlanStepType] = set()
        for item in raw_steps:
            if not isinstance(item, dict) or set(item) != {"step_type"}:
                raise AgentModelOutputRejected("A model step contained unknown fields.")
            try:
                step_type = PlanStepType(item["step_type"])
            except (TypeError, ValueError) as exc:
                raise AgentModelOutputRejected("The model selected an unknown step type.") from exc
            if step_type not in safe_catalogue or step_type in selected_types:
                raise AgentModelOutputRejected("The model selected an unsupported step type.")
            selected_types.add(step_type)
            selected.append(safe_catalogue[step_type])
        return tuple(selected)

    def cancel(self) -> None:
        cancel = getattr(self.model, "cancel", None)
        if callable(cancel):
            cancel()


def create_model_for_snapshot(
    snapshot: AgentRunSnapshot,
    credential_resolver: AgentCredentialResolver | None,
    *,
    transport: JsonTransport | None = None,
) -> OpenAICompatibleStructuredModel:
    """Create a provider client only at the planning stage, never at window open."""
    if snapshot.settings.mode is AgentModelMode.OFFLINE:
        raise AgentModelUnavailable("Offline deterministic mode does not use a model.")
    key = _resolve_credential(snapshot, credential_resolver)
    return OpenAICompatibleStructuredModel(
        snapshot.settings,
        api_key=key,
        cloud_egress_authorized=snapshot.cloud_egress_consent,
        transport=transport,
    )


def canonical_payload_for_context(context: PlanningContext) -> str:
    """Return exact allowlisted wire JSON for a preflight, without sending it."""
    from .agent_planning import GraphState, _default_proposal

    catalogue = _default_proposal(cast(GraphState, context_to_state(context)))
    available = tuple(dict.fromkeys(step.step_type for step in catalogue))
    return canonical_planning_payload(context, available).to_json()


def run_model_connection_test(
    settings: AgentModelSettings,
    credential_resolver: AgentCredentialResolver | None,
    *,
    transport: JsonTransport | None = None,
) -> None:
    """Make one explicit, NEAT-data-free settings test request."""
    selected = validate_agent_settings(settings)
    if selected.mode is AgentModelMode.OFFLINE:
        raise AgentModelUnavailable("Offline deterministic mode has no connection to test.")
    if selected.mode is AgentModelMode.OPENAI_CLOUD:
        credential = credential_resolver.get() if credential_resolver is not None else None
        if not credential:
            raise MissingAgentCredential("No Agent-specific OpenAI key is configured.")
    else:
        credential = None
    OpenAICompatibleStructuredModel(
        selected,
        api_key=credential,
        cloud_egress_authorized=True,
        transport=transport,
    ).test_connection()


__all__ = [
    "AGENT_KEYRING_SERVICE",
    "AgentCredentialResolver",
    "AgentCredentialStore",
    "AgentKeyringCredentialStore",
    "AgentModelError",
    "AgentModelOutputRejected",
    "AgentModelSettingsRepository",
    "AgentModelUnavailable",
    "HttpxJsonTransport",
    "LLMPlanProposer",
    "MissingAgentCredential",
    "ModelCompletion",
    "OpenAICompatibleStructuredModel",
    "TransportResponse",
        "canonical_payload_for_context",
        "create_model_for_snapshot",
    "default_agent_settings_path",
    "normalize_local_endpoint",
    "run_model_connection_test",
    "validate_agent_settings",
]

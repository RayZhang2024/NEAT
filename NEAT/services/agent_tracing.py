"""Opt-in LangSmith tracing for Agent LangGraph workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Literal, TypedDict, cast

LANGSMITH_REGION_ENDPOINTS = {
    "us": "https://api.smith.langchain.com",
    "eu": "https://eu.api.smith.langchain.com",
    "apac": "https://apac.api.smith.langchain.com",
    "aws-us": "https://aws.api.smith.langchain.com",
}


class LangSmithConnectionError(RuntimeError):
    """A safe, credential-free LangSmith connection-test failure."""


class _AgentAddress(TypedDict):
    kind: Literal["AGENT"]
    id: str
    environment: Literal["LOCAL", "DEVELOPMENT", "STAGING", "PRODUCTION"]


def _new_client(
    *,
    api_key: str,
    region: str,
    workspace_id: str,
    tracing_error_callback: Callable[[Exception], None] | None = None,
) -> Any:
    from langsmith import Client

    endpoint = LANGSMITH_REGION_ENDPOINTS[region]
    options: dict[str, Any] = {
        "api_key": api_key,
        "api_url": endpoint,
        "hide_inputs": False,
        "hide_outputs": False,
        "hide_metadata": False,
        "anonymizer": None,
        "omit_traced_runtime_info": False,
        "timeout_ms": 10_000,
    }
    if workspace_id:
        options["workspace_id"] = workspace_id
    if tracing_error_callback is not None:
        options["tracing_error_callback"] = tracing_error_callback
    return Client(**options)


def _address(agent_id: str, environment: str) -> _AgentAddress:
    """Build the structured agent address required by the LangSmith SDK."""
    return {
        "kind": "AGENT",
        "id": agent_id,
        "environment": cast(
            Literal["LOCAL", "DEVELOPMENT", "STAGING", "PRODUCTION"],
            environment.upper(),
        ),
    }


def _new_langchain_agent_tracer(
    *, client: Any, address: _AgentAddress, tags: list[str]
) -> Any:
    """Create a LangChain callback tracer that preserves agent addressing.

    LangChain Core's default tracer chooses a project name (usually ``default``)
    and does not copy the LangSmith address from ``tracing_context`` to its
    ``RunTree`` objects. LangGraph uses that callback tracer for node spans, so
    give it an address-aware tracer explicitly.
    """
    from langchain_core.tracers.langchain import LangChainTracer

    class _AddressedLangChainTracer(LangChainTracer):
        def __init__(self) -> None:
            super().__init__(client=client, tags=tags)
            # LangChainTracer falls back to the ``default`` project when this is
            # unset. Addressed traces must have no project destination.
            self.project_name = None

        def _start_trace(self, run: Any) -> None:
            run.address = address
            run.session_name = None
            super()._start_trace(run)

    return _AddressedLangChainTracer()


@contextmanager
def agent_langsmith_context(
    *,
    enabled: bool,
    api_key: str | None,
    agent_id: str,
    environment: str,
    region: str,
    workspace_id: str = "",
    workflow: str,
) -> Iterator[bool]:
    """Set the addressed LangSmith context for one opt-in workflow stage."""
    try:
        from langchain_core.tracers.context import tracing_v2_callback_var
        from langsmith.run_helpers import tracing_context
    except (ImportError, OSError):
        yield False
        return

    if not enabled or not api_key:
        with tracing_context(enabled=False):
            yield False
        return

    try:
        client = _new_client(
            api_key=api_key,
            region=region,
            workspace_id=workspace_id,
        )
    except Exception:  # noqa: BLE001 - tracing setup must not block local analysis
        with tracing_context(enabled=False):
            yield False
        return

    address = _address(agent_id, environment)
    tags = ["neat-agent", workflow]
    try:
        tracer = _new_langchain_agent_tracer(
            client=client,
            address=address,
            tags=tags,
        )
    except Exception:  # noqa: BLE001 - tracing setup must not block local analysis
        with tracing_context(enabled=False):
            yield False
        return

    token = tracing_v2_callback_var.set(tracer)
    try:
        with tracing_context(
            enabled=True,
            address=address,
            tags=tags,
            client=client,
        ):
            yield True
    finally:
        tracing_v2_callback_var.reset(token)


def trace_llm_call(
    *,
    name: str,
    provider: str,
    model_id: str,
    inputs: Mapping[str, object],
    operation: Callable[[], Any],
) -> Any:
    """Trace one direct provider request as an LLM child run when enabled."""
    try:
        from langsmith import traceable
        from langsmith.run_helpers import get_tracing_context
    except (ImportError, OSError):
        return operation()
    if not get_tracing_context().get("enabled"):
        return operation()

    operation_started = False
    operation_result: list[Any] = []

    def invoke(_inputs: Mapping[str, object]) -> Any:
        nonlocal operation_started
        operation_started = True
        result = operation()
        operation_result.append(result)
        return result

    try:
        traced_invoke = traceable(
            name=name,
            run_type="llm",
            metadata={"ls_provider": provider, "ls_model_name": model_id},
        )(invoke)
    except Exception:  # noqa: BLE001 - tracing setup must not block the model call
        return operation()
    try:
        return traced_invoke(dict(inputs))
    except Exception:
        if operation_result:
            return operation_result[0]
        if operation_started:
            raise
        return operation()


def test_langsmith_trace_connection(
    *,
    api_key: str,
    agent_id: str,
    environment: str,
    region: str,
    workspace_id: str = "",
) -> None:
    """Upload one synthetic trace and wait for LangSmith's response."""
    errors: list[Exception] = []
    try:
        from langsmith import traceable
        from langsmith.run_helpers import tracing_context

        client = _new_client(
            api_key=api_key,
            region=region,
            workspace_id=workspace_id,
            tracing_error_callback=errors.append,
        )

        @traceable(name="NEAT connection test", run_type="chain")
        def _emit_test_trace() -> dict[str, str]:
            return {"status": "ok"}

        with tracing_context(
            enabled=True,
            address=_address(agent_id, environment),
            tags=["neat-agent", "connection-test"],
            client=client,
        ):
            _emit_test_trace()
        client.flush(timeout=15.0)
    except Exception as exc:  # noqa: BLE001 - sanitize SDK/network errors
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 403:
            raise LangSmithConnectionError(
                "LangSmith rejected the trace (HTTP 403). Check the API key's region "
                "and workspace access."
            ) from None
        if status_code == 401:
            raise LangSmithConnectionError(
                "LangSmith rejected the API key (HTTP 401). Check that the key is active."
            ) from None
        raise LangSmithConnectionError(
            f"LangSmith trace test failed ({type(exc).__name__})."
        ) from None

    if errors:
        status_code = getattr(getattr(errors[0], "response", None), "status_code", None)
        if status_code == 403:
            raise LangSmithConnectionError(
                "LangSmith rejected the trace (HTTP 403). Check the API key's region "
                "and workspace access."
            )
        if status_code == 401:
            raise LangSmithConnectionError(
                "LangSmith rejected the API key (HTTP 401). Check that the key is active."
            )
        raise LangSmithConnectionError(
            f"LangSmith trace upload failed ({type(errors[0]).__name__})."
        )


__all__ = [
    "LANGSMITH_REGION_ENDPOINTS",
    "LangSmithConnectionError",
    "agent_langsmith_context",
    "test_langsmith_trace_connection",
    "trace_llm_call",
]

"""Discover OpenAI-compatible model servers running on localhost."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class LocalModelEngine:
    engine_id: str
    label: str
    base_url: str


LOCAL_MODEL_ENGINES: tuple[LocalModelEngine, ...] = (
    LocalModelEngine(
        engine_id="ollama",
        label="Ollama",
        base_url="http://localhost:11434/v1",
    ),
    LocalModelEngine(
        engine_id="lm_studio",
        label="LM Studio",
        base_url="http://localhost:1234/v1",
    ),
)


class LocalModelDiscoveryError(RuntimeError):
    """A local server did not provide a valid OpenAI-style model list."""


def list_local_models(
    base_url: str,
    *,
    timeout_seconds: float = 2.0,
) -> tuple[str, ...]:
    """Return model IDs from the local server's OpenAI-compatible endpoint."""

    endpoint = urljoin(base_url.rstrip("/") + "/", "models")
    request = Request(
        endpoint,
        method="GET",
        headers={"User-Agent": "NEAT-AI-Assistant"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise LocalModelDiscoveryError(
            f"The local model server returned HTTP {exc.code}."
        ) from None
    except (URLError, TimeoutError):
        raise LocalModelDiscoveryError(
            "The local model server is not reachable."
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
        raise LocalModelDiscoveryError(
            "The local model server returned an invalid model list."
        ) from None

    records = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = {
        str(record.get("id", "")).strip()
        for record in records
        if isinstance(record, dict)
        and 0 < len(str(record.get("id", "")).strip()) <= 200
    }
    return tuple(sorted(model_ids, key=str.casefold))


def discover_local_models(
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, tuple[str, ...]]:
    """Probe Ollama and LM Studio concurrently without contacting remote hosts."""

    discovered: dict[str, tuple[str, ...]] = {}
    with ThreadPoolExecutor(max_workers=len(LOCAL_MODEL_ENGINES)) as executor:
        futures = {
            executor.submit(
                list_local_models,
                engine.base_url,
                timeout_seconds=timeout_seconds,
            ): engine
            for engine in LOCAL_MODEL_ENGINES
        }
        for future in as_completed(futures):
            engine = futures[future]
            try:
                models = future.result()
            except LocalModelDiscoveryError:
                continue
            discovered[engine.engine_id] = models
    return discovered


__all__ = [
    "LOCAL_MODEL_ENGINES",
    "LocalModelDiscoveryError",
    "LocalModelEngine",
    "discover_local_models",
    "list_local_models",
]

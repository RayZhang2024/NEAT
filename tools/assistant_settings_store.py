"""Persist non-secret NEAT assistant choices outside the repository."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from tools.assistant_providers import (
    AssistantAccessMode,
    AssistantProvider,
    AssistantSettings,
    load_assistant_settings,
)


SETTINGS_SCHEMA_VERSION = 2


def default_assistant_settings_path() -> Path:
    """Return the per-user assistant settings path."""

    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        base_directory = Path(local_app_data)
    else:
        base_directory = Path.home() / ".local" / "share"
    return base_directory / "NEAT" / "assistant_settings.json"


class AssistantSettingsRepository:
    """Read and atomically write provider/model choices without credentials."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or default_assistant_settings_path()).expanduser()

    def load(self) -> AssistantSettings:
        if not self.path.exists():
            return load_assistant_settings()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return load_assistant_settings(
                access_mode=AssistantAccessMode(payload["access_mode"]),
                provider=AssistantProvider(payload["provider"]),
                model=str(payload["model"]),
                base_url=str(payload.get("base_url", "")),
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return load_assistant_settings()

    def save(self, settings: AssistantSettings) -> Path:
        payload = {
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "access_mode": settings.access_mode.value,
            "provider": settings.provider.value,
            "model": settings.model,
            "base_url": settings.base_url,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)
        return self.path


__all__ = [
    "AssistantSettingsRepository",
    "SETTINGS_SCHEMA_VERSION",
    "default_assistant_settings_path",
]

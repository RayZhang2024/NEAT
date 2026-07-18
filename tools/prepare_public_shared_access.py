"""Prepare the public shared-access configuration for a packaged NEAT build."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tools.assistant_shared_client import (
    SHARED_ACCESS_TOKEN_ENV,
    SHARED_SERVICE_URL_ENV,
    SharedServiceSettings,
    _persisted_windows_user_setting,
)


PUBLIC_SERVICE_URL_ENV = "NEAT_PUBLIC_SHARED_SERVICE_URL"
PUBLIC_ACCESS_TOKEN_ENV = "NEAT_PUBLIC_SHARED_ACCESS_TOKEN"
DEFAULT_OUTPUT_PATH = Path("build") / "private" / "shared_access.json"


def _setting(public_name: str, local_name: str) -> str:
    return (
        os.environ.get(public_name, "").strip()
        or os.environ.get(local_name, "").strip()
        or _persisted_windows_user_setting(local_name)
    )


def prepare_public_shared_access(
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Validate and write the extractable public-client build credential."""

    service_url = _setting(PUBLIC_SERVICE_URL_ENV, SHARED_SERVICE_URL_ENV)
    access_token = _setting(PUBLIC_ACCESS_TOKEN_ENV, SHARED_ACCESS_TOKEN_ENV)
    if not service_url or not access_token:
        raise RuntimeError(
            "Public shared access is missing. Set "
            "NEAT_PUBLIC_SHARED_SERVICE_URL and "
            "NEAT_PUBLIC_SHARED_ACCESS_TOKEN before packaging NEAT."
        )

    settings = SharedServiceSettings(
        service_url=service_url,
        access_token=access_token,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "service_url": settings.service_url,
                "access_token": settings.access_token,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print(f"Prepared public shared-access configuration: {output_path}")
    return output_path


if __name__ == "__main__":
    prepare_public_shared_access()

"""Expose the revision of a source checkout without changing the release version."""

from functools import lru_cache
from pathlib import Path
import re
import subprocess
import sys


@lru_cache(maxsize=1)
def development_commit() -> str | None:
    """Return an eight-character Git revision for a NEAT source checkout.

    Packaged executables and installed wheels should continue to identify
    themselves solely by the version declared in pyproject.toml. Do not use
    the caller's current working directory, which may be unrelated to NEAT.
    """
    if getattr(sys, "frozen", False):
        return None

    project_root = Path(__file__).resolve().parent.parent
    command = ["git", "-C", str(project_root), "rev-parse"]
    try:
        checkout = subprocess.run(
            [*command, "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        if Path(checkout).resolve() != project_root:
            return None
        revision = subprocess.run(
            [*command, "--short=8", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None

    return revision if re.fullmatch(r"[0-9a-fA-F]{7,40}", revision) else None

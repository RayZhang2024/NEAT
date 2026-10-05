"""Remove only generated Briefcase Windows packaging output for this version."""

from __future__ import annotations

import argparse
import shutil
import tomllib
from pathlib import Path


def clean_briefcase_state(project_root: Path) -> list[Path]:
    root = Path(project_root).resolve(strict=True)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    candidates = [
        root / "build" / "NEAT" / "windows",
        root / "dist" / f"NEAT-{version}.msi",
        root / "dist" / f"NEAT-v{version}.msi",
    ]
    removed: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Refusing to clean a path outside the project: {resolved}")
        if candidate.is_dir():
            shutil.rmtree(candidate)
            removed.append(candidate)
        elif candidate.is_file():
            candidate.unlink()
            removed.append(candidate)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    for path in clean_briefcase_state(args.project_root):
        print(f"Removed stale Briefcase output: {path}")
    print("Briefcase Windows packaging state is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate runtime files and metadata in built wheel and sdist archives."""

from __future__ import annotations

import argparse
import configparser
from email.parser import Parser
import tarfile
import tomllib
import zipfile
from pathlib import Path

from tools.assistant_retrieval import KNOWLEDGE_FILENAMES


ENTRY_POINTS = {
    "neat",
    "neat-assistant-server",
    "neat-assistant-laptop-setup",
}
PACKAGE_FILES = (
    "NEAT/__init__.py",
    "NEAT/domain/__init__.py",
    "NEAT/services/fitting_engine.py",
    "tools/__init__.py",
    "tools/assistant_retrieval.py",
)


def validate_wheel(path: Path, expected_version: str) -> str:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {
            *PACKAGE_FILES,
            "NEAT/assets/launch_splash.png",
            *(f"NEAT/knowledge/{filename}" for filename in KNOWLEDGE_FILENAMES),
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"Wheel is missing required files: {', '.join(missing)}")
        if any(name.startswith("docs/assistant/") for name in names):
            raise ValueError("Wheel must not depend on repository docs/assistant files")

        metadata_files = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        entry_point_files = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(metadata_files) != 1 or len(entry_point_files) != 1:
            raise ValueError("Wheel must contain one distribution metadata and entry-point file")
        metadata = Parser().parsestr(archive.read(metadata_files[0]).decode("utf-8"))
        if (
            (metadata.get("Name") or "").lower() != "neat"
            or metadata.get("Version") != expected_version
        ):
            raise ValueError("Wheel distribution metadata does not match pyproject.toml")
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(entry_point_files[0]).decode("utf-8"))
        scripts = set(parser["console_scripts"]) if parser.has_section("console_scripts") else set()
        missing_scripts = sorted(ENTRY_POINTS - scripts)
        if missing_scripts:
            raise ValueError(f"Wheel is missing console scripts: {', '.join(missing_scripts)}")
        return metadata["Version"]


def validate_sdist(path: Path, expected_version: str) -> None:
    with tarfile.open(path, "r:gz") as archive:
        members = {
            name.replace("\\", "/"): member
            for name, member in ((item.name, item) for item in archive.getmembers())
        }
        names = set(members)
        required = {
            *PACKAGE_FILES,
            "NEAT/assets/launch_splash.png",
            *(f"NEAT/knowledge/{filename}" for filename in KNOWLEDGE_FILENAMES),
            "pyproject.toml",
            "PKG-INFO",
        }
        missing = sorted(
            suffix
            for suffix in required
            if not any(name.endswith("/" + suffix) for name in names)
        )
        if missing:
            raise ValueError(f"Sdist is missing required files: {', '.join(missing)}")
        if not any(name.endswith("/pyproject.toml") for name in names):
            raise ValueError("Sdist is missing pyproject.toml")
        pyproject_name = next(name for name in names if name.endswith("/pyproject.toml"))
        pyproject_file = archive.extractfile(members[pyproject_name])
        if pyproject_file is None:
            raise ValueError("Sdist pyproject.toml is unreadable")
        pyproject = pyproject_file.read().decode("utf-8")
        missing_scripts = sorted(
            script for script in ENTRY_POINTS if f"{script} =" not in pyproject
        )
        if missing_scripts:
            raise ValueError(
                "Sdist pyproject is missing console scripts: "
                + ", ".join(missing_scripts)
            )
        metadata_name = next(name for name in names if name.endswith("/PKG-INFO"))
        metadata_file = archive.extractfile(members[metadata_name])
        if metadata_file is None:
            raise ValueError("Sdist PKG-INFO is unreadable")
        metadata = Parser().parsestr(metadata_file.read().decode("utf-8"))
        if (
            (metadata.get("Name") or "").lower() != "neat"
            or metadata.get("Version") != expected_version
        ):
            raise ValueError("Sdist metadata does not match wheel name and version")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    expected_version = project["project"]["version"]
    wheel_version = validate_wheel(args.wheel, expected_version)
    validate_sdist(args.sdist, wheel_version)
    print(f"Validated wheel: {args.wheel.name}")
    print(f"Validated sdist: {args.sdist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify PEP 639 MIT metadata in the built wheel and source archive."""

from __future__ import annotations

import argparse
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path


def _metadata_headers(metadata_bytes: bytes) -> dict[str, list[str]]:
    message = Parser().parsestr(metadata_bytes.decode("utf-8"))
    result: dict[str, list[str]] = {}
    for key in message.keys():
        result.setdefault(key.lower(), []).extend(message.get_all(key, []))
    return result


def _verify_metadata(metadata_bytes: bytes, archive_label: str) -> None:
    headers = _metadata_headers(metadata_bytes)
    if headers.get("license-expression") != ["MIT"]:
        raise ValueError(f"{archive_label} must report License-Expression: MIT")
    if "LICENSE" not in headers.get("license-file", []):
        raise ValueError(f"{archive_label} must declare License-File: LICENSE")


def validate_license_artifacts(artifact_dir: Path, project_root: Path) -> None:
    artifacts = Path(artifact_dir)
    root = Path(project_root)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project.get("license") != "MIT" or project.get("license-files") != ["LICENSE"]:
        raise ValueError("pyproject.toml must declare the MIT expression and LICENSE file")
    if not (root / "LICENSE").is_file():
        raise ValueError("Repository LICENSE file is missing")

    wheels = list(artifacts.glob("*.whl"))
    sdists = list(artifacts.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one .tar.gz sdist")

    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        license_names = [name for name in wheel.namelist() if name.endswith(".dist-info/licenses/LICENSE")]
        if len(metadata_names) != 1 or len(license_names) != 1:
            raise ValueError("Wheel must contain one metadata file and its packaged LICENSE")
        _verify_metadata(wheel.read(metadata_names[0]), str(wheels[0]))
        if not wheel.read(license_names[0]).strip():
            raise ValueError("Wheel's packaged LICENSE is empty")

    with tarfile.open(sdists[0], "r:gz") as sdist:
        metadata_members = [
            member
            for member in sdist.getmembers()
            if member.name.endswith("/PKG-INFO") and member.name.count("/") == 1
        ]
        license_members = [
            member
            for member in sdist.getmembers()
            if member.name.endswith("/LICENSE") and member.name.count("/") == 1
        ]
        if len(metadata_members) != 1 or len(license_members) != 1:
            raise ValueError("Sdist must contain one PKG-INFO file and its LICENSE")
        metadata_file = sdist.extractfile(metadata_members[0])
        license_file = sdist.extractfile(license_members[0])
        if metadata_file is None or license_file is None:
            raise ValueError("Sdist metadata/license could not be read")
        _verify_metadata(metadata_file.read(), str(sdists[0]))
        if not license_file.read().strip():
            raise ValueError("Sdist's packaged LICENSE is empty")

    print(f"PEP 639 MIT metadata and LICENSE verified in {wheels[0].name} and {sdists[0].name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    validate_license_artifacts(args.artifacts_dir, args.project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

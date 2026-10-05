"""Build and verify Windows release wrappers around one frozen NEAT payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
import tomllib
import xml.etree.ElementTree as ET


MANIFEST_SCHEMA_VERSION = 1


class DistributionError(ValueError):
    """Raised when a distribution artifact differs from its source payload."""


def _normalized_relative_path(path: Path) -> str:
    return unicodedata.normalize("NFC", path.as_posix())


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_files(payload_root: Path) -> list[Path]:
    """Return regular payload files sorted by normalized relative path."""

    root = Path(payload_root).resolve(strict=True)
    if not root.is_dir():
        raise DistributionError(f"Payload root is not a directory: {root}")
    files: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise DistributionError(f"Payload must not contain symlinks: {path}")
        if path.is_file():
            relative = _normalized_relative_path(path.relative_to(root))
            files.append((relative, path))
    files.sort(key=lambda item: item[0])
    return [path for _, path in files]


def create_manifest(payload_root: Path) -> dict[str, Any]:
    """Create a canonical manifest relative to the PyInstaller folder root."""

    root = Path(payload_root).resolve(strict=True)
    files = payload_files(root)
    if not any(path.relative_to(root).as_posix() == "NEAT.exe" for path in files):
        raise DistributionError(f"Expected NEAT.exe in payload root: {root}")
    entries = [
        {
            "path": _normalized_relative_path(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": _hash_file(path),
        }
        for path in files
    ]
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "files": entries}


def write_manifest(payload_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = create_manifest(payload_root)
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DistributionError(f"Cannot read payload manifest: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or not isinstance(manifest.get("files"), list)
    ):
        raise DistributionError("Unsupported or malformed payload manifest")
    paths: list[str] = []
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise DistributionError("Malformed payload manifest entry")
        relative = entry["path"]
        if not isinstance(relative, str) or "\\" in relative:
            raise DistributionError(f"Manifest path is not normalized: {relative!r}")
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or not posix.parts:
            raise DistributionError(f"Unsafe manifest path: {relative!r}")
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise DistributionError(f"Invalid manifest file size: {relative!r}")
        if (
            not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
        ):
            raise DistributionError(f"Invalid SHA-256 for {relative!r}")
        paths.append(relative)
    if paths != sorted(set(paths)):
        raise DistributionError("Manifest paths must be unique and sorted")
    if "NEAT.exe" not in paths:
        raise DistributionError("Manifest does not include NEAT.exe")
    return manifest


def _compare_manifests(expected: dict[str, Any], actual: dict[str, Any], label: str) -> None:
    expected_files = {entry["path"]: entry for entry in expected["files"]}
    actual_files = {entry["path"]: entry for entry in actual["files"]}
    missing = sorted(expected_files.keys() - actual_files.keys())
    extra = sorted(actual_files.keys() - expected_files.keys())
    changed = sorted(
        path
        for path in expected_files.keys() & actual_files.keys()
        if expected_files[path] != actual_files[path]
    )
    if missing or extra or changed:
        details = []
        if missing:
            details.append(f"missing={missing[:10]}")
        if extra:
            details.append(f"extra={extra[:10]}")
        if changed:
            details.append(f"changed={changed[:10]}")
        raise DistributionError(f"{label} differs from payload manifest: " + "; ".join(details))


def verify_directory(payload_root: Path, manifest_path: Path) -> None:
    expected = read_manifest(manifest_path)
    actual = create_manifest(payload_root)
    _compare_manifests(expected, actual, str(payload_root))


def create_portable_zip(payload_root: Path, manifest_path: Path, archive_path: Path) -> None:
    root = Path(payload_root).resolve(strict=True)
    destination = Path(archive_path).resolve()
    if destination.is_relative_to(root):
        raise DistributionError("Portable ZIP must be outside the payload directory")
    verify_directory(root, manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("NEAT/", b"")
        for path in payload_files(root):
            relative = _normalized_relative_path(path.relative_to(root))
            archive.write(path, f"NEAT/{relative}")
    verify_portable_zip(destination, manifest_path)


def verify_portable_zip(archive_path: Path, manifest_path: Path) -> None:
    expected = read_manifest(manifest_path)
    actual_entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                if member.filename != "NEAT/":
                    raise DistributionError(f"Unexpected directory in portable ZIP: {member.filename}")
                continue
            if not member.filename.startswith("NEAT/"):
                raise DistributionError(f"Portable ZIP has content outside NEAT/: {member.filename}")
            relative = member.filename.removeprefix("NEAT/")
            posix = PurePosixPath(relative)
            if posix.is_absolute() or ".." in posix.parts or "\\" in relative:
                raise DistributionError(f"Unsafe portable ZIP path: {member.filename}")
            digest = hashlib.sha256()
            size = 0
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(block)
                    digest.update(block)
            actual_entries.append(
                {"path": relative, "size": size, "sha256": digest.hexdigest()}
            )
    actual_entries.sort(key=lambda entry: entry["path"])
    _compare_manifests(
        expected,
        {"schema_version": MANIFEST_SCHEMA_VERSION, "files": actual_entries},
        str(archive_path),
    )


def verify_briefcase_output(project_root: Path) -> None:
    """Check the generated MSI metadata and that Briefcase used external-app mode."""

    root = Path(project_root).resolve(strict=True)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    briefcase = project["tool"]["briefcase"]
    app = briefcase["app"]["neat"]
    app_name = next(iter(briefcase["app"]))
    windows = app["windows"]
    version = project["project"]["version"]
    expected_msi = root / "dist" / f"NEAT-{version}.msi"
    if not expected_msi.is_file():
        raise DistributionError(f"Briefcase did not produce the expected MSI: {expected_msi}")

    bundle_identifier = f"{briefcase['bundle']}.{app_name}"
    if bundle_identifier != "io.github.rayzhang2024.neat":
        raise DistributionError(f"Unexpected Briefcase bundle identity: {bundle_identifier}")
    expected_upgrade_code = str(
        uuid.uuid5(uuid.NAMESPACE_DNS, ".".join(bundle_identifier.split(".")[::-1]))
    )

    app_template = root / "build" / "NEAT" / "windows" / "app"
    wxs_path = app_template / "neat.wxs"
    license_path = app_template / "LICENSE.rtf"
    if not wxs_path.is_file() or not license_path.is_file() or not license_path.stat().st_size:
        raise DistributionError("Briefcase MSI template or project license is missing")
    namespace = {"wix": "http://wixtoolset.org/schemas/v4/wxs"}
    try:
        package = ET.parse(wxs_path).getroot().find("wix:Package", namespace)
    except ET.ParseError as exc:
        raise DistributionError(f"Briefcase generated invalid WiX XML: {exc}") from exc
    if package is None:
        raise DistributionError("Briefcase WiX template has no Package element")
    if package.get("Name") != "NEAT" or package.get("Version") != version:
        raise DistributionError("MSI product name/version do not match NEAT project metadata")
    if package.get("UpgradeCode", "").lower() != expected_upgrade_code.lower():
        raise DistributionError("MSI upgrade identity does not match the stable NEAT bundle")
    if package.get("Scope") != "perUserOrMachine":
        raise DistributionError(f"Unexpected MSI package scope: {package.get('Scope')!r}")

    launcher = next(
        (
            element
            for element in ET.parse(wxs_path).iter()
            if element.tag.endswith("Shortcut") and element.get("Target") == "[INSTALLFOLDER]NEAT.exe"
        ),
        None,
    )
    if windows.get("install_launcher") is not True or launcher is None:
        raise DistributionError("The MSI does not provide the required NEAT Start Menu launcher")
    payload_source = str((root / windows["external_package_path"] / "**").resolve())
    if payload_source.replace("/", "\\").casefold() not in wxs_path.read_text(encoding="utf-8").replace("/", "\\").casefold():
        raise DistributionError("The MSI template does not reference the validated PyInstaller payload")

    # External-app mode creates installer metadata but no second Python app tree.
    for generated in (app_template / "src", app_template / "app_packages", app_template / "support"):
        if generated.exists():
            raise DistributionError(f"Briefcase created an unexpected runtime/dependency tree: {generated}")
    print(f"Verified Briefcase external MSI: {expected_msi}")
    print(f"Stable app identity: {bundle_identifier}; project version: {version}")
    print("Verified external payload mode: no Briefcase Python/support/dependency tree")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    manifest_parser = commands.add_parser("manifest", help="write the canonical payload manifest")
    manifest_parser.add_argument("--payload", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)

    portable_parser = commands.add_parser("portable", help="create and verify the portable ZIP")
    portable_parser.add_argument("--payload", type=Path, required=True)
    portable_parser.add_argument("--manifest", type=Path, required=True)
    portable_parser.add_argument("--output", type=Path, required=True)

    zip_parser = commands.add_parser("verify-zip", help="verify a portable ZIP against a manifest")
    zip_parser.add_argument("--archive", type=Path, required=True)
    zip_parser.add_argument("--manifest", type=Path, required=True)

    directory_parser = commands.add_parser(
        "verify-directory", help="verify a directory against a payload manifest"
    )
    directory_parser.add_argument("--payload", type=Path, required=True)
    directory_parser.add_argument("--manifest", type=Path, required=True)

    briefcase_parser = commands.add_parser(
        "verify-briefcase", help="verify the generated external-app MSI configuration"
    )
    briefcase_parser.add_argument("--project-root", type=Path, default=Path.cwd())

    args = parser.parse_args()
    try:
        if args.command == "manifest":
            manifest = write_manifest(args.payload, args.output)
            print(f"Manifest records {len(manifest['files'])} files: {args.output}")
        elif args.command == "portable":
            create_portable_zip(args.payload, args.manifest, args.output)
            print(f"Portable payload verified: {args.output}")
        elif args.command == "verify-zip":
            verify_portable_zip(args.archive, args.manifest)
            print(f"Portable payload matches manifest: {args.archive}")
        elif args.command == "verify-briefcase":
            verify_briefcase_output(args.project_root)
        else:
            verify_directory(args.payload, args.manifest)
            print(f"Directory payload matches manifest: {args.payload}")
    except (DistributionError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate that a production release tag matches installed NEAT metadata."""

from __future__ import annotations

import argparse
import re
from importlib.metadata import version


_RELEASE_TAG = re.compile(r"v(?P<version>\d+\.\d+\.\d+)")


def validate_release_version(tag: str, package_version: str) -> str:
    """Return the validated version or raise ``ValueError`` with a clear cause."""

    match = _RELEASE_TAG.fullmatch(str(tag))
    if match is None:
        raise ValueError("Release tag must use the form vX.Y.Z")
    tag_version = match.group("version")
    if tag_version != package_version:
        raise ValueError(
            f"Release tag version {tag_version} does not match package version "
            f"{package_version}"
        )
    return tag_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag, for example v4.8.2")
    args = parser.parse_args(argv)
    validated = validate_release_version(args.tag, version("NEAT"))
    print(f"Release version validated: {validated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

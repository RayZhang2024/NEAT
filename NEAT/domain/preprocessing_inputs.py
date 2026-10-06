"""Typed in-memory inputs shared by classic preprocessing operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import numpy as np


@dataclass(frozen=True, slots=True, eq=False, init=False, repr=False)
class LoadedImageRun:
    """A structural snapshot of one loaded classic image run.

    The frame mapping and provenance collections are snapshotted and read-only.
    NumPy arrays themselves are intentionally shared by reference and remain
    mutable; this avoids copying or freezing potentially large image payloads.
    """

    primary_source: str
    frames: Mapping[str, np.ndarray] = field(repr=False)
    source_folders: tuple[str, ...]
    load_errors: tuple[str, ...]

    def __init__(
        self,
        primary_source: str,
        frames: Mapping[str, np.ndarray],
        source_folders: Sequence[str] | None = None,
        load_errors: Sequence[str] = (),
    ) -> None:
        if not isinstance(primary_source, str):
            raise TypeError("primary_source must be a string")
        if not primary_source:
            raise ValueError("primary_source must not be empty")
        if not isinstance(frames, Mapping):
            raise TypeError("frames must be a mapping of frame keys to NumPy arrays")

        frame_snapshot = dict(frames)
        for key, array in frame_snapshot.items():
            if not isinstance(key, str):
                raise TypeError("frame keys must be strings")
            if not key:
                raise ValueError("frame keys must not be empty")
            if not isinstance(array, np.ndarray):
                raise TypeError("frame values must be NumPy arrays")

        folder_snapshot: tuple[str, ...]
        if source_folders is None:
            folder_snapshot = (primary_source,)
        else:
            folder_snapshot = _snapshot_strings(
                source_folders, "source_folders", require_non_empty=True
            )
            if not folder_snapshot:
                raise ValueError("source_folders must not be empty when provided")

        error_snapshot = _snapshot_strings(
            load_errors, "load_errors", require_non_empty=False
        )

        object.__setattr__(self, "primary_source", primary_source)
        object.__setattr__(self, "frames", MappingProxyType(frame_snapshot))
        object.__setattr__(self, "source_folders", folder_snapshot)
        object.__setattr__(self, "load_errors", error_snapshot)

    def __repr__(self) -> str:
        return (
            f"LoadedImageRun(primary_source={self.primary_source!r}, "
            f"frame_count={len(self.frames)}, "
            f"source_folders={self.source_folders!r}, "
            f"load_error_count={len(self.load_errors)})"
        )


def _snapshot_strings(
    values: Sequence[str], field_name: str, *, require_non_empty: bool
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be an ordered sequence of strings")

    snapshot = tuple(values)
    for value in snapshot:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must contain only strings")
        if require_non_empty and not value:
            raise ValueError(f"{field_name} values must not be empty")
    return cast(tuple[str, ...], snapshot)

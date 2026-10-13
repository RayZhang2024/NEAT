"""Typed, bounded proposals for optional OpenAI visual pre-fitting review."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .observation import RoiBounds


class VisionStatus(str, Enum):
    NEEDS_REVIEW = "needs_review"
    UNAVAILABLE = "vision_unavailable"
    BLOCKED = "blocked_invalid_evidence"
    CANCELLED = "cancelled"
    STALE = "stale"


@dataclass(frozen=True)
class VisionOutcome:
    run_id: str
    dataset_id: str
    revision: int
    status: VisionStatus
    message: str
    roi: RoiBounds | None = None
    roi_reason: str = ""
    confidence: str = ""
    ambiguity_flags: tuple[str, ...] = ()
    roi_preview_png: bytes | None = None
    spectrum_png: bytes | None = None
    phase_id: str = ""
    hkls: tuple[tuple[int, int, int], ...] = ()
    edge_reasons: tuple[str, ...] = ()
    display_window: tuple[float, float] | None = None
    overlay_png: bytes | None = None
    alternatives: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    inspected_frames: tuple[int, ...] = ()
    inspected_phases: tuple[str, ...] = ()
    request_stages: tuple[str, ...] = ()
    phase_overlays: tuple[tuple[str, bytes], ...] = ()
    phase_reason: str = ""
    display_interval_diagnostic: str = ""


__all__ = ["VisionOutcome", "VisionStatus"]

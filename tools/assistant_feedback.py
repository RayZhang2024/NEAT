"""Privacy-limited local feedback logging for the NEAT assistant."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4


SCHEMA_VERSION = 1
VALID_RATINGS = frozenset({"helpful", "not_helpful"})
WINDOWS_PATH_PATTERN = re.compile(
    r"(?i)(?:file:///|[a-z]:[\\/]|\\\\)[^\s<>\"|?*]+"
)
MAX_LOGGED_QUESTION_CHARACTERS = 2000


def default_feedback_path() -> Path:
    """Return a per-user local path without writing inside the repository."""

    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        base_directory = Path(local_app_data)
    else:
        base_directory = Path.home() / ".local" / "share"
    return base_directory / "NEAT" / "assistant_feedback.jsonl"


def build_feedback_record(
    *,
    rating: str,
    question: str,
    citations: Sequence[Mapping[str, object]],
    route: str,
    requires_human_review: bool,
    neat_version: str,
) -> dict[str, object]:
    """Build an allow-listed record that excludes answers and application context."""

    normalized_rating = str(rating or "").strip().lower()
    if normalized_rating not in VALID_RATINGS:
        raise ValueError(f"Unsupported feedback rating: {rating!r}")

    normalized_question = str(question or "").strip()
    if not normalized_question:
        raise ValueError("Feedback requires the original question")
    normalized_question = WINDOWS_PATH_PATTERN.sub(
        "[local path removed]",
        normalized_question,
    )[:MAX_LOGGED_QUESTION_CHARACTERS]

    safe_citations = []
    for citation in citations:
        source_id = str(citation.get("source_id", "")).strip()
        heading_path = str(citation.get("heading_path", "")).strip()
        if source_id or heading_path:
            safe_citations.append(
                {
                    "source_id": source_id,
                    "heading_path": heading_path,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "feedback_id": str(uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rating": normalized_rating,
        "question": normalized_question,
        "citations": safe_citations,
        "route": str(route or "unknown"),
        "requires_human_review": bool(requires_human_review),
        "neat_version": str(neat_version or "unknown"),
    }


def append_feedback_record(
    record: Mapping[str, object],
    *,
    destination: Path | None = None,
) -> Path:
    """Append one UTF-8 JSON record and return the destination path."""

    output_path = Path(destination or default_feedback_path()).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        dict(record),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.write("\n")
    return output_path


def record_assistant_feedback(**fields) -> Path:
    """Validate and append one allow-listed assistant feedback entry."""

    return append_feedback_record(build_feedback_record(**fields))


__all__ = [
    "SCHEMA_VERSION",
    "VALID_RATINGS",
    "append_feedback_record",
    "build_feedback_record",
    "default_feedback_path",
    "record_assistant_feedback",
]

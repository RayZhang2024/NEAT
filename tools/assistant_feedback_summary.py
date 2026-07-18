"""Create a local human-readable summary of NEAT assistant feedback."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping


def load_feedback_records(path: Path) -> tuple[list[dict], int]:
    """Load valid feedback objects and count unreadable lines."""

    records: list[dict] = []
    invalid_lines = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except (TypeError, ValueError):
                invalid_lines += 1
                continue
            if not isinstance(record, dict):
                invalid_lines += 1
                continue
            if record.get("rating") not in {"helpful", "not_helpful"}:
                invalid_lines += 1
                continue
            records.append(record)
    return records, invalid_lines


def summarize_feedback(
    records: Iterable[Mapping[str, object]],
    *,
    invalid_lines: int = 0,
) -> dict[str, object]:
    """Return deterministic aggregate counts for a feedback collection."""

    items = [dict(record) for record in records]
    ratings = Counter(str(item.get("rating", "unknown")) for item in items)
    routes = Counter(str(item.get("route", "unknown")) for item in items)
    sources: Counter[str] = Counter()
    for item in items:
        for citation in item.get("citations", []) or []:
            if not isinstance(citation, Mapping):
                continue
            source_id = str(citation.get("source_id", "")).strip()
            if source_id:
                sources[source_id] += 1

    total = len(items)
    helpful = ratings["helpful"]
    helpful_rate = helpful / total if total else None
    latest_by_question: dict[str, dict] = {}
    for item in items:
        question_key = " ".join(
            str(item.get("question", "")).casefold().split()
        )
        if question_key:
            latest_by_question[question_key] = item
    latest_records = list(latest_by_question.values())
    latest_ratings = Counter(
        str(item.get("rating", "unknown")) for item in latest_records
    )
    latest_total = len(latest_records)
    latest_helpful = latest_ratings["helpful"]
    latest_helpful_rate = (
        latest_helpful / latest_total if latest_total else None
    )
    return {
        "total": total,
        "helpful": helpful,
        "not_helpful": ratings["not_helpful"],
        "helpful_rate": helpful_rate,
        "latest_total": latest_total,
        "latest_helpful": latest_helpful,
        "latest_not_helpful": latest_ratings["not_helpful"],
        "latest_helpful_rate": latest_helpful_rate,
        "invalid_lines": int(invalid_lines),
        "routes": dict(sorted(routes.items())),
        "sources": dict(sources.most_common()),
        "not_helpful_records": [
            item for item in latest_records if item.get("rating") == "not_helpful"
        ],
    }


def _escape_markdown_cell(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_feedback_summary(summary: Mapping[str, object]) -> str:
    """Render a concise Markdown pilot report."""

    total = int(summary.get("total", 0))
    helpful = int(summary.get("helpful", 0))
    not_helpful = int(summary.get("not_helpful", 0))
    helpful_rate = summary.get("helpful_rate")
    rate_text = "N/A" if helpful_rate is None else f"{float(helpful_rate):.1%}"
    latest_helpful_rate = summary.get("latest_helpful_rate")
    latest_rate_text = (
        "N/A"
        if latest_helpful_rate is None
        else f"{float(latest_helpful_rate):.1%}"
    )

    lines = [
        "# NEAT Assistant Pilot Feedback Summary",
        "",
        "## Current result (latest rating for each question)",
        "",
        f"- Distinct rated questions: {int(summary.get('latest_total', 0))}",
        f"- Currently helpful: {int(summary.get('latest_helpful', 0))}",
        f"- Currently not helpful: {int(summary.get('latest_not_helpful', 0))}",
        f"- Current helpful rate: {latest_rate_text}",
        "",
        "## Rating history",
        "",
        f"- Rated answers: {total}",
        f"- Helpful: {helpful}",
        f"- Not helpful: {not_helpful}",
        f"- Historical helpful rate: {rate_text}",
        f"- Ignored malformed log lines: {int(summary.get('invalid_lines', 0))}",
        "",
        "## Coverage by route",
        "",
        "| Route | Ratings |",
        "|---|---:|",
    ]
    routes = summary.get("routes", {}) or {}
    if routes:
        for route, count in routes.items():
            lines.append(f"| {_escape_markdown_cell(route)} | {int(count)} |")
    else:
        lines.append("| No feedback yet | 0 |")

    lines.extend(
        [
            "",
            "## Most frequently cited sources",
            "",
            "| Source | Times cited |",
            "|---|---:|",
        ]
    )
    sources = summary.get("sources", {}) or {}
    if sources:
        for source_id, count in list(sources.items())[:10]:
            lines.append(
                f"| `{_escape_markdown_cell(source_id)}` | {int(count)} |"
            )
    else:
        lines.append("| No cited sources yet | 0 |")

    lines.extend(["", "## Currently unresolved Not-helpful answers", ""])
    not_helpful_records = summary.get("not_helpful_records", []) or []
    if not_helpful_records:
        lines.extend(
            [
                "| Time (UTC) | Route | Question | Verified sources |",
                "|---|---|---|---|",
            ]
        )
        for record in not_helpful_records:
            source_ids = []
            for citation in record.get("citations", []) or []:
                if isinstance(citation, Mapping):
                    source_id = str(citation.get("source_id", "")).strip()
                    if source_id:
                        source_ids.append(source_id)
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown_cell(record.get("created_at_utc", "")),
                        _escape_markdown_cell(record.get("route", "unknown")),
                        _escape_markdown_cell(record.get("question", "")),
                        _escape_markdown_cell(", ".join(source_ids) or "None"),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No answers were rated Not helpful.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Review every Not-helpful answer individually; the rating alone does not identify the cause.",
            "- A missing or irrelevant verified source usually indicates a documentation or retrieval gap.",
            "- Relevant sources with a poor answer usually indicate an answer-generation or clarity problem.",
            "- Scientific overclaiming or unsafe certainty should be corrected before expanding the pilot.",
            "- For an initial pilot, collect at least 20 rated answers across several routes before treating the helpful rate as meaningful.",
            "",
        ]
    )
    return "\n".join(lines)


def save_feedback_summary(summary: Mapping[str, object], path: Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_feedback_summary(summary), encoding="utf-8")
    return output_path


__all__ = [
    "load_feedback_records",
    "render_feedback_summary",
    "save_feedback_summary",
    "summarize_feedback",
]

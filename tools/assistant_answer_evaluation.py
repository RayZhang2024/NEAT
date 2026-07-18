"""Reusable reporting helpers for live NEAT assistant answer evaluation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from tools.assistant_retrieval import tokenize


REPORT_SCHEMA_VERSION = 1
MANUAL_REVIEW_FIELDS = (
    "correctness",
    "groundedness",
    "clarity",
    "scientific_scope",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key_point_match(answer: str, key_point: str) -> dict:
    answer_tokens = set(tokenize(answer))
    point_tokens = set(tokenize(key_point))
    matched = sorted(answer_tokens.intersection(point_tokens))
    coverage = len(matched) / len(point_tokens) if point_tokens else 0.0
    return {
        "key_point": key_point,
        "token_coverage": round(coverage, 3),
        "heuristic_match": coverage >= 0.6,
        "matched_tokens": matched,
    }


def build_answer_evaluation_record(question: Mapping[str, object], result) -> dict:
    """Combine a live answer with deterministic and heuristic review fields."""

    expected_sources = [str(item).lower() for item in question["expected_sources"]]
    citations = [
        {
            "number": citation.number,
            "source_id": citation.source_id,
            "heading_path": citation.heading_path,
        }
        for citation in result.citations
    ]
    cited_sources = {item["source_id"].lower() for item in citations}
    source_hit = bool(cited_sources.intersection(expected_sources))
    behavior = str(question["expected_behavior"])
    escalation_alignment = None
    if behavior == "clarify_and_escalate":
        escalation_alignment = bool(
            result.requires_human_review and result.route == "escalation"
        )

    key_point_checks = [
        _key_point_match(result.answer, str(key_point))
        for key_point in question["key_points"]
    ]
    heuristic_key_point_rate = (
        sum(check["heuristic_match"] for check in key_point_checks)
        / len(key_point_checks)
        if key_point_checks
        else 0.0
    )
    automatic_pass = source_hit and escalation_alignment is not False

    return {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "expected_behavior": behavior,
        "expected_sources": question["expected_sources"],
        "expected_key_points": question["key_points"],
        "answer": result.answer,
        "route": result.route,
        "route_confidence": result.route_confidence,
        "requires_human_review": result.requires_human_review,
        "citations": citations,
        "automatic_checks": {
            "expected_source_cited": source_hit,
            "mandatory_escalation_aligned": escalation_alignment,
            "heuristic_key_point_rate": round(heuristic_key_point_rate, 3),
            "key_points": key_point_checks,
            "automatic_pass": automatic_pass,
            "note": (
                "Key-point coverage is a lexical heuristic and is not a scientific "
                "correctness score."
            ),
        },
        "manual_review": {
            **{field: None for field in MANUAL_REVIEW_FIELDS},
            "notes": "",
        },
        "status": "answered",
        "evaluated_at": utc_now(),
    }


def build_error_record(question: Mapping[str, object], message: str) -> dict:
    return {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "expected_behavior": question["expected_behavior"],
        "expected_sources": question["expected_sources"],
        "expected_key_points": question["key_points"],
        "status": "error",
        "error": str(message),
        "evaluated_at": utc_now(),
        "manual_review": {
            **{field: None for field in MANUAL_REVIEW_FIELDS},
            "notes": "",
        },
    }


def summarize_results(results: Sequence[Mapping[str, object]]) -> dict:
    answered = [item for item in results if item.get("status") == "answered"]
    errors = [item for item in results if item.get("status") == "error"]
    source_hits = sum(
        bool(item.get("automatic_checks", {}).get("expected_source_cited"))
        for item in answered
    )
    automatic_passes = sum(
        bool(item.get("automatic_checks", {}).get("automatic_pass"))
        for item in answered
    )
    key_point_rates = [
        float(item.get("automatic_checks", {}).get("heuristic_key_point_rate", 0.0))
        for item in answered
    ]
    return {
        "result_count": len(results),
        "answered_count": len(answered),
        "error_count": len(errors),
        "expected_source_citation_rate": (
            source_hits / len(answered) if answered else 0.0
        ),
        "automatic_pass_rate": (
            automatic_passes / len(answered) if answered else 0.0
        ),
        "mean_heuristic_key_point_rate": (
            sum(key_point_rates) / len(key_point_rates) if key_point_rates else 0.0
        ),
        "manual_reviews_completed": sum(
            all(item.get("manual_review", {}).get(field) is not None for field in MANUAL_REVIEW_FIELDS)
            for item in answered
        ),
    }


def create_report(*, model: str, knowledge_fingerprint: str) -> dict:
    now = utc_now()
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "model": model,
        "knowledge_fingerprint": knowledge_fingerprint,
        "started_at": now,
        "updated_at": now,
        "summary": summarize_results([]),
        "results": [],
    }


def update_report(report: dict, record: Mapping[str, object]) -> None:
    existing_index = next(
        (
            index
            for index, item in enumerate(report["results"])
            if item.get("id") == record.get("id")
        ),
        None,
    )
    if existing_index is None:
        report["results"].append(dict(record))
    else:
        report["results"][existing_index] = dict(record)
    report["updated_at"] = utc_now()
    report["summary"] = summarize_results(report["results"])


def save_report(report: Mapping[str, object], path: Path) -> None:
    """Atomically save progress so interrupted live runs can resume safely."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported answer-evaluation report schema")
    if not isinstance(report.get("results"), list):
        raise ValueError("Answer-evaluation report has no results list")
    return report


def _markdown_text(value: object) -> str:
    return str(value).replace("\r", "").strip()


def render_markdown_report(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# NEAT Assistant Answer Evaluation",
        "",
        f"- Model: `{report['model']}`",
        f"- Knowledge fingerprint: `{report['knowledge_fingerprint']}`",
        f"- Answered: {summary['answered_count']}",
        f"- Errors: {summary['error_count']}",
        (
            "- Expected-source citation rate: "
            f"{summary['expected_source_citation_rate']:.1%}"
        ),
        (
            "- Heuristic key-point rate: "
            f"{summary['mean_heuristic_key_point_rate']:.1%} "
            "(not a correctness score)"
        ),
        "",
        "## Human review rubric",
        "",
        "Score each dimension from 0 to 2:",
        "",
        "- Correctness: 0 incorrect, 1 partly correct, 2 correct.",
        "- Groundedness: 0 unsupported, 1 partly supported, 2 fully supported.",
        "- Clarity: 0 confusing, 1 usable, 2 clear and actionable.",
        "- Scientific scope: 0 unsafe/overclaimed, 1 minor issue, 2 appropriately bounded.",
        "",
    ]

    for item in report["results"]:
        lines.extend([f"## {item['id']}: {_markdown_text(item['question'])}", ""])
        if item.get("status") == "error":
            lines.extend([f"**Error:** {_markdown_text(item.get('error', 'Unknown'))}", ""])
            continue

        citations = "; ".join(
            f"{citation['heading_path']} (`{citation['source_id']}`)"
            for citation in item["citations"]
        )
        checks = item["automatic_checks"]
        lines.extend(
            [
                f"**Route:** `{item['route']}`",
                "",
                f"**Verified sources:** {citations}",
                "",
                f"**Expected source cited:** {'Yes' if checks['expected_source_cited'] else 'No'}",
                "",
                "**Answer**",
                "",
                _markdown_text(item["answer"]),
                "",
                "**Expected key points**",
                "",
                *[f"- {point}" for point in item["expected_key_points"]],
                "",
                "**Human scores (0–2)**",
                "",
                "- Correctness: ",
                "- Groundedness: ",
                "- Clarity: ",
                "- Scientific scope: ",
                "- Notes: ",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def save_markdown_report(report: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(report), encoding="utf-8")

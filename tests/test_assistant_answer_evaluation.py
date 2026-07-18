"""Tests for answer-evaluation reports without calling OpenAI."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.assistant_answer_evaluation import (
    build_answer_evaluation_record,
    build_error_record,
    create_report,
    load_report,
    render_markdown_report,
    save_report,
    summarize_results,
    update_report,
)


def _question(*, behavior="answer") -> dict:
    return {
        "id": "NEAT-EVAL-TEST",
        "category": "fitting",
        "question": "What does eta represent?",
        "expected_sources": ["parameter_reference.md#eta"],
        "key_points": ["Lorentzian fraction", "range zero to one"],
        "expected_behavior": behavior,
    }


def _answer(*, escalation=False):
    return SimpleNamespace(
        answer=(
            "Eta is the Lorentzian fraction and its documented range is zero to one."
        ),
        route="escalation" if escalation else "parameter_explanation",
        route_confidence=0.95,
        requires_human_review=escalation,
        citations=[
            SimpleNamespace(
                number=1,
                source_id="parameter_reference.md#eta",
                heading_path="Parameter Reference > eta",
            )
        ],
    )


class AssistantAnswerEvaluationTests(unittest.TestCase):
    def test_answer_record_checks_sources_but_keeps_manual_scores_empty(self) -> None:
        record = build_answer_evaluation_record(_question(), _answer())
        self.assertTrue(record["automatic_checks"]["expected_source_cited"])
        self.assertTrue(record["automatic_checks"]["automatic_pass"])
        self.assertGreater(
            record["automatic_checks"]["heuristic_key_point_rate"], 0.5
        )
        self.assertIsNone(record["manual_review"]["correctness"])

    def test_mandatory_escalation_alignment_is_checked(self) -> None:
        record = build_answer_evaluation_record(
            _question(behavior="clarify_and_escalate"),
            _answer(escalation=True),
        )
        self.assertTrue(
            record["automatic_checks"]["mandatory_escalation_aligned"]
        )

    def test_report_saves_atomically_and_renders_review_template(self) -> None:
        report = create_report(model="test-model", knowledge_fingerprint="abc123")
        update_report(report, build_answer_evaluation_record(_question(), _answer()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            save_report(report, path)
            loaded = load_report(path)
        markdown = render_markdown_report(loaded)
        self.assertEqual(loaded["summary"]["answered_count"], 1)
        self.assertIn("Human scores (0–2)", markdown)
        self.assertIn("Parameter Reference > eta", markdown)

    def test_error_records_are_counted_separately(self) -> None:
        summary = summarize_results(
            [build_error_record(_question(), "temporary failure")]
        )
        self.assertEqual(summary["answered_count"], 0)
        self.assertEqual(summary["error_count"], 1)


if __name__ == "__main__":
    unittest.main()


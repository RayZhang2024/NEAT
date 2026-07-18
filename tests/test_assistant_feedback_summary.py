"""Tests for local assistant pilot feedback summaries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.assistant_feedback_summary import (
    load_feedback_records,
    render_feedback_summary,
    summarize_feedback,
)


class AssistantFeedbackSummaryTests(unittest.TestCase):
    def test_load_ignores_malformed_and_unsupported_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"rating": "helpful", "question": "Good"}),
                        "{not json",
                        json.dumps({"rating": "maybe", "question": "Unknown"}),
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            records, invalid_lines = load_feedback_records(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(invalid_lines, 2)

    def test_summary_calculates_rates_routes_and_sources(self) -> None:
        records = [
            {
                "rating": "helpful",
                "route": "how_to",
                "question": "How?",
                "citations": [{"source_id": "faq.md#workflow"}],
            },
            {
                "rating": "not_helpful",
                "route": "troubleshooting",
                "question": "Why?",
                "citations": [{"source_id": "faq.md#workflow"}],
            },
        ]
        summary = summarize_feedback(records)

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["helpful"], 1)
        self.assertEqual(summary["not_helpful"], 1)
        self.assertEqual(summary["helpful_rate"], 0.5)
        self.assertEqual(summary["latest_total"], 2)
        self.assertEqual(summary["latest_helpful_rate"], 0.5)
        self.assertEqual(summary["routes"]["how_to"], 1)
        self.assertEqual(summary["sources"]["faq.md#workflow"], 2)

    def test_latest_rating_resolves_an_earlier_negative_rating(self) -> None:
        summary = summarize_feedback(
            [
                {
                    "rating": "not_helpful",
                    "question": "How do I configure the Edge Table?",
                },
                {
                    "rating": "helpful",
                    "question": "  HOW do I configure the Edge Table? ",
                },
            ]
        )

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["helpful_rate"], 0.5)
        self.assertEqual(summary["latest_total"], 1)
        self.assertEqual(summary["latest_helpful_rate"], 1.0)
        self.assertEqual(summary["not_helpful_records"], [])

    def test_markdown_lists_not_helpful_questions_without_answers(self) -> None:
        summary = summarize_feedback(
            [
                {
                    "rating": "not_helpful",
                    "route": "troubleshooting",
                    "question": "Why did this fail?",
                    "created_at_utc": "2026-07-16T00:00:00+00:00",
                    "citations": [{"source_id": "troubleshooting.md#fit"}],
                }
            ]
        )
        rendered = render_feedback_summary(summary)

        self.assertIn("Current helpful rate: 0.0%", rendered)
        self.assertIn("Why did this fail?", rendered)
        self.assertIn("troubleshooting.md#fit", rendered)
        self.assertNotIn("generated answer", rendered.lower())


if __name__ == "__main__":
    unittest.main()

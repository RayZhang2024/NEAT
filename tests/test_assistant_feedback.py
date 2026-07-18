"""Tests for privacy-limited local assistant feedback."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.assistant_feedback import (
    append_feedback_record,
    build_feedback_record,
)


class AssistantFeedbackTests(unittest.TestCase):
    def test_record_contains_only_allow_listed_feedback_fields(self) -> None:
        record = build_feedback_record(
            rating="helpful",
            question="How do I choose a macro-pixel?",
            citations=[
                {
                    "source_id": "faq.md#what-is-a-macro-pixel",
                    "heading_path": "FAQ > What is a macro-pixel?",
                    "filename": "faq.md",
                    "private_path": "C:/private/data.fits",
                }
            ],
            route="how_to",
            requires_human_review=False,
            neat_version="4.7.4",
        )

        self.assertEqual(record["rating"], "helpful")
        self.assertEqual(
            record["citations"],
            [
                {
                    "source_id": "faq.md#what-is-a-macro-pixel",
                    "heading_path": "FAQ > What is a macro-pixel?",
                }
            ],
        )
        self.assertNotIn("answer", record)
        self.assertNotIn("context", record)
        self.assertNotIn("private_path", str(record))

    def test_windows_paths_typed_in_a_question_are_redacted(self) -> None:
        record = build_feedback_record(
            rating="not_helpful",
            question=(
                r"Why can NEAT not open C:\Users\person\private\sample.fits "
                r"or \\server\experiment\sample.fits?"
            ),
            citations=[],
            route="bug_report",
            requires_human_review=True,
            neat_version="4.7.4",
        )

        self.assertNotIn(r"C:\Users", record["question"])
        self.assertNotIn(r"\\server", record["question"])
        self.assertEqual(record["question"].count("[local path removed]"), 2)

    def test_record_is_appended_as_utf8_json_line(self) -> None:
        record = build_feedback_record(
            rating="not_helpful",
            question="What does eta mean?",
            citations=[],
            route="parameter_explanation",
            requires_human_review=False,
            neat_version="4.7.4",
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "feedback.jsonl"
            returned_path = append_feedback_record(
                record,
                destination=destination,
            )
            saved = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(returned_path, destination)
        self.assertEqual(saved["feedback_id"], record["feedback_id"])
        self.assertEqual(saved["question"], "What does eta mean?")

    def test_invalid_rating_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_feedback_record(
                rating="maybe",
                question="Question",
                citations=[],
                route="how_to",
                requires_human_review=False,
                neat_version="4.7.4",
            )


if __name__ == "__main__":
    unittest.main()

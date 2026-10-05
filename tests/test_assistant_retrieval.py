"""Tests for the NEAT assistant retrieval baseline."""

from __future__ import annotations

import unittest
from pathlib import Path

from NEAT.package_resources import assistant_knowledge_root
from tools.assistant_retrieval import (
    BM25Retriever,
    evaluate_retriever,
    github_anchor,
    load_evaluation_questions,
    load_knowledge_base,
    validate_expected_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIRECTORY = assistant_knowledge_root()
QUESTIONS_PATH = PROJECT_ROOT / "docs" / "assistant" / "evaluation_questions.json"


class AssistantRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sections = load_knowledge_base(KNOWLEDGE_DIRECTORY)
        cls.questions = load_evaluation_questions(QUESTIONS_PATH)
        cls.retriever = BM25Retriever(cls.sections)

    def test_github_anchor_matches_curated_source_ids(self) -> None:
        self.assertEqual(github_anchor("Window half (`n`)"), "window-half-n")
        self.assertEqual(github_anchor("`eta`"), "eta")
        self.assertEqual(
            github_anchor("Manual anchor: Wavelength or ToF value"),
            "manual-anchor-wavelength-or-tof-value",
        )

    def test_every_expected_source_exists(self) -> None:
        self.assertEqual(
            validate_expected_sources(self.sections, self.questions),
            [],
        )

    def test_curated_technical_sources_are_explicitly_loaded(self) -> None:
        filenames = {section.filename for section in self.sections}
        self.assertTrue(
            {
                "software_ui_guide.md",
                "menu_reference.md",
                "fitting_ui_reference.md",
                "preprocessing_technical.md",
                "fitting_technical.md",
                "mapping_technical.md",
                "postprocessing_technical.md",
                "known_limitations.md",
            }.issubset(filenames)
        )

    def test_macro_pixel_question_retrieves_macro_pixel_faq(self) -> None:
        results = self.retriever.search(
            "What is a macro-pixel and why would I make it larger?",
            limit=3,
        )
        source_ids = {result.section.source_id for result in results}
        self.assertIn("faq.md#what-is-a-macro-pixel", source_ids)

    def test_baseline_retrieval_hit_at_three(self) -> None:
        report = evaluate_retriever(self.retriever, self.questions, limit=3)
        self.assertGreaterEqual(report["hit_at_limit"], 0.75)


if __name__ == "__main__":
    unittest.main()

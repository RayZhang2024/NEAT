"""Tests for NEAT assistant intent and safety routing."""

from __future__ import annotations

import unittest
from pathlib import Path

from tools.assistant_query_pipeline import ESCALATION_SOURCE_ID, RoutedRetriever
from tools.assistant_retrieval import SearchResult, load_knowledge_base
from tools.assistant_router import QuestionRoute, QuestionRouter
from tools.evaluate_assistant_router import evaluate_router, load_routing_questions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "docs" / "assistant"
ROUTING_QUESTIONS = KNOWLEDGE_DIRECTORY / "routing_questions.json"


class _FakeRetriever:
    def __init__(self, sections):
        self.sections = sections

    def search(self, query: str, *, limit: int = 3):
        unrelated = [
            section
            for section in self.sections
            if section.source_id != ESCALATION_SOURCE_ID
        ][:limit]
        return [SearchResult(section=section, score=0.5) for section in unrelated]


class AssistantRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.router = QuestionRouter()
        cls.sections = load_knowledge_base(KNOWLEDGE_DIRECTORY)
        cls.questions = load_routing_questions(ROUTING_QUESTIONS)

    def test_guarantee_request_requires_escalation(self) -> None:
        decision = self.router.route(
            "Give me settings that guarantee my strain map is scientifically correct."
        )
        self.assertEqual(decision.route, QuestionRoute.ESCALATION)
        self.assertTrue(decision.requires_human_review)

    def test_traceback_routes_to_bug_report(self) -> None:
        decision = self.router.route("I get a ModuleNotFoundError traceback on startup")
        self.assertEqual(decision.route, QuestionRoute.BUG_REPORT)

    def test_noisy_fit_routes_to_troubleshooting(self) -> None:
        decision = self.router.route("My Bragg-edge fit is noisy and unstable")
        self.assertEqual(decision.route, QuestionRoute.TROUBLESHOOTING)

    def test_eta_question_routes_to_parameter_explanation(self) -> None:
        decision = self.router.route("What does eta represent?")
        self.assertEqual(decision.route, QuestionRoute.PARAMETER_EXPLANATION)

    def test_inferring_stress_from_strain_requires_scientific_review(self) -> None:
        decision = self.router.route(
            "Can I trust the mm scale and infer stress directly from the strain map?"
        )
        self.assertEqual(decision.route, QuestionRoute.SCIENTIFIC_INTERPRETATION)
        self.assertTrue(decision.requires_human_review)

    def test_escalation_source_is_pinned_before_semantic_results(self) -> None:
        routed = RoutedRetriever(_FakeRetriever(self.sections)).route_and_search(
            "Give me settings that guarantee a scientifically correct map.",
            limit=3,
        )
        self.assertEqual(routed.results[0].section.source_id, ESCALATION_SOURCE_ID)

    def test_routing_evaluation_quality_and_safety_recall(self) -> None:
        report = evaluate_router(self.router, self.questions)
        self.assertGreaterEqual(report["accuracy"], 0.9)
        self.assertEqual(report["escalation_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()

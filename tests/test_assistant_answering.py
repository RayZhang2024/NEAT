"""Tests for grounded NEAT assistant answer generation."""

from __future__ import annotations

import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tools.assistant_answering import (
    ConversationTurn,
    GroundedAnswerGenerator,
    GroundedPromptBuilder,
)
from tools.assistant_query_pipeline import ESCALATION_SOURCE_ID, RoutedRetriever
from tools.assistant_retrieval import BM25Retriever, load_knowledge_base
from tools.assistant_service import NEATAssistantService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "docs" / "assistant"


class _RecordingModel:
    def __init__(self, response: str = "A grounded answer.") -> None:
        self.response = response
        self.messages = None
        self.config = None

    def invoke(self, messages, config=None):
        self.messages = messages
        self.config = config
        return AIMessage(content=self.response)


class AssistantAnsweringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sections = load_knowledge_base(KNOWLEDGE_DIRECTORY)
        cls.retriever = BM25Retriever(sections)

    def test_prompt_contains_policy_question_context_and_approved_sources(self) -> None:
        routed = RoutedRetriever(self.retriever).route_and_search(
            "What does eta represent?", limit=2
        )
        prepared = GroundedPromptBuilder().prepare(
            "What does eta represent?",
            routed,
            context={"current_module": "fitting", "raw_object": object()},
        )
        self.assertIsInstance(prepared.messages[0], SystemMessage)
        request = prepared.messages[-1].content
        self.assertIn("ROUTE-SPECIFIC RESPONSE POLICY", request)
        self.assertIn("What does eta represent?", request)
        self.assertIn("current_module: fitting", request)
        self.assertNotIn("raw_object", request)
        self.assertIn("<approved_source", request)

    def test_only_supported_history_roles_are_sent_and_history_is_limited(self) -> None:
        routed = RoutedRetriever(self.retriever).route_and_search(
            "How do I start fitting?", limit=1
        )
        history = [
            ConversationTurn("user", "first"),
            ConversationTurn("assistant", "second"),
            ConversationTurn("tool", "must not appear"),
            ConversationTurn("user", "last"),
        ]
        prepared = GroundedPromptBuilder(maximum_history_turns=3).prepare(
            "How do I start fitting?", routed, history=history
        )
        self.assertFalse(any("must not appear" in str(message.content) for message in prepared.messages))
        self.assertTrue(any(isinstance(message, HumanMessage) and message.content == "last" for message in prepared.messages))

    def test_citations_come_from_retrieval_not_the_model(self) -> None:
        routed = RoutedRetriever(self.retriever).route_and_search(
            "What is a macro-pixel?", limit=2
        )
        model = _RecordingModel("Use fake source [99].")
        answer = GroundedAnswerGenerator(model).answer(
            "What is a macro-pixel?", routed
        )
        expected = [result.section.source_id for result in routed.results]
        self.assertEqual([item.source_id for item in answer.citations], expected)
        self.assertEqual([item.number for item in answer.citations], [1, 2])

    def test_escalation_remains_human_review_and_pins_guidance(self) -> None:
        model = _RecordingModel("Human review is required.")
        service = NEATAssistantService(self.retriever, model)
        answer = service.ask(
            "Give me settings that guarantee a scientifically correct strain map.",
            source_limit=3,
        )
        self.assertEqual(answer.route, "escalation")
        self.assertTrue(answer.requires_human_review)
        self.assertEqual(answer.citations[0].source_id, ESCALATION_SOURCE_ID)

    def test_model_receives_langchain_messages_and_config(self) -> None:
        model = _RecordingModel()
        service = NEATAssistantService(self.retriever, model)
        answer = service.ask(
            "How do I load a project?",
            config={"tags": ["unit-test"]},
        )
        self.assertEqual(answer.answer, "A grounded answer.")
        self.assertIsInstance(model.messages[0], SystemMessage)
        self.assertIsInstance(model.messages[-1], HumanMessage)
        self.assertEqual(model.config, {"tags": ["unit-test"]})

    def test_empty_model_response_is_rejected(self) -> None:
        routed = RoutedRetriever(self.retriever).route_and_search(
            "How do I load a project?", limit=1
        )
        with self.assertRaises(RuntimeError):
            GroundedAnswerGenerator(_RecordingModel(" ")).answer(
                "How do I load a project?", routed
            )


if __name__ == "__main__":
    unittest.main()


"""End-to-end retrieval and answer service for the NEAT assistant."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from tools.assistant_answering import (
    ChatModel,
    ConversationTurn,
    GroundedAnswer,
    GroundedAnswerGenerator,
)
from tools.assistant_query_pipeline import RoutedRetriever
from tools.assistant_retrieval import SectionRetriever
from tools.assistant_router import QuestionRouter


class NEATAssistantService:
    """Combine safety routing, approved retrieval and grounded generation."""

    def __init__(
        self,
        retriever: SectionRetriever,
        model: ChatModel,
        *,
        router: Optional[QuestionRouter] = None,
    ) -> None:
        self.routed_retriever = RoutedRetriever(retriever, router=router)
        self.answer_generator = GroundedAnswerGenerator(model)

    def ask(
        self,
        question: str,
        *,
        context: Optional[Mapping[str, object]] = None,
        history: Optional[Sequence[ConversationTurn]] = None,
        source_limit: int = 3,
        config: Optional[Mapping[str, object]] = None,
    ) -> GroundedAnswer:
        routed = self.routed_retriever.route_and_search(
            question,
            limit=source_limit,
            context=context,
        )
        return self.answer_generator.answer(
            question,
            routed,
            context=context,
            history=history,
            config=config,
        )


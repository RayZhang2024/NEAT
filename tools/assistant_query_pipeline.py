"""Route NEAT questions and retrieve sources under the selected policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from tools.assistant_retrieval import (
    KnowledgeSection,
    SearchResult,
    SectionRetriever,
)
from tools.assistant_router import QuestionRoute, QuestionRouter, RoutingDecision


ESCALATION_SOURCE_ID = (
    "troubleshooting.md#when-should-the-assistant-escalate-instead-of-answering"
)


@dataclass(frozen=True)
class RoutedSearchResult:
    """Question route plus the policy-aware retrieved knowledge sections."""

    decision: RoutingDecision
    results: Sequence[SearchResult]


class RoutedRetriever:
    """Apply intent and safety routing before invoking a section retriever."""

    def __init__(
        self,
        retriever: SectionRetriever,
        *,
        router: Optional[QuestionRouter] = None,
    ) -> None:
        self.retriever = retriever
        self.router = router or QuestionRouter()
        self._sections = {
            section.source_id: section for section in retriever.sections
        }

    def route_and_search(
        self,
        question: str,
        *,
        limit: int = 3,
        context: Optional[Mapping[str, object]] = None,
    ) -> RoutedSearchResult:
        if limit <= 0:
            raise ValueError("Search limit must be positive")

        decision = self.router.route(question, context=context)
        retrieved = list(self.retriever.search(question, limit=limit))

        if decision.route is QuestionRoute.ESCALATION:
            retrieved = self._pin_source(
                retrieved,
                source_id=ESCALATION_SOURCE_ID,
                limit=limit,
            )

        return RoutedSearchResult(decision=decision, results=retrieved)

    def _pin_source(
        self,
        results: Sequence[SearchResult],
        *,
        source_id: str,
        limit: int,
    ) -> list[SearchResult]:
        section: Optional[KnowledgeSection] = self._sections.get(source_id)
        if section is None:
            return list(results[:limit])

        remaining = [
            result for result in results if result.section.source_id != source_id
        ]
        return [SearchResult(section=section, score=1.0), *remaining][:limit]


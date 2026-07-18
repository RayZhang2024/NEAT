"""Grounded, provider-neutral answer generation for the NEAT assistant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except ImportError as exc:  # pragma: no cover - optional assistant installation
    raise ImportError(
        "Answer generation requires the optional assistant dependencies. "
        "Install them with: python -m pip install -e \".[assistant]\""
    ) from exc

from tools.assistant_query_pipeline import RoutedSearchResult


SYSTEM_INSTRUCTIONS = """You are the support assistant for NEAT Bragg-edge imaging software.

Rules you must follow:
- Answer NEAT-specific questions only from the APPROVED NEAT SOURCES supplied below.
- Do not invent controls, menu items, parameters, defaults, capabilities, or scientific results.
- If the sources are insufficient, say what is unknown and recommend contacting a NEAT developer.
- Keep documented NEAT behaviour separate from general scientific interpretation.
- Never claim that convergence proves scientific validity or that settings guarantee a correct result.
- Do not claim to have inspected files, images, plots, logs, or application state unless their contents are explicitly supplied.
- Follow the route-specific response policy.
- Do not create a bibliography or source identifiers. The application attaches verified citations separately.
- Be concise, practical, and transparent about uncertainty.
"""


@dataclass(frozen=True)
class ConversationTurn:
    """One safe conversational turn supplied to the answer model."""

    role: str
    content: str


@dataclass(frozen=True)
class SourceCitation:
    """Citation metadata derived from retrieval rather than model output."""

    number: int
    source_id: str
    filename: str
    heading: str
    heading_path: str
    anchor: str


@dataclass(frozen=True)
class PreparedAnswerRequest:
    """Messages and verified source metadata ready for a chat model."""

    messages: Sequence[BaseMessage]
    citations: Sequence[SourceCitation]
    routed_search: RoutedSearchResult


@dataclass(frozen=True)
class GroundedAnswer:
    """Answer text plus application-controlled route and citations."""

    answer: str
    route: str
    route_confidence: float
    requires_human_review: bool
    citations: Sequence[SourceCitation]
    shared_remaining: Optional[int] = None
    shared_daily_limit: Optional[int] = None
    shared_reset_at_utc: Optional[str] = None


class ChatModel(Protocol):
    """Minimal interface implemented by LangChain-compatible chat models."""

    def invoke(self, input, config=None):
        """Return a LangChain message or message-like response."""
        ...


def _format_application_context(context: Mapping[str, object]) -> str:
    if not context:
        return "No application context was supplied."

    safe_lines = []
    for key in sorted(context):
        value = context[key]
        if value is None or isinstance(value, (str, int, float, bool)):
            safe_lines.append(f"- {key}: {value}")
    return "\n".join(safe_lines) or "No safe application context was supplied."


def _source_block(number: int, source) -> str:
    section = source.section
    return (
        f"<approved_source number=\"{number}\" "
        f"source_id=\"{section.source_id}\">\n"
        f"Heading: {section.heading_path}\n"
        f"Content:\n{section.text}\n"
        "</approved_source>"
    )


class GroundedPromptBuilder:
    """Build a strict prompt from a routed search result."""

    def __init__(self, *, maximum_history_turns: int = 6) -> None:
        if maximum_history_turns < 0:
            raise ValueError("maximum_history_turns cannot be negative")
        self.maximum_history_turns = maximum_history_turns

    def prepare(
        self,
        question: str,
        routed_search: RoutedSearchResult,
        *,
        context: Optional[Mapping[str, object]] = None,
        history: Optional[Sequence[ConversationTurn]] = None,
    ) -> PreparedAnswerRequest:
        question = str(question or "").strip()
        if not question:
            raise ValueError("A non-empty question is required")
        if not routed_search.results:
            raise ValueError("At least one approved source is required")

        citations = [
            SourceCitation(
                number=index,
                source_id=result.section.source_id,
                filename=result.section.filename,
                heading=result.section.heading,
                heading_path=result.section.heading_path,
                anchor=result.section.anchor,
            )
            for index, result in enumerate(routed_search.results, start=1)
        ]

        messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_INSTRUCTIONS)]
        for turn in list(history or [])[-self.maximum_history_turns :]:
            content = str(turn.content or "").strip()
            if not content:
                continue
            if turn.role.lower() == "user":
                messages.append(HumanMessage(content=content))
            elif turn.role.lower() == "assistant":
                messages.append(AIMessage(content=content))

        sources = "\n\n".join(
            _source_block(index, result)
            for index, result in enumerate(routed_search.results, start=1)
        )
        decision = routed_search.decision
        final_request = f"""ROUTE: {decision.route.value}
ROUTE-SPECIFIC RESPONSE POLICY: {decision.response_policy}
HUMAN REVIEW REQUIRED: {str(decision.requires_human_review).lower()}

APPLICATION CONTEXT:
{_format_application_context(context or {})}

USER QUESTION:
{question}

APPROVED NEAT SOURCES:
{sources}

Write the answer now. If human review is required, state that clearly."""
        messages.append(HumanMessage(content=final_request))
        return PreparedAnswerRequest(
            messages=messages,
            citations=citations,
            routed_search=routed_search,
        )


def _message_text(response) -> str:
    if isinstance(response, str):
        return response.strip()

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    return ""


class GroundedAnswerGenerator:
    """Invoke a chat model while preserving deterministic safety metadata."""

    def __init__(
        self,
        model: ChatModel,
        *,
        prompt_builder: Optional[GroundedPromptBuilder] = None,
    ) -> None:
        self.model = model
        self.prompt_builder = prompt_builder or GroundedPromptBuilder()

    def answer(
        self,
        question: str,
        routed_search: RoutedSearchResult,
        *,
        context: Optional[Mapping[str, object]] = None,
        history: Optional[Sequence[ConversationTurn]] = None,
        config: Optional[Mapping[str, object]] = None,
    ) -> GroundedAnswer:
        prepared = self.prompt_builder.prepare(
            question,
            routed_search,
            context=context,
            history=history,
        )
        response = self.model.invoke(prepared.messages, config=config)
        answer_text = _message_text(response)
        if not answer_text:
            raise RuntimeError("The answer model returned no text")

        decision = routed_search.decision
        return GroundedAnswer(
            answer=answer_text,
            route=decision.route.value,
            route_confidence=decision.confidence,
            requires_human_review=decision.requires_human_review,
            citations=prepared.citations,
        )

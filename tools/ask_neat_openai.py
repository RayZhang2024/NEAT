"""Ask the grounded NEAT assistant using an OpenAI chat model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.assistant_openai import (
    DEFAULT_OPENAI_MODEL,
    MissingOpenAIAPIKey,
    create_openai_chat_model,
    describe_openai_error,
    load_openai_settings,
)
from tools.assistant_retrieval import load_knowledge_base
from tools.assistant_semantic_retrieval import ChromaSemanticRetriever
from tools.assistant_service import NEATAssistantService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "docs" / "assistant"
INDEX_DIRECTORY = PROJECT_ROOT / ".assistant_cache" / "chroma"


def _display_answer(result, *, model_name: str) -> None:
    print(result.answer)
    print(f"\nModel: {model_name}")
    print(f"Route: {result.route}")
    print(
        "Human review: "
        f"{'required' if result.requires_human_review else 'not required'}"
    )
    print("Verified NEAT sources:")
    for citation in result.citations:
        print(
            f"  [{citation.number}] {citation.heading_path} "
            f"({citation.source_id})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask the grounded NEAT assistant through OpenAI."
    )
    parser.add_argument("question")
    parser.add_argument(
        "--model",
        help=(
            "OpenAI model ID. Defaults to NEAT_ASSISTANT_MODEL or "
            f"{DEFAULT_OPENAI_MODEL}."
        ),
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    try:
        settings = load_openai_settings(model=args.model)
    except MissingOpenAIAPIKey as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        print(
            "PowerShell setup for future terminals:\n"
            "  setx OPENAI_API_KEY \"your_api_key_here\"\n\n"
            "For this PowerShell window only:\n"
            "  $env:OPENAI_API_KEY = \"your_api_key_here\"",
            file=sys.stderr,
        )
        return 2

    try:
        sections = load_knowledge_base(KNOWLEDGE_DIRECTORY)
        retriever = ChromaSemanticRetriever(
            sections,
            persist_directory=INDEX_DIRECTORY,
        )
        model = create_openai_chat_model(settings)
        result = NEATAssistantService(retriever, model).ask(
            args.question,
            source_limit=args.top_k,
            config={"tags": ["neat-assistant", "openai"]},
        )
    except Exception as exc:
        print(describe_openai_error(exc), file=sys.stderr)
        return 1

    _display_answer(result, model_name=settings.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

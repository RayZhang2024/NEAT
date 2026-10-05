"""Preview a complete grounded answer without requiring an external API key."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from langchain_core.messages import AIMessage

from NEAT.package_resources import assistant_knowledge_root
from tools.assistant_retrieval import load_knowledge_base
from tools.assistant_semantic_retrieval import ChromaSemanticRetriever
from tools.assistant_service import NEATAssistantService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIRECTORY = PROJECT_ROOT / ".assistant_cache" / "chroma"


class ExtractivePreviewModel:
    """Offline wiring check that extracts guidance from the first source."""

    def invoke(self, messages, config=None):
        request = str(messages[-1].content)
        route_match = re.search(r"ROUTE: ([a-z_]+)", request)
        source_match = re.search(
            r'<approved_source[^>]*>\s*Heading: (.+?)\s*Content:\s*(.+?)\s*</approved_source>',
            request,
            flags=re.DOTALL,
        )
        route = route_match.group(1) if route_match else "unknown"
        if not source_match:
            return AIMessage(content="The approved NEAT sources were insufficient.")

        heading = source_match.group(1).strip()
        body = source_match.group(2).strip()
        paragraphs = [part.strip() for part in body.split("\n\n") if part.strip()]
        useful_paragraphs = [
            paragraph
            for paragraph in paragraphs
            if paragraph != heading
            and not paragraph.lower().startswith("source:")
        ]
        guidance = " ".join(useful_paragraphs[:3]) or body
        prefix = (
            "Human review is required for this question. "
            if "HUMAN REVIEW REQUIRED: true" in request
            else ""
        )
        return AIMessage(
            content=(
                f"{prefix}Offline grounded preview ({route}). "
                f"Relevant NEAT guidance from {heading}: {guidance}"
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview the grounded NEAT assistant pipeline offline."
    )
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    sections = load_knowledge_base(assistant_knowledge_root())
    retriever = ChromaSemanticRetriever(
        sections,
        persist_directory=INDEX_DIRECTORY,
    )
    result = NEATAssistantService(retriever, ExtractivePreviewModel()).ask(
        args.question,
        source_limit=args.top_k,
    )

    print(result.answer)
    print(f"\nRoute: {result.route}")
    print(
        "Human review: "
        f"{'required' if result.requires_human_review else 'not required'}"
    )
    print("Verified sources:")
    for citation in result.citations:
        print(f"  [{citation.number}] {citation.source_id}")
    print("\nNote: this preview is extractive and does not call a production LLM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

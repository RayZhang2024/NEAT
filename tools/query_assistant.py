"""Inspect the current routed semantic retrieval pipeline for one question."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.assistant_query_pipeline import RoutedRetriever
from tools.assistant_retrieval import load_knowledge_base
from tools.assistant_semantic_retrieval import ChromaSemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "docs" / "assistant"
INDEX_DIRECTORY = PROJECT_ROOT / ".assistant_cache" / "chroma"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Route one NEAT question and inspect retrieved sources."
    )
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    sections = load_knowledge_base(KNOWLEDGE_DIRECTORY)
    semantic = ChromaSemanticRetriever(
        sections,
        persist_directory=INDEX_DIRECTORY,
    )
    routed = RoutedRetriever(semantic).route_and_search(
        args.question,
        limit=args.top_k,
    )

    decision = routed.decision
    print(f"Route: {decision.route.value}")
    print(f"Confidence: {decision.confidence:.0%}")
    print(f"Human review: {'required' if decision.requires_human_review else 'not required'}")
    print(f"Reason: {decision.reason}")
    print(f"Response policy: {decision.response_policy}")
    print("\nRetrieved sources:")
    for index, result in enumerate(routed.results, start=1):
        print(
            f"  {index}. {result.section.source_id} "
            f"(score={result.score:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


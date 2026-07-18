"""Compare lexical BM25 and local LangChain-Chroma semantic retrieval."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tools.assistant_retrieval import (
    BM25Retriever,
    evaluate_retriever,
    load_evaluation_questions,
    load_knowledge_base,
    validate_expected_sources,
)
from tools.assistant_semantic_retrieval import ChromaSemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "docs" / "assistant"
DEFAULT_QUESTIONS_PATH = DEFAULT_KNOWLEDGE_DIRECTORY / "evaluation_questions.json"
DEFAULT_INDEX_DIRECTORY = PROJECT_ROOT / ".assistant_cache" / "chroma"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare NEAT assistant BM25 and semantic retrieval quality."
    )
    parser.add_argument(
        "--knowledge-directory",
        type=Path,
        default=DEFAULT_KNOWLEDGE_DIRECTORY,
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--index-directory", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-embed the current knowledge collection even if its index exists.",
    )
    parser.add_argument(
        "--show-misses",
        action="store_true",
        help="Show semantic misses and their retrieved sources.",
    )
    return parser


def hit_ids(report: dict) -> set[str]:
    return {
        result["id"] for result in report["results"] if result["rank"] is not None
    }


def print_comparison(bm25: dict, semantic: dict) -> None:
    limit = bm25["limit"]
    print("NEAT assistant retrieval comparison")
    print(f"Knowledge sections: {bm25['section_count']}")
    print(f"Evaluation questions: {bm25['question_count']}")
    print()
    print(f"{'Retriever':<22} {'Hit@1':>8} {f'Hit@{limit}':>8} {'MRR':>8}")
    print("-" * 50)
    print(
        f"{'BM25 lexical':<22} {bm25['hit_at_1']:>7.1%} "
        f"{bm25['hit_at_limit']:>7.1%} {bm25['mean_reciprocal_rank']:>8.3f}"
    )
    print(
        f"{'MiniLM + Chroma':<22} {semantic['hit_at_1']:>7.1%} "
        f"{semantic['hit_at_limit']:>7.1%} "
        f"{semantic['mean_reciprocal_rank']:>8.3f}"
    )

    bm25_hits = hit_ids(bm25)
    semantic_hits = hit_ids(semantic)
    gained = sorted(semantic_hits - bm25_hits)
    lost = sorted(bm25_hits - semantic_hits)
    print(f"\nSemantic gains over BM25: {', '.join(gained) if gained else 'None'}")
    print(f"Semantic regressions: {', '.join(lost) if lost else 'None'}")


def print_semantic_misses(report: dict) -> None:
    misses = [result for result in report["results"] if result["rank"] is None]
    print(f"\nSemantic misses: {len(misses)}")
    for result in misses:
        retrieved = ", ".join(item["source"] for item in result["retrieved"])
        print(f"\n  {result['id']}: {result['question']}")
        print(f"    Expected: {', '.join(result['expected_sources'])}")
        print(f"    Retrieved: {retrieved}")


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    sections = load_knowledge_base(args.knowledge_directory.resolve())
    questions = load_evaluation_questions(args.questions.resolve())
    missing_sources = validate_expected_sources(sections, questions)
    if missing_sources:
        raise SystemExit(
            "Evaluation references missing knowledge sections: "
            + ", ".join(missing_sources)
        )

    bm25_started = time.perf_counter()
    bm25_report = evaluate_retriever(
        BM25Retriever(sections), questions, limit=args.top_k
    )
    bm25_seconds = time.perf_counter() - bm25_started

    index_started = time.perf_counter()
    semantic_retriever = ChromaSemanticRetriever(
        sections,
        persist_directory=args.index_directory,
        rebuild=args.rebuild,
    )
    index_seconds = time.perf_counter() - index_started

    semantic_started = time.perf_counter()
    semantic_report = evaluate_retriever(
        semantic_retriever, questions, limit=args.top_k
    )
    semantic_seconds = time.perf_counter() - semantic_started

    print_comparison(bm25_report, semantic_report)
    print(
        f"\nSemantic model: {semantic_retriever.embedding_model} "
        f"({'reused index' if semantic_retriever.reused_existing_index else 'built index'})"
    )
    print(f"Index directory: {semantic_retriever.persist_directory}")
    print(f"BM25 evaluation time: {bm25_seconds:.3f} s")
    print(f"Semantic index startup/build time: {index_seconds:.3f} s")
    print(f"Semantic evaluation time: {semantic_seconds:.3f} s")

    if args.show_misses:
        print_semantic_misses(semantic_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


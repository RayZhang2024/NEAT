"""Run the NEAT assistant retrieval benchmark or inspect one query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from NEAT.package_resources import assistant_knowledge_root
from tools.assistant_retrieval import (
    BM25Retriever,
    evaluate_retriever,
    load_evaluation_questions,
    load_knowledge_base,
    validate_expected_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "docs" / "assistant" / "evaluation_questions.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the dependency-free NEAT assistant retrieval baseline."
    )
    parser.add_argument(
        "--knowledge-directory",
        type=Path,
        help="Optional filesystem override for the package-owned knowledge corpus.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--query",
        help="Search one question instead of running the evaluation dataset.",
    )
    parser.add_argument(
        "--show-misses",
        action="store_true",
        help="Print questions whose expected source was not found in the top results.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete benchmark result as JSON.",
    )
    parser.add_argument(
        "--minimum-hit-rate",
        type=float,
        default=0.0,
        help="Exit unsuccessfully when Hit@K is below this value (0 to 1).",
    )
    return parser


def print_query_results(retriever: BM25Retriever, query: str, top_k: int) -> None:
    print(f"Query: {query}")
    for index, result in enumerate(retriever.search(query, limit=top_k), start=1):
        preview = " ".join(result.section.text.split())[:240]
        print(f"\n{index}. {result.section.source_id}  score={result.score:.4f}")
        print(f"   {preview}")


def print_summary(report: dict, show_misses: bool) -> None:
    limit = report["limit"]
    print("NEAT assistant retrieval baseline")
    print(f"Knowledge sections: {report['section_count']}")
    print(f"Evaluation questions: {report['question_count']}")
    print(f"Hit@1: {report['hit_at_1']:.1%}")
    print(f"Hit@{limit}: {report['hit_at_limit']:.1%}")
    print(f"MRR: {report['mean_reciprocal_rank']:.3f}")

    print("\nHit@K by category:")
    for category, metrics in report["category_hit_at_limit"].items():
        print(
            f"  {category}: {metrics['hits']}/{metrics['total']} "
            f"({metrics['rate']:.0%})"
        )

    misses = [result for result in report["results"] if result["rank"] is None]
    print(f"\nMisses: {len(misses)}")
    if show_misses:
        for result in misses:
            retrieved = ", ".join(item["source"] for item in result["retrieved"])
            print(f"\n  {result['id']}: {result['question']}")
            print(f"    Expected: {', '.join(result['expected_sources'])}")
            print(f"    Retrieved: {retrieved}")


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if not 0.0 <= args.minimum_hit_rate <= 1.0:
        raise SystemExit("--minimum-hit-rate must be between 0 and 1")

    knowledge_root = (
        args.knowledge_directory.resolve()
        if args.knowledge_directory is not None
        else assistant_knowledge_root()
    )
    sections = load_knowledge_base(knowledge_root)
    retriever = BM25Retriever(sections)

    if args.query:
        print_query_results(retriever, args.query, args.top_k)
        return 0

    questions = load_evaluation_questions(args.questions.resolve())
    missing_sources = validate_expected_sources(sections, questions)
    if missing_sources:
        print("Evaluation references missing knowledge sections:")
        for source in missing_sources:
            print(f"  {source}")
        return 2

    report = evaluate_retriever(retriever, questions, limit=args.top_k)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_summary(report, args.show_misses)

    return int(report["hit_at_limit"] < args.minimum_hit_rate)


if __name__ == "__main__":
    raise SystemExit(main())


"""Run the curated NEAT questions through the live grounded OpenAI assistant."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tools.assistant_answer_evaluation import (
    build_answer_evaluation_record,
    build_error_record,
    create_report,
    load_report,
    save_markdown_report,
    save_report,
    update_report,
)
from tools.assistant_openai import (
    MissingOpenAIAPIKey,
    create_openai_chat_model,
    describe_openai_error,
    load_openai_settings,
)
from tools.assistant_retrieval import (
    load_evaluation_questions,
    load_knowledge_base,
    validate_expected_sources,
)
from tools.assistant_semantic_retrieval import (
    ChromaSemanticRetriever,
    knowledge_fingerprint,
)
from tools.assistant_service import NEATAssistantService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "docs" / "assistant"
QUESTIONS_PATH = KNOWLEDGE_DIRECTORY / "evaluation_questions.json"
INDEX_DIRECTORY = PROJECT_ROOT / ".assistant_cache" / "chroma"
OUTPUT_DIRECTORY = PROJECT_ROOT / ".assistant_cache" / "evaluations"


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIRECTORY / f"answer_evaluation_{timestamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate live grounded NEAT assistant answers."
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required acknowledgement that this run makes billable OpenAI requests.",
    )
    parser.add_argument("--model", help="Override NEAT_ASSISTANT_MODEL for this run.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N selected questions.")
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Evaluate only these question IDs, for example NEAT-EVAL-001.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, help="JSON report path.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing --output report and skip answered questions.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed API request instead of recording and continuing.",
    )
    return parser


def select_questions(questions: list[dict], ids, limit) -> list[dict]:
    selected = questions
    if ids:
        requested = set(ids)
        selected = [item for item in selected if item["id"] in requested]
        missing = sorted(requested.difference(item["id"] for item in selected))
        if missing:
            raise ValueError("Unknown evaluation IDs: " + ", ".join(missing))
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    return selected


def safe_error_message(exc: Exception) -> str:
    message = describe_openai_error(exc)
    if message.startswith("Unexpected OpenAI error type"):
        detail = str(exc).strip()
        if detail:
            return f"{message} Detail: {detail[:500]}"
    return message


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    sections = load_knowledge_base(KNOWLEDGE_DIRECTORY)
    questions = load_evaluation_questions(QUESTIONS_PATH)
    missing_sources = validate_expected_sources(sections, questions)
    if missing_sources:
        raise SystemExit("Missing expected sources: " + ", ".join(missing_sources))
    try:
        selected = select_questions(questions, args.ids, args.limit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_path = (args.output or default_output_path()).resolve()
    markdown_path = output_path.with_suffix(".md")
    print(f"Selected questions: {len(selected)}")
    print(f"JSON report: {output_path}")
    print(f"Review report: {markdown_path}")
    if not args.confirm_live:
        print(
            "No API requests were made. Add --confirm-live to acknowledge that "
            "the evaluation uses OpenAI API credit."
        )
        return 2

    try:
        settings = load_openai_settings(model=args.model)
    except MissingOpenAIAPIKey as exc:
        raise SystemExit(str(exc)) from exc

    fingerprint = knowledge_fingerprint(sections)
    if args.resume:
        if not output_path.exists():
            raise SystemExit("--resume requires an existing --output report")
        report = load_report(output_path)
        if report["model"] != settings.model:
            raise SystemExit("Cannot resume with a different model")
        if report["knowledge_fingerprint"] != fingerprint:
            raise SystemExit("Cannot resume after the knowledge base has changed")
    else:
        if args.output and output_path.exists():
            raise SystemExit("Output already exists; use --resume or another path")
        report = create_report(
            model=settings.model,
            knowledge_fingerprint=fingerprint,
        )

    answered_ids = {
        item["id"] for item in report["results"] if item.get("status") == "answered"
    }
    remaining = [item for item in selected if item["id"] not in answered_ids]
    print(f"Model: {settings.model}")
    print(f"Remaining live requests: {len(remaining)}")

    retriever = ChromaSemanticRetriever(
        sections,
        persist_directory=INDEX_DIRECTORY,
    )
    service = NEATAssistantService(
        retriever,
        create_openai_chat_model(settings),
    )

    error_count = 0
    for index, question in enumerate(remaining, start=1):
        print(f"[{index}/{len(remaining)}] {question['id']}: {question['question']}")
        try:
            answer = service.ask(question["question"], source_limit=args.top_k)
            record = build_answer_evaluation_record(question, answer)
            hit = record["automatic_checks"]["expected_source_cited"]
            print(f"  answered; expected source cited: {'yes' if hit else 'NO'}")
        except Exception as exc:
            error_count += 1
            message = safe_error_message(exc)
            print(f"  ERROR: {message}")
            record = build_error_record(question, message)

        update_report(report, record)
        save_report(report, output_path)
        save_markdown_report(report, markdown_path)
        if record["status"] == "error" and args.stop_on_error:
            break

    summary = report["summary"]
    print("\nEvaluation complete")
    print(f"Answered: {summary['answered_count']}")
    print(f"Errors: {summary['error_count']}")
    print(
        "Expected-source citation rate: "
        f"{summary['expected_source_citation_rate']:.1%}"
    )
    print(
        "Heuristic key-point rate: "
        f"{summary['mean_heuristic_key_point_rate']:.1%} "
        "(human review still required)"
    )
    return int(error_count > 0)


if __name__ == "__main__":
    raise SystemExit(main())


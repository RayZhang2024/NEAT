"""Evaluate NEAT assistant intent routing and escalation recall."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.assistant_router import QuestionRoute, QuestionRouter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = (
    PROJECT_ROOT / "docs" / "assistant" / "routing_questions.json"
)


def load_routing_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Routing evaluation dataset must be a JSON list")

    valid_routes = {route.value for route in QuestionRoute}
    seen = set()
    for index, record in enumerate(data, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Routing record {index} must be an object")
        for field in ("id", "question", "expected_route"):
            if not record.get(field):
                raise ValueError(f"Routing record {index} is missing {field}")
        if record["id"] in seen:
            raise ValueError(f"Duplicate routing ID: {record['id']}")
        seen.add(record["id"])
        if record["expected_route"] not in valid_routes:
            raise ValueError(
                f"Unknown expected route in {record['id']}: "
                f"{record['expected_route']}"
            )
    return data


def evaluate_router(router: QuestionRouter, questions: list[dict]) -> dict:
    results = []
    correct = 0
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()

    for record in questions:
        decision = router.route(record["question"])
        expected = record["expected_route"]
        predicted = decision.route.value
        is_correct = predicted == expected
        correct += int(is_correct)
        totals[expected] += 1
        hits[expected] += int(is_correct)
        results.append(
            {
                "id": record["id"],
                "question": record["question"],
                "expected_route": expected,
                "predicted_route": predicted,
                "correct": is_correct,
                "reason": decision.reason,
            }
        )

    count = len(questions)
    escalation = QuestionRoute.ESCALATION.value
    return {
        "question_count": count,
        "accuracy": correct / count if count else 0.0,
        "escalation_recall": hits[escalation] / totals[escalation]
        if totals[escalation]
        else 0.0,
        "per_route": {
            route: {
                "hits": hits[route],
                "total": totals[route],
                "accuracy": hits[route] / totals[route],
            }
            for route in sorted(totals)
        },
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the NEAT question router.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--question", help="Inspect one question instead of evaluating.")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--minimum-accuracy", type=float, default=0.0)
    parser.add_argument("--minimum-escalation-recall", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    router = QuestionRouter()

    if args.question:
        decision = router.route(args.question)
        print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))
        return 0

    questions = load_routing_questions(args.questions.resolve())
    report = evaluate_router(router, questions)
    print("NEAT assistant routing evaluation")
    print(f"Questions: {report['question_count']}")
    print(f"Overall accuracy: {report['accuracy']:.1%}")
    print(f"Escalation recall: {report['escalation_recall']:.1%}")
    print("\nAccuracy by route:")
    for route, metrics in report["per_route"].items():
        print(
            f"  {route}: {metrics['hits']}/{metrics['total']} "
            f"({metrics['accuracy']:.0%})"
        )

    errors = [result for result in report["results"] if not result["correct"]]
    print(f"\nRouting errors: {len(errors)}")
    if args.show_errors:
        for error in errors:
            print(f"\n  {error['id']}: {error['question']}")
            print(f"    Expected: {error['expected_route']}")
            print(f"    Predicted: {error['predicted_route']}")
            print(f"    Reason: {error['reason']}")

    failed = (
        report["accuracy"] < args.minimum_accuracy
        or report["escalation_recall"] < args.minimum_escalation_recall
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())


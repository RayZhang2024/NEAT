"""Command-line entry point for the local NEAT assistant feedback summary."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tools.assistant_feedback import default_feedback_path
from tools.assistant_feedback_summary import (
    load_feedback_records,
    save_feedback_summary,
    summarize_feedback,
)


def default_summary_path(feedback_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return feedback_path.with_name(f"assistant_feedback_summary_{timestamp}.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize local NEAT assistant Helpful/Not helpful ratings."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_feedback_path(),
        help="Feedback JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown output path. Defaults beside the feedback log.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    feedback_path = args.input.expanduser().resolve()
    if not feedback_path.exists():
        raise SystemExit(
            "No feedback log was found. Use the assistant and rate at least one "
            f"answer first.\nExpected path: {feedback_path}"
        )

    records, invalid_lines = load_feedback_records(feedback_path)
    summary = summarize_feedback(records, invalid_lines=invalid_lines)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else default_summary_path(feedback_path)
    )
    save_feedback_summary(summary, output_path)

    print(f"Rated answers: {summary['total']}")
    print(f"Distinct rated questions: {summary['latest_total']}")
    if summary["latest_helpful_rate"] is None:
        print("Current helpful rate: N/A")
    else:
        print(f"Current helpful rate: {summary['latest_helpful_rate']:.1%}")
    print(f"Currently not helpful: {summary['latest_not_helpful']}")
    print(f"Summary report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

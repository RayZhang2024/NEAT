"""Dependency-free retrieval baseline for the NEAT assistant knowledge base."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Protocol, Sequence

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable


KNOWLEDGE_FILENAMES = (
    "faq.md",
    "troubleshooting.md",
    "parameter_reference.md",
    "software_ui_guide.md",
    "menu_reference.md",
    "fitting_ui_reference.md",
    "preprocessing_technical.md",
    "fitting_technical.md",
    "mapping_technical.md",
    "postprocessing_technical.md",
    "known_limitations.md",
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "should",
    "so",
    "that",
    "the",
    "this",
    "to",
    "use",
    "what",
    "when",
    "which",
    "why",
    "with",
    "would",
}

_TOKEN_ALIASES = {
    "normalisation": "normalise",
    "normalised": "normalise",
    "normalising": "normalise",
    "normalization": "normalise",
    "normalized": "normalise",
    "normalizing": "normalise",
    "normalize": "normalise",
    "fitted": "fit",
    "fitting": "fit",
    "failed": "fail",
    "failing": "fail",
    "fails": "fail",
    "failure": "fail",
    "failures": "fail",
    "noisy": "noise",
    "unstable": "instability",
    "images": "image",
    "frames": "frame",
    "folders": "folder",
    "pixels": "pixel",
    "datasets": "dataset",
    "parameters": "parameter",
    "settings": "setting",
    "edges": "edge",
    "windows": "window",
    "results": "result",
    "coordinates": "coordinate",
    "values": "value",
    "files": "file",
    "runs": "run",
    "masks": "mask",
}


@dataclass(frozen=True)
class KnowledgeSection:
    """One retrievable Markdown section."""

    source_id: str
    filename: str
    anchor: str
    heading: str
    heading_path: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    """A scored retrieval result."""

    section: KnowledgeSection
    score: float


class SectionRetriever(Protocol):
    """Common interface shared by lexical and semantic retrievers."""

    sections: Sequence[KnowledgeSection]

    def search(self, query: str, *, limit: int = 3) -> list[SearchResult]:
        """Return the highest-ranked sections for a query."""
        ...


def github_anchor(heading: str) -> str:
    """Return the GitHub-style anchor used by the curated Markdown headings."""

    value = unicodedata.normalize("NFKC", heading).strip().lower()
    value = re.sub(r"[^\w\- ]", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "-", value)
    return re.sub(r"-+", "-", value).strip("-")


def tokenize(text: str) -> list[str]:
    """Normalize NEAT terminology into tokens for lexical retrieval."""

    value = unicodedata.normalize("NFKC", text).lower()
    value = (
        value.replace("η", " eta ")
        .replace("σ", " sigma ")
        .replace("τ", " tau ")
        .replace("µ", " micro ")
        .replace("μ", " micro ")
    )

    tokens = []
    for raw_token in _TOKEN_RE.findall(value):
        if raw_token in _STOP_WORDS:
            continue
        tokens.append(_TOKEN_ALIASES.get(raw_token, raw_token))
    return tokens


def load_markdown_sections(
    path: Path | Traversable,
    *,
    minimum_level: int = 2,
    maximum_level: int = 3,
) -> list[KnowledgeSection]:
    """Split a Markdown file into retrievable heading-based sections."""

    lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[KnowledgeSection] = []
    heading_stack: dict[int, str] = {}
    anchor_counts: Counter[str] = Counter()
    current: Optional[dict] = None

    def finish_current() -> None:
        nonlocal current
        if current is None:
            return

        body = "\n".join(current["body"]).strip()
        if body:
            heading_path = " > ".join(
                heading_stack[level]
                for level in sorted(heading_stack)
                if level <= current["level"]
            )
            full_text = f"{heading_path}\n\n{body}" if heading_path else body
            sections.append(
                KnowledgeSection(
                    source_id=current["source_id"],
                    filename=path.name,
                    anchor=current["anchor"],
                    heading=current["heading"],
                    heading_path=heading_path,
                    text=full_text,
                )
            )
        current = None

    for line in lines:
        match = _HEADING_RE.match(line)
        if not match:
            if current is not None:
                current["body"].append(line)
            continue

        finish_current()
        level = len(match.group(1))
        heading = match.group(2).strip()

        for stack_level in tuple(heading_stack):
            if stack_level >= level:
                del heading_stack[stack_level]
        heading_stack[level] = heading

        base_anchor = github_anchor(heading)
        duplicate_number = anchor_counts[base_anchor]
        anchor_counts[base_anchor] += 1
        anchor = base_anchor if duplicate_number == 0 else f"{base_anchor}-{duplicate_number}"

        if minimum_level <= level <= maximum_level:
            current = {
                "level": level,
                "heading": heading,
                "anchor": anchor,
                "source_id": f"{path.name.lower()}#{anchor}",
                "body": [],
            }

    finish_current()
    return sections


def load_knowledge_base(directory: Path | Traversable) -> list[KnowledgeSection]:
    """Load the approved NEAT assistant Markdown files."""

    sections: list[KnowledgeSection] = []
    missing = []
    for filename in KNOWLEDGE_FILENAMES:
        path = directory.joinpath(filename)
        if not path.is_file():
            missing.append(filename)
            continue
        sections.extend(load_markdown_sections(path))

    if missing:
        raise FileNotFoundError(
            f"Missing knowledge files in {directory}: {', '.join(missing)}"
        )
    if not sections:
        raise RuntimeError(f"No retrievable knowledge sections found in {directory}")
    return sections


class BM25Retriever:
    """Small BM25 retriever used as the dependency-free quality baseline."""

    def __init__(
        self,
        sections: Sequence[KnowledgeSection],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        heading_repetitions: int = 2,
    ) -> None:
        if not sections:
            raise ValueError("At least one knowledge section is required")

        self.sections = list(sections)
        self.k1 = float(k1)
        self.b = float(b)
        self._document_tokens: list[list[str]] = []
        self._term_frequencies: list[Counter[str]] = []
        document_frequency: defaultdict[str, int] = defaultdict(int)

        for section in self.sections:
            heading_tokens = tokenize(section.heading_path)
            tokens = heading_tokens * max(1, int(heading_repetitions)) + tokenize(
                section.text
            )
            self._document_tokens.append(tokens)
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            for term in frequencies:
                document_frequency[term] += 1

        self._average_length = sum(map(len, self._document_tokens)) / len(
            self._document_tokens
        )
        document_count = len(self._document_tokens)
        self._inverse_document_frequency = {
            term: math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, *, limit: int = 3) -> list[SearchResult]:
        """Return the highest-scoring sections for a user query."""

        if limit <= 0:
            return []

        query_frequencies = Counter(tokenize(query))
        scored: list[SearchResult] = []
        for section, tokens, frequencies in zip(
            self.sections,
            self._document_tokens,
            self._term_frequencies,
        ):
            document_length = len(tokens)
            normalization = self.k1 * (
                1.0
                - self.b
                + self.b * document_length / max(self._average_length, 1.0)
            )
            score = 0.0
            for term, query_frequency in query_frequencies.items():
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                inverse_frequency = self._inverse_document_frequency.get(term, 0.0)
                term_score = inverse_frequency * (
                    term_frequency * (self.k1 + 1.0)
                ) / (term_frequency + normalization)
                score += term_score * (1.0 + math.log(query_frequency))

            scored.append(SearchResult(section=section, score=score))

        scored.sort(key=lambda result: (-result.score, result.section.source_id))
        return scored[: min(limit, len(scored))]


def load_evaluation_questions(path: Path) -> list[dict]:
    """Load and minimally validate the curated retrieval evaluation dataset."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a JSON list")

    required = {
        "id",
        "category",
        "question",
        "expected_sources",
        "key_points",
        "expected_behavior",
    }
    seen_ids = set()
    for index, record in enumerate(data, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Evaluation record {index} must be an object")
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"Evaluation record {index} is missing: {', '.join(sorted(missing))}"
            )
        if record["id"] in seen_ids:
            raise ValueError(f"Duplicate evaluation ID: {record['id']}")
        seen_ids.add(record["id"])

    return data


def validate_expected_sources(
    sections: Iterable[KnowledgeSection], questions: Iterable[dict]
) -> list[str]:
    """Return evaluation source IDs that are absent from the knowledge base."""

    available = {section.source_id.lower() for section in sections}
    missing = set()
    for record in questions:
        for source_id in record["expected_sources"]:
            if source_id.lower() not in available:
                missing.add(source_id)
    return sorted(missing)


def evaluate_retriever(
    retriever: SectionRetriever,
    questions: Sequence[dict],
    *,
    limit: int = 3,
) -> dict:
    """Evaluate retrieval and return serializable aggregate metrics."""

    if limit <= 0:
        raise ValueError("Evaluation limit must be positive")

    results = []
    reciprocal_rank_total = 0.0
    hit_at_1 = 0
    hit_at_limit = 0
    category_totals: Counter[str] = Counter()
    category_hits: Counter[str] = Counter()

    for record in questions:
        retrieved = retriever.search(record["question"], limit=limit)
        expected = {source.lower() for source in record["expected_sources"]}
        rank = None
        for index, result in enumerate(retrieved, start=1):
            if result.section.source_id.lower() in expected:
                rank = index
                break

        category = record["category"]
        category_totals[category] += 1
        if rank == 1:
            hit_at_1 += 1
        if rank is not None:
            hit_at_limit += 1
            reciprocal_rank_total += 1.0 / rank
            category_hits[category] += 1

        results.append(
            {
                "id": record["id"],
                "category": category,
                "question": record["question"],
                "expected_sources": record["expected_sources"],
                "rank": rank,
                "retrieved": [
                    {
                        "source": result.section.source_id,
                        "score": round(result.score, 6),
                    }
                    for result in retrieved
                ],
            }
        )

    total = len(questions)
    return {
        "question_count": total,
        "section_count": len(retriever.sections),
        "limit": limit,
        "hit_at_1": hit_at_1 / total if total else 0.0,
        "hit_at_limit": hit_at_limit / total if total else 0.0,
        "mean_reciprocal_rank": reciprocal_rank_total / total if total else 0.0,
        "category_hit_at_limit": {
            category: {
                "hits": category_hits[category],
                "total": category_totals[category],
                "rate": category_hits[category] / category_totals[category],
            }
            for category in sorted(category_totals)
        },
        "results": results,
    }

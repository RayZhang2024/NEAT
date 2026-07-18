"""Deterministic intent and safety routing for NEAT assistant questions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Optional


class QuestionRoute(str, Enum):
    """Supported response paths before knowledge retrieval."""

    HOW_TO = "how_to"
    PARAMETER_EXPLANATION = "parameter_explanation"
    TROUBLESHOOTING = "troubleshooting"
    SCIENTIFIC_INTERPRETATION = "scientific_interpretation"
    BUG_REPORT = "bug_report"
    ESCALATION = "escalation"


RESPONSE_POLICIES = {
    QuestionRoute.HOW_TO: (
        "Provide concise numbered instructions grounded in approved NEAT sources. "
        "Mention prerequisites and cite the relevant section."
    ),
    QuestionRoute.PARAMETER_EXPLANATION: (
        "Define the parameter, explain its effect and trade-offs, and distinguish "
        "documented behaviour from dataset-dependent recommendations."
    ),
    QuestionRoute.TROUBLESHOOTING: (
        "Give checks in a safe diagnostic order. Ask for missing settings or error "
        "details when needed, and do not invent a guaranteed fix."
    ),
    QuestionRoute.SCIENTIFIC_INTERPRETATION: (
        "Separate what NEAT calculates from what the result may mean physically. "
        "State assumptions, uncertainty and the need for expert scientific review."
    ),
    QuestionRoute.BUG_REPORT: (
        "Collect the NEAT version, traceback, reproduction steps, input format and "
        "relevant settings. Offer documented checks and prepare escalation if unresolved."
    ),
    QuestionRoute.ESCALATION: (
        "Do not claim guaranteed scientific correctness or prescribe universal settings. "
        "Explain the limitation, provide only documented checks, and refer the decision "
        "to an experienced beamline scientist or NEAT developer."
    ),
}


@dataclass(frozen=True)
class RoutingDecision:
    """Routing result consumed by retrieval and, later, answer generation."""

    route: QuestionRoute
    confidence: float
    reason: str
    requires_human_review: bool
    response_policy: str

    def to_dict(self) -> dict:
        result = asdict(self)
        result["route"] = self.route.value
        return result


_ESCALATION_PATTERNS = (
    re.compile(r"\bguarantee(?:d|s)?\b"),
    re.compile(r"\bscientifically\s+(?:correct|valid|accurate|proven)\b"),
    re.compile(r"\b(?:always|never)\s+(?:work|fail|wrong)\b"),
    re.compile(
        r"\bwithout\s+(?:an?\s+)?(?:expert|scientist|human)(?:\s+\w+){0,2}\s+"
        r"(?:review|validation|checking)\b"
    ),
    re.compile(r"\bsafe\s+to\s+(?:publish|report|use)\b"),
    re.compile(r"\bdefinitive(?:ly)?\b"),
    re.compile(r"\bprove\b.*\b(?:stress|strain|phase|microstructure)\b"),
)

_BUG_PATTERNS = (
    re.compile(r"\btraceback\b"),
    re.compile(r"\b(?:module|file)notfounderror\b"),
    re.compile(r"\b(?:importerror|valueerror|typeerror|runtimeerror|exception)\b"),
    re.compile(r"\bcrash(?:ed|es|ing)?\b"),
    re.compile(r"\bfreez(?:e|es|ing|es up)\b"),
    re.compile(r"\b(?:will not|won't|wont)\s+(?:start|open|launch)\b"),
    re.compile(r"\bcloses?\s+unexpectedly\b"),
    re.compile(r"\breproducible\s+bug\b"),
)

_TROUBLESHOOTING_PATTERNS = (
    re.compile(r"\bfail(?:ed|s|ing|ure|ures)?\b"),
    re.compile(r"\berror\b"),
    re.compile(r"\babort(?:ed|s|ing)?\b"),
    re.compile(r"\breject(?:ed|s|ing)?\b"),
    re.compile(r"\bmissing\b"),
    re.compile(r"\bunstable\b"),
    re.compile(r"\b(?:noisy|noise)\b"),
    re.compile(r"\btoo\s+(?:smooth|noisy|slow)\b"),
    re.compile(r"\b(?:cannot|can't|cant|unable to)\b"),
    re.compile(r"\bno\s+(?:theoretical\s+)?(?:bragg[- ]?)?edges?\b"),
    re.compile(r"\bno\s+(?:data|spectrum|result|results)\b"),
    re.compile(r"\b(?:does not|doesn't|doesnt)\s+(?:appear|work|update|start)\b"),
    re.compile(r"\b(?:invalid|wrong|skipped)\b"),
)

_SCIENTIFIC_PATTERNS = (
    re.compile(r"\binterpret(?:ation|ed|ing)?\b"),
    re.compile(r"\bphysical(?:ly)?\s+(?:mean|meaning|interpretation)\b"),
    re.compile(r"\bscientific\s+(?:meaning|conclusion|conclusions)\b"),
    re.compile(r"\bconclude\b"),
    re.compile(r"\bresidual\s+stress\b"),
    re.compile(r"\bmicrostructur(?:e|al|ally)\b"),
    re.compile(r"\bconvert\b.*\bstrain\b.*\bstress\b"),
    re.compile(r"\binfer\b.*\bstress\b"),
    re.compile(r"\bstress\b.*\b(?:from|using)\b.*\bstrain\b"),
    re.compile(r"\b(?:choose|select)\b.*\bd0\b.*\breference\b"),
    re.compile(r"\bwhat\b.*\b(?:map|value|result)\b.*\b(?:tell|say)\b"),
)

_PARAMETER_EXPLANATION_PATTERNS = (
    re.compile(r"^what\s+(?:is|are|does|do)\b"),
    re.compile(r"\bdifference\s+between\b"),
    re.compile(r"\b(?:mean|means|represent|represents)\b"),
    re.compile(r"\bwhat\s+happens\s+if\b"),
    re.compile(r"\bpurpose\s+of\b"),
    re.compile(r"\bexplain\b"),
)

_PARAMETER_TERMS = (
    "macro-pixel",
    "macro pixel",
    "window half",
    "adjacent",
    "eta",
    "sigma",
    "tau",
    "d0",
    "flight path",
    "time delay",
    "left window",
    "right window",
    "edge window",
    "pixel skip",
    "step x",
    "step y",
    "fit edges",
    "fit pattern",
    "summation",
    "normalisation",
    "normalization",
    "filtering",
    "full process",
    "clean function",
    "phase",
)


def normalize_question(question: str) -> str:
    """Normalize punctuation and whitespace while preserving diagnostic terms."""

    value = unicodedata.normalize("NFKC", str(question or "")).lower()
    value = value.replace("’", "'").replace("`", "")
    return re.sub(r"\s+", " ", value).strip()


def _first_match(text: str, patterns) -> Optional[str]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


class QuestionRouter:
    """Priority-ordered, auditable router for NEAT support questions."""

    def route(
        self,
        question: str,
        *,
        context: Optional[Mapping[str, object]] = None,
    ) -> RoutingDecision:
        text = normalize_question(question)
        context = context or {}
        diagnostic_text = normalize_question(str(context.get("error_text", "")))
        combined_diagnostics = f"{text} {diagnostic_text}".strip()

        if not text:
            return self._decision(
                QuestionRoute.HOW_TO,
                0.2,
                "No question text was supplied; ask the user to enter a NEAT question.",
            )

        match = _first_match(text, _ESCALATION_PATTERNS)
        if match:
            return self._decision(
                QuestionRoute.ESCALATION,
                0.99,
                f"Safety/escalation phrase detected: {match!r}.",
            )

        match = _first_match(combined_diagnostics, _BUG_PATTERNS)
        if match:
            return self._decision(
                QuestionRoute.BUG_REPORT,
                0.98,
                f"Software failure signature detected: {match!r}.",
            )

        match = _first_match(text, _TROUBLESHOOTING_PATTERNS)
        if match:
            return self._decision(
                QuestionRoute.TROUBLESHOOTING,
                0.92,
                f"Troubleshooting symptom detected: {match!r}.",
            )

        match = _first_match(text, _SCIENTIFIC_PATTERNS)
        if match:
            return self._decision(
                QuestionRoute.SCIENTIFIC_INTERPRETATION,
                0.9,
                f"Scientific-interpretation phrase detected: {match!r}.",
            )

        explanation_match = _first_match(text, _PARAMETER_EXPLANATION_PATTERNS)
        parameter_term = next((term for term in _PARAMETER_TERMS if term in text), None)
        if explanation_match and parameter_term:
            return self._decision(
                QuestionRoute.PARAMETER_EXPLANATION,
                0.88,
                f"Explanation request for NEAT term {parameter_term!r}.",
            )

        return self._decision(
            QuestionRoute.HOW_TO,
            0.72,
            "No higher-priority risk, diagnostic, scientific or parameter rule matched.",
        )

    @staticmethod
    def _decision(
        route: QuestionRoute,
        confidence: float,
        reason: str,
    ) -> RoutingDecision:
        return RoutingDecision(
            route=route,
            confidence=confidence,
            reason=reason,
            requires_human_review=route
            in {QuestionRoute.SCIENTIFIC_INTERPRETATION, QuestionRoute.ESCALATION},
            response_policy=RESPONSE_POLICIES[route],
        )

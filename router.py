"""Rule-based intent router for chat text."""

from __future__ import annotations

import re
from collections.abc import Iterable


INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cancel": (
        "cancel",
        "cancellation",
        "call off",
        "can't make",
        "cannot make",
        "drop appointment",
        "remove appointment",
    ),
    "reschedule": (
        "reschedule",
        "move my appointment",
        "move appointment",
        "change time",
        "change date",
        "postpone",
        "push back",
        "rebook",
    ),
    "schedule": (
        "schedule",
        "book",
        "set up",
        "setup",
        "arrange",
        "new appointment",
        "make an appointment",
    ),
}

NEGATION_PREFIX_PATTERN = r"(?:do\s+not|don't|dont|not|never)"
NEGATION_TARGET_GAP_PATTERN = (
    r"(?:\s+(?:want|wants|wanted|wish|wishes|need|needs|needed|"
    r"intend|intends|intended|plan|plans|planned|planning|try|trying|"
    r"going|mean|meant)(?:\s+[a-z0-9']+){0,3}\s+to)?"
)
CANCEL_CONJUNCTION_PATTERN = r"(?:\s*,?\s+(?:or|and|nor)\s+)"


def _keyword_pattern(keyword: str) -> str:
    return rf"(?<!\w){re.escape(keyword)}(?!\w)"


def _contains_keyword(text: str, keyword: str) -> bool:
    return re.search(_keyword_pattern(keyword), text) is not None


def _is_negated_cancel_keyword(text: str, keyword: str) -> bool:
    if re.search(
        rf"(?<!\w){NEGATION_PREFIX_PATTERN}"
        rf"{NEGATION_TARGET_GAP_PATTERN}\s+{_keyword_pattern(keyword)}",
        text,
    ):
        return True

    for previous_keyword in INTENT_KEYWORDS["cancel"]:
        if previous_keyword == keyword:
            continue

        if re.search(
            rf"(?<!\w){NEGATION_PREFIX_PATTERN}"
            rf"{NEGATION_TARGET_GAP_PATTERN}\s+"
            rf"{_keyword_pattern(previous_keyword)}"
            rf"{CANCEL_CONJUNCTION_PATTERN}{_keyword_pattern(keyword)}",
            text,
        ):
            return True

    if keyword == "cancellation" and re.search(
        rf"(?<!\w)(?:no|not\s+a)\s+{_keyword_pattern(keyword)}",
        text,
    ):
        return True

    return False


def _count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    hits = 0
    for keyword in keywords:
        if keyword in INTENT_KEYWORDS["cancel"] and _is_negated_cancel_keyword(
            text,
            keyword,
        ):
            continue

        if _contains_keyword(text, keyword):
            hits += 1
    return hits


def route_intent(text: str) -> tuple[str, str]:
    """Classify intent by keyword matching without using an LLM."""
    normalized = text.strip().lower()
    if not normalized:
        return "schedule", "No content provided; defaulted to schedule."

    scores = {
        intent: _count_keyword_hits(normalized, keywords)
        for intent, keywords in INTENT_KEYWORDS.items()
    }

    max_score = max(scores.values())
    if max_score == 0:
        return "schedule", "No cancel/reschedule keywords found; defaulted to schedule."

    # Prefer rescheduling in mixed-intent messages to avoid destructive cancels.
    for intent in ("reschedule", "cancel", "schedule"):
        if scores[intent] == max_score:
            return intent, f"Matched {max_score} keyword(s) for intent '{intent}'."

    return "schedule", "Fallback to schedule."

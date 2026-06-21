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

NEGATION_AWARE_INTENTS = {"cancel", "reschedule", "schedule"}
IMPLIED_CANCEL_KEYWORDS = {"can't make", "cannot make"}
IMPLIED_INTENT_KEYWORDS = {
    "cancel": IMPLIED_CANCEL_KEYWORDS,
    "schedule": {"new appointment", "make an appointment"},
}
NEGATION_PREFIX_PATTERN = r"(?:do\s+not|don't|dont|not|never)"
NEGATION_TARGET_GAP_PATTERN = (
    r"(?:\s+(?:want|wants|wanted|wish|wishes|need|needs|needed|"
    r"intend|intends|intended|plan|plans|planned|planning|try|trying|"
    r"going|mean|meant)(?:\s+[a-z0-9']+){0,3}\s+to)?"
)
ACTION_CHOICE_CONJUNCTION_PATTERN = r"(?:\s*,?\s+(?:or|nor|and)\s+)"
SAME_INTENT_CONJUNCTION_PATTERN = r"(?:\s*,?\s+(?:or|and|nor)\s+)"
NEGATED_OBJECT_GAP_PATTERN = r"(?:\s+(?!(?:or|nor|and)\b)[a-z0-9']+)*"


def _keyword_pattern(keyword: str) -> str:
    return rf"(?<!\w){re.escape(keyword)}(?!\w)"


def _contains_keyword(text: str, keyword: str) -> bool:
    return re.search(_keyword_pattern(keyword), text) is not None


def _action_keywords() -> tuple[str, ...]:
    return tuple(keyword for keywords in INTENT_KEYWORDS.values() for keyword in keywords)


def _action_keyword_alternation() -> str:
    return "|".join(
        _keyword_pattern(keyword)
        for keyword in sorted(_action_keywords(), key=len, reverse=True)
    )


def _has_negation_before_keyword(text: str, keyword: str) -> bool:
    if re.search(
        rf"(?<!\w){NEGATION_PREFIX_PATTERN}"
        rf"{NEGATION_TARGET_GAP_PATTERN}\s+{_keyword_pattern(keyword)}",
        text,
    ):
        return True

    return False


def _has_negated_conjunction(
    text: str,
    keyword: str,
    previous_keywords: Iterable[str],
    conjunction_pattern: str,
) -> bool:
    for previous_keyword in previous_keywords:
        if previous_keyword == keyword:
            continue

        if re.search(
            rf"(?<!\w){NEGATION_PREFIX_PATTERN}"
            rf"{NEGATION_TARGET_GAP_PATTERN}\s+"
            rf"{_keyword_pattern(previous_keyword)}"
            rf"{conjunction_pattern}{_keyword_pattern(keyword)}",
            text,
        ):
            return True

        if re.search(
            rf"(?<!\w){NEGATION_PREFIX_PATTERN}"
            rf"{NEGATION_TARGET_GAP_PATTERN}\s+"
            rf"{_keyword_pattern(previous_keyword)}"
            rf"{NEGATED_OBJECT_GAP_PATTERN}"
            rf"{conjunction_pattern}"
            rf"{_keyword_pattern(keyword)}",
            text,
        ):
            return True

    return False


def _has_negated_action_list(text: str, keyword: str) -> bool:
    action_keyword_pattern = rf"(?:{_action_keyword_alternation()})"
    list_tail_patterns = (
        rf"(?:\s*,\s*{action_keyword_pattern})*"
        rf"\s*,?\s+(?:or|nor|and)\s+{action_keyword_pattern}",
        rf"(?:\s*,\s*{action_keyword_pattern}){{2,}}",
    )

    for previous_keyword in _action_keywords():
        if previous_keyword == keyword:
            continue

        for list_tail_pattern in list_tail_patterns:
            for match in re.finditer(
                rf"(?<!\w){NEGATION_PREFIX_PATTERN}"
                rf"{NEGATION_TARGET_GAP_PATTERN}\s+"
                rf"{_keyword_pattern(previous_keyword)}"
                rf"(?P<tail>{list_tail_pattern})",
                text,
            ):
                if re.search(_keyword_pattern(keyword), match.group("tail")):
                    return True

    return False


def _has_explicit_negated_intent(text: str, intent: str) -> bool:
    for keyword in INTENT_KEYWORDS[intent]:
        if keyword in IMPLIED_INTENT_KEYWORDS.get(intent, set()):
            continue

        if _has_negation_before_keyword(text, keyword):
            return True

        if _has_negated_conjunction(
            text,
            keyword,
            INTENT_KEYWORDS[intent],
            SAME_INTENT_CONJUNCTION_PATTERN,
        ):
            return True

        if _has_negated_conjunction(
            text,
            keyword,
            _action_keywords(),
            ACTION_CHOICE_CONJUNCTION_PATTERN,
        ):
            return True

        if _has_negated_action_list(text, keyword):
            return True

        if keyword == "cancellation" and re.search(
            rf"(?<!\w)(?:no|not\s+a)\s+{_keyword_pattern(keyword)}",
            text,
        ):
            return True

    return False


def _is_negated_keyword(text: str, intent: str, keyword: str) -> bool:
    if intent not in NEGATION_AWARE_INTENTS:
        return False

    if (
        keyword in IMPLIED_INTENT_KEYWORDS.get(intent, set())
        and _has_explicit_negated_intent(text, intent)
    ):
        return True

    if _has_negation_before_keyword(text, keyword):
        return True

    if _has_negated_conjunction(
        text,
        keyword,
        INTENT_KEYWORDS[intent],
        SAME_INTENT_CONJUNCTION_PATTERN,
    ):
        return True

    if _has_negated_conjunction(
        text,
        keyword,
        _action_keywords(),
        ACTION_CHOICE_CONJUNCTION_PATTERN,
    ):
        return True

    if _has_negated_action_list(text, keyword):
        return True

    if keyword == "cancellation" and re.search(
        rf"(?<!\w)(?:no|not\s+a)\s+{_keyword_pattern(keyword)}",
        text,
    ):
        return True

    return False


def _count_keyword_hits(text: str, intent: str, keywords: Iterable[str]) -> int:
    hits = 0
    for keyword in keywords:
        if intent == "cancel" and keyword in IMPLIED_CANCEL_KEYWORDS:
            continue

        if _is_negated_keyword(
            text,
            intent,
            keyword,
        ):
            continue

        if _contains_keyword(text, keyword):
            hits += 1
    return hits


def _count_negated_keyword_hits(text: str, intent: str, keywords: Iterable[str]) -> int:
    return sum(
        1
        for keyword in keywords
        if _contains_keyword(text, keyword) and _is_negated_keyword(text, intent, keyword)
    )


def route_intent(text: str) -> tuple[str, str]:
    """Classify intent by keyword matching without using an LLM."""
    normalized = text.strip().lower()
    if not normalized:
        return "no_action", "No content provided; no action selected."

    negated_scores = {
        intent: _count_negated_keyword_hits(normalized, intent, keywords)
        for intent, keywords in INTENT_KEYWORDS.items()
    }
    scores = {
        intent: _count_keyword_hits(normalized, intent, keywords)
        for intent, keywords in INTENT_KEYWORDS.items()
    }

    max_score = max(scores.values())
    if max_score == 0:
        if any(negated_scores.values()):
            return "no_action", "Only negated action keyword(s) found; no action selected."

        return "no_action", "No action keywords found; no action selected."

    # Prefer rescheduling in mixed-intent messages to avoid destructive cancels.
    for intent in ("reschedule", "cancel", "schedule"):
        if scores[intent] == max_score:
            return intent, f"Matched {max_score} keyword(s) for intent '{intent}'."

    return "schedule", "Fallback to schedule."

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
APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "`": "'",
        "\u00b4": "'",
        "\u2018": "'",
        "\u2019": "'",
        "\u201b": "'",
        "\u02bc": "'",
        "\u2032": "'",
        "\uff07": "'",
    }
)
NEGATION_GAP_TOKEN_PATTERN = r"[a-z0-9']+(?:[.-][a-z0-9']+)*(?:\.)?,?"
NEGATION_DELEGATED_GAP_TOKEN_PATTERN = (
    r"(?!(?:to|but|however|instead|yet|please)\b)"
    r"(?:(?:mr|mrs|ms|dr)\.|[a-z0-9']+(?:[.-][a-z0-9']+)*)"
)
NEGATION_SAME_CLAUSE_GAP_TOKEN_PATTERN = (
    r"(?!(?:to|but|however|instead|yet)\b)"
    r"[a-z0-9']+(?:[.-][a-z0-9']+)*"
)
NEGATION_EMPHASIS_PATTERN = (
    r"(?:\s*,?\s*(?:ever|under\s+(?:any|no)\s+circumstances|"
    r"for\s+any\s+reason|i\s+repeat)\s*,?)*"
)
NEGATION_PREFIX_PATTERN = (
    r"(?:do\s+not|don't|dont|won't|wont|not|never)(?:\s+ever\b)?"
    rf"{NEGATION_EMPHASIS_PATTERN}"
)
NO_INTENT_NOUN_PATTERN = (
    r"(?:plans?|intentions?|intents?|desires?|need|needs?|reason|reasons?)"
)
DIRECT_OBJECT_NEGATION_VERB_PATTERN = (
    r"(?:want|wants|wanted|need|needs|needed|wish|wishes|wished|"
    r"desire|desires|desired)"
)
DIRECT_OBJECT_ARTICLE_PATTERN = r"(?:a|an|any|another|the)"
OPERATIONAL_NEGATION_VERB_PATTERN = (
    r"(?:process|processes|processed|proceed|proceeds|proceeded|"
    r"initiate|initiates|initiated|submit|submits|submitted|start|starts|started|"
    r"begin|begins|began|begun|handle|handles|handled|complete|completes|completed|"
    r"perform|performs|performed)"
)
NEGATION_TARGET_GAP_PATTERN = (
    r"(?:"
    r"\s+to"
    r"|\s+(?:want|wants|wanted|wish|wishes|need|needs|needed|"
    r"intend|intends|intended|plan|plans|planned|planning|try|trying|"
    r"look|looks|looked|looking|seek|seeks|seeking|sought|"
    rf"going|mean|meant)(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,3}}\s+to"
    r"|\s+(?:ask|asks|asked|tell|tells|told|instruct|instructs|instructed|"
    r"request|requests|requested|advise|advises|advised|direct|directs|directed|"
    r"order|orders|ordered|authorize|authorizes|authorized|urge|urges|urged|get|"
    r"gets|got|have|has|had|make|makes|made)"
    rf"(?:\s+{NEGATION_DELEGATED_GAP_TOKEN_PATTERN}){{0,6}}"
    rf"{NEGATION_EMPHASIS_PATTERN}(?:,?\s+to|\s+please\s+to)?"
    rf"|\s+{OPERATIONAL_NEGATION_VERB_PATTERN}"
    rf"(?:\s+{NEGATION_SAME_CLAUSE_GAP_TOKEN_PATTERN}){{0,6}}(?:\s+to)?"
    rf"|\s+(?:let|allow|permit)"
    rf"(?:\s+{NEGATION_DELEGATED_GAP_TOKEN_PATTERN}){{1,6}}"
    rf"{NEGATION_EMPHASIS_PATTERN}(?:,?\s+to|\s+please\s+to)?"
    r")?"
)
ACTION_CHOICE_CONJUNCTION_PATTERN = r"(?:\s*,?\s+(?:or|nor|and)\s+)"
SAME_INTENT_CONJUNCTION_PATTERN = r"(?:\s*,?\s+(?:or|and|nor)\s+)"
NEGATED_OBJECT_GAP_PATTERN = (
    rf"(?:\s+(?!(?:or|nor|and)\b){NEGATION_GAP_TOKEN_PATTERN})*"
)
INFORMATIONAL_CANCELLATION_PATTERN = (
    r"\bcancellation\s+(?:polic(?:y|ies)|fees?|rules?|terms?|details?|info|information)\b"
    r"|(?:\b(?:what(?:'s|\s+(?:is|are))|how(?:\s+does)?|where\s+can\s+i\s+find|"
    r"can\s+you\s+explain|could\s+you\s+explain|please\s+explain|"
    r"tell\s+me\s+about|do\s+you\s+have|is\s+there|when\s+is)\b"
    r"(?:\s+[a-z0-9']+){0,6}\s+cancellation\s+"
    r"(?:process(?:es)?|procedures?|steps?|instructions?|deadlines?|windows?|"
    r"requirements?|options?|works?)\b)"
)
INFORMATIONAL_ACTION_CONTEXT_PREFIX = (
    r"(?:"
    r"what(?:'s|\s+is)|what\s+are|where\s+can\s+i\s+find|"
    r"can\s+you\s+(?:tell|show)\s+me|could\s+you\s+(?:tell|show)\s+me|"
    r"please\s+(?:tell|show)\s+me|tell\s+me|"
    r"please\s+explain|can\s+you\s+explain|could\s+you\s+explain"
    r")"
)
INFORMATIONAL_ACTION_NOUN_PATTERN = (
    r"(?:process(?:es)?|procedures?|steps?|instructions?|requirements?|options?)"
)


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


def _has_informational_action_context(text: str, keyword: str) -> bool:
    keyword_pattern = _keyword_pattern(keyword)
    return (
        re.search(
            rf"\bhow\s+(?:do|can|could|should|would)\s+i\s+{keyword_pattern}",
            text,
        )
        is not None
        or re.search(rf"\bhow\s+to\s+{keyword_pattern}", text) is not None
        or re.search(
            rf"\b(?:can|could)\s+you\s+(?:tell|show)\s+me\s+how\s+to\s+"
            rf"{keyword_pattern}",
            text,
        )
        is not None
        or re.search(
            rf"\bplease\s+(?:tell|show)\s+me\s+how\s+to\s+{keyword_pattern}",
            text,
        )
        is not None
        or re.search(
            rf"\b{INFORMATIONAL_ACTION_CONTEXT_PREFIX}\b"
            rf"(?:\s+[a-z0-9']+){{0,8}}\s+"
            rf"{INFORMATIONAL_ACTION_NOUN_PATTERN}"
            rf"(?:\s+[a-z0-9']+){{0,5}}\s+(?:to|for)\s+{keyword_pattern}",
            text,
        )
        is not None
        or re.search(
            rf"\b{INFORMATIONAL_ACTION_CONTEXT_PREFIX}\b"
            rf"(?:\s+[a-z0-9']+){{0,8}}\s+"
            rf"{INFORMATIONAL_ACTION_NOUN_PATTERN}"
            rf"(?:\s+[a-z0-9']+){{0,5}}\s+{keyword_pattern}",
            text,
        )
        is not None
    )


def _unless_clause_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"(?<!\w)unless\b", text):
        after_unless = text[match.end() :]
        boundary = re.search(r"(?:[.;!?]|,\s*(?:then|please|but|and|or)\b)", after_unless)
        end = match.end() + boundary.start() if boundary else len(text)
        spans.append((match.start(), end))

    return tuple(spans)


def _has_unless_condition_context(text: str, keyword: str) -> bool:
    keyword_matches = tuple(re.finditer(_keyword_pattern(keyword), text))
    if not keyword_matches:
        return False

    unless_spans = _unless_clause_spans(text)
    return bool(unless_spans) and all(
        any(start <= match.start() and match.end() <= end for start, end in unless_spans)
        for match in keyword_matches
    )


def _is_ignored_keyword_context(text: str, intent: str, keyword: str) -> bool:
    if _has_informational_action_context(text, keyword):
        return True

    if _has_unless_condition_context(text, keyword):
        return True

    return (
        intent == "cancel"
        and keyword == "cancellation"
        and re.search(INFORMATIONAL_CANCELLATION_PATTERN, text) is not None
    )


def _has_negated_belief_before_keyword(text: str, keyword: str) -> bool:
    keyword_pattern = _keyword_pattern(keyword)
    if re.search(
        rf"(?<!\w){NEGATION_PREFIX_PATTERN}\s+"
        r"(?:think|believe|feel|suppose|expect)"
        rf"(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,4}}\s+"
        r"(?:need|needs|needed|want|wants|wanted|should|would|have|has|had|"
        r"ought|reason|reasons)"
        rf"(?:\s+to)?\s+{keyword_pattern}",
        text,
    ):
        return True

    return (
        re.search(
            rf"(?<!\w){NEGATION_PREFIX_PATTERN}\s+"
            r"(?:think|believe|feel|suppose|expect)"
            rf"(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,5}}\s+"
            rf"{keyword_pattern}(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,3}}\s+"
            r"(?:is|are|will\s+be|would\s+be)\s+needed\b",
            text,
        )
        is not None
    )


def _has_negated_direct_object_before_keyword(text: str, keyword: str) -> bool:
    return (
        re.search(
            rf"(?<!\w){NEGATION_PREFIX_PATTERN}\s+"
            rf"{DIRECT_OBJECT_NEGATION_VERB_PATTERN}"
            rf"(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,2}}\s+"
            rf"(?:{DIRECT_OBJECT_ARTICLE_PATTERN}\s+)?"
            rf"{_keyword_pattern(keyword)}",
            text,
        )
        is not None
    )


def _has_no_intent_noun_before_keyword(text: str, keyword: str) -> bool:
    return (
        re.search(
            rf"(?<!\w)(?:no|not\s+any)\s+{NO_INTENT_NOUN_PATTERN}"
            rf"(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,3}}\s+"
            rf"(?:to|for)\s+(?:{DIRECT_OBJECT_ARTICLE_PATTERN}\s+)?"
            rf"{_keyword_pattern(keyword)}",
            text,
        )
        is not None
    )


def _has_negation_before_keyword(text: str, keyword: str) -> bool:
    if _has_negated_belief_before_keyword(text, keyword):
        return True

    if _has_negated_direct_object_before_keyword(text, keyword):
        return True

    if re.search(
        rf"(?<!\w){NEGATION_PREFIX_PATTERN}"
        rf"{NEGATION_TARGET_GAP_PATTERN}\s+{_keyword_pattern(keyword)}",
        text,
    ):
        return True

    if _has_no_intent_noun_before_keyword(text, keyword):
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

        if re.search(
            rf"(?<!\w)(?:no|not\s+a|not\s+an)\s+{_keyword_pattern(keyword)}",
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

    if re.search(
        rf"(?<!\w)(?:no|not\s+a|not\s+an)\s+{_keyword_pattern(keyword)}",
        text,
    ):
        return True

    return False


def _count_keyword_hits(text: str, intent: str, keywords: Iterable[str]) -> int:
    hits = 0
    for keyword in keywords:
        if intent == "cancel" and keyword in IMPLIED_CANCEL_KEYWORDS:
            continue

        if not _contains_keyword(text, keyword):
            continue

        if _is_ignored_keyword_context(text, intent, keyword):
            continue

        if _is_negated_keyword(
            text,
            intent,
            keyword,
        ):
            continue

        hits += 1
    return hits


def _count_negated_keyword_hits(text: str, intent: str, keywords: Iterable[str]) -> int:
    return sum(
        1
        for keyword in keywords
        if _contains_keyword(text, keyword)
        and not _is_ignored_keyword_context(text, intent, keyword)
        and _is_negated_keyword(text, intent, keyword)
    )


def route_intent(text: str) -> tuple[str, str]:
    """Classify intent by keyword matching without using an LLM."""
    normalized = text.strip().lower().translate(APOSTROPHE_TRANSLATION)
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

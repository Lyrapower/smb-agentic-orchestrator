"""Rule-based intent router for chat text."""

from __future__ import annotations

import re
import unicodedata
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
NEGATION_DELEGATED_GAP_TOKEN_NO_COMMA_PATTERN = (
    r"(?!(?:to|but|however|instead|yet|please)\b)"
    r"(?:(?:mr|mrs|ms|dr)\.|[a-z0-9']+(?:[.-][a-z0-9']+)*)"
)
NEGATION_DELEGATED_GAP_TOKEN_PATTERN = (
    rf"{NEGATION_DELEGATED_GAP_TOKEN_NO_COMMA_PATTERN},?"
)
NEGATION_SAME_CLAUSE_GAP_TOKEN_NO_COMMA_PATTERN = (
    r"(?!(?:to|but|however|instead|yet|please)\b)"
    r"[a-z0-9']+(?:[.-][a-z0-9']+)*"
)
NEGATION_SAME_CLAUSE_GAP_TOKEN_PATTERN = (
    rf"{NEGATION_SAME_CLAUSE_GAP_TOKEN_NO_COMMA_PATTERN},?"
)
NEGATION_EMPHASIS_PATTERN = (
    r"(?:\s*,?\s*(?:ever|under\s+(?:any|no)\s+circumstances|"
    r"for\s+any\s+reason|i\s+repeat)\s*,?)*"
)
# Hedge/intensifier adverbs that commonly sit between a negation and an intent
# bridge ("don't really want to cancel") without changing refusal meaning.
NEGATION_ADVERB_PATTERN = (
    r"(?:really|actually|even|particularly|currently|also|still|always|"
    r"just|quite|truly|simply|honestly|especially|generally|normally|"
    r"usually|necessarily|exactly|literally)"
)
NEGATION_PREFIX_PATTERN = (
    r"(?:do\s+not|don't|dont|donot|won't|wont|"
    r"shouldn't|shouldnt|mustn't|mustnt|wouldn't|wouldnt|couldn't|couldnt|"
    r"didn't|didnt|wasn't|wasnt|weren't|werent|"
    r"haven't|havent|hadn't|hadnt|hasn't|hasnt|"
    r"isn't|isnt|aren't|arent|"
    r"ain't|aint|"
    r"not|never|no\s+longer)(?:\s+ever\b)?"
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
# Determiners/prepositions that may sit between a negated want/need verb and its
# noun object. Arbitrary content words are intentionally excluded so mixed
# phrases like "I don't want Tuesday, please cancel" keep the affirmative action.
DIRECT_OBJECT_PRE_NOUN_PATTERN = (
    rf"(?:"
    rf"(?:{DIRECT_OBJECT_ARTICLE_PATTERN}\s+)"
    rf"|(?:(?:for|of)\s+(?:{DIRECT_OBJECT_ARTICLE_PATTERN}\s+)?)"
    rf"|(?:(?:my|our|your|his|her|their)\s+)"
    rf")"
)
OPERATIONAL_NEGATION_VERB_PATTERN = (
    r"(?:process|processes|processed|proceed|proceeds|proceeded|"
    r"initiate|initiates|initiated|submit|submits|submitted|start|starts|started|"
    r"begin|begins|began|begun|handle|handles|handled|complete|completes|completed|"
    r"perform|performs|performed)"
)
NEGATION_TARGET_GAP_PATTERN = (
    r"(?:"
    r"\s+to"
    # Colloquial contractions of "want to" / "going to", with optional hedges.
    rf"|(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+(?:wanna|gonna)"
    rf"|(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+"
    r"(?:want|wants|wanted|wish|wishes|need|needs|needed|"
    r"intend|intends|intended|plan|plans|planned|planning|try|trying|"
    r"look|looks|looked|looking|seek|seeks|seeking|sought|"
    rf"going|mean|meant|about)(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,3}}\s+to"
    r"|\s+(?:ask|asks|asked|tell|tells|told|instruct|instructs|instructed|"
    r"request|requests|requested|advise|advises|advised|direct|directs|directed|"
    r"order|orders|ordered|authorize|authorizes|authorized|urge|urges|urged|get|"
    r"gets|got|have|has|had|make|makes|made)"
    r"(?:"
    rf"(?:\s+{NEGATION_DELEGATED_GAP_TOKEN_PATTERN}){{0,6}}"
    rf"{NEGATION_EMPHASIS_PATTERN}(?:,?\s+to|\s+please\s+to)"
    rf"|"
    rf"(?:\s+{NEGATION_DELEGATED_GAP_TOKEN_NO_COMMA_PATTERN}){{0,6}}"
    rf"{NEGATION_EMPHASIS_PATTERN}"
    r")"
    rf"|\s+{OPERATIONAL_NEGATION_VERB_PATTERN}"
    r"(?:"
    rf"(?:\s+{NEGATION_SAME_CLAUSE_GAP_TOKEN_PATTERN}){{0,6}}\s+to"
    rf"|"
    rf"(?:\s+{NEGATION_SAME_CLAUSE_GAP_TOKEN_NO_COMMA_PATTERN}){{0,6}}"
    r")"
    rf"|\s+(?:let|allow|permit)"
    r"(?:"
    rf"(?:\s+{NEGATION_DELEGATED_GAP_TOKEN_PATTERN}){{1,6}}"
    rf"{NEGATION_EMPHASIS_PATTERN}(?:,?\s+to|\s+please\s+to)"
    rf"|"
    rf"(?:\s+{NEGATION_DELEGATED_GAP_TOKEN_NO_COMMA_PATTERN}){{1,6}}"
    rf"{NEGATION_EMPHASIS_PATTERN}"
    r")"
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


def _normalize_route_text(text: str) -> str:
    """Normalize chat text so appositive punctuation does not break negation."""
    normalized = text.strip().lower()
    # Unwrap markdown code spans/fences of any length (`not`, ``not``, ```not```)
    # before leftover lone backticks are treated as apostrophes (don`t -> don't).
    normalized = re.sub(r"`+([^`]+)`+", r" \1 ", normalized)
    # Translate apostrophe lookalikes before NFKC: NFKC would turn acute accent
    # (don´t) into space+combining acute, which later Mn stripping breaks into
    # "don t" and loses negation.
    normalized = normalized.translate(APOSTROPHE_TRANSLATION)
    # Compatibility-normalize so fullwidth Latin (ｎｏｔ / ｃａｎｃｅｌ) and
    # fullwidth punctuation from IME/CJK paste fold to ASCII before matching.
    normalized = unicodedata.normalize("NFKC", normalized)
    # Strip HTML/XML tags from rich-text/email paste before leftover <> marks are
    # spaced out; otherwise <b>not</b> becomes "b not /b" and breaks negation.
    normalized = re.sub(r"</?[a-z][a-z0-9]*(?:\s[^>]*)?/?>", " ", normalized)
    # Treat grouping/quote marks like surrounding words so negation can span them.
    # Include ASCII angle brackets used for email-style appositives, European
    # guillemets, low-9 quotes, and CJK corner/lenticular/angle brackets
    # (including halfwidth forms that NFKC folds into these code points).
    normalized = re.sub(
        r"[()\[\]{}\"\u201c\u201d\u201e\u201a\u201f"
        r"\u00ab\u00bb\u2039\u203a<>"
        r"\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f"
        r"\u3010\u3011\u3014\u3015\u3016\u3017]",
        " ",
        normalized,
    )
    # Strip markdown emphasis so forms like *not* / _not_ / ~~not~~ still negate.
    normalized = re.sub(r"[*_~]+", " ", normalized)
    # Soft hyphens are invisible line-break markers inside words; strip so
    # "can\u00adcel" still matches "cancel". That can also glue "do\u00adnot" into
    # "donot", which is treated as a negation prefix synonym below.
    # Remaining Unicode format chars (Cf) from copy/paste (ZWSP, LRM/RLM, bidi
    # isolates/embeddings, BOM, etc.) are replaced with spaces so
    # "do not\u200ecancel" stays a negated phrase instead of gluing the action
    # keyword past the negation boundary.
    # Combining/enclosing marks (Mn/Me), including variation selectors, are
    # folded away after NFD so "do not\u0301 cancel" / "do not\ufe0f cancel"
    # still negate and affirmative "can\u0301cel" still matches.
    normalized = normalized.replace("\u00ad", "")
    normalized = unicodedata.normalize("NFD", normalized)
    normalized = "".join(
        " " if unicodedata.category(ch) == "Cf" else ch
        for ch in normalized
        if unicodedata.category(ch) not in {"Mn", "Me"}
    )
    # Ellipses (ASCII and unicode) are pauses, not tokens.
    normalized = re.sub(r"\.{2,}|\u2026", " ", normalized)
    # Convert slash-joined phrases like Sarah/my assistant into separate tokens.
    normalized = re.sub(r"(?<=[a-z0-9])/(?=[a-z0-9])", " ", normalized)
    normalized = re.sub(r"\s/\s", " ", normalized)
    # Convert dash/colon/semicolon appositives to commas so comma-bridge rules apply.
    # Keep hyphenated names (Mary-Jane) and clock times (3:30).
    # Include figure dash, en/em dashes, horizontal bar, minus, and ASCII --.
    normalized = re.sub(
        r"\s*[\u2012\u2013\u2014\u2015\u2212]+\s*",
        ", ",
        normalized,
    )
    normalized = re.sub(r"\s*--+\s*", ", ", normalized)
    normalized = re.sub(r"\s+-\s+", ", ", normalized)
    normalized = re.sub(r"[:;](?!\d)\s*", ", ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


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
        rf"(?<!\w){NEGATION_PREFIX_PATTERN}"
        rf"(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+"
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
            rf"(?<!\w){NEGATION_PREFIX_PATTERN}"
            rf"(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+"
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
            rf"(?<!\w){NEGATION_PREFIX_PATTERN}"
            rf"(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+"
            rf"{DIRECT_OBJECT_NEGATION_VERB_PATTERN}\s+"
            rf"{DIRECT_OBJECT_PRE_NOUN_PATTERN}"
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
    normalized = _normalize_route_text(text)
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

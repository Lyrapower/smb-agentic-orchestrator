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
    r"(?:\s*,?\s*(?:ever|"
    r"under\s+(?:any|no)\s+circumstances?(?!\w)|"
    r"in\s+no\s+circumstances?(?!\w)|"
    r"on\s+no\s+account(?!\w)|"
    r"for\s+any\s+reason|i\s+repeat)\s*,?)*"
)
# Hedge/intensifier adverbs that commonly sit between a negation and an intent
# bridge ("don't really want to cancel") without changing refusal meaning.
NEGATION_ADVERB_PATTERN = (
    r"(?:really|actually|even|particularly|currently|also|still|always|"
    r"just|quite|truly|simply|honestly|especially|generally|normally|"
    r"usually|necessarily|exactly|literally)"
)
# Same hedges after an already-matched bridge ("going to really cancel",
# "gonna really have them cancel"). Keep these inside matched bridges so a
# bare adverb after negation ("Please don't actually cancel") does not gain a
# new no-bridge match.
NEGATION_TRAILING_ADVERB_PATTERN = rf"(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}"
# Whole-token auxiliaries that may sit between a negation and an already
# recognized intent-bridge or delegated verb ("don't be asking them to cancel",
# "haven't been going to cancel"). Keep these bounded so "been able to cancel"
# stays a help request rather than a refusal.
NEGATION_BE_AUXILIARY_PATTERN = r"(?:be|been)"
# Optional be/been plus trailing hedges ("gonna be really asking",
# "going to really be having them cancel").
NEGATION_OPTIONAL_BE_WITH_ADVERBS_PATTERN = (
    rf"(?:\s+{NEGATION_BE_AUXILIARY_PATTERN}{NEGATION_TRAILING_ADVERB_PATTERN})?"
)
# "needn't" is the written contraction of already-handled "need not" /
# "don't need to". "shan't" is the written contraction of already-handled
# "shall not". "oughtn't" is the written contraction of already-handled
# "ought not". "daren't" is the written contraction of already-handled
# "dare not". "mayn't" is the written contraction of already-handled
# "may not". "mightn't" is the written contraction of already-handled
# "might not". Mixed "I needn't wait, please cancel" / "I shan't wait,
# please cancel" / "I oughtn't wait, please cancel" / "I daren't wait,
# please cancel" / "I mayn't wait, please cancel" / "I mightn't wait,
# please cancel" must stay executable.
NEGATION_PREFIX_PATTERN = (
    r"(?:do\s+not|don't|dont|donot|doesn't|doesnt|won't|wont|"
    r"shouldn't|shouldnt|mustn't|mustnt|needn't|neednt|shan't|shant|"
    r"oughtn't|oughtnt|daren't|darent|mayn't|maynt|mightn't|mightnt|"
    r"wouldn't|wouldnt|couldn't|couldnt|"
    r"didn't|didnt|wasn't|wasnt|weren't|werent|"
    r"haven't|havent|hadn't|hadnt|hasn't|hasnt|"
    r"isn't|isnt|aren't|arent|"
    r"ain't|aint|"
    r"not|never|no\s+longer)(?:\s+ever\b)?"
    rf"{NEGATION_EMPHASIS_PATTERN}"
)
# Standalone "under no circumstances <action>" is itself a refusal. Singular
# "under no circumstance", "in no circumstances", and "on no account" are the
# same prohibition. The same phrases after don't/never match via
# NEGATION_EMPHASIS_PATTERN. Do not add these idioms to
# NEGATION_PREFIX_PATTERN: that would invert "Under no circumstances I refuse
# to cancel" into an executable cancel, the same way "Don't refuse to cancel"
# must stay executable. Do not treat "under any circumstances" as this prefix;
# without a negation word it is affirmative. Inversion auxiliaries cover
# "should you" / "will I" / "do I" / "am I", not "can"/"could". Extra subject
# tokens stop before "not" so "do not forget to cancel" stays executable.
# "Don't go and cancel" uses the go-and/come-and bridge in the shared gap,
# not this prefix.
# Trailing comma belongs to the idiom ("circumstances, cancel"). Do not eat the
# following space; the action keyword still needs a whitespace boundary.
UNDER_NO_CIRCUMSTANCES_IDIOM_PATTERN = (
    r"(?:(?:under|in)\s+no\s+circumstances?|on\s+no\s+account)(?!\w)"
    r"(?:\s*,)?"
)
UNDER_NO_CIRCUMSTANCES_SUBJECT_PATTERN = (
    r"(?:anyone|someone|everybody|everyone|their|they|this|that|she|you|"
    r"your|our|his|her|the|we|he|it|an|my|i|a)(?!\w)"
)
UNDER_NO_CIRCUMSTANCES_SUBJECT_TOKEN_PATTERN = (
    r"(?!(?:to|but|however|instead|yet|please|not)\b)"
    r"[a-z0-9']+(?:[.-][a-z0-9']+)*"
)
UNDER_NO_CIRCUMSTANCES_INVERSION_PATTERN = (
    r"(?:\s+(?:should|would|will|shall|must|do|does|did|am|is|are|was|were)"
    rf"(?:\s+{UNDER_NO_CIRCUMSTANCES_SUBJECT_PATTERN}"
    rf"(?:\s+{UNDER_NO_CIRCUMSTANCES_SUBJECT_TOKEN_PATTERN}){{0,3}})?)?"
)
UNDER_NO_CIRCUMSTANCES_PREFIX_PATTERN = (
    rf"{UNDER_NO_CIRCUMSTANCES_IDIOM_PATTERN}"
    rf"{UNDER_NO_CIRCUMSTANCES_INVERSION_PATTERN}"
    r"(?:\s+ever\b)?"
    rf"{NEGATION_EMPHASIS_PATTERN}"
)
# Speech-act refusals: the verb itself is the negation ("I refuse to cancel"),
# so these must not be added to NEGATION_PREFIX_PATTERN. "Don't refuse to
# cancel" / "I cannot refuse to cancel" / "I don't think I should refuse to
# cancel" invert the speech act and stay executable, same as "do not forget
# to cancel". Longer inflections are listed first so "refusing" does not
# stick on the "refuse" alternative.
SPEECH_ACT_REFUSAL_VERB_PATTERN = (
    r"(?:refusing|refuses|refused|refuse|"
    r"declining|declines|declined|decline)"
)
SPEECH_ACT_REFUSAL_PREFIX_PATTERN = (
    rf"{SPEECH_ACT_REFUSAL_VERB_PATTERN}(?:\s*,\s*|\s+)to\b"
)
SPEECH_ACT_REFUSAL_SUBJECT_PRONOUN_PATTERN = r"(?:you|we|they|he|she|i)"
# Noun-object speech acts ("I refuse the cancellation", "I decline a
# cancellation") have no "to", so the infinitive prefix above misses them and
# the action noun stays live. Keep the object to an optional determiner plus
# the action keyword. An open gap would treat "I refuse, please cancel" as a
# refusal. Inverted "Don't refuse the cancellation" still stays executable
# because the verb itself is negated. Do not add these verbs to
# NEGATION_PREFIX_PATTERN.
SPEECH_ACT_NOUN_OBJECT_DETERMINER_PATTERN = (
    r"(?:a|an|any|another|the|this|that|my|our|your|his|her|their)"
)
# Noun refusals after "no" / "not any" / contracted "haven't/hasn't/hadn't any"
# / "isn't/aren't/wasn't/weren't any". "wish(es)" is the noun counterpart of
# already-covered "desire(s)" so "I have no wish to cancel" matches the same
# way "I have no desire to cancel" already does. "I haven't any wish/plans
# to cancel" is the written contraction of already-handled "I have not any
# ...". "There aren't any plans to cancel" / "There isn't any wish to cancel"
# is the same contraction of already-handled "There are not any ..." /
# "There is not any ...". Keep this to-taking/for-noun only; mixed
# "I have no wish Tuesday, please cancel" / "I haven't any wish Tuesday,
# please cancel" / "There aren't any plans Tuesday, please cancel" must stay
# executable.
NO_INTENT_NOUN_PATTERN = (
    r"(?:plans?|intentions?|intents?|desires?|wish(?:es)?|need|needs?|reason|reasons?)"
)
NO_INTENT_NOUN_DETERMINER_PATTERN = (
    r"(?:no|not\s+any|(?:have|has|had|is|are|was|were)n(?:o)?'?t\s+any)"
)
DIRECT_OBJECT_NEGATION_VERB_PATTERN = (
    r"(?:want|wants|wanted|need|needs|needed|wish|wishes|wished|"
    r"desire|desires|desired)"
)
DIRECT_OBJECT_ARTICLE_PATTERN = r"(?:a|an|any|another|the)"
# Determiners allowed after "go ahead/through with" so "with the cancellation"
# attaches while "with Tuesday, please cancel" stays a mixed-message action.
GO_AHEAD_DETERMINER_PATTERN = (
    rf"(?:{DIRECT_OBJECT_ARTICLE_PATTERN}|my|our|your|his|her|their|this|that)"
)
# "go and <action>" / "come and <action>" are the same proceed-to-action idiom
# as "go ahead and <action>" ("Don't go and cancel", "Please don't come and
# cancel", "Under no circumstances go and cancel"). Require an immediate "and"
# so "don't go, please cancel", "don't go, and cancel", and "don't come
# Tuesday, please cancel" stay mixed-message actions. Do not match
# going/coming, and do not add a generic "and" after other verbs
# ("don't look and cancel" / "don't want and cancel" stay executable).
GO_AND_COME_AND_PATTERN = r"(?:go|come)(?!\w)\s+and"
# Verbs that take a "to <action>" complement. Shared by the generic to-path and
# by nested "VERB to try and" / "VERB to go ahead/through with" so hoping/looking
# and the rest of this set attach the same way going/want/plan/intend already do.
# Do not add a generic "and" after these verbs; "look and" / "want and" mixed
# messages must stay executable.
INTENT_BRIDGE_TO_TAKING_VERB_PATTERN = (
    r"(?:want|wants|wanted|wanting|like|likes|liked|liking|"
    r"wish|wishes|wished|wishing|need|needs|needed|needing|"
    r"desire|desires|desired|desiring|"
    r"intend|intends|intended|intending|plan|plans|planned|planning|"
    r"try|trying|attempt|attempts|attempting|"
    r"look|looks|looked|looking|seek|seeks|seeking|sought|"
    r"going|mean|meant|meaning|hope|hopes|hoped|hoping|about|supposed|willing|ready|prepared|"
    r"allowed|permitted|require|requires|required|requiring|"
    r"obligate|obligates|obligated|obligating|oblige|obliges|obliged|obliging|"
    r"expect|expects|expected|expecting|"
    r"force|forces|forced|forcing|compel|compels|compelled|compelling|"
    r"mandate|mandates|mandated|mandating|"
    r"pressure|pressures|pressured|pressuring|"
    r"coerce|coerces|coerced|coercing|being)"
)
# Delegated speech/causative verbs that take a person object then "to <action>".
# Shared by the generic delegated path and by nested "ask them to try and" /
# "tell them to go ahead/through with". Progressive having/letting/making are
# the same causative refusals as have/let/make ("I'm not having/letting/making
# them cancel"); allowing/permitting are the same as allow/permit
# ("I'm not allowing/permitting them to cancel"). Asking/telling gerunds
# already work. Do not add getting: "I'm not getting to cancel" is help-seeking.
NEGATION_DELEGATED_VERB_PATTERN = (
    r"(?:ask|asks|asked|asking|tell|tells|told|telling|"
    r"instruct|instructs|instructed|instructing|"
    r"request|requests|requested|requesting|advise|advises|advised|advising|"
    r"direct|directs|directed|directing|order|orders|ordered|ordering|"
    r"authorize|authorizes|authorized|authorizing|urge|urges|urged|urging|get|"
    r"gets|got|have|has|had|having|make|makes|made|making|letting|"
    r"allowing|permitting)"
)
# Optional nested "VERB (object) to" before try-and / go-ahead / go-through-with.
# To-taking bridges keep the existing 0-3 generic gaps. Delegated/let verbs use
# the delegated gap, which will not skip past "to", so "ask them to try and"
# attaches while "ask them to wait, please cancel" stays mixed-message executable.
# Optional be/been after the first "to" covers "going to be asking them to try
# and cancel" / "going to be going ahead"; trailing hedges cover "going to
# really ask them to try and cancel". "able" is not a delegated verb, so
# "going to be able to try and cancel" does not use this nest.
NEGATION_NESTED_TO_COMPLEMENT_INNER_PATTERN = (
    rf"(?:(?:{INTENT_BRIDGE_TO_TAKING_VERB_PATTERN})"
    rf"(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,3}}\s+to"
    rf"{NEGATION_TRAILING_ADVERB_PATTERN}"
    rf"{NEGATION_OPTIONAL_BE_WITH_ADVERBS_PATTERN}"
    rf"(?:\s+(?:{NEGATION_DELEGATED_VERB_PATTERN}|let|allow|permit)"
    rf"(?:\s+{NEGATION_DELEGATED_GAP_TOKEN_PATTERN}){{0,6}}\s+to)?"
    rf"|(?:{NEGATION_DELEGATED_VERB_PATTERN}|let|allow|permit)"
    rf"(?:\s+{NEGATION_DELEGATED_GAP_TOKEN_PATTERN}){{0,6}}\s+to)"
)
NEGATION_NESTED_TO_COMPLEMENT_PREFIX_PATTERN = (
    rf"(?:{NEGATION_NESTED_TO_COMPLEMENT_INNER_PATTERN}\s+)?"
)
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
    r"(?:"
    r"\s+to"
    # Colloquial contractions of "want to" / "going to", with optional hedges.
    # Also "gonna/wanna try to/and cancel" and "gonna/wanna go ahead and/to/with
    # cancel", which the bare contraction path cannot reach because it expects
    # the action keyword immediately after. Nested "gonna/wanna ask them to
    # (try and / go ahead / go and / come and) cancel" needs the same
    # delegated/to-taking complement as the non-contraction path; "to" is
    # excluded from delegated gaps, so the bare contraction cannot skip from
    # "gonna" to "cancel".
    # Optional be/been after gonna/wanna covers "gonna be asking them to cancel"
    # / "gonna be trying to cancel"; trailing hedges cover "gonna really cancel"
    # / "gonna really have them cancel" / "gonna be really asking". "able" is
    # not a bridge verb, so "gonna be able to cancel" stays a help request.
    rf"|(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+(?:wanna|gonna)"
    rf"{NEGATION_TRAILING_ADVERB_PATTERN}"
    rf"{NEGATION_OPTIONAL_BE_WITH_ADVERBS_PATTERN}"
    rf"(?:"
    rf"\s+{NEGATION_NESTED_TO_COMPLEMENT_INNER_PATTERN}"
    rf"(?:"
    rf"\s+(?:try|trying)(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,3}}\s+(?:to|and)"
    rf"|"
    rf"\s+go(?:ing)?\s+ahead"
    rf"(?:\s+(?:and|to|with(?:\s+{GO_AHEAD_DETERMINER_PATTERN})?))?"
    rf"|"
    rf"\s+go(?:ing)?\s+through\s+with(?:\s+{GO_AHEAD_DETERMINER_PATTERN})?"
    rf"|"
    rf"\s+{GO_AND_COME_AND_PATTERN}"
    rf")?"
    rf"|"
    rf"\s+(?:{NEGATION_DELEGATED_VERB_PATTERN}|let|allow|permit)"
    rf"(?:\s+{NEGATION_DELEGATED_GAP_TOKEN_NO_COMMA_PATTERN}){{1,6}}"
    rf"|"
    rf"\s+(?:try|trying)(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,3}}\s+(?:to|and)"
    rf"|"
    rf"\s+go(?:ing)?\s+ahead"
    rf"(?:\s+(?:and|to|with(?:\s+{GO_AHEAD_DETERMINER_PATTERN})?))?"
    rf"|"
    rf"\s+go(?:ing)?\s+through\s+with(?:\s+{GO_AHEAD_DETERMINER_PATTERN})?"
    rf"|"
    rf"\s+{GO_AND_COME_AND_PATTERN}"
    rf")?"
    rf"|(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}"
    rf"(?:\s+{NEGATION_BE_AUXILIARY_PATTERN})?\s+"
    rf"{INTENT_BRIDGE_TO_TAKING_VERB_PATTERN}"
    rf"(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,3}}\s+to"
    # Nested "going/want/about to have/let/make them cancel" has no second "to".
    # "I'm not going to ask them to cancel" already works via the second "to",
    # and "I'm not gonna have them cancel" already works via the contraction
    # path, but "to" is excluded from delegated gaps so the generic to-path
    # cannot skip from "going to have them" to "cancel". Optional object gaps
    # reuse the no-to delegated shape; "have to cancel" still uses the second
    # "to", and mixed "have them wait, please cancel" stays executable.
    # Optional be/been after that "to" covers "going to be asking them to
    # cancel" / "going to be having them cancel". Trailing hedges cover
    # "going to really cancel" / "going to really have them cancel" /
    # "going to be really asking". After be, a to-taking verb covers
    # "going to be really trying to cancel" the same way "gonna be really
    # trying" already does. Keep be bound to a delegated/to-taking verb so
    # "going to be able to cancel" does not gain a new match here.
    rf"{NEGATION_TRAILING_ADVERB_PATTERN}"
    rf"(?:"
    rf"(?:\s+(?:{NEGATION_BE_AUXILIARY_PATTERN}{NEGATION_TRAILING_ADVERB_PATTERN}\s+)?"
    rf"(?:{NEGATION_DELEGATED_VERB_PATTERN}|let|allow|permit)"
    rf"(?:"
    rf"(?:\s+{NEGATION_DELEGATED_GAP_TOKEN_PATTERN}){{0,6}}\s+to"
    rf"|"
    rf"(?:\s+{NEGATION_DELEGATED_GAP_TOKEN_NO_COMMA_PATTERN}){{1,6}}"
    rf"))"
    rf"|"
    rf"(?:\s+{NEGATION_BE_AUXILIARY_PATTERN}{NEGATION_TRAILING_ADVERB_PATTERN}\s+"
    rf"{INTENT_BRIDGE_TO_TAKING_VERB_PATTERN}"
    rf"(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,3}}\s+to)"
    rf")?"
    # Colloquial "try and cancel" is synonymous with already-handled "try to cancel".
    # Nested "hoping/looking/going to try and cancel" needs the same to-taking
    # bridge before try-and; the generic to-path only accepts "to", so "try and"
    # never attached. Delegated "ask/tell/have them to try and cancel" needs the
    # same nested prefix; "to" is excluded from delegated gaps, so the generic
    # delegated path stops at "to try" instead of the action keyword.
    rf"|(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}"
    rf"(?:\s+{NEGATION_BE_AUXILIARY_PATTERN})?\s+"
    rf"{NEGATION_NESTED_TO_COMPLEMENT_PREFIX_PATTERN}"
    rf"{NEGATION_TRAILING_ADVERB_PATTERN}"
    r"(?:try|trying)"
    rf"(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,3}}\s+and"
    # "go ahead and/to/with" and "go through with" are proceed-to-action idioms,
    # not a generic "go". Nested "hoping/looking/going to go ahead and cancel"
    # needs the same to-taking bridge before the idiom; progressive
    # "going ahead/through with" is the same shape as already-handled "trying".
    # Delegated "ask/tell/allow them to go ahead/through with" uses the same
    # nested prefix as try-and.
    rf"|(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}"
    rf"(?:\s+{NEGATION_BE_AUXILIARY_PATTERN})?\s+"
    rf"{NEGATION_NESTED_TO_COMPLEMENT_PREFIX_PATTERN}"
    rf"{NEGATION_TRAILING_ADVERB_PATTERN}"
    r"(?:go(?:ing)?\s+ahead"
    rf"(?:\s+(?:and|to|with(?:\s+{GO_AHEAD_DETERMINER_PATTERN})?))?"
    rf"|go(?:ing)?\s+through\s+with(?:\s+{GO_AHEAD_DETERMINER_PATTERN})?)"
    # "go and" / "come and" are the bare form of that idiom. Same nested
    # prefix so "going to go and" / "ask them to come and" attach, while a
    # comma or an extra word before "and" does not.
    rf"|(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+"
    rf"{NEGATION_NESTED_TO_COMPLEMENT_PREFIX_PATTERN}"
    rf"{NEGATION_TRAILING_ADVERB_PATTERN}"
    rf"{GO_AND_COME_AND_PATTERN}"
    rf"|(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}"
    rf"(?:\s+{NEGATION_BE_AUXILIARY_PATTERN})?"
    rf"\s+{NEGATION_DELEGATED_VERB_PATTERN}"
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
    r")"
    rf"{NEGATION_TRAILING_ADVERB_PATTERN}"
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
            rf"(?<!\w){NO_INTENT_NOUN_DETERMINER_PATTERN}\s+{NO_INTENT_NOUN_PATTERN}"
            rf"(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,3}}\s+"
            rf"(?:to|for)\s+(?:{DIRECT_OBJECT_ARTICLE_PATTERN}\s+)?"
            rf"{_keyword_pattern(keyword)}",
            text,
        )
        is not None
    )


def _speech_act_verb_is_negated(text: str, verb_start: int) -> bool:
    """True when refuse/decline is itself under a negation or inability."""
    before = text[:verb_start]
    if re.search(
        rf"(?<!\w){NEGATION_PREFIX_PATTERN}{NEGATION_TARGET_GAP_PATTERN}"
        rf"(?:(?:\s+{SPEECH_ACT_REFUSAL_SUBJECT_PRONOUN_PATTERN})"
        rf"|(?:\s+{NEGATION_BE_AUXILIARY_PATTERN}"
        rf"(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}})?\s+able\s+to"
        rf")?\s+$",
        before,
    ):
        return True

    if re.search(
        rf"(?<!\w){NEGATION_PREFIX_PATTERN}"
        rf"(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+"
        r"(?:think|believe|feel|suppose|expect)"
        rf"(?:\s+{NEGATION_GAP_TOKEN_PATTERN}){{0,4}}\s+"
        r"(?:need|needs|needed|want|wants|wanted|should|would|have|has|had|"
        r"ought|reason|reasons)"
        rf"(?:\s+to)?\s+$",
        before,
    ):
        return True

    return (
        re.search(
            rf"(?<!\w)(?:can't|cannot|can\s+not|unable|not\s+able)"
            rf"(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}"
            rf"\s+(?:to\s+)?$",
            before,
        )
        is not None
        or re.search(
            rf"(?<!\w)not(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+"
            rf"(?:able|unable)(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+to\s+$",
            before,
        )
        is not None
    )


def _speech_act_refusal_tails(text: str) -> tuple[str, ...]:
    if "refus" not in text and "declin" not in text:
        return ()

    tails: list[str] = []
    for match in re.finditer(
        rf"(?<!\w){SPEECH_ACT_REFUSAL_PREFIX_PATTERN}",
        text,
    ):
        if _speech_act_verb_is_negated(text, match.start()):
            continue
        tails.append(text[match.end() :])
    return tuple(tails)


def _unnegated_speech_act_verb_ends(text: str) -> tuple[int, ...]:
    """End offsets of refuse/decline verbs that are not themselves negated."""
    if "refus" not in text and "declin" not in text:
        return ()

    ends: list[int] = []
    for match in re.finditer(
        rf"(?<!\w){SPEECH_ACT_REFUSAL_VERB_PATTERN}(?!\w)",
        text,
    ):
        if _speech_act_verb_is_negated(text, match.start()):
            continue
        ends.append(match.end())
    return tuple(ends)


def _has_speech_act_noun_object_refusal(text: str, keyword: str) -> bool:
    keyword_pattern = _keyword_pattern(keyword)
    determiner = SPEECH_ACT_NOUN_OBJECT_DETERMINER_PATTERN
    for end in _unnegated_speech_act_verb_ends(text):
        if re.search(
            rf"^(?:\s+{determiner})?\s+{keyword_pattern}",
            text[end:],
        ):
            return True
    return False


def _has_speech_act_refusal_before_keyword(text: str, keyword: str) -> bool:
    keyword_pattern = _keyword_pattern(keyword)
    for tail in _speech_act_refusal_tails(text):
        if re.search(
            rf"^{NEGATION_TRAILING_ADVERB_PATTERN}{NEGATION_TARGET_GAP_PATTERN}"
            rf"\s+{keyword_pattern}",
            tail,
        ):
            return True

    return _has_speech_act_noun_object_refusal(text, keyword)


def _has_speech_act_noun_object_conjunction(
    text: str,
    keyword: str,
    previous_keywords: Iterable[str],
    conjunction_pattern: str,
) -> bool:
    ends = _unnegated_speech_act_verb_ends(text)
    if not ends:
        return False

    keyword_pattern = _keyword_pattern(keyword)
    determiner = SPEECH_ACT_NOUN_OBJECT_DETERMINER_PATTERN
    object_prefix = rf"(?:\s+{determiner})?"
    for end in ends:
        tail = text[end:]
        for previous_keyword in previous_keywords:
            if previous_keyword == keyword:
                continue

            if re.search(
                rf"^{object_prefix}\s+{_keyword_pattern(previous_keyword)}"
                rf"{NEGATED_OBJECT_GAP_PATTERN}"
                rf"{conjunction_pattern}"
                rf"(?:{determiner}\s+)?"
                rf"{keyword_pattern}",
                tail,
            ):
                return True

    return False


def _has_speech_act_refused_conjunction(
    text: str,
    keyword: str,
    previous_keywords: Iterable[str],
    conjunction_pattern: str,
) -> bool:
    if _has_speech_act_noun_object_conjunction(
        text,
        keyword,
        previous_keywords,
        conjunction_pattern,
    ):
        return True

    tails = _speech_act_refusal_tails(text)
    if not tails:
        return False

    keyword_pattern = _keyword_pattern(keyword)
    for tail in tails:
        for previous_keyword in previous_keywords:
            if previous_keyword == keyword:
                continue

            if re.search(
                rf"^{NEGATION_TRAILING_ADVERB_PATTERN}{NEGATION_TARGET_GAP_PATTERN}\s+"
                rf"{_keyword_pattern(previous_keyword)}"
                rf"{conjunction_pattern}{keyword_pattern}",
                tail,
            ):
                return True

            if re.search(
                rf"^{NEGATION_TRAILING_ADVERB_PATTERN}{NEGATION_TARGET_GAP_PATTERN}\s+"
                rf"{_keyword_pattern(previous_keyword)}"
                rf"{NEGATED_OBJECT_GAP_PATTERN}"
                rf"{conjunction_pattern}{keyword_pattern}",
                tail,
            ):
                return True

    return False


def _has_speech_act_noun_object_action_list(text: str, keyword: str) -> bool:
    ends = _unnegated_speech_act_verb_ends(text)
    if not ends:
        return False

    determiner = SPEECH_ACT_NOUN_OBJECT_DETERMINER_PATTERN
    action_keyword_pattern = (
        rf"(?:(?:{determiner}\s+)?(?:{_action_keyword_alternation()}))"
    )
    list_tail_patterns = (
        rf"(?:\s*,\s*{action_keyword_pattern})*"
        rf"\s*,?\s+(?:or|nor|and)\s+{action_keyword_pattern}",
        rf"(?:\s*,\s*{action_keyword_pattern}){{2,}}",
    )
    keyword_pattern = _keyword_pattern(keyword)
    object_prefix = rf"(?:\s+{determiner})?"
    for end in ends:
        tail = text[end:]
        for previous_keyword in _action_keywords():
            if previous_keyword == keyword:
                continue

            for list_tail_pattern in list_tail_patterns:
                match = re.search(
                    rf"^{object_prefix}\s+{_keyword_pattern(previous_keyword)}"
                    rf"(?P<tail>{list_tail_pattern})",
                    tail,
                )
                if match is not None and re.search(keyword_pattern, match.group("tail")):
                    return True

    return False


def _has_speech_act_refused_action_list(text: str, keyword: str) -> bool:
    if _has_speech_act_noun_object_action_list(text, keyword):
        return True

    tails = _speech_act_refusal_tails(text)
    if not tails:
        return False

    action_keyword_pattern = rf"(?:{_action_keyword_alternation()})"
    list_tail_patterns = (
        rf"(?:\s*,\s*{action_keyword_pattern})*"
        rf"\s*,?\s+(?:or|nor|and)\s+{action_keyword_pattern}",
        rf"(?:\s*,\s*{action_keyword_pattern}){{2,}}",
    )
    keyword_pattern = _keyword_pattern(keyword)
    for tail in tails:
        for previous_keyword in _action_keywords():
            if previous_keyword == keyword:
                continue

            for list_tail_pattern in list_tail_patterns:
                match = re.search(
                    rf"^{NEGATION_TRAILING_ADVERB_PATTERN}"
                    rf"{NEGATION_TARGET_GAP_PATTERN}\s+"
                    rf"{_keyword_pattern(previous_keyword)}"
                    rf"(?P<tail>{list_tail_pattern})",
                    tail,
                )
                if match is not None and re.search(keyword_pattern, match.group("tail")):
                    return True

    return False


def _has_prohibitive_idiom_marker(text: str) -> bool:
    """Fast reject before the standalone prohibitive-idiom scan.

    "circumstance" covers singular and plural. "no account" is tighter than
    "account", which shows up in unrelated appointment text.
    """
    return "circumstance" in text or "no account" in text


def _under_no_circumstances_forward_attaches(text: str, idiom_end: int) -> bool:
    """True when the idiom negates an action that follows it."""
    tail = text[idiom_end:]
    keyword_pattern = rf"(?:{_action_keyword_alternation()})"
    determiner = rf"(?:{SPEECH_ACT_NOUN_OBJECT_DETERMINER_PATTERN}\s+)?"
    prefix_rest = (
        rf"{UNDER_NO_CIRCUMSTANCES_INVERSION_PATTERN}"
        r"(?:\s+ever\b)?"
        rf"{NEGATION_EMPHASIS_PATTERN}"
    )
    if re.search(
        rf"^{prefix_rest}{NEGATION_TARGET_GAP_PATTERN}\s+{determiner}{keyword_pattern}",
        tail,
    ):
        return True

    return (
        re.search(
            rf"^{prefix_rest}"
            rf"(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+"
            rf"{DIRECT_OBJECT_NEGATION_VERB_PATTERN}\s+"
            rf"{DIRECT_OBJECT_PRE_NOUN_PATTERN}"
            rf"{keyword_pattern}",
            tail,
        )
        is not None
    )


def _under_no_circumstances_suffix_applies(text: str, idiom_end: int) -> bool:
    """True when a preceding keyword, not a following word, owns the idiom.

    Clause-final "cancel under no circumstances" refuses cancel. A following
    action ("under no circumstances reschedule") is a forward refusal. Other
    following words ("under no circumstances wait") leave the earlier action
    executable.
    """
    if _under_no_circumstances_forward_attaches(text, idiom_end):
        return False

    rest = re.sub(r"^[\s,;:.!?]+", "", text[idiom_end:])
    if not rest:
        return True

    return re.match(r"(?:please|but|however|instead|yet)\b", rest) is not None


def _has_under_no_circumstances_after_keyword(text: str, keyword: str) -> bool:
    """True when the idiom follows the keyword in the same clause.

    "I will cancel under no circumstances" refuses the keyword. A later action
    the idiom attaches to stays the forward target, so "Please cancel, under
    no circumstances reschedule" keeps cancel.
    """
    keyword_pattern = _keyword_pattern(keyword)
    determiner = rf"(?:{SPEECH_ACT_NOUN_OBJECT_DETERMINER_PATTERN}\s+)?"
    action_keyword_pattern = rf"(?:{determiner}(?:{_action_keyword_alternation()}))"
    gap = rf"(?:\s+{UNDER_NO_CIRCUMSTANCES_SUBJECT_TOKEN_PATTERN}){{0,3}}"
    list_bridge = (
        rf"(?:(?:\s*,\s*{action_keyword_pattern})*"
        rf"\s*,?\s+(?:or|nor|and)\s+{action_keyword_pattern}"
        rf"|(?:\s*,\s*{action_keyword_pattern})+)"
    )
    patterns = (
        rf"{keyword_pattern}{gap}\s*,?\s+(?P<idiom>{UNDER_NO_CIRCUMSTANCES_IDIOM_PATTERN})",
        rf"{keyword_pattern}(?P<mid>{list_bridge}){gap}\s*,?\s+"
        rf"(?P<idiom>{UNDER_NO_CIRCUMSTANCES_IDIOM_PATTERN})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            if not _under_no_circumstances_suffix_applies(text, match.end("idiom")):
                continue
            return True

    return False


def _has_under_no_circumstances_before_keyword(text: str, keyword: str) -> bool:
    if not _has_prohibitive_idiom_marker(text):
        return False

    keyword_pattern = _keyword_pattern(keyword)
    determiner = rf"(?:{SPEECH_ACT_NOUN_OBJECT_DETERMINER_PATTERN}\s+)?"
    if re.search(
        rf"(?<!\w){UNDER_NO_CIRCUMSTANCES_PREFIX_PATTERN}"
        rf"{NEGATION_TARGET_GAP_PATTERN}\s+{determiner}{keyword_pattern}",
        text,
    ):
        return True

    if re.search(
        rf"(?<!\w){UNDER_NO_CIRCUMSTANCES_PREFIX_PATTERN}"
        rf"(?:\s+{NEGATION_ADVERB_PATTERN}){{0,2}}\s+"
        rf"{DIRECT_OBJECT_NEGATION_VERB_PATTERN}\s+"
        rf"{DIRECT_OBJECT_PRE_NOUN_PATTERN}"
        rf"{keyword_pattern}",
        text,
    ):
        return True

    return _has_under_no_circumstances_after_keyword(text, keyword)


def _has_under_no_circumstances_conjunction(
    text: str,
    keyword: str,
    previous_keywords: Iterable[str],
    conjunction_pattern: str,
) -> bool:
    if not _has_prohibitive_idiom_marker(text):
        return False

    keyword_pattern = _keyword_pattern(keyword)
    determiner = rf"(?:{SPEECH_ACT_NOUN_OBJECT_DETERMINER_PATTERN}\s+)?"
    for previous_keyword in previous_keywords:
        if previous_keyword == keyword:
            continue

        if re.search(
            rf"(?<!\w){UNDER_NO_CIRCUMSTANCES_PREFIX_PATTERN}"
            rf"{NEGATION_TARGET_GAP_PATTERN}\s+"
            rf"{determiner}{_keyword_pattern(previous_keyword)}"
            rf"{conjunction_pattern}{determiner}{keyword_pattern}",
            text,
        ):
            return True

        if re.search(
            rf"(?<!\w){UNDER_NO_CIRCUMSTANCES_PREFIX_PATTERN}"
            rf"{NEGATION_TARGET_GAP_PATTERN}\s+"
            rf"{determiner}{_keyword_pattern(previous_keyword)}"
            rf"{NEGATED_OBJECT_GAP_PATTERN}"
            rf"{conjunction_pattern}"
            rf"{determiner}{keyword_pattern}",
            text,
        ):
            return True

    return False


def _has_under_no_circumstances_action_list(text: str, keyword: str) -> bool:
    if not _has_prohibitive_idiom_marker(text):
        return False

    determiner = rf"(?:{SPEECH_ACT_NOUN_OBJECT_DETERMINER_PATTERN}\s+)?"
    action_keyword_pattern = rf"(?:{determiner}(?:{_action_keyword_alternation()}))"
    list_tail_patterns = (
        rf"(?:\s*,\s*{action_keyword_pattern})*"
        rf"\s*,?\s+(?:or|nor|and)\s+{action_keyword_pattern}",
        rf"(?:\s*,\s*{action_keyword_pattern}){{2,}}",
    )
    keyword_pattern = _keyword_pattern(keyword)
    for previous_keyword in _action_keywords():
        if previous_keyword == keyword:
            continue

        for list_tail_pattern in list_tail_patterns:
            for match in re.finditer(
                rf"(?<!\w){UNDER_NO_CIRCUMSTANCES_PREFIX_PATTERN}"
                rf"{NEGATION_TARGET_GAP_PATTERN}\s+"
                rf"{determiner}{_keyword_pattern(previous_keyword)}"
                rf"(?P<tail>{list_tail_pattern})",
                text,
            ):
                if re.search(keyword_pattern, match.group("tail")):
                    return True

    return False


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

    if _has_speech_act_refusal_before_keyword(text, keyword):
        return True

    if _has_under_no_circumstances_before_keyword(text, keyword):
        return True

    return False


def _has_negated_conjunction(
    text: str,
    keyword: str,
    previous_keywords: Iterable[str],
    conjunction_pattern: str,
) -> bool:
    if _has_under_no_circumstances_conjunction(
        text,
        keyword,
        previous_keywords,
        conjunction_pattern,
    ):
        return True

    if _has_speech_act_refused_conjunction(
        text,
        keyword,
        previous_keywords,
        conjunction_pattern,
    ):
        return True

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
    if _has_under_no_circumstances_action_list(text, keyword):
        return True

    if _has_speech_act_refused_action_list(text, keyword):
        return True

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

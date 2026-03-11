import argparse
import json
import random
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import yaml
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

# ––– project‑local helpers ––––––––––––––––––––––––––––––––––––––––––––––––
from __init__ import parse_object_names, build_actions_dict


# ---------------------------------------------------------------------------#
# 0)  NLTK helpers                                                           #
# ---------------------------------------------------------------------------#
def _ensure_nltk_resource(pkg: str, locator: str):
    try:
        nltk.data.find(locator)
    except LookupError:
        nltk.download(pkg, quiet=True)


_ensure_nltk_resource("punkt_tab", "tokenizers/punkt_tab")
_ensure_nltk_resource("punkt", "tokenizers/punkt")
_ensure_nltk_resource(
    "averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"
)
_ensure_nltk_resource("wordnet", "corpora/wordnet")

_LEM = WordNetLemmatizer()
_DETOK = TreebankWordDetokenizer()

_REQUIRES_GERUND = {
    "avoid",
    "keep",
    "stop",
    "start",
    "finish",
    "by",
    "while",
    "after",
    "before",
    "until",
}
_PUNCT = {".", ",", ";", ":", "!", "?", "—", "-", "–", "…"}
_PLACEHOLDER_RE = re.compile(r"^prop_\d+$")


def _same_word(sent_tok: str, targ_tok: str) -> bool:
    """True if the tokens match literally or by verb‑lemma."""
    s = sent_tok.rstrip(string.punctuation).lower()
    t = targ_tok.lower()
    return s == t or _LEM.lemmatize(s, "v") == t


#
# -------------------------#
# 1)  Scenario loader                                                        #
# ---------------------------------------------------------------------------#
def load_scenario(
    scenario_name: str, yaml_path: str = "dataset_generators/scenarios.yaml"
) -> Tuple[Dict, Dict, Dict, List[str], Dict]:
    """
    Return
        cfg            – full YAML dict
        object_dict    – {canon -> [synonyms]}
        actions_dict   – {verb  -> [NL synonyms]}
        locations      – list[str]
        actions_cfg    – cfg["actions"] (incl. 'params' lists)
    """
    cfg_all = yaml.safe_load(Path(yaml_path).read_text())
    if scenario_name not in cfg_all:
        raise ValueError(f"Scenario '{scenario_name}' not found in {yaml_path}.")

    cfg = cfg_all[scenario_name]

    object_dict = parse_object_names("dataset_generators/object_names.txt")
    actions_dict = {
        k: v for k, v in build_actions_dict().items() if k in cfg["actions"]
    }

    return cfg, object_dict, actions_dict, cfg.get("locations", []), cfg["actions"]


# ---------------------------------------------------------------------------#
# 2)  Simple inflection helper                                               #
# ---------------------------------------------------------------------------#
def _to_gerund(word: str) -> str:
    if word.endswith("ie"):
        return word[:-2] + "ying"
    if word.endswith("e") and not word.endswith("ee"):
        return word[:-1] + "ing"
    if len(word) > 2 and re.match(r"[aeiou][^aeiouywx]$", word[-2:]):
        return word + "ing"
    return word + "ing"


# ---------------------------------------------------------------------------
# 3)  LTL skeletons & NL templates (unchanged)
# ---------------------------------------------------------------------------

LTL_TEMPLATES_STATE = [
    ("F_NOT", 1, lambda P: ["finally", "(", "not", P[0], ")"]),
    ("G_NOT", 1, lambda P: ["globally", "(", "not", P[0], ")"]),
    ("F_AND", 2, lambda P: ["finally", "(", P[0], "and", P[1], ")"]),
    ("G_AND", 2, lambda P: ["globally", "(", P[0], "and", P[1], ")"]),
    ("F_OR", 2, lambda P: ["finally", "(", P[0], "or", P[1], ")"]),
    ("G_OR", 2, lambda P: ["globally", "(", P[0], "or", P[1], ")"]),
    ("X", 1, lambda P: ["next", P[0]]),
    ("U", 2, lambda P: ["(", P[0], "until", P[1], ")"]),
]

SEMANTIC_TEMPLATES = {
    "F_NOT": [lambda s: f"eventually, avoid {s[0]}."],
    "G_NOT": [lambda s: f"always avoid {s[0]}."],
    "F_AND": [lambda s: f"eventually {s[0]} and {s[1]}."],
    "G_AND": [lambda s: f"always maintain both {s[0]} and {s[1]}."],
    "F_OR": [lambda s: f"eventually {s[0]} or {s[1]}."],
    "G_OR": [lambda s: f"always have either {s[0]} or {s[1]}."],
    "NOT": [lambda s: f"never {s[0]}.", lambda s: f"avoid {s[0]} at all costs."],
    "AND": [
        lambda s: f"{s[0]} and {s[1]}.",
        lambda s: f"ensure both {s[0]} and {s[1]}.",
    ],
    "OR": [lambda s: f"{s[0]} or {s[1]}."],
}

GENERIC_TEMPLATES = [
    lambda s: "In this task, " + ", then ".join(s) + ".",
    lambda s: "Please " + " and then ".join(s) + ".",
]

VERB_LIKE_STARTS = {
    "never",
    "avoid",
    "always",
    "grab",
    "ensure",
    "please",
    "eventually",
    "do",
    "keep",
    "ultimately",
    "make",
}
EGO_REFS = ["The robot", "You", "Our agent", "The system", "This controller"]


def _trace_for_NOT(p, good=True):
    # ~p   ⇒ good: p false   | bad: p true
    return [[]] if good else [[p]]


def _trace_for_F_NOT(p, good=True):
    # F (¬p)
    return [[p], [], []] if good else [[p]] * 3


def _trace_for_G_NOT(p, good=True):
    # G (¬p)
    return [[]] * 3 if good else [[p]] + [[]] * 2


def _trace_for_X(p, good=True):
    # X p
    return [[], [p]] if good else [[p]]


def _trace_for_NOT_A_UNTIL_B(a, b, good=True):
    # ¬a  U  b
    if good:
        return [
            [],  # t0  a false
            [],  # t1  a false
            [b],
        ]  # t2  b true        (repeats)
    else:
        return [[a]] * 3  # never sees b


# -- two‑prop helpers --------------------------------------------------------
def _trace_for_F_AND(p, q, good=True):
    # F (p ∧ q)
    if good:
        return [[], [p], [p, q]]  # first time t2
    else:
        return [[], [p], []]  # never both


def _trace_for_G_AND(p, q, good=True):
    # G (p ∧ q)
    return [[p, q]] * 3 if good else [[p], [p], [q]]


def _trace_for_F_OR(p, q, good=True):
    # F (p ∨ q)
    return [[], [p]] if good else [[]] * 3


def _trace_for_G_OR(p, q, good=True):
    # G (p ∨ q)
    if good:
        return [[p], [q], [p]]  # always at least one
    else:
        return [[p], [], []]  # gap at t1


def _trace_for_AND(p, q, good=True):
    return [[p, q]] if good else [[p]]


def _trace_for_OR(p, q, good=True):
    return [[p]] if good else [[]]


_TRACE_BUILDERS = {
    "NOT": lambda L, g: _trace_for_NOT(L[0], g),
    "F_NOT": lambda L, g: _trace_for_F_NOT(L[0], g),
    "G_NOT": lambda L, g: _trace_for_G_NOT(L[0], g),
    "X": lambda L, g: _trace_for_X(L[0], g),
    "NOT_A_UNTIL_B": lambda L, g: _trace_for_NOT_A_UNTIL_B(L[0], L[1], g),
    "F_AND": lambda L, g: _trace_for_F_AND(L[0], L[1], g),
    "G_AND": lambda L, g: _trace_for_G_AND(L[0], L[1], g),
    "F_OR": lambda L, g: _trace_for_F_OR(L[0], L[1], g),
    "G_OR": lambda L, g: _trace_for_G_OR(L[0], L[1], g),
    "AND": lambda L, g: _trace_for_AND(L[0], L[1], g),
    "OR": lambda L, g: _trace_for_OR(L[0], L[1], g),
}

# Added for nl2spec templates
_TRACE_BUILDERS.update(
    {
        # 1 G(a ⇒ ◇b)
        "G_IMPL_F": lambda L, g: (
            [[L[0]], [], [L[1]]]  # a at t0, b by t2
            if g
            else [[L[0]], [], []]
        ),  # a occurs, b never
        # 2 G ¬(a ∧ b)
        "G_NOT_AND": lambda L, g: (
            [[L[0]], [L[1]], []]  # never together
            if g
            else [[L[0], L[1]]]
        ),  # both at t0
        # 3 G(a ⇒ XXX b)
        "G_IMPL_XXX": lambda L, g: (
            [[L[0]], [], [], [L[1]]]  # b exactly 3 steps later
            if g
            else [[L[0]], [], [], []]
        ),
        # 4 a U (GF b)
        "U_GF": lambda L, g: (
            [[L[0]], [L[0]], [L[1]]]  # a‑a‑b (then b repeats)
            if g
            else [[L[0]]] * 3
        ),  # no b ever
        # 5 (F b) ⇒ (¬b U (a ∧ ¬b))
        "F_B_IMPL_A_BEFORE": lambda L, g: (
            [[L[0]], [L[1]]]  # a before first b
            if g
            else [[L[1]]]
        ),  # b first, no prior a
        # 6 G(a ⇒ b)
        "G_IMPL": lambda L, g: (
            [[L[0], L[1]], [L[1]]]  # whenever a, b too
            if g
            else [[L[0]], []]
        ),
        # 7 G(a ∧ b)
        "G_AND_PAIR": lambda L, g: (
            [[L[0], L[1]]]  # both always
            if g
            else [[L[0]], []]
        ),
        # 8 G a  ∧  G(b ⇒ ¬c)
        "G_A_AND_G_B_IMPL_NOT_C": lambda L, g: (
            [[L[0]], [L[0], L[1]], [L[0]]]  # b occurs w/o c
            if g
            else [[L[0], L[2]]]
        ),  # b plus c violates
        # 9 (G(a ⇒ ◇b)) ⇒ G F c
        "G_IMPL_F_IMPL_GF": lambda L, g: (
            [[L[0]], [L[1], L[2]]]  # premise true, c fulfils GF
            if g
            else [[L[0]], [L[1]]]
        ),  # premise true, no c ever
        # 10 GF a ⇒ GF b
        "GF_IMPL_GF": lambda L, g: (
            [[]]  # no a (premise false)
            if g
            else [[L[0]]]
        ),  # a infinitely often, b never
        # 11 GF a ∨ GF b
        "GF_OR_GF": lambda L, g: (
            [[L[0]]]  # a infinitely often
            if g
            else [[]]
        ),  # neither ever occurs
        # 12 F G ¬a
        "FG_NOT": lambda L, g: (
            [[L[0]], [], []]  # a stops after t0
            if g
            else [[L[0]], [], [L[0]]]
        ),  # a re‑appears
        # 13 G(¬(a∧b) ⇒ ◇c)
        "G_NOT_AND_IMPL_F": lambda L, g: (
            [[L[0], L[1]]]  # antecedent false
            if g
            else [[]]
        ),  # antecedent true, no c ever
        # 14 ¬(a∧b)  &  G(a ∨ b)
        "EXCLUSIVE_ALWAYS_ONE": lambda L, g: (
            [[L[0]], [L[1]], [L[0]]]  # one true, never both/none
            if g
            else [[], []]
        ),  # both false at t0
        # 15 G((a↔b) ⇒ (b↔c))
        "G_EQ_IMPL_EQ": lambda L, g: (
            [[L[0], L[1], L[2]]]  # all equal
            if g
            else [[L[0], L[1]]]
        ),  # a=b true, b≠c
        # 16 handled earlier (NOT_A_UNTIL_B)
        # 17 G(a ⇒ X G ¬b)
        "G_A_IMPL_XG_NOT_B": lambda L, g: (
            [[L[0]], [], []]  # a then b never again
            if g
            else [[L[0]], [], [L[1]]]
        ),
        # 18 a releases b   (formula variant)
        "A_RELEASES_B": lambda L, g: (
            [[L[1]]]  # G b  branch
            if g
            else [[L[1]], []]
        ),  # b drops without a trigger
        # 19 same as G_NOT_AND
        "G_NOT_AND_ALT": lambda L, g: _TRACE_BUILDERS["G_NOT_AND"](L, g),
        # 20 G(a ∧ X b ⇒ X X c)
        "TWO_STEP_TRIGGER": lambda L, g: (
            [[L[0]], [L[1]], [L[2]]]  # a, then b, then c
            if g
            else [[L[0]], [L[1]], []]
        ),
        # 21 G(a ⇒ X ◇b)
        "NEXT_EVENTUAL": lambda L, g: (
            [[L[0]], [], [L[1]]]  # b eventually after next
            if g
            else [[L[0]], [], []]
        ),
        # 22 EVERY FIFTH STEP a   (a … ¬a×4 … a …)
        "EVERY_FIFTH_STEP": lambda L, g: (
            [[L[0]], [], [], [], [], [L[0]]]  # good 0‑5 prefix
            if g
            else [[L[0]], [], [], [L[0]]]
        ),  # repeats too early
        # 23 GF a ∨ X b
        "GF_A_OR_NEXT_B": lambda L, g: (
            [[L[0]]]  # GF a
            if g
            else [[], []]
        ),  # neither condition true
        # 24 G a
        "G_ALWAYS_A": lambda L, g: (
            [[L[0]]]  # a always
            if g
            else [[L[0]], []]
        ),
        # 25 G(a ⇒ (b ∨ X b))
        "A_IMPL_B_WITHIN_1": lambda L, g: (
            [[L[0], L[1]]]  # b same tick
            if g
            else [[L[0]], []]
        ),  # no b within 1
        # 26 G(a ∨ b ∨ c)
        "G_ONE_OF_ABC": lambda L, g: (
            [[L[0]], [L[1]], [L[2]]]  # at least one each tick
            if g
            else [[]]
        ),
        # 27 G(a ⇒ ◇b)
        "A_IMPL_EVENTUAL_B": lambda L, g: (
            [[L[0]], [], [L[1]]]  # b eventually
            if g
            else [[L[0]], [], []]
        ),
        # 28 “almost always” a (≤1 tick gap)
        "ALMOST_ALWAYS_A": lambda L, g: (
            [[L[0]], [], [L[0]]]  # single‑tick gap
            if g
            else [[L[0]], [], [], []]
        ),  # two‑tick gap
        # 29 ¬a ≤ 2 ticks
        "NOT_A_AT_MOST_TWO": lambda L, g: (
            [[L[0]], [], [], [L[0]]]  # gap length 2
            if g
            else [[L[0]], [], [], [], [L[0]]]
        ),  # gap length 3
        # 30 a every 3rd tick (max once per 3)
        "A_EVERY_THIRD_STEP": lambda L, g: (
            [[L[0]], [], [], [L[0]]]  # good spacing
            if g
            else [[L[0]], [], [L[0]]]
        ),  # repeats too early
        # 31 G(a ⇒ X b)
        "NEXT_FOLLOW": lambda L, g: (
            [[L[0]], [L[1]]]  # b directly after a
            if g
            else [[L[0]], []]
        ),
        # 32 ◇(a ∧ b)
        "EVENTUALLY_BOTH": lambda L, g: (
            [[L[0], L[1]]]  # both together
            if g
            else [[], [], []]
        ),
        # 33 ◇a ∧ ◇b
        "BOTH_EVENTUAL": lambda L, g: (
            [[L[0]], [L[1]]]  # each eventually
            if g
            else [[], [], []]
        ),
        # 34 G(a ↔ X b)
        "STEPWISE_EQUALITY": lambda L, g: (
            [[L[0]], [L[1]]]  # matches next‑step
            if g
            else [[L[0]], []]
        ),
        # 35 b ⇒ X((c U a) ∨ G c)
        "RESPONSE_UNTIL_OR_ALWAYS": lambda L, g: (
            [[L[1]], [L[2]], [L[2]], [L[0]]]  # b, then c until a
            if g
            else [[L[1]], []]
        ),  # b, but no c next
        # 36 (a U b) ∨ G a
        "UNTIL_OR_ALWAYS": lambda L, g: (
            [[L[0]], [L[0]], [L[1]]]  # until branch
            if g
            else [[], []]
        ),  # neither holds
    }
)


def _make_traces(key: str, labels: List[str]):
    """
    Return (good_trace, bad_trace) for the given template key.
    Fallback strategy: put every prop true at t0 (good) vs. all false (bad).
    """
    if key in _TRACE_BUILDERS:
        good = _TRACE_BUILDERS[key](labels, True)
        bad = _TRACE_BUILDERS[key](labels, False)
    else:
        good = [labels]  # naive but satisfies most conjunction‑free specs
        bad = [[]] * 3  # clearly violates every positive requirement
    return good, bad


ADDITIONAL_LTL_TEMPLATES_STATE = [
    # 1  Every a is eventually followed by b     globally (a implies finally b)
    ("G_IMPL_F", 2, lambda P: "globally ( {} implies finally {} )".format(*P).split()),
    # 2  Never a and b at the same time          globally (not (a and b))
    ("G_NOT_AND", 2, lambda P: "globally ( not ( {} and {} ) )".format(*P).split()),
    # 3  a ⇒ b three steps later                 globally (a implies next next next b)
    (
        "G_IMPL_XXX",
        2,
        lambda P: "globally ( {} implies next ( next ( next {} ) ) )".format(
            *P
        ).split(),
    ),
    # 4  a until  globally finally b            a until (globally (finally b))
    ("U_GF", 2, lambda P: "{} until ( globally ( finally {} ) )".format(*P).split()),
    # 5  If finally b then …                    (finally b) implies (not b until (a and not b))
    (
        "F_B_IMPL_A_BEFORE",
        2,
        lambda P: "( finally {} ) implies ( not {} until ( {} and not {} ) )".format(
            P[1], P[1], P[0], P[1]
        ).split(),
    ),
    # 6  Whenever a then b                      globally (a implies b)
    ("G_IMPL", 2, lambda P: "globally ( {} implies {} )".format(*P).split()),
    # 7  a & b everywhere                       globally (a and b)
    ("G_AND_PAIR", 2, lambda P: "globally ( {} and {} )".format(*P).split()),
    # 8  a always  &  (b ⇒ not c)               globally a  and  globally (b implies not c)
    (
        "G_A_AND_G_B_IMPL_NOT_C",
        3,
        lambda P: "globally {} and globally ( {} implies not {} )".format(
            P[0], P[1], P[2]
        ).split(),
    ),
    # 9  (a → finally b) ⇒ globally finally c
    (
        "G_IMPL_F_IMPL_GF",
        3,
        lambda P: "globally ( {} implies finally {} ) implies globally finally {}".format(
            P[0], P[1], P[2]
        ).split(),
    ),
    # 10 finally‑often a ⇒ finally‑often b      globally finally a implies globally finally b
    (
        "GF_IMPL_GF",
        2,
        lambda P: "globally finally {} implies globally finally {}".format(*P).split(),
    ),
    # 11 either a or b happens infinitely often
    (
        "GF_OR_GF",
        2,
        lambda P: "globally finally {} or globally finally {}".format(*P).split(),
    ),
    # 12 a eventually stops forever             finally globally not a
    ("FG_NOT", 1, lambda P: "finally globally not {}".format(P[0]).split()),
    # 13 if not(a and b) then finally c
    (
        "G_NOT_AND_IMPL_F",
        3,
        lambda P: "globally ( not ( {} and {} ) implies finally {} )".format(
            *P
        ).split(),
    ),
    # 14 never (a and b)  &  always (a or b)
    (
        "EXCLUSIVE_ALWAYS_ONE",
        2,
        lambda P: "globally ( not ( {} and {} ) ) and globally ( {} or {} )".format(
            P[0], P[1], P[0], P[1]
        ).split(),
    ),
    # 15 equality preservation (double implies)
    (
        "G_EQ_IMPL_EQ",
        3,
        lambda P: "globally ( ( {} double_implies {} ) implies ( {} double_implies {} ) )".format(
            P[0], P[1], P[1], P[2]
        ).split(),
    ),
    # 16 a only after b                          not a until b
    ("NOT_A_UNTIL_B", 2, lambda P: "not {} until {}".format(*P).split()),
    # 17 once a then never b again               globally (a implies next globally not b)
    (
        "G_A_IMPL_XG_NOT_B",
        2,
        lambda P: "globally ( {} implies next globally not {} )".format(*P).split(),
    ),
    # 18 a releases b                            (b until (b and not a)) or globally b
    (
        "A_RELEASES_B",
        2,
        lambda P: "( {} until ( {} and not {} ) ) or globally {}".format(
            P[1], P[1], P[0], P[1]
        ).split(),
    ),
    # 19 same as #2 (variant)
    ("G_NOT_AND_ALT", 2, lambda P: "globally not ( {} and {} )".format(*P).split()),
    # 20 a & next b ⇒ next next c
    (
        "TWO_STEP_TRIGGER",
        3,
        lambda P: "globally ( {} and next {} implies next next {} )".format(*P).split(),
    ),
    # 21 a ⇒ next finally b
    (
        "NEXT_EVENTUAL",
        2,
        lambda P: "globally ( {} implies next finally {} )".format(*P).split(),
    ),
    # 22 every fifth step a
    (
        "EVERY_FIFTH_STEP",
        1,
        lambda P: (
            f"{P[0]} and "
            "globally ( "
            f"{P[0]} implies next not {P[0]} and next next not {P[0]} and "
            f"next next next not {P[0]} and next next next next not {P[0]} and "
            f"next next next next next {P[0]} )"
        ).split(),
    ),
    # 23 finally‑often a  or  next b
    (
        "GF_A_OR_NEXT_B",
        2,
        lambda P: "globally finally {} or next {}".format(*P).split(),
    ),
    # 24 always a                                globally a
    ("G_ALWAYS_A", 1, lambda P: "globally {}".format(P[0]).split()),
    # 25 a ⇒ (b now or next b)
    (
        "A_IMPL_B_WITHIN_1",
        2,
        lambda P: "globally ( {} implies ( {} or next {} ) )".format(
            P[0], P[1], P[1]
        ).split(),
    ),
    # 26 always (a or b or c)
    ("G_ONE_OF_ABC", 3, lambda P: "globally ( {} or {} or {} )".format(*P).split()),
    # 27 a ⇒ finally b
    (
        "A_IMPL_EVENTUAL_B",
        2,
        lambda P: "globally ( {} implies finally {} )".format(*P).split(),
    ),
    # 28 soft always‑a (2‑tick grace)
    (
        "ALMOST_ALWAYS_A",
        1,
        lambda P: "not globally ( not ( {} and next {} ) )".format(P[0], P[0]).split(),
    ),
    # 29 not‑a ≤ 2 ticks  (same formula, diff NL)
    (
        "NOT_A_AT_MOST_TWO",
        1,
        lambda P: "not globally ( not ( {} and next {} ) )".format(P[0], P[0]).split(),
    ),
    # 30 a every three ticks
    (
        "A_EVERY_THIRD_STEP",
        1,
        lambda P: "globally ( {} implies ( next not {} or next next not {} or next next next {} ) )".format(
            P[0], P[0], P[0], P[0]
        ).split(),
    ),
    # 31 every a directly followed by b
    ("NEXT_FOLLOW", 2, lambda P: "globally ( {} implies next {} )".format(*P).split()),
    # 32 eventually a and b together
    ("EVENTUALLY_BOTH", 2, lambda P: "finally ( {} and {} )".format(*P).split()),
    # 33 finally a  and  finally b
    ("BOTH_EVENTUAL", 2, lambda P: "finally {} and finally {}".format(*P).split()),
    # 34 always (a double_implies next b)
    (
        "STEPWISE_EQUALITY",
        2,
        lambda P: "globally ( {} double_implies next {} )".format(*P).split(),
    ),
    # 35 b ⇒ next (c until a  or  globally c)
    (
        "RESPONSE_UNTIL_OR_ALWAYS",
        3,
        lambda P: "{} implies next ( ( {} until {} ) or globally {} )".format(
            P[1], P[2], P[0], P[2]
        ).split(),
    ),
    # 36 (a until b) or globally a
    (
        "UNTIL_OR_ALWAYS",
        2,
        lambda P: "( {} until {} ) or globally {}".format(P[0], P[1], P[0]).split(),
    ),
]


# ----  Natural‑language templates ---------------------------------------

ADDITIONAL_SEMANTIC_TEMPLATES = {
    "G_IMPL_F": [lambda s: f"Globally, if {s[0]} occurs then finally {s[1]} happens."],
    "G_NOT_AND": [
        lambda s: f"Globally, it is not the case that both {s[0]} and {s[1]} hold simultaneously."
    ],
    "G_IMPL_XXX": [
        lambda s: f"Whenever {s[0]} holds, {s[1]} holds exactly three steps later."
    ],
    "U_GF": [
        lambda s: f"{s[0]} must keep holding until, from some point on, {s[1]} holds infinitely often."
    ],
    "F_B_IMPL_A_BEFORE": [
        lambda s: f"If {s[1]} ever holds, {s[0]} must have held beforehand."
    ],
    "G_IMPL": [lambda s: f"Whenever {s[0]} holds, {s[1]} holds as well."],
    "G_AND_PAIR": [lambda s: f"Both {s[0]} and {s[1]} hold at every step."],
    "G_A_AND_G_B_IMPL_NOT_C": [
        lambda s: f"{s[0]} holds always, and whenever {s[1]} holds, {s[2]} does not."
    ],
    "G_IMPL_F_IMPL_GF": [
        lambda s: f"If every {s[0]} is eventually followed by {s[1]}, then {s[2]} must occur infinitely often."
    ],
    "GF_IMPL_GF": [
        lambda s: f"If {s[0]} happens infinitely often, then so does {s[1]}."
    ],
    "GF_OR_GF": [lambda s: f"Either {s[0]} or {s[1]} happens infinitely often."],
    "FG_NOT": [lambda s: f"From some point onwards, {s[0]} never occurs again."],
    "G_NOT_AND_IMPL_F": [
        lambda s: f"Whenever neither {s[0]} nor {s[1]} holds, {s[2]} eventually holds."
    ],
    "EXCLUSIVE_ALWAYS_ONE": [
        lambda s: f"{s[0]} and {s[1]} never coincide, yet one of them is always true."
    ],
    "G_EQ_IMPL_EQ": [
        lambda s: f"Whenever {s[0]} and {s[1]} are equal, {s[1]} and {s[2]} are equal as well."
    ],
    "NOT_A_UNTIL_B": [lambda s: f"{s[0]} can only happen after {s[1]}."],
    "G_A_IMPL_XG_NOT_B": [
        lambda s: f"Once {s[0]} has occurred, {s[1]} will never occur again."
    ],
    "A_RELEASES_B": [
        lambda s: f"{s[0]} releases {s[1]} — after {s[0]} ends, {s[1]} must hold continuously."
    ],
    "G_NOT_AND_ALT": [
        lambda s: f"{s[0]} and {s[1]} are mutually exclusive at all times."
    ],
    "TWO_STEP_TRIGGER": [
        lambda s: f"If {s[0]} holds and {s[1]} holds next, then {s[2]} holds in the step after that."
    ],
    "NEXT_EVENTUAL": [
        lambda s: f"Whenever {s[0]} holds, from the next step onwards {s[1]} will eventually hold."
    ],
    "EVERY_FIFTH_STEP": [lambda s: f"{s[0]} holds exactly every fifth step."],
    "GF_A_OR_NEXT_B": [
        lambda s: f"Either {s[0]} happens infinitely often, or {s[1]} happens in the next step."
    ],
    "G_ALWAYS_A": [lambda s: f"{s[0]} holds at all times."],
    "A_IMPL_B_WITHIN_1": [
        lambda s: f"When {s[0]} happens, {s[1]} must hold now or in the next step."
    ],
    "G_ONE_OF_ABC": [
        lambda s: f"At every moment, at least one of {s[0]}, {s[1]}, or {s[2]} holds."
    ],
    "A_IMPL_EVENTUAL_B": [
        lambda s: f"Whenever {s[0]} holds, eventually {s[1]} will hold."
    ],
    "ALMOST_ALWAYS_A": [
        lambda s: f"{s[0]} should always hold, with at most a two-step grace period for recovery."
    ],
    "NOT_A_AT_MOST_TWO": [
        lambda s: f"Not {s[0]} may last at most two consecutive steps."
    ],
    "A_EVERY_THIRD_STEP": [lambda s: f"{s[0]} occurs at most once every three steps."],
    "NEXT_FOLLOW": [
        lambda s: f"Every {s[0]} is directly followed by {s[1]} in the next step."
    ],
    "EVENTUALLY_BOTH": [
        lambda s: f"Eventually, both {s[0]} and {s[1]} hold simultaneously."
    ],
    "BOTH_EVENTUAL": [lambda s: f"{s[0]} and {s[1]} will each happen at some point."],
    "STEPWISE_EQUALITY": [
        lambda s: f"At every step, {s[0]} equals the value of {s[1]} in the next step."
    ],
    "RESPONSE_UNTIL_OR_ALWAYS": [
        lambda s: f"If {s[1]} holds, then in the next step {s[2]} persists until {s[0]} holds, or else {s[2]} holds forever."
    ],
    "UNTIL_OR_ALWAYS": [
        lambda s: f"{s[0]} must hold until {s[1]} does, or else {s[0]} holds forever."
    ],
}

LTL_TEMPLATES_STATE.extend(ADDITIONAL_LTL_TEMPLATES_STATE)
for k, v in ADDITIONAL_SEMANTIC_TEMPLATES.items():
    SEMANTIC_TEMPLATES.setdefault(k, []).extend(v)


# ---------------------------------------------------------------------------#
# 4)  Sentence utilities                                                     #
# ---------------------------------------------------------------------------#
def add_ego_reference(sentence: str, ego: str) -> str:
    first = re.sub(r"[^a-zA-Z]", "", sentence.split()[0]).lower()
    if first in VERB_LIKE_STARTS:
        return f"{ego} must {sentence}"
    return sentence


def correct_sentence(sentence: str) -> str:
    toks = word_tokenize(sentence)
    tags = pos_tag(toks)

    # gerund agreement
    i = 0
    while i < len(tags) - 1:
        tok, _ = tags[i]
        if tok.lower() in _REQUIRES_GERUND:
            j = i + 1
            while j < len(tags) and tags[j][0] in _PUNCT:
                j += 1
            if j < len(tags):
                nxt_tok, nxt_tag = tags[j]
                if _PLACEHOLDER_RE.match(nxt_tok) or nxt_tok.lower().endswith("ing"):
                    pass
                elif nxt_tag in {"VB", "VBP", "NN", "NNS"}:
                    toks[j] = _to_gerund(nxt_tok.lower())
        i += 1

    toks = ["I" if t == "i" else t for t in toks]
    if toks:
        toks[0] = toks[0].capitalize()
    for k, t in enumerate(toks[:-1]):
        if t in {".", "!", "?"}:
            toks[k + 1] = toks[k + 1].capitalize()
    return _DETOK.detokenize(toks)


def pick_sentence_template(key: str):
    return random.choice(SEMANTIC_TEMPLATES.get(key, GENERIC_TEMPLATES))


def build_entry_for_props(
    idx: int,
    props: Dict,  # prop_k  -> info
    actions_cfg: Dict,
) -> Dict:
    # ------------------------------------------------------------------ #
    # 1)  pick a skeleton                                                 #
    # ------------------------------------------------------------------ #
    arity = len(props)
    viable = [tpl for tpl in LTL_TEMPLATES_STATE if tpl[1] == arity] or [
        tpl for tpl in LTL_TEMPLATES_STATE if tpl[1] <= arity
    ]
    key, need, skeleton = random.choice(viable)
    labels_used = list(props.keys())[:need]  # preserve order

    # ------------------------------------------------------------------ #
    # 2)  LTL formula (grounded + masked)                                 #
    # ------------------------------------------------------------------ #
    def _atom(info):
        return (
            f"{info['action_canon']}(" + ",".join(info["args_canon"]) + ")"
            if info["args_canon"]
            else info["action_canon"]
        )

    g_ltl = skeleton([_atom(props[lbl]) for lbl in labels_used])
    m_ltl = skeleton(labels_used)

    # ─── inside build_entry_for_props ───────────────────────────────────────────
    def _nl_segment(info):
        """
        Build the natural‑language phrase for one proposition, **including**
        articles (‘the’) and a preposition (‘at’ / ‘on’) where appropriate.
        """
        verb = info["action_canon"]
        params = actions_cfg[verb]["params"]
        v_ref = info["action_ref"]

        # ------------------------------------------------------------------ #
        # 0‑argument actions  (idle, get_help, go_home, …)
        # ------------------------------------------------------------------ #
        if not params:
            return v_ref  # “idle”

        # ------------------------------------------------------------------ #
        # 1‑argument actions  (item, person, threat, target, …)
        # ------------------------------------------------------------------ #
        if (
            params == ["item"]
            or params == ["person"]
            or params == ["threat"]
            or params == ["target"]
        ):  # ← NEW
            obj = info["args_ref"][0]
            return f"{v_ref} the {obj}"  # “photograph the jaywalker”

        # ------------------------------------------------------------------ #
        # 2‑argument (item,location)   → “deliver the box to the shelf”
        # ------------------------------------------------------------------ #
        if params == ["item", "location"]:
            obj, loc = info["args_ref"]
            return f"{v_ref} the {obj} to the {loc}"

        # ------------------------------------------------------------------ #
        # 2‑argument (traffic_target,lane)  → “photograph the pedestrian on west 8th avenue”
        # ------------------------------------------------------------------ #
        if params == ["traffic_target", "lane"]:
            tgt, lane = info["args_ref"]
            prep = (
                "on" if any(k in lane for k in ["street", "avenue", "road"]) else "at"
            )
            return f"{v_ref} the {tgt} {prep} {lane}"

        # ------------------------------------------------------------------ #
        # Fallback for exotic signatures
        # ------------------------------------------------------------------ #
        return f"{v_ref} " + " ".join(info["args_ref"])

    segs_ground = [_nl_segment(props[l]) for l in labels_used]
    tpl = pick_sentence_template(key)
    ego = random.choice(EGO_REFS)

    g_raw = correct_sentence(add_ego_reference(tpl(segs_ground), ego))
    g_tok = g_raw.split()

    # ------------------------------------------------------------------ #
    # 4)  After gerund fixes, update props + regenerate segments          #
    # ------------------------------------------------------------------ #
    for info in props.values():
        bare = info["action_ref"].split()[0]
        ger = _to_gerund(bare)
        if bare not in g_raw and ger in g_raw:
            info["action_ref"] = ger
    segs_ground = [_nl_segment(props[l]) for l in labels_used]  # refresh

    # ------------------------------------------------------------------ #
    # 4)  Refresh segments *with* gerund fixes                            #
    # ------------------------------------------------------------------ #
    def _segment_tokens(lbl: str) -> List[str]:
        """Lower‑cased, punctuation‑stripped tokens of the segment *post‑fix*."""
        seg_txt = _nl_segment(props[lbl])
        seg_txt = correct_sentence(seg_txt)  # applies gerund rule
        toks = word_tokenize(seg_txt.lower())
        return [
            t.rstrip(string.punctuation) for t in toks if t not in string.punctuation
        ]

    seg_tokens = {lbl: _segment_tokens(lbl) for lbl in labels_used}

    # ------------------------------------------------------------------ #
    # 5)  Label tokens & build masked sentence                           #
    # ------------------------------------------------------------------ #
    pid_map = {lbl: i + 1 for i, lbl in enumerate(labels_used)}  # lbl → 1‑based id
    mapping = {lbl: f"prop_{pid_map[lbl]}" for lbl in labels_used}

    ids = [0] * len(g_tok)
    masked = []
    i = 0
    while i < len(g_tok):
        g_word = g_tok[i].rstrip(string.punctuation).lower()
        matched = False
        for lbl in labels_used:
            tok_seq = seg_tokens[lbl]
            L = len(tok_seq)
            if i + L <= len(g_tok) and all(
                _same_word(g_tok[i + j], tok_seq[j]) for j in range(L)
            ):
                for j in range(L):
                    ids[i + j] = pid_map[lbl]
                masked.append(mapping[lbl])  # single token in masked sent.
                i += L
                matched = True
                break
        if not matched:  # normal word
            masked.append(g_tok[i])
            i += 1

    good_trace, bad_trace = _make_traces(
        key, [_atom(props[lbl]) for lbl in labels_used]
    )

    # ------------------------------------------------------------------ #
    # 6)  Finalise entry                                                 #
    # ------------------------------------------------------------------ #
    prop_final = {mapping[l]: props[l] for l in labels_used}
    masked_tl = [mapping.get(t, t) for t in m_ltl]

    return {
        "id": idx,
        "sentence": g_tok,  # full natural language
        "tl": g_ltl,  # grounded LTL
        "masked_tl": masked_tl,  # prop_i tokens
        "grounded_sentence": masked,  # NL with prop_i
        "lifted_sentence_prop_ids": ids,  # 0 / 1 / 2 / …
        "prop_dict": prop_final,
        "good_trace": good_trace,  # satisfiable example
        "bad_trace": bad_trace,  # minimal counter‑example
    }


# ---------------------------------------------------------------------------
# 6)  Build the whole dataset                                                #
# ---------------------------------------------------------------------------


def _sample_argument(kind: str, objects: Dict, locs: List[str]) -> Tuple[str, str]:
    """Return (canonical, NL‑reference) for an argument of the given kind."""
    if kind == "item":
        key = random.choice(list(objects.keys()))
        canon = key.replace(" ", "_")
        ref = random.choice(objects[key]).replace("_", " ")

    elif kind == "location":
        canon = random.choice(locs)
        ref = canon.replace("_", " ")

    elif kind == "ego":
        canon, ref = "ego", "yourself"

    # --- target‑specific kinds -------------------------------------------
    elif kind == "person":
        mod1 = random.choice(["injured", "safe"])
        canon = mod1 + "_" + random.choice(["victim", "rescuer", "hostile"])
        ref = canon.replace("_", " ")

    elif kind == "threat":
        mod1 = random.choice(["active", "inactive", "impending", "probable", "nearest"])
        canon = (
            mod1
            + "_"
            + random.choice(
                ["gas_leak", "unstable_beam", "fire_source", "debris", "flood"]
            )
        )
        ref = canon.replace("_", " ")

    elif kind == "light":
        pos = random.choice(locs)  # north / south / …
        canon = f"light_{pos}"
        ref = f"{pos} light".replace("_", " ")

    elif kind == "lane":
        dir = random.choice(
            [
                "north",
                "south",
                "east",
                "west",
                "northwest",
                "northeast",
                "southwest",
                "southeast",
            ]
        )
        num = random.choice(
            ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
        )
        road = random.choice(["street", "avenue"])
        canon = dir + "_" + num + "_" + road  # random.choice(["lane_a", "lane_b"])
        ref = canon.replace("_", " ")

    elif kind == "traffic_target":
        canon = random.choice(
            [
                "person",
                "pedestrian",
                "vehicle",
                "car",
                "motorcycle",
                "cyclist",
                "jaywalker",
                "collision",
            ]
        )
        ref = canon

    elif kind == "sr_target":
        # choose randomly from the *existing* pools
        if random.random() < 0.5:  # 50 % person
            mod1 = random.choice(["injured", "safe", "unsafe"])
            canon = (
                mod1 + "_" + random.choice(["person", "civilian", "victim", "rescuer"])
            )
            ref = canon.replace("_", " ")

        else:  # 50 % threat
            canon = random.choice(
                ["gas_leak", "unstable_beam", "fire_source", "debris", "flood"]
            )
        ref = canon.replace("_", " ")
    elif kind == "color":
        canon = random.choice(["red", "yellow", "green"])
        ref = canon

    else:
        raise ValueError(f"Unknown param kind '{kind}'")

    return canon, ref


# ---------------------------------------------------------------------------
# 6)  Build the whole dataset  (only this function changed)                  #
# ---------------------------------------------------------------------------
def _entry_is_valid(entry: Dict) -> bool:
    """Return True iff the entry fulfils both dataset invariants."""
    n_props = len(entry["prop_dict"])
    required_ids = set(range(1, n_props + 1))
    if not required_ids.issubset(entry["lifted_sentence_prop_ids"]):
        return False

    required_toks = {f"prop_{i}" for i in required_ids}
    if not required_toks.issubset(set(entry["grounded_sentence"])):
        return False

    return True


# Token limit for LTL formulas
MAX_LTL_TOKENS = 80


def count_ltl_tokens(ltl_formula) -> int:
    """计算LTL公式的token数量"""
    # LTL公式可能是list或str
    if isinstance(ltl_formula, list):
        return len(ltl_formula)
    # LTL公式以空格分隔，每个token就是一个token
    tokens = ltl_formula.split()
    return len(tokens)


# ---------------------------------------------------------------------------
# 6)  Build the whole dataset  (with validation & resampling)                #
# ---------------------------------------------------------------------------


def build_dataset_entries(
    object_dict: Dict,
    actions_dict: Dict,
    locations: List[str],
    actions_cfg: Dict,
    num_entries: int,
    max_props: int = 3,
) -> List[Dict]:
    """
    Create *exactly* ``num_entries`` valid dataset entries.

    An entry is thrown away and re‑sampled whenever
      – it misses at least one prop‑id in lifted_sentence_prop_ids, **or**
      – its grounded_sentence does not mention every prop_k token, **or**
      – its LTL formula exceeds 80 tokens.
    """
    dataset, label_pool = [], [f"prop_{i + 1}" for i in range(max_props)]
    token_limit_rejected_count = 0  # 因token数量超限被拒绝的计数

    # keep generating until we have enough *valid* examples
    while len(dataset) < num_entries:
        # ------------- (a)  sample propositions ---------------------------
        want_labels = random.sample(label_pool, k=random.randint(1, max_props))
        props, atoms_used = {}, set()

        for lbl in want_labels:
            # keep looking for a *new* atomic proposition
            for _ in range(50):  # avoid infinite loops
                verb = random.choice(list(actions_dict.keys()))
                a_canon, a_ref = [], []
                for kind in actions_cfg[verb]["params"]:
                    c, r = _sample_argument(kind, object_dict, locations)
                    a_canon.append(c)
                    a_ref.append(r)

                atom = f"{verb}(" + ",".join(a_canon) + ")" if a_canon else verb
                if atom in atoms_used:
                    continue
                atoms_used.add(atom)

                props[lbl] = {
                    "action_canon": verb,
                    "action_ref": random.choice(actions_dict[verb]),
                    "args_canon": a_canon,
                    "args_ref": a_ref,
                }
                break  # found a fresh one

        # ------------- (b)  build entry & validate ------------------------
        tmp_idx = len(dataset)  # provisional id
        entry = build_entry_for_props(tmp_idx, props, actions_cfg)

        # 检查LTL公式token数量
        ltl_token_count = count_ltl_tokens(entry["tl"])
        if ltl_token_count > MAX_LTL_TOKENS:
            # 超过80个token，重新生成
            token_limit_rejected_count += 1
            continue

        if _entry_is_valid(entry):
            dataset.append(entry)  # keep it
        # else: drop silently and try again

    # 打印token限制统计信息
    if token_limit_rejected_count > 0:
        print(f"[LTL-gen] Token限制: 超过80个token被拒绝 {token_limit_rejected_count} 条")

    # make sure the ids are sequential 0…N‑1 (they are, but keep it explicit)
    for new_id, entry in enumerate(dataset):
        entry["id"] = new_id

    return dataset


# ---------------------------------------------------------------------------#
# 7)  Main                                                                   #
# ---------------------------------------------------------------------------#
def main():
    p = argparse.ArgumentParser(description="Scenario‑aware NL ↔ LTL generator")
    p.add_argument("-s", "--scenario", default="warehouse")
    p.add_argument("-n", "--num_entries", type=int, default=5)
    p.add_argument(
        "-o",
        "--output",
        default="new_generated_datasets/LTL/warehouse.jsonl",
        help="Output JSONL file",
    )
    args = p.parse_args()

    cfg, obj_dict, act_dict, locs, act_cfg = load_scenario(args.scenario)
    entries = build_dataset_entries(
        obj_dict, act_dict, locs, act_cfg, num_entries=args.num_entries, max_props=10
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(
        f"[LTL‑gen] ({args.scenario}) wrote {args.num_entries} examples → {out.as_posix()}"
    )


if __name__ == "__main__":
    main()

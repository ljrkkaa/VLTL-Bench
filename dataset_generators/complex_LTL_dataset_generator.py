import argparse
import json
import random
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import yaml

# ––– project‑local helpers ––––––––––––––––––––––––––––––––––––––––––––––––
from __init__ import parse_object_names, build_actions_dict


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


def _same_word(sent_tok: str, targ_tok: str) -> bool:
    """True if the tokens match literally (case-insensitive)."""
    s = sent_tok.rstrip(string.punctuation).lower()
    t = targ_tok.lower()
    return s == t


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

# --- Complex Long-horizon Templates (Mission Composition) ---
COMPLEX_LTL_TEMPLATES = [
    # 37 [Sequence] 顺序访问 A -> B -> C
    # LTL: finally (a and finally (b and finally c))
    (
        "SEQ_VISIT_3",
        3,
        lambda P: "finally ( {} and finally ( {} and finally {} ) )".format(*P).split(),
    ),
    # 38 [Sequence with Global Avoidance] 顺序访问 A -> B，同时全程避开 C
    # LTL: (finally (a and finally b)) and globally (not c)
    (
        "SEQ_VISIT_2_AVOID_1",
        3,
        lambda P: "( finally ( {} and finally {} ) ) and globally ( not {} )".format(
            P[0], P[1], P[2]
        ).split(),
    ),
    # 39 [Strict Ordering] 在 A 发生之前，B 不能发生 (Precedence)
    # LTL: not b until a
    ("PRECEDENCE_STRONG", 2, lambda P: "not {} until {}".format(P[1], P[0]).split()),
    # 40 [Looping/Patrol] 无限次访问 A，且无限次访问 B
    # LTL: (globally finally a) and (globally finally b)
    (
        "PATROL_2_POINTS",
        2,
        lambda P: "( globally finally {} ) and ( globally finally {} )".format(
            *P
        ).split(),
    ),
    # 41 [Safe Response] 如果 A 发生，则必须在 B 发生之前响应 C
    # LTL: globally (a implies (not b until c))
    (
        "SAFE_RESPONSE",
        3,
        lambda P: "globally ( {} implies ( not {} until {} ) )".format(*P).split(),
    ),
    # 42 [Triggered Stability] 一旦 A 发生，B 就必须永远保持
    # LTL: globally (a implies globally b)
    (
        "STABILIZATION",
        2,
        lambda P: "globally ( {} implies globally {} )".format(*P).split(),
    ),
]

# =========================================================================
# DWYER SPECIFICATION PATTERNS - Extended Template Set
# =========================================================================
# Parameter Mapping Conventions:
# - Primary Propositions: P, S, T (indices 0, 1, 2)
# - Scope Propositions: Q, R (indices 3, 4 when present)
#
# Pattern Structure:
# 1. Single proposition patterns (Absence, Existence, Universality, Bounded)
# 2. Two proposition patterns (Precedence, Response)
# 3. Chain patterns (Precedence/Response chains)
# =========================================================================


def _W(p, q):
    """
    Expand Weak Until: p W q === (p U q) || (globally p)
    Used to maintain compatibility with original Dwyer patterns while
    generating only Strong Until (U) formulas.
    """
    return f"( ( ( {p} ) until ( {q} ) ) or ( globally ( {p} ) ) )"


DWYER_TEMPLATES = [
    # =========================================================================
    # 1. Absence (P is false)
    #    Target: P (index 0)
    # =========================================================================
    ("ABSENCE_GLOBAL", 1, lambda P: f"globally ( not {P[0]} )".split()),
    (
        "ABSENCE_BEFORE_R",
        2,
        lambda P: f"( finally {P[1]} ) implies ( ( not {P[0]} ) until {P[1]} )".split(),
    ),
    (
        "ABSENCE_AFTER_Q",
        2,
        lambda P: f"globally ( {P[1]} implies globally ( not {P[0]} ) )".split(),
    ),
    (
        "ABSENCE_BETWEEN_Q_R",
        3,
        lambda P: f"globally ( ( {P[1]} and not {P[2]} and finally {P[2]} ) implies ( ( not {P[0]} ) until {P[2]} ) )".split(),
    ),
    (
        "ABSENCE_AFTER_Q_UNTIL_R",
        3,
        lambda P: f"globally ( ( {P[1]} and not {P[2]} ) implies {_W(f'not {P[0]}', P[2])} )".split(),
    ),
    # =========================================================================
    # 2. Existence (P becomes true)
    #    Target: P (index 0)
    # =========================================================================
    ("EXISTENCE_GLOBAL", 1, lambda P: f"finally {P[0]}".split()),
    (
        "EXISTENCE_BEFORE_R",
        2,
        lambda P: _W(f"not {P[1]}", f"{P[0]} and not {P[1]}").split(),
    ),
    (
        "EXISTENCE_AFTER_Q",
        2,
        lambda P: f"( globally not {P[1]} ) or ( finally ( {P[1]} and finally {P[0]} ) )".split(),
    ),
    (
        "EXISTENCE_BETWEEN_Q_R",
        3,
        lambda P: f"globally ( ( {P[1]} and not {P[2]} ) implies {_W(f'not {P[2]}', f'{P[0]} and not {P[2]}')} )".split(),
    ),
    (
        "EXISTENCE_AFTER_Q_UNTIL_R",
        3,
        lambda P: f"globally ( ( {P[1]} and not {P[2]} ) implies ( ( not {P[2]} ) until ( {P[0]} and not {P[2]} ) ) )".split(),
    ),
    # =========================================================================
    # 3. Bounded Existence (P occurs at most 2 times)
    #    Logic: !P W (P W (!P W (P W []!P)))
    # =========================================================================
    (
        "BOUNDED_2_GLOBAL",
        1,
        lambda P: _W(
            f"not {P[0]}", _W(P[0], _W(f"not {P[0]}", _W(P[0], f"globally not {P[0]}")))
        ).split(),
    ),
    (
        "BOUNDED_2_BEFORE_R",
        2,
        lambda P: (
            f"( finally {P[1]} ) implies "
            + _W(
                f"not {P[0]} and not {P[1]}",
                f"{P[1]} or ( "
                + _W(
                    f"{P[0]} and not {P[1]}",
                    f"{P[1]} or ( "
                    + _W(
                        f"not {P[0]} and not {P[1]}",
                        f"{P[1]} or ( "
                        + _W(
                            f"{P[0]} and not {P[1]}",
                            f"{P[1]} or ( ( not {P[0]} ) until {P[1]} )",
                        )
                        + " ) ",
                    )
                    + " ) ",
                )
                + " ) ",
            )
        ).split(),
    ),
    (
        "BOUNDED_2_AFTER_Q",
        2,
        lambda P: (
            f"( finally {P[1]} ) implies ( ( not {P[1]} ) until ( {P[1]} and "
            + _W(
                f"not {P[0]}",
                _W(P[0], _W(f"not {P[0]}", _W(P[0], f"globally not {P[0]}"))),
            )
            + " ) )"
        ).split(),
    ),
    (
        "BOUNDED_2_BETWEEN_Q_R",
        3,
        lambda P: (
            f"globally ( ( {P[1]} and finally {P[2]} ) implies ( ( not {P[0]} and not {P[2]} ) until ( {P[2]} or ( ( {P[0]} and not {P[2]} ) until "
            f"( {P[2]} or ( ( not {P[0]} and not {P[2]} ) until ( {P[2]} or ( ( {P[0]} and not {P[2]} ) until "
            f"( {P[2]} or ( ( not {P[0]} ) until {P[2]} ) ) ) ) ) ) ) )"
        ).split(),
    ),
    # =========================================================================
    # 4. Universality (P is true)
    #    Target: P (index 0)
    # =========================================================================
    ("UNIVERSALITY_GLOBAL", 1, lambda P: f"globally {P[0]}".split()),
    (
        "UNIVERSALITY_BEFORE_R",
        2,
        lambda P: f"( finally {P[1]} ) implies ( {P[0]} until {P[1]} )".split(),
    ),
    (
        "UNIVERSALITY_AFTER_Q",
        2,
        lambda P: f"globally ( {P[1]} implies globally {P[0]} )".split(),
    ),
    (
        "UNIVERSALITY_BETWEEN_Q_R",
        3,
        lambda P: f"globally ( ( {P[1]} and not {P[2]} and finally {P[2]} ) implies ( {P[0]} until {P[2]} ) )".split(),
    ),
    (
        "UNIVERSALITY_AFTER_Q_UNTIL_R",
        3,
        lambda P: f"globally ( ( {P[1]} and not {P[2]} ) implies {_W(P[0], P[2])} )".split(),
    ),
    # =========================================================================
    # 5. Precedence (S precedes P)
    #    Mapping: P (Subject) = x[0], S (Precedes) = x[1]
    # =========================================================================
    ("PRECEDENCE_GLOBAL", 2, lambda P: _W(f"not {P[0]}", P[1]).split()),
    (
        "PRECEDENCE_BEFORE_R",
        3,
        lambda P: f"( finally {P[2]} ) implies ( ( not {P[0]} ) until ( {P[1]} or {P[2]} ) )".split(),
    ),
    (
        "PRECEDENCE_AFTER_Q",
        3,
        lambda P: f"( globally not {P[2]} ) or ( finally ( {P[2]} and {_W(f'not {P[0]}', P[1])} ) )".split(),
    ),
    (
        "PRECEDENCE_BETWEEN_Q_R",
        4,
        lambda P: f"globally ( ( {P[2]} and not {P[3]} and finally {P[3]} ) implies ( ( not {P[0]} ) until ( {P[1]} or {P[3]} ) ) )".split(),
    ),
    (
        "PRECEDENCE_AFTER_Q_UNTIL_R",
        4,
        lambda P: f"globally ( ( {P[2]} and not {P[3]} ) implies {_W(f'not {P[0]}', f'{P[1]} or {P[3]}')} )".split(),
    ),
    # =========================================================================
    # 6. Response (S responds to P)
    #    Mapping: P (Trigger) = x[0], S (Response) = x[1]
    # =========================================================================
    (
        "RESPONSE_GLOBAL",
        2,
        lambda P: f"globally ( {P[0]} implies finally {P[1]} )".split(),
    ),
    (
        "RESPONSE_BEFORE_R",
        3,
        lambda P: f"( finally {P[2]} ) implies ( ( {P[0]} implies ( ( not {P[2]} ) until ( {P[1]} and not {P[2]} ) ) ) until {P[2]} )".split(),
    ),
    (
        "RESPONSE_AFTER_Q",
        3,
        lambda P: f"globally ( {P[2]} implies globally ( {P[0]} implies finally {P[1]} ) )".split(),
    ),
    (
        "RESPONSE_BETWEEN_Q_R",
        4,
        lambda P: f"globally ( ( {P[2]} and not {P[3]} and finally {P[3]} ) implies ( ( {P[0]} implies ( ( not {P[3]} ) until ( {P[1]} and not {P[3]} ) ) ) until {P[3]} ) )".split(),
    ),
    (
        "RESPONSE_AFTER_Q_UNTIL_R",
        4,
        lambda P: f"globally ( ( {P[2]} and not {P[3]} ) implies {_W(f'{P[0]} implies ( ( not {P[3]} ) until ( {P[1]} and not {P[3]} ) )', P[3])} )".split(),
    ),
    # =========================================================================
    # 7. Precedence Chain (2 causes - 1 effect) -> S, T precedes P
    #    Mapping: P=x[0], S=x[1], T=x[2]
    # =========================================================================
    (
        "PRECEDENCE_CHAIN_21_GLOBAL",
        3,
        lambda P: f"( finally {P[0]} ) implies ( ( not {P[0]} ) until ( {P[1]} and not {P[0]} and next ( ( not {P[0]} ) until {P[2]} ) ) )".split(),
    ),
    # =========================================================================
    # 8. Precedence Chain (1 cause - 2 effects) -> P precedes S, T
    #    Mapping: P=x[0], S=x[1], T=x[2]
    # =========================================================================
    (
        "PRECEDENCE_CHAIN_12_GLOBAL",
        3,
        lambda P: f"( finally ( {P[1]} and next finally {P[2]} ) ) implies ( ( not {P[1]} ) until {P[0]} )".split(),
    ),
    # =========================================================================
    # 9. Response Chain (P responds to S, T) -> Trigger S then T implies Response P
    #    Mapping: P=x[0], S=x[1], T=x[2]
    # =========================================================================
    (
        "RESPONSE_CHAIN_21_GLOBAL",
        3,
        lambda P: f"globally ( ( {P[1]} and next finally {P[2]} ) implies next ( finally ( {P[2]} and finally {P[0]} ) ) )".split(),
    ),
    # =========================================================================
    # 10. Response Chain (S, T responds to P) -> Trigger P implies S then T
    #     Mapping: P=x[0], S=x[1], T=x[2]
    # =========================================================================
    (
        "RESPONSE_CHAIN_12_GLOBAL",
        3,
        lambda P: f"globally ( {P[0]} implies finally ( {P[1]} and next finally {P[2]} ) )".split(),
    ),
    # =========================================================================
    # 11. Constrained Chain (S, T without Z responds to P)
    #     Mapping: P=x[0], S=x[1], T=x[2], Z=x[3]
    # =========================================================================
    (
        "CONSTRAINED_CHAIN_GLOBAL",
        4,
        lambda P: f"globally ( {P[0]} implies finally ( {P[1]} and not {P[3]} and next ( ( not {P[3]} ) until {P[2]} ) ) )".split(),
    ),
]

# =========================================================================
# DWYER SEMANTIC TEMPLATES - Natural Language Patterns
# =========================================================================

DWYER_SEMANTIC_TEMPLATES = {
    # Absence Patterns
    "ABSENCE_GLOBAL": [
        lambda s: f"Never {s[0]}.",
        lambda s: f"At no time should {s[0]} occur.",
        lambda s: f"{s[0]} is strictly forbidden throughout the entire mission.",
    ],
    "ABSENCE_BEFORE_R": [
        lambda s: f"Before {s[1]}, {s[0]} must never happen.",
        lambda s: f"If {s[1]} occurs, ensure that {s[0]} has not happened before it.",
    ],
    "ABSENCE_AFTER_Q": [
        lambda s: f"After {s[1]}, {s[0]} must never occur.",
        lambda s: f"Once {s[1]} happens, {s[0]} is prohibited for the rest of the mission.",
    ],
    "ABSENCE_BETWEEN_Q_R": [
        lambda s: f"Between {s[1]} and {s[2]}, ensure that {s[0]} never happens.",
        lambda s: f"During the interval from {s[1]} to {s[2]}, {s[0]} is forbidden.",
    ],
    "ABSENCE_AFTER_Q_UNTIL_R": [
        lambda s: f"After {s[1]} until {s[2]}, {s[0]} must never occur.",
        lambda s: f"From the moment {s[1]} occurs, {s[0]} is prohibited until {s[2]} happens.",
    ],
    # Existence Patterns
    "EXISTENCE_GLOBAL": [
        lambda s: f"{s[0]} must eventually happen.",
        lambda s: f"At some point, {s[0]} should occur.",
    ],
    "EXISTENCE_BEFORE_R": [
        lambda s: f"Before {s[1]}, {s[0]} must occur at least once.",
        lambda s: f"If {s[1]} occurs, {s[0]} must have happened beforehand.",
    ],
    "EXISTENCE_AFTER_Q": [
        lambda s: f"After {s[1]}, {s[0]} must eventually happen.",
        lambda s: f"Once {s[1]} occurs, ensure that {s[0]} happens at some point.",
    ],
    "EXISTENCE_BETWEEN_Q_R": [
        lambda s: f"Between {s[1]} and {s[2]}, {s[0]} must occur at least once.",
        lambda s: f"During the interval from {s[1]} to {s[2]}, {s[0]} should happen.",
    ],
    "EXISTENCE_AFTER_Q_UNTIL_R": [
        lambda s: f"After {s[1]} until {s[2]}, {s[0]} must occur.",
        lambda s: f"From {s[1]} to {s[2]}, ensure {s[0]} happens.",
    ],
    # Bounded Existence Patterns
    "BOUNDED_2_GLOBAL": [
        lambda s: f"{s[0]} may happen at most twice during the entire mission.",
        lambda s: f"Limit {s[0]} to no more than two occurrences.",
    ],
    "BOUNDED_2_BEFORE_R": [
        lambda s: f"Before {s[1]}, {s[0]} can happen at most twice.",
        lambda s: f"Ensure {s[0]} occurs no more than two times before {s[1]}.",
    ],
    "BOUNDED_2_AFTER_Q": [
        lambda s: f"After {s[1]}, {s[0]} may occur at most twice.",
        lambda s: f"Once {s[1]} occurs, limit {s[0]} to two maximum occurrences.",
    ],
    "BOUNDED_2_BETWEEN_Q_R": [
        lambda s: f"Between {s[1]} and {s[2]}, {s[0]} can happen at most twice.",
        lambda s: f"During the interval from {s[1]} to {s[2]}, limit {s[0]} to two occurrences.",
    ],
    # Universality Patterns
    "UNIVERSALITY_GLOBAL": [
        lambda s: f"Always {s[0]}.",
        lambda s: f"{s[0]} must hold at every step of the mission.",
    ],
    "UNIVERSALITY_BEFORE_R": [
        lambda s: f"Before {s[1]}, {s[0]} must always hold.",
        lambda s: f"Until {s[1]} occurs, maintain {s[0]} continuously.",
    ],
    "UNIVERSALITY_AFTER_Q": [
        lambda s: f"After {s[1]}, {s[0]} must always hold.",
        lambda s: f"Once {s[1]} occurs, {s[0]} must be maintained forever after.",
    ],
    "UNIVERSALITY_BETWEEN_Q_R": [
        lambda s: f"Between {s[1]} and {s[2]}, {s[0]} must always hold.",
        lambda s: f"Throughout the interval from {s[1]} to {s[2]}, maintain {s[0]} continuously.",
    ],
    "UNIVERSALITY_AFTER_Q_UNTIL_R": [
        lambda s: f"After {s[1]} until {s[2]}, {s[0]} must always hold.",
        lambda s: f"From {s[1]} to {s[2]}, {s[0]} must be continuously maintained.",
    ],
    # Precedence Patterns
    "PRECEDENCE_GLOBAL": [
        lambda s: f"{s[1]} must happen before {s[0]}.",
        lambda s: f"{s[0]} can only occur after {s[1]} has happened.",
    ],
    "PRECEDENCE_BEFORE_R": [
        lambda s: f"Before {s[2]}, either {s[1]} or {s[0]} must happen, with {s[1]} preceding {s[0]}.",
        lambda s: f"If {s[2]} occurs, ensure {s[1]} has happened before {s[0]}.",
    ],
    "PRECEDENCE_AFTER_Q": [
        lambda s: f"After {s[2]}, {s[1]} must precede {s[0]}.",
        lambda s: f"Once {s[2]} occurs, {s[0]} may only happen after {s[1]}.",
    ],
    "PRECEDENCE_BETWEEN_Q_R": [
        lambda s: f"Between {s[2]} and {s[3]}, {s[1]} must happen before {s[0]}.",
        lambda s: f"During the interval from {s[2]} to {s[3]}, ensure {s[1]} precedes {s[0]}.",
    ],
    "PRECEDENCE_AFTER_Q_UNTIL_R": [
        lambda s: f"After {s[2]} until {s[3]}, {s[1]} must happen before {s[0]}.",
        lambda s: f"From {s[2]} to {s[3]}, {s[0]} may only occur after {s[1]}.",
    ],
    # Response Patterns
    "RESPONSE_GLOBAL": [
        lambda s: f"Whenever {s[0]} happens, {s[1]} must eventually happen.",
        lambda s: f"If {s[0]} occurs, {s[1]} must follow at some point.",
    ],
    "RESPONSE_BEFORE_R": [
        lambda s: f"Before {s[2]}, whenever {s[0]} happens, {s[1]} must respond.",
        lambda s: f"If {s[0]} occurs before {s[2]}, ensure {s[1]} happens in response.",
    ],
    "RESPONSE_AFTER_Q": [
        lambda s: f"After {s[2]}, whenever {s[0]} happens, {s[1]} must eventually happen.",
        lambda s: f"Once {s[2]} occurs, {s[0]} should always be followed by {s[1]}.",
    ],
    "RESPONSE_BETWEEN_Q_R": [
        lambda s: f"Between {s[2]} and {s[3]}, whenever {s[0]} happens, {s[1]} must respond.",
        lambda s: f"During the interval from {s[2]} to {s[3]}, ensure {s[1]} follows {s[0]}.",
    ],
    "RESPONSE_AFTER_Q_UNTIL_R": [
        lambda s: f"After {s[2]} until {s[3]}, whenever {s[0]} happens, {s[1]} must respond.",
        lambda s: f"From {s[2]} to {s[3]}, {s[0]} should always trigger {s[1]}.",
    ],
    # Chain Patterns
    "PRECEDENCE_CHAIN_21_GLOBAL": [
        lambda s: f"Before {s[0]}, both {s[1]} and then {s[2]} must happen in sequence.",
        lambda s: f"{s[0]} can only occur after {s[1]} has happened, followed by {s[2]}.",
    ],
    "PRECEDENCE_CHAIN_12_GLOBAL": [
        lambda s: f"Before {s[0]}, {s[1]} and {s[2]} must both occur (in that order).",
        lambda s: f"Ensure {s[1]} and then {s[2]} happen before {s[0]}.",
    ],
    "RESPONSE_CHAIN_21_GLOBAL": [
        lambda s: f"Whenever {s[1]} happens and is followed by {s[2]}, {s[0]} must eventually respond.",
        lambda s: f"If {s[1]} occurs and then {s[2]}, ensure {s[0]} happens in response.",
    ],
    "RESPONSE_CHAIN_12_GLOBAL": [
        lambda s: f"Whenever {s[0]} happens, it must eventually be followed by {s[1]} and then {s[2]}.",
        lambda s: f"If {s[0]} occurs, ensure {s[1]} and subsequently {s[2]} happen.",
    ],
    "CONSTRAINED_CHAIN_GLOBAL": [
        lambda s: f"Whenever {s[0]} happens, {s[1]} must happen, and then {s[2]} must follow, while avoiding {s[3]}.",
        lambda s: f"If {s[0]} occurs, respond with {s[1]} and then {s[2]}, but never {s[3]} in between.",
    ],
}

# Extend existing template lists with Dwyer patterns
LTL_TEMPLATES_STATE.extend(DWYER_TEMPLATES)
for k, v in DWYER_SEMANTIC_TEMPLATES.items():
    SEMANTIC_TEMPLATES.setdefault(k, []).extend(v)


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

# --- Complex Semantic Templates (Mission Composition) ---
COMPLEX_SEMANTIC_TEMPLATES = {
    "SEQ_VISIT_3": [
        lambda s: f"First {s[0]}, then {s[1]}, and finally {s[2]}.",
        lambda s: f"Complete the following sequence: {s[0]}, followed by {s[1]}, and ending with {s[2]}.",
    ],
    "SEQ_VISIT_2_AVOID_1": [
        lambda s: f"Visit {s[0]} and then {s[1]}, while always avoiding {s[2]}.",
        lambda s: f"Ensure that {s[2]} never happens, but make sure to {s[0]} followed by {s[1]}.",
    ],
    "PRECEDENCE_STRONG": [
        lambda s: f"{s[1]} must not happen until {s[0]} has occurred.",
        lambda s: f"Do not allow {s[1]} unless {s[0]} has already finished.",
    ],
    "PATROL_2_POINTS": [
        lambda s: f"Keep visiting {s[0]} and {s[1]} infinitely often.",
        lambda s: f"Continually patrol between {s[0]} and {s[1]}.",
    ],
    "SAFE_RESPONSE": [
        lambda s: f"Whenever {s[0]} happens, you must perform {s[2]} before {s[1]} is allowed to occur.",
    ],
    "STABILIZATION": [
        lambda s: f"Once {s[0]} happens, ensure that {s[1]} holds forever after.",
        lambda s: f"If {s[0]} is triggered, maintain {s[1]} for the rest of the mission.",
    ],
}

LTL_TEMPLATES_STATE.extend(ADDITIONAL_LTL_TEMPLATES_STATE)
LTL_TEMPLATES_STATE.extend(COMPLEX_LTL_TEMPLATES)
for k, v in ADDITIONAL_SEMANTIC_TEMPLATES.items():
    SEMANTIC_TEMPLATES.setdefault(k, []).extend(v)
for k, v in COMPLEX_SEMANTIC_TEMPLATES.items():
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
    """Simple sentence correction without NLTK (faster)."""
    # Basic capitalization
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence


def pick_sentence_template(key: str):
    return random.choice(SEMANTIC_TEMPLATES.get(key, GENERIC_TEMPLATES))


# =========================================================================
# Safe Renumbering Helper (Prevents ID Collision Bug)
# =========================================================================
def safe_renumber(text: str, mapping: Dict[str, str]) -> str:
    """
    使用单次正则扫描进行安全的重命名，防止顺序替换导致的 ID 碰撞。
    例如：防止 {prop_2->prop_3, prop_3->prop_2} 变成全部是 prop_3

    Args:
        text: 需要重命名的文本
        mapping: 旧ID -> 新ID 的映射字典

    Returns:
        重命名后的文本
    """
    if not mapping:
        return text

    # 1. 构建正则：匹配所有需要替换的旧 prop (如 \bprop_1\b|\bprop_2\b)
    # 按长度降序排列，防止 prop_10 被错误匹配成 prop_1
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b")

    # 2. 使用回调函数一次性完成替换
    return pattern.sub(lambda m: mapping[m.group()], text)


# =========================================================================
# BLUEPRINT-BASED MISSION COMPOSITION (Non-Recursive)
# =========================================================================
# Instead of recursive splitting and remapping, we use predefined blueprints.
# All props already have unique IDs (prop_1, prop_2, ...) - no remapping needed!
# =========================================================================


def _atom(info):
    """Generate canonical atom string from proposition info."""
    return (
        f"{info['action_canon']}({','.join(info['args_canon'])})"
        if info["args_canon"]
        else info["action_canon"]
    )


def _nl_segment(info, actions_cfg):
    """Generate NL segment for a proposition."""
    verb = info["action_canon"]
    params = actions_cfg[verb]["params"]
    v_ref = info["action_ref"]

    if not params:
        return v_ref
    if (
        params == ["item"]
        or params == ["person"]
        or params == ["threat"]
        or params == ["target"]
    ):
        return f"{v_ref} the {info['args_ref'][0]}"
    if params == ["item", "location"]:
        return f"{v_ref} the {info['args_ref'][0]} to the {info['args_ref'][1]}"
    if params == ["traffic_target", "lane"]:
        prep = (
            "on"
            if any(k in info["args_ref"][1] for k in ["street", "avenue", "road"])
            else "at"
        )
        return f"{v_ref} the {info['args_ref'][0]} {prep} {info['args_ref'][1]}"
    return f"{v_ref} " + " ".join(info["args_ref"])


def _build_atomic_entry(prop_key: str, prop_info: Dict, actions_cfg) -> Dict:
    """
    Build a complete atomic entry with temporal operators (for standalone use).
    e.g., "finally search(cake)", "globally not avoid_obstacle"
    """
    ego = random.choice(EGO_REFS)

    # Select template (prefer arity=1 for atomic chunks)
    viable = [t for t in LTL_TEMPLATES_STATE if t[1] <= 1]
    viable.sort(key=lambda x: x[1], reverse=True)
    key, _, skeleton = random.choice(viable[: max(1, len(viable) // 2)])

    # Generate LTL
    ltl = " ".join(skeleton([_atom(prop_info)]))
    masked_tl = " ".join(skeleton([prop_key]))

    # Generate NL
    tpl = pick_sentence_template(key)
    nl_sent = correct_sentence(
        add_ego_reference(tpl([_nl_segment(prop_info, actions_cfg)]), ego)
    )

    return {
        "tl": ltl,
        "masked_tl": masked_tl,
        "sentence": nl_sent,
        "grounded_sentence": nl_sent,
    }


# =========================================================================
# PURE ATOM EXTRACTION (For Blueprint Composition)
# =========================================================================
# These functions extract raw atoms WITHOUT temporal operators.
# Temporal logic is applied at the blueprint level, NOT at the atom level.
# This prevents invalid formulas like "not finally ..."
# =========================================================================


def _pure_atom(prop_info: Dict) -> str:
    """
    Extract pure action atom WITHOUT any temporal operators.
    e.g., "search(cake)", "avoid_obstacle"
    Used for composing temporal formulas at blueprint level.
    """
    args = ",".join(prop_info["args_canon"])
    return f"{prop_info['action_canon']}({args})" if args else prop_info["action_canon"]


def _verb_phrase(prop_info: Dict, actions_cfg: Dict) -> str:
    """
    Extract verb phrase for NL composition (no "must", no "eventually").
    e.g., "watch the sharpie", "search for the cake"
    Used for building natural sentences like "Do not [phrase] until..."
    """
    return _nl_segment(prop_info, actions_cfg)


# =========================================================================
# BLUEPRINT DEFINITIONS (Fixed: Use Pure Atoms + Temporal Logic at Blueprint Level)
# =========================================================================


def _blueprint_simple_list(
    idx: int, props: Dict, prop_keys: List[str], actions_cfg: Dict
) -> Dict:
    """
    Blueprint 1: Simple AND List with Temporal Operators
    Structure: (<>A) and (<>B) and (<>C)...
    Sentence: "Eventually A. Also, B. Also, C."
    """
    # Apply "finally" to each atom (pure atoms + temporal operator)
    final_tl = " and ".join([f"( finally {_pure_atom(props[k])} )" for k in prop_keys])
    final_masked_tl = " and ".join([f"( finally {k} )" for k in prop_keys])

    # NL with "eventually"
    connectors = [" Also, ", " Moreover, ", " Additionally, ", " Furthermore, "]
    phrases = [_verb_phrase(props[k], actions_cfg) for k in prop_keys]
    final_sentence = f"Eventually {phrases[0]}." + "".join(
        [random.choice(connectors) + phrases[i] + "." for i in range(1, len(phrases))]
    )

    # Grounded
    final_grounded = final_sentence

    return {
        "id": idx,
        "tl": final_tl,
        "masked_tl": final_masked_tl,
        "sentence": final_sentence,
        "grounded_sentence": final_grounded,
    }


def _blueprint_safety_liveness(
    idx: int, props: Dict, prop_keys: List[str], actions_cfg: Dict
) -> Dict:
    """
    Blueprint 2: Safety + Liveness
    Structure: ([]!A) and (<>B) and (<>C)...
    Sentence: "Never A. Eventually B. Eventually C."
    """
    safety_key = prop_keys[-1]
    liveness_keys = prop_keys[:-1]

    # Safety: globally not
    safety_tl = f"( globally not {_pure_atom(props[safety_key])} )"
    safety_masked = f"( globally not {safety_key} )"
    safety_phrase = _verb_phrase(props[safety_key], actions_cfg)
    safety_sentence = f"Never {safety_phrase}."

    # Liveness: finally each
    liveness_tl = " and ".join(
        [f"( finally {_pure_atom(props[k])} )" for k in liveness_keys]
    )
    liveness_masked = " and ".join([f"( finally {k} )" for k in liveness_keys])
    liveness_phrases = [_verb_phrase(props[k], actions_cfg) for k in liveness_keys]
    liveness_sentence = " Eventually ".join([p + "." for p in liveness_phrases])

    # Combine
    final_tl = f"( {safety_tl} ) and ( {liveness_tl} )"
    final_masked_tl = f"( {safety_masked} ) and ( {liveness_masked} )"
    final_sentence = f"{safety_sentence} {liveness_sentence}"

    return {
        "id": idx,
        "tl": final_tl,
        "masked_tl": final_masked_tl,
        "sentence": final_sentence,
        "grounded_sentence": final_sentence,
    }


def _blueprint_precedence_chain(
    idx: int, props: Dict, prop_keys: List[str], actions_cfg: Dict
) -> Dict:
    """
    Blueprint 3: Precedence Chain (Correct implementation)
    Structure: (!A U B) and (!B U C) and ... and (<>last)
    Sentence: "Do not A until B. Do not B until C. Finally, do D."

    Prevents: "not finally ... until ..." (the bug we fixed)
    """
    n = len(prop_keys)

    # Build chain: (!K_i U K_{i+1}) for i in 0..n-2
    tl_parts = []
    masked_parts = []

    for i in range(n - 1):
        k_curr = prop_keys[i]
        k_next = prop_keys[i + 1]
        tl_parts.append(
            f"( ( not {_pure_atom(props[k_curr])} ) until {_pure_atom(props[k_next])} )"
        )
        masked_parts.append(f"( ( not {k_curr} ) until {k_next} )")

    # Add liveness for last element
    last_key = prop_keys[-1]
    tl_parts.append(f"( finally {_pure_atom(props[last_key])} )")
    masked_parts.append(f"( finally {last_key} )")

    final_tl = " and ".join(tl_parts)
    final_masked_tl = " and ".join(masked_parts)

    # NL: "Do not [phrase_i] until [phrase_{i+1}]. ..."
    phrases = [_verb_phrase(props[k], actions_cfg) for k in prop_keys]
    sent_parts = []
    for i in range(n - 1):
        sent_parts.append(f"Do not {phrases[i]} until {phrases[i + 1]}.")
    sent_parts.append(f"Finally, do {phrases[-1]}.")

    final_sentence = " ".join(sent_parts)

    return {
        "id": idx,
        "tl": final_tl,
        "masked_tl": final_masked_tl,
        "sentence": final_sentence,
        "grounded_sentence": final_sentence,
    }


def _blueprint_trigger_response(
    idx: int, props: Dict, prop_keys: List[str], actions_cfg: Dict
) -> Dict:
    """
    Blueprint 4: Trigger -> Response Chain
    Structure: (<>Trigger) implies ( (<>Response1) U (<>Response2) ... )
    Sentence: "Whenever A happens, do B until C happens."

    Uses pure atoms: "avoid_obstacle" not "finally avoid_obstacle"
    """
    trigger_key = prop_keys[0]
    response_keys = prop_keys[1:]

    # LTL: (finally Trigger) implies (Response chain with Until)
    trigger_atom = f"( finally {_pure_atom(props[trigger_key])} )"
    trigger_masked = f"( finally {trigger_key} )"

    # Response chain: build until chain or and chain
    if len(response_keys) == 1:
        resp_atom = f"( finally {_pure_atom(props[response_keys[0]])} )"
        resp_masked = f"( finally {response_keys[0]} )"
    else:
        # Random: AND or UNTIL chain
        if random.random() < 0.5:
            resp_atom = " and ".join(
                [f"( finally {_pure_atom(props[k])} )" for k in response_keys]
            )
            resp_masked = " and ".join([f"( finally {k} )" for k in response_keys])
        else:
            # Until chain
            resp_atom = f"( finally {_pure_atom(props[response_keys[0]])} )"
            resp_masked = f"( finally {response_keys[0]} )"
            for k in response_keys[1:]:
                resp_atom = f"( {resp_atom} until ( finally {_pure_atom(props[k])} ) )"
                resp_masked = f"( {resp_masked} until ( finally {k} ) )"

    final_tl = f"( {trigger_atom} ) implies ( {resp_atom} )"
    final_masked_tl = f"( {trigger_masked} ) implies ( {resp_masked} )"

    # NL: "Whenever [phrase_trigger], [do response until ... / do response1 and response2]"
    trigger_phrase = _verb_phrase(props[trigger_key], actions_cfg)
    response_phrases = [_verb_phrase(props[k], actions_cfg) for k in response_keys]

    if len(response_keys) == 1:
        response_sent = f"do {response_phrases[0]}"
    else:
        if random.random() < 0.5:
            response_sent = "do " + " and ".join(response_phrases)
        else:
            response_sent = f"do {response_phrases[0]} until {response_phrases[-1]}"

    final_sentence = f"Whenever {trigger_phrase} happens, {response_sent}."

    return {
        "id": idx,
        "tl": final_tl,
        "masked_tl": final_masked_tl,
        "sentence": final_sentence,
        "grounded_sentence": final_sentence,
    }


def _blueprint_alternative_plans(
    idx: int, props: Dict, prop_keys: List[str], actions_cfg: Dict
) -> Dict:
    """
    Blueprint 5: Alternative Plans (OR)
    Structure: (<>A) or (<>B) or (<>C)...
    Sentence: "Either eventually A, or eventually B, or eventually C."
    """
    final_tl = " or ".join([f"( finally {_pure_atom(props[k])} )" for k in prop_keys])
    final_masked_tl = " or ".join([f"( finally {k} )" for k in prop_keys])

    phrases = [_verb_phrase(props[k], actions_cfg) for k in prop_keys]
    final_sentence = "Either eventually " + ", or eventually ".join(phrases) + "."

    return {
        "id": idx,
        "tl": final_tl,
        "masked_tl": final_masked_tl,
        "sentence": final_sentence,
        "grounded_sentence": final_sentence,
    }


def _blueprint_stability(
    idx: int, props: Dict, prop_keys: List[str], actions_cfg: Dict
) -> Dict:
    """
    Blueprint 6: Stability/Stickiness
    Structure: (<>Trigger) implies (globally Effect)
    Sentence: "Once A happens, always B."
    """
    if len(prop_keys) < 2:
        return _blueprint_simple_list(idx, props, prop_keys, actions_cfg)

    trigger_key = prop_keys[0]
    effect_key = prop_keys[1]

    # LTL: (finally Trigger) implies (globally Effect)
    final_tl = f"( ( finally {_pure_atom(props[trigger_key])} ) implies ( globally {_pure_atom(props[effect_key])} ) )"
    final_masked_tl = f"( ( finally {trigger_key} ) implies ( globally {effect_key} ) )"

    trigger_phrase = _verb_phrase(props[trigger_key], actions_cfg)
    effect_phrase = _verb_phrase(props[effect_key], actions_cfg)

    final_sentence = f"Once {trigger_phrase} happens, always {effect_phrase}."

    return {
        "id": idx,
        "tl": final_tl,
        "masked_tl": final_masked_tl,
        "sentence": final_sentence,
        "grounded_sentence": final_sentence,
    }


# =========================================================================
# 8. HYBRID MISSION FACTORY (New Architecture)
# =========================================================================


# --- 1. Logic Fragments ---

def _fragment_patrol(props: List[str]) -> str:
    """Generate patrol mode: []<> (p1 && <> p2 && <> p3...)"""
    if not props:
        return ""
    if len(props) == 1:
        return f"globally ( finally {props[0]} )"
    
    # Nested finally structure: p1 and finally (p2 and finally p3)
    inner = props[-1]
    for p in reversed(props[:-1]):
        inner = f"( {p} and finally {inner} )"
    return f"globally ( finally {inner} )"


def _fragment_avoidance(props: List[str]) -> str:
    """Generate safety/avoidance mode: []!p1 && []!p2"""
    return " and ".join([f"( globally not {p} )" for p in props])


def _fragment_sequence(props: List[str]) -> str:
    """Generate sequence mode: <> (p1 && <> (p2 && <> p3))"""
    if not props:
        return ""
    if len(props) == 1:
        return f"finally {props[0]}"
        
    inner = props[-1]
    for p in reversed(props[:-1]):
        inner = f"( {p} and finally {inner} )"
    return f"finally {inner}"


def _fragment_reaction(props: List[str]) -> str:
    """Generate response constraint: [] (Trigger -> X (Stay U Target))"""
    if len(props) < 3:
        return ""
    return f"globally ( {props[0]} implies next ( {props[1]} until {props[2]} ) )"


# --- 2. Basic Template Bridge ---

def _get_basic_template_fragment(prop_keys: List[str], props: Dict, actions_cfg: Dict) -> Tuple[str, Callable, str]:
    """
    Selects a basic template from LTL_TEMPLATES_STATE that matches the arity.
    Returns (LTL_str, NL_lambda, Masked_LTL_str)
    """
    n = len(prop_keys)
    
    # 1. Filter templates by exact arity match
    candidates = [t for t in LTL_TEMPLATES_STATE if t[1] == n]
    
    # Fallback: if no exact match, try arity=2 and take first 2 props (if n > 2)
    # or arity=1 (if n=1)
    if not candidates:
        if n > 2:
            candidates = [t for t in LTL_TEMPLATES_STATE if t[1] == 2]
            used_keys = prop_keys[:2]
        else:
            candidates = [t for t in LTL_TEMPLATES_STATE if t[1] == 1]
            used_keys = prop_keys[:1]
    else:
        used_keys = prop_keys

    if not candidates:
        # Emergency fallback to simple sequence
        return (
            _fragment_sequence([_pure_atom(props[k]) for k in prop_keys]),
            lambda s: " then ".join(s),
            _fragment_sequence(prop_keys)
        )

    # 2. Pick a template
    key, arity, skeleton = random.choice(candidates)
    
    # 3. Generate LTL & Masked
    # skeleton returns list of tokens, join with space
    # BUT: skeleton expects pure atoms? 
    # Logic in _build_atomic_entry uses _atom(prop_info) which includes args.
    # Here we also use _pure_atom(props[k]) which includes args. 
    # Because LTL_TEMPLATES_STATE skeletons like ["finally", "(", "not", P[0], ")"] expect P[0] to be the atom string.
    
    atom_strs = [_pure_atom(props[k]) for k in used_keys]
    ltl_str = " ".join(skeleton(atom_strs))
    
    masked_strs = used_keys
    masked_ltl_str = " ".join(skeleton(masked_strs))
    
    # 4. Get NL template lambda
    # We return the lambda so it can be called with verb phrases later
    # SEMANTIC_TEMPLATES[key] is a list of lambdas
    nl_templates = SEMANTIC_TEMPLATES.get(key, GENERIC_TEMPLATES)
    nl_lambda = random.choice(nl_templates)
    
    return ltl_str, nl_lambda, masked_ltl_str


# --- 3. Hybrid Fusion Blueprint ---

def _blueprint_hybrid_fusion(idx: int, props: Dict, prop_keys: List[str], actions_cfg: Dict) -> Dict:
    """
    Hybrid Mission Factory:
    Splits props into 3 groups (Complex, Basic, Safety) and fuses them.
    """
    n = len(prop_keys)
    # Shuffle keys to ensure random distribution
    # Note: prop_keys is a list of strings 'prop_1', 'prop_2'... 
    # We should work on a copy to not affect caller? Caller passes a list, we can shuffle it.
    # But usually we want deterministic behavior if seeded. random.shuffle is fine.
    # However, to avoid side effects if this list is reused (it's not), good practice to copy.
    keys_shuffled = prop_keys[:]
    random.shuffle(keys_shuffled)
    
    # 1. Dynamic Partitioning
    # We need at least some props. If n is small (e.g. 2), we can't split into 3 useful groups.
    # Minimum for full hybrid: 2 (complex) + 2 (basic) + 1 (safety) = 5?
    # Logic from prompt: "n=7 -> 3, 2, 2"
    
    g_complex, g_basic, g_safety = [], [], []

    if n < 3:
        # Too small for hybrid, just dump to complex
        g_complex = keys_shuffled
    else:
        # Try to give 2+ to complex, maybe some to basic, rest to safety
        # Partition logic:
        # random cut points
        if n >= 5:
            cut1 = random.randint(2, n - 3) # ensure at least 2 for complex, leave 3
            cut2 = random.randint(cut1 + 1, n - 1) # ensure at least 1 for basic
        elif n == 4:
            cut1 = 2
            cut2 = 3
        else: # n=3
            cut1 = 1
            cut2 = 2
            
        g_complex = keys_shuffled[:cut1]
        g_basic = keys_shuffled[cut1:cut2]
        g_safety = keys_shuffled[cut2:]

    ltl_segs, nl_segs, msk_segs = [], [], []

    # 2. Complex Mission Segment
    if g_complex:
        # Patrol if >= 2 props and coin flip, else Sequence
        is_patrol = random.random() > 0.5 and len(g_complex) >= 2
        func = _fragment_patrol if is_patrol else _fragment_sequence
        
        # LTL & Masked
        ltl_segs.append(func([_pure_atom(props[k]) for k in g_complex]))
        msk_segs.append(func(g_complex))
        
        # NL
        verb_phrases = [_verb_phrase(props[k], actions_cfg) for k in g_complex]
        verb_chain = " then ".join(verb_phrases)
        
        if is_patrol:
            # "Continually visit X then Y..."
            if len(g_complex) > 1:
                nl_segs.append(f"Continually visit {verb_chain}")
            else:
                nl_segs.append(f"Continually visit {verb_phrases[0]}")
        else:
            # "First X then Y..."
            nl_segs.append(f"First {verb_chain}")

    # 3. Basic Template Segment
    if g_basic:
        basic_ltl, basic_nl_fn, basic_msk = _get_basic_template_fragment(g_basic, props, actions_cfg)
        ltl_segs.append(basic_ltl)
        msk_segs.append(basic_msk)
        
        basic_phrases = [_verb_phrase(props[k], actions_cfg) for k in g_basic]
        try:
            nl_segs.append(basic_nl_fn(basic_phrases))
        except IndexError:
             nl_segs.append("Basic task: " + " and ".join(basic_phrases))

    # 4. Safety Constraint Segment
    if g_safety:
        safety_ltl = _fragment_avoidance([_pure_atom(props[k]) for k in g_safety])
        safety_msk = _fragment_avoidance(g_safety)
        
        ltl_segs.append(safety_ltl)
        msk_segs.append(safety_msk)
        
        safety_phrases = [_verb_phrase(props[k], actions_cfg) for k in g_safety]
        nl_segs.append("meanwhile, always avoid " + " and ".join(safety_phrases))

    # 5. Assembly
    # Filter out empty strings if any fragment returned empty
    ltl_segs = [s for s in ltl_segs if s]
    msk_segs = [s for s in msk_segs if s]
    nl_segs = [s for s in nl_segs if s]

    final_tl = " and ".join([f"( {s} )" for s in ltl_segs])
    final_masked = " and ".join([f"( {s} )" for s in msk_segs])
    
    raw_nl = ". ".join(nl_segs)
    if not raw_nl.endswith("."):
        raw_nl += "."
    
    final_sentence = correct_sentence(raw_nl)

    return {
        "id": idx,
        "tl": final_tl,
        "masked_tl": final_masked,
        "sentence": final_sentence,
        "grounded_sentence": final_sentence,
        "task_type": "hybrid_mission",
        "num_props": n
    }

# Blueprint registry: (function, min_props, weight)
BLUEPRINTS = [
    (_blueprint_simple_list, 2, 20),
    (_blueprint_safety_liveness, 2, 20),
    (_blueprint_precedence_chain, 2, 20),
    (_blueprint_trigger_response, 2, 15),
    (_blueprint_alternative_plans, 2, 15),
    (_blueprint_stability, 2, 10),
    (_blueprint_hybrid_fusion, 3, 50),
]


def build_entry_blueprint(idx: int, props: Dict, actions_cfg: Dict) -> Dict:
    """
    Blueprint-based mission composition (FIXED VERSION).

    Key Fix: Uses _pure_atom() and _verb_phrase() instead of pre-generated LTL strings.
    This prevents invalid formulas like "not finally ..." or "until (finally ...)".

    Architecture:
    1. Extract pure atoms (action(args)) for LTL composition
    2. Extract verb phrases for NL composition
    3. Apply temporal operators (finally, globally, until, implies) at blueprint level
    """
    prop_keys = list(props.keys())
    n = len(prop_keys)

    # Select blueprint
    available = [bp for bp in BLUEPRINTS if n >= bp[1]]
    total_weight = sum(bp[2] for bp in available)
    weights = [bp[2] / total_weight for bp in available]
    blueprint_fn = random.choices([bp[0] for bp in available], weights=weights)[0]

    return blueprint_fn(idx, props, prop_keys, actions_cfg)


# =========================================================================
# Meta Composition Templates for Mission Composition
# =========================================================================
# These templates define how to combine two sub-tasks with different logical operators
# Each returns: (ltl_combined, nl_combined)
# =========================================================================

META_COMPOSITION_TEMPLATES = [
    # AND: Both A and B must hold (existing behavior)
    lambda a, b: (
        f"( {a['tl']} ) and ( {b['tl']} )",
        f"{a['sentence']}. Also, {b['sentence']}",
    ),
    # OR: Either A or B must hold
    lambda a, b: (
        f"( {a['tl']} ) or ( {b['tl']} )",
        f"Either {a['sentence']}, or {b['sentence']}",
    ),
    # IMPLIES: If A holds, then B must hold
    lambda a, b: (
        f"( {a['tl']} ) implies ( {b['tl']} )",
        f"If {a['sentence']}, then {b['sentence']} must also hold.",
    ),
    # NOT A UNTIL B: B cannot happen before A completes
    lambda a, b: (
        f"( not {b['tl']} ) until ( {a['tl']} )",
        f"{b['sentence']} must not happen until {a['sentence']} has occurred.",
    ),
    # SEQUENCE: A must complete before B starts (using Until)
    lambda a, b: (
        f"( {a['tl']} ) until ( {b['tl']} )",
        f"First {a['sentence']}, then {b['sentence']}.",
    ),
]


def build_entry_for_props(
    idx: int,
    props: Dict,  # prop_k  -> info
    actions_cfg: Dict,
    depth: int = 0,  # recursion depth to prevent infinite loops
) -> Dict:
    # 递归方式组合子任务，保持原有算法
    MAX_DEPTH = 3
    if depth < MAX_DEPTH and len(props) >= 2 and random.random() < 0.7:
        prop_keys = list(props.keys())
        random.shuffle(prop_keys)
        cut = random.randint(1, len(prop_keys) - 1)
        keys_a = prop_keys[:cut]
        keys_b = prop_keys[cut:]

        props_a = {k: props[k] for k in keys_a}
        props_b = {k: props[k] for k in keys_b}

        entry_a = build_entry_for_props(idx, props_a, actions_cfg, depth=depth + 1)
        entry_b = build_entry_for_props(idx, props_b, actions_cfg, depth=depth + 1)

        props_in_a = len(props_a)
        props_in_b = len(props_b)

        offset = props_in_a
        remap_dict = {
            f"prop_{i}": f"prop_{offset + i}" for i in range(1, props_in_b + 1)
        }

        entry_b_remapped = {
            "tl": safe_renumber(entry_b["tl"], remap_dict),
            "masked_tl": safe_renumber(entry_b["masked_tl"], remap_dict),
            "grounded_sentence": safe_renumber(
                entry_b["grounded_sentence"], remap_dict
            ),
            "sentence": entry_b["sentence"],
        }

        composer = random.choice(META_COMPOSITION_TEMPLATES)
        tl_parts, nl_parts = composer(entry_a, entry_b_remapped)

        final_tl = tl_parts
        final_masked_tl = (
            f"( {entry_a['masked_tl']} ) or ( {entry_b_remapped['masked_tl']} )"
        )
        final_sentence = nl_parts

        grounded_a = entry_a["grounded_sentence"]
        grounded_b = entry_b_remapped["grounded_sentence"]
        if grounded_a.endswith("."):
            grounded_a = grounded_a[:-1]
        final_grounded_sentence = f"{grounded_a}. Also, {grounded_b}"

        all_text = final_masked_tl + " " + final_grounded_sentence
        prop_matches = re.findall(r"prop_\d+", all_text)
        unique_props_ordered = []
        seen = set()
        for p in prop_matches:
            if p not in seen:
                unique_props_ordered.append(p)
                seen.add(p)

        prop_renumber = {
            old: f"prop_{i + 1}" for i, old in enumerate(unique_props_ordered)
        }

        final_masked_tl = safe_renumber(final_masked_tl, prop_renumber)
        final_grounded_sentence = safe_renumber(final_grounded_sentence, prop_renumber)
        final_tl = safe_renumber(final_tl, prop_renumber)

        return {
            "id": idx,
            "sentence": final_sentence,
            "tl": final_tl,
            "masked_tl": final_masked_tl,
            "grounded_sentence": final_grounded_sentence,
        }

    # 选择合适的模板骨架
    arity = len(props)
    viable = [tpl for tpl in LTL_TEMPLATES_STATE if tpl[1] <= arity]
    viable.sort(key=lambda x: x[1], reverse=True)
    key, need, skeleton = random.choice(viable[: max(1, len(viable) // 2)])
    labels_used = list(props.keys())[:need]

    # 生成 LTL（带原子与 masked）
    g_ltl = skeleton([_atom(props[lbl]) for lbl in labels_used])
    m_ltl = skeleton(labels_used)

    # 生成自然语言句子并进行动名词修正
    segs_ground = [_nl_segment(props[l], actions_cfg) for l in labels_used]
    tpl = pick_sentence_template(key)
    ego = random.choice(EGO_REFS)
    g_raw = correct_sentence(add_ego_reference(tpl(segs_ground), ego))
    g_tok = g_raw.split()

    for info in props.values():
        bare = info["action_ref"].split()[0]
        ger = _to_gerund(bare)
        if bare not in g_raw and ger in g_raw:
            info["action_ref"] = ger
    segs_ground = [_nl_segment(props[l], actions_cfg) for l in labels_used]

    def _segment_tokens(lbl: str) -> List[str]:
        seg_txt = _nl_segment(props[lbl], actions_cfg).lower()
        return seg_txt.replace(".", "").replace(",", "").split()

    seg_tokens = {lbl: _segment_tokens(lbl) for lbl in labels_used}

    pid_map = {lbl: i + 1 for i, lbl in enumerate(labels_used)}  # lbl → 1‑based id
    mapping = {lbl: f"prop_{pid_map[lbl]}" for lbl in labels_used}

    masked = []
    i = 0
    while i < len(g_tok):
        matched = False
        for lbl in labels_used:
            tok_seq = seg_tokens[lbl]
            L = len(tok_seq)
            if i + L <= len(g_tok) and all(
                _same_word(g_tok[i + j], tok_seq[j]) for j in range(L)
            ):
                masked.append(mapping[lbl])  # single token in masked sent.
                i += L
                matched = True
                break
        if not matched:  # normal word
            masked.append(g_tok[i])
            i += 1

    masked_tl = [mapping.get(t, t) for t in m_ltl]

    return {
        "id": idx,
        "sentence": " ".join(str(x) for x in g_tok),
        "tl": " ".join(str(x) for x in g_ltl),
        "masked_tl": " ".join(str(x) for x in masked_tl),
        "grounded_sentence": " ".join(str(x) for x in masked),
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
# 6)  Build the whole dataset                                                #
# ---------------------------------------------------------------------------


def build_dataset_entries(
    object_dict: Dict,
    actions_dict: Dict,
    locations: List[str],
    actions_cfg: Dict,
    num_entries: int,
    max_props: int = 8,
) -> List[Dict]:
    """
    Create *exactly* ``num_entries`` dataset entries with up to ``max_props`` propositions.
    """
    dataset, label_pool = [], [f"prop_{i + 1}" for i in range(max_props)]

    while len(dataset) < num_entries:
        # ------------- (a)  sample propositions ---------------------------
        # Sample between 1 and max_props (bias towards higher values for more complex LTL)
        want_labels = random.sample(
            label_pool, k=random.randint(max_props - 2, max_props)
        )
        props = {}

        for lbl in want_labels:
            # Generate a unique proposition directly
            verb = random.choice(list(actions_dict.keys()))
            a_canon, a_ref = [], []
            for kind in actions_cfg[verb]["params"]:
                c, r = _sample_argument(kind, object_dict, locations)
                a_canon.append(c)
                a_ref.append(r)

            props[lbl] = {
                "action_canon": verb,
                "action_ref": random.choice(actions_dict[verb]),
                "args_canon": a_canon,
                "args_ref": a_ref,
            }

        # ------------- (b)  build entry (blueprint-based, non-recursive) ---------
        tmp_idx = len(dataset)  # provisional id
        entry = build_entry_blueprint(tmp_idx, props, actions_cfg)
        dataset.append(entry)

    # make sure the ids are sequential 0…N‑1
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
        "-m",
        "--max_props",
        type=int,
        default=8,
        help="Maximum number of propositions per entry",
    )
    p.add_argument(
        "-o",
        "--output",
        default="new_generated_datasets/LTL/warehouse.jsonl",
        help="Output JSONL file",
    )
    args = p.parse_args()

    cfg, obj_dict, act_dict, locs, act_cfg = load_scenario(args.scenario)
    entries = build_dataset_entries(
        obj_dict,
        act_dict,
        locs,
        act_cfg,
        num_entries=args.num_entries,
        max_props=args.max_props,
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

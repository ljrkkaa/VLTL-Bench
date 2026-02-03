#!/usr/bin/env python3
"""Minimal VERL reward function (verif_acc + similarity + exact match).

Public API:
    compute_score(data_source, solution_str, ground_truth, extra_info=None)
    compute_with_details(prediction, ground_truth, good_trace=None, bad_trace=None)

Behavior:
    reward = (verif_acc + sim + exact_match) / 3 when parse succeeds
    reward = 0 when parse fails (bad_parse = 1)

Supported LTL operators for finite-trace evaluation: G, F, X, U, and, or, not.
Any parsed node outside this subset triggers bad_parse=1 and reward=0.

Finite-trace semantics:
    - Atomic propositions: true iff atom string present in state set
    - X φ: evaluated at min(t+1, len(trace)-1) (stutter-at-end)
    - F φ: true if φ holds at some k in [t, end]
    - G φ: true if φ holds at all k in [t, end]
    - φ U ψ: true if ∃k ∈ [t, end] with ψ(k) and φ(j) for all j ∈ [t, k)
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
from pyModelChecking.LTL import Parser, AtomicProposition as AP, Not, And, Or, Imply, X, F, G, U



# ---------------------------------------------------------------------------
# Cleaning utilities
# ---------------------------------------------------------------------------

_NUMBERISH = re.compile(r"^\d+\.?$")
_AP_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_PREFIXES = ("LTL:", "3. *FINAL:* ", "*FINAL:* ", "FINAL: ")
_SUFFIXES = ("*FINISH*", "FINISH", "*END*", "END")


def _clean_prediction(text: str) -> str:
    parts = text.strip().split(" ", 1)
    if len(parts) == 2 and _NUMBERISH.match(parts[0].rstrip(".")):
        text = parts[1]

    for prefix in _PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]

    for suffix in _SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]

    return text.strip()


# ---------------------------------------------------------------------------
# Token normalization
# ---------------------------------------------------------------------------

TOKEN_MAP = {
    "globally": "G",
    "always": "G",
    "[]": "G",
    "finally": "F",
    "eventually": "F",
    "<>": "F",
    "next": "X",
    "until": "U",
    "not": "not",
    "¬": "not",
    "!": "not",
    "&": "and",
    "∧": "and",
    "and": "and",
    "|": "or",
    "∨": "or",
    "or": "or",
    "imply": "-->",
    "implies": "-->",
    "->": "-->",
    "⇒": "-->",
    "-->": "-->",
    "double_implies": "-->",
}


def _tokenise(text: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    multi = ["-->", "->", "[]", "<>"]
    specials = {"(", ")"}
    length = len(text)
    while i < length:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        matched = False
        for pattern in multi:
            if text.startswith(pattern, i):
                tokens.append(pattern)
                i += len(pattern)
                matched = True
                break
        if matched:
            continue
        if ch in specials:
            tokens.append(ch)
            i += 1
            continue
        if ch.isalnum() or ch == "_":
            start = i
            while i < length and (text[i].isalnum() or text[i] == "_"):
                i += 1
            tokens.append(text[start:i])
            continue
        tokens.append(ch)
        i += 1
    return tokens


def _normalize_tokens(tokens: List[str]) -> List[str]:
    normalized: List[str] = []
    for token in tokens:
        low = token.lower()
        if low in TOKEN_MAP:
            normalized.append(TOKEN_MAP[low])
        elif token in ("(", ")"):
            normalized.append(token)
        elif _AP_OK.match(token):
            normalized.append(token)
        else:
            normalized.append(f"'{token}'")
    return normalized


def _elim_impl_tokens(tokens: List[str]) -> List[str]:
    while True:
        depth = [0] * len(tokens)
        d = 0
        for i, tok in enumerate(tokens):
            if tok == "(":
                d += 1
            depth[i] = d
            if tok == ")":
                d -= 1

        for idx, tok in enumerate(tokens):
            if tok not in ("->", "-->"):
                continue
            di = depth[idx]
            j = idx - 1
            while j >= 0 and depth[j] >= di:
                j -= 1
            lhs_start = j + 1
            k = idx + 1
            while k < len(tokens) and depth[k] >= di:
                k += 1
            rhs_end = k

            lhs = tokens[lhs_start:idx]
            rhs = tokens[idx + 1 : rhs_end]
            replacement = ["(", "(", "not", *lhs, ")", "or", "(", *rhs, ")", ")"]
            tokens = tokens[:lhs_start] + replacement + tokens[rhs_end:]
            break
        else:
            return tokens


# ---------------------------------------------------------------------------
# Parser and evaluator helpers
# ---------------------------------------------------------------------------

_PARSER: Optional[Parser] = None


def _get_parser() -> Parser:
    global _PARSER
    if _PARSER is None:
        _PARSER = Parser()
    return _PARSER


@lru_cache(maxsize=8192)
def _parse_formula(text: str):
    parser = _get_parser()
    return parser(text)


def _normalize_formula_string(prediction: str) -> str:
    tokens = _tokenise(prediction)
    normalized = _normalize_tokens(tokens)
    eliminated = _elim_impl_tokens(normalized)
    return " ".join(eliminated)


def _validate_formula(
    formula_str: str, good_trace: List[Set[str]], bad_trace: List[Set[str]]
):
    ast = _parse_formula(formula_str)
    good_sat = _eval_ast(ast, good_trace)
    bad_sat = _eval_ast(ast, bad_trace)
    return good_sat, bad_sat


def _eval_ast(ast, trace: List[Set[str]], t: int = 0) -> bool:
    if not trace:
        return False

    if isinstance(ast, AtomicProposition):
        return str(ast) in trace[t]
    if isinstance(ast, Not):
        return not _eval_ast(ast.subformula(0), trace, t)
    if isinstance(ast, And):
        return _eval_ast(ast.subformula(0), trace, t) and _eval_ast(
            ast.subformula(1), trace, t
        )
    if isinstance(ast, Or):
        return _eval_ast(ast.subformula(0), trace, t) or _eval_ast(
            ast.subformula(1), trace, t
        )
    if isinstance(ast, Imply):
        return (not _eval_ast(ast.subformula(0), trace, t)) or _eval_ast(
            ast.subformula(1), trace, t
        )
    if isinstance(ast, X):
        nxt = min(t + 1, len(trace) - 1)
        return _eval_ast(ast.subformula(0), trace, nxt)
    if isinstance(ast, F):
        return any(_eval_ast(ast.subformula(0), trace, k) for k in range(t, len(trace)))
    if isinstance(ast, G):
        return all(_eval_ast(ast.subformula(0), trace, k) for k in range(t, len(trace)))
    if isinstance(ast, U):
        phi = ast.subformula(0)
        psi = ast.subformula(1)
        for k in range(t, len(trace)):
            if _eval_ast(psi, trace, k):
                return all(_eval_ast(phi, trace, j) for j in range(t, k))
        return False
    raise ValueError(f"Unsupported operator: {type(ast).__name__}")


def _normalize_trace(raw: Optional[Sequence[Iterable[str]]]) -> List[Set[str]]:
    if not raw:
        return []
    return [set(step) for step in raw]


# ---------------------------------------------------------------------------
# String metrics
# ---------------------------------------------------------------------------


def _best_substring_similarity(prediction: str, target: str) -> float:
    prediction = prediction or ""
    target = target or ""
    sm = SequenceMatcher
    t_len = len(target)
    p_len = len(prediction)
    if p_len < t_len:
        return sm(None, prediction, target).ratio()
    best = 0.0
    for i in range(p_len - t_len + 1):
        sub = prediction[i : i + t_len]
        best = max(best, sm(None, sub, target).ratio())
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_with_details(
    prediction: str,
    ground_truth: str,
    good_trace: Optional[Sequence[Iterable[str]]] = None,
    bad_trace: Optional[Sequence[Iterable[str]]] = None,
) -> Dict[str, Any]:
    clean_pred = _clean_prediction(prediction)
    gt_clean = _clean_prediction(ground_truth)
    details: Dict[str, Any] = {
        "reward": 0.0,
        "verif_acc": 0.0,
        "sim": 0.0,
        "exact_match": 0.0,
        "bad_parse": 0,
        "good_sat": False,
        "bad_sat": False,
        "clean_prediction": clean_pred,
        "formula_str": "",
        "error": None,
    }

    sim = _best_substring_similarity(clean_pred, gt_clean)
    exact = 1.0 if clean_pred == gt_clean else 0.0
    details["sim"] = sim
    details["exact_match"] = exact

    good_labels = _normalize_trace(good_trace)
    bad_labels = _normalize_trace(bad_trace)

    if not good_labels or not bad_labels:
        verif_acc = 0.0
        details["verif_acc"] = verif_acc
        details["reward"] = (verif_acc + sim + exact) / 3
        return details

    try:
        formula_str = _normalize_formula_string(clean_pred)
        details["formula_str"] = formula_str
        good_sat, bad_sat = _validate_formula(formula_str, good_labels, bad_labels)
        details["good_sat"] = bool(good_sat)
        details["bad_sat"] = bool(bad_sat)
        verif_acc = 0.5 * ((1.0 if good_sat else 0.0) + (0.0 if bad_sat else 1.0))
        details["verif_acc"] = verif_acc
        details["reward"] = (verif_acc + sim + exact) / 3
    except Exception as exc:  # pylint:disable=broad-except
        details["bad_parse"] = 1
        details["error"] = str(exc)
        details["reward"] = 0.0

    return details


def compute_score(
    data_source: str,  # unused but kept for VERL compatibility
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    good_trace = (extra_info or {}).get("good_trace", [])
    bad_trace = (extra_info or {}).get("bad_trace", [])
    details = compute_with_details(solution_str, ground_truth, good_trace, bad_trace)
    return details["reward"]


__all__ = [
    "compute_score",
    "compute_with_details",
]

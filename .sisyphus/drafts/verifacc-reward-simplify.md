# Draft: Simplify verl_reward_function.py (VerifAcc reward)

## Requirements (confirmed)
- User asked: introduce `verl_reward_function.py` code, then analyze where it can be simplified; prefer the simplest possible logic.

## Requirements (newly confirmed)
- Keep only: VERL interface + metrics computed in `NL2TL_eval.py`.
- Remove: extra configs/surfaces not needed; remove `__main__` demo.
- Reward should combine multiple metrics as a weighted sum.

## Interface decisions (confirmed)
- Atomic proposition naming: action names used consistently (e.g., `idle`, `get_help`); no `prop_dict` reverse-mapping needed.
- Public API: keep `compute_score()` plus a `compute_with_details()` for logging/debug.

## Weighting decision (partially confirmed)
- Metrics to include: `verif_acc`, `best_substring_similarity`, `exact_match`, `bad_parse` (parse_penalty).
- Weights: all equal (pending exact formula because penalty needs sign + possible clamping).
- Parse-failure strategy: A) semantic terms become 0; still compute similarity/exact_match.

## Weighting decision (confirmed)
- User chose option 3 for parse handling/penalty:
  - Base reward is average of available positive terms: `(verif_acc + sim + exact_match) / 3`.
  - If parse fails (bad_parse=1), set semantic terms to 0 (per A), and then **set reward to 0** (no negatives).

## Parse-failure policy (confirmed)
- User confirmed 3a: on parse failure, `reward=0` (do NOT fall back to sim/exact_match).

## What seems required (from local tests)
- `test_verl_reward_function.py` imports/uses these symbols from `verl_reward_function`:
  - Public: `VerifAccRewardFunction`, `VerifAccRewardConfig`, `VerificationResult`, `RewardLevel`
  - Public functions: `compute_score`, `compute_verif_acc_reward`
  - Config factories: `get_strict_verif_acc_config`, `get_lenient_verif_acc_config`, `get_asymmetric_verif_acc_config`, `get_sparse_verif_acc_config`
  - Helper functions: `_preprocess_prediction`, `_tokenise`, `_normalise_tokens`
- Tests assert behavior for:
  - strict parse error → returns `reward_parse_error` (e.g. -0.5)
  - non-strict parse error → fallback to string similarity (0..1)
  - missing/empty traces → fallback to string similarity (0..1)
  - batch APIs return numpy arrays, plus batch-with-details returns list[dict]

## Tooling constraints observed
- `ast-grep` tool invocation failed due to GLIBC version mismatch in this environment; rely on grep + manual inspection (or other available tooling) instead.

## Technical Decisions
- (Pending) Keep vs remove: semantic validation (pyModelChecking), string-similarity fallback, batch APIs, detailed-result APIs, demo/test main block.

## Research Findings
- `test_integration.py` imports `prepare_verl_batch_data` from `nl2tl_reward_function` (not this file), but our `verl_reward_function.py` also defines a `prepare_verl_batch_data` + `extract_extra_info_from_parquet_row` API shape that may be expected elsewhere.
- (Pending) Need broader repo scan for real runtime imports beyond tests.

## Research Findings (pyModelChecking)
- pyModelChecking provides LTL parsing + model checking over a Kripke structure, but we did **not** find a built-in function to directly evaluate an LTL formula on an explicit finite trace.
- Their semantics are aligned with infinite-path model checking; representing a finite trace typically requires stutter/self-loop completion on the last state to satisfy total-transition requirements, which changes semantics vs LTLf.
- Conclusion for simplification: if we keep finite-trace semantics like current `_eval` (with `X` clamped at end), we should keep a small custom evaluator; switching to `modelcheck` would be heavier and introduces semantic differences.

## NL2TL_eval.py metrics to align with
- Translation metrics in notebook:
  - exact match accuracy (binary)
  - best-substring similarity (SequenceMatcher best window)
- Verification metrics in notebook cell:
  - ok_good: good_trace satisfies
  - ok_bad: bad_trace does NOT satisfy
  - ok_both: good_sat and not bad_sat
  - bad_parse count
- It also maps trace labels via `prop_dict` (action strings like `idle()` → prop_id), which our current `verl_reward_function.py` does **not** implement.

## Open Questions
- What is the *minimum* required external interface in this repo (only `compute_score`? also batch/details?)
- Is pyModelChecking guaranteed available in your runtime, or must we support “no parser installed” fallback?
- Is string-similarity fallback acceptable, or should parse errors be strict (e.g., 0 or negative) and never fallback?
- Weighted sum: which metrics exactly, and what weights? (need to decide default + scaling)
- When traces exist but parse fails: do we still include similarity term or force 0 for semantic terms?
- Do we need the `prop_dict` reverse-mapping (action_canon/args_canon → prop_id) like `NL2TL_eval.py` to make traces compatible with formulas?

## Scope Boundaries
- INCLUDE: refactor/simplify logic while preserving required behavior.
- EXCLUDE (for now): changing the VerifAcc definition itself.

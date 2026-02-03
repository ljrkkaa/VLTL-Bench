# Simplify `verl_reward_function.py` to minimal VERL reward + NL2TL metrics

## TL;DR

> **Quick Summary**: Replace the current config-heavy `verl_reward_function.py` with a minimal module that computes (1) semantic VerifAcc on good/bad traces (using pyModelChecking + a small finite-trace evaluator) and (2) NL2TL_eval-style string metrics (exact match + best-substring similarity), then returns a single scalar reward.
>
> **Deliverables**:
> - Minimal `compute_score()` VERL interface
> - Minimal `compute_with_details()` for logging
> - Remove configs/classes/batch/demo; keep only private helpers needed
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES (2 waves)
> **Critical Path**: Usage-scan & contract → implement minimal metrics → validate via executable smoke checks

---

## Context

### Original Request
- “先介绍 verl_reward_function.py 然后一起分析需要简化的地方，逻辑越简单越好。”
- Later clarified:
  - “只保留 VERL 接口和 NL2TL_eval.py 中计算的指标，其他配置删掉，main 不需要。”
  - Reward should combine metrics; final decisions:
    - parse success: `reward = (verif_acc + sim + exact_match) / 3`
    - parse failure: `reward = 0` (no fallback to sim/exact_match)
  - Runtime: pyModelChecking is installed
  - No tests requested (we will provide agent-executable verification commands instead)
  - Atomic propositions are action names consistently; do **not** implement prop_dict reverse mapping.

### Interview Summary (decisions)
- Keep only public APIs:
  - `compute_score(data_source, solution_str, ground_truth, extra_info=None, ...) -> float`
  - `compute_with_details(prediction, ground_truth, good_trace, bad_trace, ...) -> dict`
- Remove public surface area: config dataclass/factories, class wrappers, batch APIs, enums/dataclasses, `__main__` demo.
- Metrics:
  - `verif_acc = 0.5 * (1[good_sat] + 1[not bad_sat])`
  - `sim` = best-substring similarity (SequenceMatcher best window) as in `NL2TL_eval.py`
  - `exact_match` = 1 if cleaned prediction equals ground_truth else 0
  - `bad_parse` = 1 if parse fails

### Research Findings
- `pyModelChecking` does not provide a simple “evaluate formula on a finite trace” helper; it focuses on Kripke model checking with infinite-path semantics.
  - Therefore, keeping a small custom finite-trace evaluator (like current `_eval`) is simplest and avoids semantic drift.
- Repo usage scan found tests importing many symbols from `verl_reward_function.py`, but no other production imports were found. **Still, we must do a repo-wide scan before removing names**.

### Metis Review (gaps addressed)
- Biggest risk is hidden downstream imports; plan includes a mandatory usage scan and either updating call sites or providing compatibility stubs.
- Finite-trace semantics must be explicit; plan includes documented semantics and smoke checks.
- Tokenization/operator support can cause silent semantic drift; plan includes scoping to operators actually present and failing deterministically otherwise.

---

## Work Objectives

### Core Objective
Make `verl_reward_function.py` minimal and deterministic: compute a scalar reward from (verif_acc, sim, exact_match) with strict parse-failure behavior, while keeping only the required VERL interface plus a details helper.

### Concrete Deliverables
- `verl_reward_function.py` exposes exactly:
  - `compute_score(...) -> float`
  - `compute_with_details(...) -> Dict[str, Any]`
- Module-private helpers for:
  - prediction cleaning
  - best-substring similarity
  - parsing LTL (pyModelChecking Parser)
  - finite-trace evaluation on good/bad traces

### Definition of Done
- `python -c "import verl_reward_function as m; print(m.compute_score)"` works.
- `compute_score()` returns:
  - `0.0` on parse failure
  - otherwise `(verif_acc + sim + exact_match) / 3` with each term in `[0,1]`
- `compute_with_details()` returns a dict including at least:
  - `reward, verif_acc, sim, exact_match, bad_parse, good_sat, bad_sat`

### Must NOT Have (Guardrails)
- No fallback reward when parsing fails (reward must be 0).
- No config classes/factories; no batch APIs; no `__main__` demo.
- No prop_dict reverse mapping.
- No Kripke/modelcheck encoding.

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest is present in repo)
- **User wants tests**: NO (manual / agent-executable smoke checks only)

### Agent-executable smoke checks

> Executor should run these commands to validate correctness (no manual steps).

1) Import + API presence
```bash
python - <<'PY'
import verl_reward_function as m
print(hasattr(m,'compute_score'), hasattr(m,'compute_with_details'))
PY
# Expect: True True
```

2) Similarity + exact match basics
```bash
python - <<'PY'
from verl_reward_function import compute_with_details
d = compute_with_details(
    prediction='globally A',
    ground_truth='globally A',
    good_trace=[{'A'}],
    bad_trace=[set()],
)
print(d['exact_match'], d['sim'])
PY
# Expect: exact_match == 1, sim close to 1
```

3) Parse failure ⇒ reward=0
```bash
python - <<'PY'
from verl_reward_function import compute_with_details
d = compute_with_details(
    prediction='invalid formula (',
    ground_truth='globally A',
    good_trace=[{'A'}],
    bad_trace=[set()],
)
print(d['bad_parse'], d['reward'])
PY
# Expect: bad_parse == 1, reward == 0.0
```

4) VerifAcc sanity: correct case should be high
```bash
python - <<'PY'
from verl_reward_function import compute_with_details
d = compute_with_details(
    prediction='G ( idle -> F get_help )',
    ground_truth='G ( idle -> F get_help )',
    good_trace=[{'idle'},{'get_help'}],
    bad_trace=[{'idle'},set()],
)
print(d['good_sat'], d['bad_sat'], d['verif_acc'], d['reward'])
PY
# Expect: good_sat True, bad_sat False, verif_acc 1.0, reward in (0,1]
```

---

## Execution Strategy

### Parallel Execution Waves

Wave 1:
- Task 1: Repo usage scan + contract extraction (what must remain public)
- Task 2: Define minimal metric contract + details schema

Wave 2 (after Wave 1):
- Task 3: Implement minimal module rewrite (delete/replace old APIs)
- Task 4: Run smoke checks and reconcile semantic edge cases

Critical Path: Task 1 → Task 3 → Task 4

---

## TODOs

### 1) Repo-wide usage scan & interface contract (must-do before deletions)

**What to do**:
- Search entire repo for any imports/usages of `verl_reward_function` and its exported symbols.
- Decide whether to:
  - update those call sites to new API, OR
  - keep tiny compatibility stubs that raise a clear error pointing to the new API.

**Must NOT do**:
- Do not assume tests are the only consumer without scanning.

**Recommended Agent Profile**:
- Category: `quick`
- Skills: none

**Parallelization**:
- Can Run In Parallel: YES (Wave 1)

**References**:
- `test_verl_reward_function.py` (current imports illustrate old public surface)

**Acceptance Criteria (agent-executable)**:
- `grep -R "from verl_reward_function" -n .` enumerates all import sites.
- A written list of which names can be removed safely.
- If `test_verl_reward_function.py` imports removed symbols, plan either (a) update that test to new API or (b) add explicit stubs that raise with guidance; choose and document which.

---

### 2) Specify minimal details schema + cleaning rules (lock the contract)

**What to do**:
- Define a stable dict schema for `compute_with_details()` including keys:
  - `reward: float`
  - `verif_acc: float`
  - `sim: float`
  - `exact_match: float|int`
  - `bad_parse: int`
  - `good_sat: bool`
  - `bad_sat: bool`
  - plus `clean_prediction` and `formula_str` optionally for debugging.
- Define `clean_prediction(s)` rule, minimally mirroring current `_preprocess_prediction` and NL2TL_eval logic:
  - strip numbering prefix like `"12."`
  - strip known prefixes (`LTL:` etc.) if still relevant
  - strip known suffixes (`FINISH` etc.)
  - `.strip()`

**Must NOT do**:
- Do not add configuration switches.

**Recommended Agent Profile**:
- Category: `quick`
- Skills: none

**Parallelization**:
- Can Run In Parallel: YES (Wave 1)

**References**:
- `NL2TL_eval.py:best_substring_similarity()` for similarity definition.
- `verl_reward_function.py:_preprocess_prediction()` for cleaning behavior.

**Acceptance Criteria**:
- Details schema written into module docstring or comment.
- Token/normalization pipeline documented in module (or docstring) including:
  - Token map: globally/always/[]→G; finally/eventually/<>→F; next→X; until→U; not/¬/!→not; &/∧→and; |/∨/or→or; imply/implies/->/⇒/-->→imply token (or desugared); double_implies→imply.
  - Implication elimination rule (A -> B → (not A) or B) applied after normalization.
  - Cleaning steps: numbering prefix removal, known prefixes/suffixes, strip.

---

### 3) Rewrite `verl_reward_function.py` to minimal implementation

**What to do**:
- Delete/replace current module content with:
  - imports: `re`, `difflib.SequenceMatcher`, `typing`, and pyModelChecking LTL Parser + AST classes.
  - `clean_prediction()`
  - `best_substring_similarity()` (copy NL2TL_eval algorithm)
- `parse_ltl()` (cached)
- `eval_ltl_on_finite_trace()` implementing finite semantics for the operator subset required by data.
  - `compute_with_details(prediction, ground_truth, good_trace, bad_trace, ...)`
  - `compute_score(data_source, solution_str, ground_truth, extra_info=None, ...)`:
    - extract `good_trace`, `bad_trace`
    - call `compute_with_details` and return `reward`

**Finite-trace semantics (explicit)**:
- Atoms: true iff atom string in `trace[t]`
- `not/and/or/imply`: boolean
- `X φ`: evaluate at `min(t+1, end)` (stutter-at-end), matching existing `verl_reward_function.py` and the evaluation cell in `NL2TL_eval.py`.
- `F φ`: exists k in [t..end] s.t. φ holds
- `G φ`: for all k in [t..end], φ holds
- `φ U ψ`: exists k in [t..end] with ψ at k and φ for all j in [t..k)
- Unsupported operators (e.g., R, W): either implement finite semantics or treat as parse/eval failure → `bad_parse=1` → reward=0; document the choice.

**Missing/empty traces policy (explicit)**:
- If `good_trace` or `bad_trace` is missing/empty, set `good_sat=False`, `bad_sat=False`, `verif_acc=0.0` and still compute `sim` and `exact_match`.
- This is **not** treated as parse failure (i.e., does not force reward=0) unless the formula parse itself fails.

**Must NOT do**:
- No fallback scoring if parse fails: set `bad_parse=1` and `reward=0`.
- No prop_dict mapping.
- No additional public APIs beyond the two functions.

**Recommended Agent Profile**:
- Category: `unspecified-high`
- Skills: none

**Parallelization**:
- Can Run In Parallel: NO (Wave 2, depends on tasks 1–2)

**References**:
- `verl_reward_function.py` (current `_eval` semantics and cleaning list)
- `NL2TL_eval.py` (best-substring similarity implementation)

**Acceptance Criteria (agent-executable)**:
- Run the smoke checks in “Verification Strategy” section; all pass.
- Module docstring lists supported operators and states unsupported ones trigger reward=0.

---

### 4) Validate operator subset + fail deterministically for unsupported operators

**What to do**:
- Scan dataset predictions / ground_truth examples (if available in repo) to identify which operators appear.
- Suggested search paths: `translation_eval/**/*.jsonl`, `VLTL-Bench/test/*.jsonl`, and any `verl_data` parquet/json if present.
- If an operator appears that we do not support (e.g., `R`), choose one:
  - implement it correctly for finite traces, OR
  - explicitly treat it as parse/eval failure and return reward=0.

**Must NOT do**:
- Do not silently mis-evaluate unsupported operators.

**Recommended Agent Profile**:
- Category: `quick`
- Skills: none

**Parallelization**:
- Can Run In Parallel: YES (Wave 2 alongside implementation if done early)

**Acceptance Criteria**:
- A short list of supported operators documented in the module.
- A captured list of operators actually observed in dataset scan (even if empty).

---

## Commit Strategy

- Single commit recommended after smoke checks pass:
  - Message: `refactor(reward): simplify verl reward to verif_acc+sim+exact_match`
  - Verification: run the smoke check commands.

---

## Success Criteria

### Final Checklist
- [ ] `compute_score()` and `compute_with_details()` exist and work
- [ ] Reward formula matches: parse success → `(verif_acc+sim+exact_match)/3`; parse fail → `0`
- [ ] No config/class/batch/demo code remains
- [ ] Operator semantics explicitly documented

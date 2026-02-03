## 2026-01-30 Initialization
- Finite-trace evaluator must use stutter-at-end semantics for X
- Reward schema: (verif_acc + sim + exact_match)/3 on parse success; reward=0 on parse failure
- Smoke checks 1-4 executed successfully on 2026-01-30
## 2026-01-30 Operator coverage scan
- Ad-hoc Python scan over 258 JSONL files across translation_eval, VLTL-Bench, lifting_eval, grounding_eval, priorwork counted operator usage after `_normalize_formula_string` transformations: F (169,663), not (110,550), G (105,515), and (97,181), or (90,821), X (35,221), U (15,124).
- Non-operator tokens flagged by tokenizer as `","`, `"'-'"`, and `"""` come from hyphen/apostrophe characters; these correspond to punctuation we already downgrade to parse failure (bad_parse=1 → reward 0). No additional logical operators (e.g., R, W) observed.

import os, json, pathlib, importlib  # 导入通用标准库模块
from pprint import pprint
import pandas as pd  # 数据处理依赖

# Path where this notebook lives
NOTEBOOK_DIR = pathlib.Path.cwd()  # 记录当前笔记本目录，方便构造相对路径

import os
import json
import pandas as pd
import difflib
from IPython.display import display

# ─── CONFIG ──────────────────────────────────────────────────────────────
root = "path/to/nl2tl eval"  # nl2tl 评估结果根目录
eval_types = ["LLM_masked_nl", "gt_lifting", "raw_nl"]  # 不同评估设置
max_entries = 500  # 样本数量限制
# ─────────────────────────────────────────────────────────────────────────

# discover dataset names from the first eval_type
datasets = sorted([
    os.path.splitext(f)[0]
    for f in os.listdir(os.path.join(root, eval_types[0]))
    if f.endswith(".jsonl")
])

# initialize DataFrames
acc_df = pd.DataFrame(index=eval_types, columns=datasets, dtype=float)  # 精确匹配率表
sim_df = pd.DataFrame(index=eval_types, columns=datasets, dtype=float)  # 子串相似度表

def best_substring_similarity(prediction: str, target: str) -> float:
    """
    Return the highest SequenceMatcher ratio between `target`
    and any substring of `prediction` of length len(target).
    If prediction is shorter than target, compare whole strings.
    """
    sm = difflib.SequenceMatcher
    t_len, p_len = len(target), len(prediction)
    if p_len < t_len:
        return sm(None, prediction, target).ratio()
    best = 0.0
    for i in range(p_len - t_len + 1):
        sub = prediction[i : i + t_len]
        best = max(best, sm(None, sub, target).ratio())
    return best

# compute metrics
for et in eval_types:
    et_dir = os.path.join(root, et)
    for ds in datasets:
        file_path = os.path.join(et_dir, f"{ds}.jsonl")
        if not os.path.isfile(file_path):
            continue

        preds, targets = [], []
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= max_entries:
                    break
                ent = json.loads(line)
                # target = grounded_sentence if present, else raw sentence
                if "masked_tl" in ent:
                    tgt = " ".join(ent["masked_tl"])
                elif et=="raw_nl":
                    tgt = " ".join(ent.get("tl", []))
                # clean prediction
                pred = ent.get("prediction", "").strip()
                parts = pred.split(" ", 1)
                if parts[0].rstrip(".").isdigit() and len(parts) > 1:
                    pred = parts[1]  # 去掉编号前缀
                preds.append(pred)
                targets.append(tgt)

        if not targets:
            continue

        # binary accuracy
        acc_df.at[et, ds] = sum(p == t for p, t in zip(preds, targets)) / len(targets)  # 精确匹配率
        # substring-based similarity
        sim_df.at[et, ds] = sum(
            best_substring_similarity(p, t)
            for p, t in zip(preds, targets)
        ) / len(targets)  # 子串最佳相似度平均

print("✅ NL2TL Translation Accuracy")
display(sim_df)


# Jupyter notebook cell: Verification evaluation with parser-error handling
# 该单元用于在存在解析错误时完成形式化验证评估

import json
import re
from pathlib import Path
from typing import List, Set, Union
from tqdm import tqdm
from functools import lru_cache
from pyModelChecking.LTL import Parser, AtomicProposition as AP, Not, And, Or, Imply, X, F, G, U
import pandas as pd

# ----------------------------------------------------------------------------
# 1 — Normalisation / implication elimination
# 1 — 规范化与蕴含消除
# ----------------------------------------------------------------------------
TOKEN_MAP = {
    "globally": "G", "always": "G", "[]": "G",
    "finally": "F", "eventually": "F", "<>": "F",
    "next": "X", "until": "U",
    "not": "not", "¬": "not", "!": "not", 
    "&": "and", "∧": "and",
    "|": "or", "∨": "or", "or": "or",
    "imply": "-->", "implies": "-->", "->": "-->",
    "⇒": "-->",
    "double_implies": "-->"
}
_PARSER = Parser()
_AP_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _normalise_tokens(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        low = t.lower()
        if low in TOKEN_MAP:
            out.append(TOKEN_MAP[low])
        elif t in ("(", ")"):
            out.append(t)
        elif _AP_OK.match(t):
            out.append(t)
        else:
            out.append(f"'{t}'")
    return " ".join(out)  # 将 token 列表重写为 LTL 友好的字符串

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
        for i, tok in enumerate(tokens):
            if tok not in ("->", "-->"):
                continue
            di = depth[i]
            j = i - 1
            while j >= 0 and depth[j] >= di:
                j -= 1
            lhs_start = j + 1
            k = i + 1
            while k < len(tokens) and depth[k] >= di:
                k += 1
            rhs_end = k
            lhs = tokens[lhs_start:i]
            rhs = tokens[i+1:rhs_end]
            new = ["(", "(", "not"] + lhs + [")", "or", "("] + rhs + [")", ")"]
            tokens = tokens[:lhs_start] + new + tokens[rhs_end:]
            break
        else:
            return tokens  # 当没有蕴含符时停止

@lru_cache(maxsize=16384)
def _parse(formula_str: str):
    return _PARSER(formula_str)  # 复用解析结果降低重复解析成本

def _eval(ast, trace: List[Set[str]], t: int = 0) -> bool:
    if isinstance(ast, AP):
        return str(ast) in trace[t]
    if isinstance(ast, Not):
        return not _eval(ast.subformula(0), trace, t)
    if isinstance(ast, And):
        return _eval(ast.subformula(0), trace, t) and _eval(ast.subformula(1), trace, t)
    if isinstance(ast, Or):
        return _eval(ast.subformula(0), trace, t) or _eval(ast.subformula(1), trace, t)
    if isinstance(ast, Imply):
        return (not _eval(ast.subformula(0), trace, t)) or _eval(ast.subformula(1), trace, t)
    if isinstance(ast, X):
        return _eval(ast.subformula(0), trace, min(t+1, len(trace)-1))
    if isinstance(ast, F):
        return any(_eval(ast.subformula(0), trace, k) for k in range(t, len(trace)))
    if isinstance(ast, G):
        return all(_eval(ast.subformula(0), trace, k) for k in range(t, len(trace)))
    if isinstance(ast, U):
        φ, ψ = ast.subformula(0), ast.subformula(1)
        for k in range(t, len(trace)):
            if _eval(ψ, trace, k):
                return all(_eval(φ, trace, j) for j in range(t, k))
        return False
    raise NotImplementedError(f"Unsupported AST node: {type(ast)}")

def _tokenise(tokens: Union[List[str], str]) -> List[str]:
    if isinstance(tokens, str):
        return re.findall(r"\w+|[()]", tokens)  # 简单分词器：拆出单词与括号
    return tokens

# ----------------------------------------------------------------------------
# 2 — Load ground-truth test entries
# 2 — 载入官方测试集用于比对
# ----------------------------------------------------------------------------
test_set_dir = Path("VLTL-Bench/test")
datasets = ["search_and_rescue", "traffic_light", "warehouse"]
test_entries = {}
for ds in datasets:
    m = {}
    with open(test_set_dir/f"{ds}.jsonl") as f:
        for line in f:
            e = json.loads(line)
            m[e["id"]] = e
    test_entries[ds] = m

# ----------------------------------------------------------------------------
# 3 — Run evaluation with parser-error handling
# 3 — 启动评估流程并记录解析错误
# ----------------------------------------------------------------------------
base_eval_dir = Path("translation_eval")
results = []

for fw_dir in base_eval_dir.iterdir():
    if not fw_dir.is_dir(): 
        continue
    framework = fw_dir.name

    # nl2tl structure
    if framework == "nl2tl":
        # continue
        for lift_dir in fw_dir.iterdir():
            lifting = lift_dir.name
            for ds in datasets:
                file = lift_dir/f"{ds}.jsonl"
                if not file.exists():
                    continue
                total = ok_good = ok_bad = ok_both = 0
                for line in tqdm(file.open(), desc=f"{framework}/{lifting}/{ds}"):
                    e = json.loads(line)
                    gt = test_entries[ds][e["id"]]
                    mapping = {
                        pid: f"{info['action_canon']}({','.join(info.get('args_canon',[]))})"
                        for pid,info in gt["prop_dict"].items()
                    }
                    rev = {v:k for k,v in mapping.items()}
                    to_labels = lambda raw: [{rev.get(ap,ap) for ap in step} for step in raw]  # 将原子命题映射回 prop_id
                    good, bad = to_labels(gt["good_trace"]), to_labels(gt["bad_trace"])
                    phi = e["prediction"]
                    if type(phi) == List: phi = "".join(phi)
                    

                    # ---- build a clean LTL string ----
                    # 将预测文本正则化为 LTL 可解析的公式
                    tokens   = _tokenise(phi)                    # list of word-tokens
                    norm_str  = _normalise_tokens(tokens)        # e.g. "globally ( prop_1 implies … )"
                    toks      = norm_str.split()                 # back to list
                    elim      = _elim_impl_tokens(toks)          # impl-elim
                    f_str     = " ".join(elim)                   # final formula string
                    try:
                        ast       = _parse(f_str)                    # parse AST
                        # ---- evaluate ---- 同时验证 good/bad 轨迹
                        good_sat = _eval(ast, good)  # 正例轨迹是否满足公式
                        bad_sat  = _eval(ast, bad)   # 反例轨迹是否违反公式

                        if good_sat:
                            ok_good += 1
                        if not bad_sat:
                            ok_bad += 1
                        if good_sat and not bad_sat:
                            ok_both += 1
                    except Exception:
                        bad_parse +=1 


                    total += 1
                    # print(bad_parse)
                results.append((
                    framework, lifting, model, ds, total,
                    ok_good/total, ok_bad/total, ok_both/total
                ))

    else:
        for lift_dir in fw_dir.iterdir():
            lifting = lift_dir.name
            for model_dir in lift_dir.iterdir():
                model = model_dir.name
                if '4.1-mini' not in model:
                    continue
                print(model)
                for ds in datasets:
                    file = model_dir / f"{ds}.jsonl"
                    if not file.exists():
                        continue

                    total = ok_good = ok_bad = ok_both = bad_parse = 0
                    for line in tqdm(file.open(), desc=f"{framework}/{lifting}/{model}/{ds}"):
                        e = json.loads(line)
                        gt = test_entries[ds][e["id"]]

                        # rebuild prop->atom mapping
                        mapping = {
                            pid: f"{info['action_canon']}({','.join(info.get('args_canon', []))})"
                            for pid, info in gt["prop_dict"].items()
                        }
                        rev_map = {atom: pid for pid, atom in mapping.items()}
                        to_labels = lambda raw: [{rev_map.get(ap, ap) for ap in step} for step in raw]  # 逆映射回复合标识
                        good, bad = to_labels(gt["good_trace"]), to_labels(gt["bad_trace"])

                        # strip any ChatGPT prefixes/suffixes
                        phi = e["prediction"]
                        if phi.startswith('LTL:'):
                            phi = phi[4:]
                        if phi.startswith('3. *FINAL:* '):
                            phi = phi[12:]
                        for suffix in ('*FINISH*', 'FINISH'):
                            if phi.endswith(suffix):
                                phi = phi[: -len(suffix)]

                        # ---- build a clean LTL string ----
                        tokens   = _tokenise(phi)                    # list of word-tokens
                        norm_str  = _normalise_tokens(tokens)        # e.g. "globally ( prop_1 implies … )"
                        toks      = norm_str.split()                 # back to list
                        elim      = _elim_impl_tokens(toks)          # impl-elim
                        f_str     = " ".join(elim)                   # final formula string
                        try:
                            ast       = _parse(f_str)                    # parse AST
                            # ---- evaluate ---- 同时验证 good/bad 轨迹
                            good_sat = _eval(ast, good)  # 正例轨迹是否满足公式
                            bad_sat  = _eval(ast, bad)   # 反例轨迹是否违反公式

                            if good_sat:
                                ok_good += 1
                            if not bad_sat:
                                ok_bad += 1
                            if good_sat and not bad_sat:
                                ok_both += 1
                        except Exception:
                            bad_parse +=1 


                        total += 1
                        # print(bad_parse)
                    results.append((
                        framework, lifting, model, ds, total,
                        ok_good/total, ok_bad/total, ok_both/total
                    ))



                        # results.append((framework, lifting, model, ds, total,
                        #                 ok_good/total, ok_bad/total, ok_both/total))

# Summarize
columns = ["framework","lifting","model","dataset","total",
           "ok_good(%)","ok_bad(%)","ok_both(%)"]
df = pd.DataFrame(results, columns=columns)
df
"""improve_dataset.py
读取 JSONL 中的某个字段，利用大模型对 LTL 公式进行质量提升和变体扩充。
融合了 5 种多智能体协同 LTL 模板风格。
支持“1变2”增广：合理的公式会同时保存原句和变体为两条独立的 JSON 记录。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

DEFAULT_INPUT_PATH = "/data/ljr/llm_experiments/ljr_ltl_datasetV1/rawltl.jsonl"

try:
    from dataset_generators.ltl_verifier import SPOT_AVAILABLE, verify_ltl_formula
except Exception:
    try:
        from ltl_verifier import SPOT_AVAILABLE, verify_ltl_formula  # type: ignore
    except Exception:
        SPOT_AVAILABLE = False

        def verify_ltl_formula(formula: str, original_props: Optional[dict] = None):  # type: ignore
            return True, None, formula, "SPOT not available"


ALLOWED_KEYWORDS = {
    "globally",
    "finally",
    "next",
    "until",
    "release",
    "and",
    "or",
    "not",
    "implies",
    "(",
    ")",
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_props(formula: str) -> List[str]:
    if not formula:
        return []
    return list(dict.fromkeys(re.findall(r"prop_\d+", formula)))


def normalize_ltl(text: str) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text.strip().replace("\n", " "))
    s = re.sub(r"\(\s*", "( ", s)
    s = re.sub(r"\s*\)", " )", s)
    return re.sub(r"\s+", " ", s).strip()


def choose_complexity(index: int) -> str:
    return ["simple", "medium", "complex", "medium"][index % 4]


def preferred_task_type_from_complexity(complexity: str, num_props: int = 0) -> str:
    """Determine task type: use 'single' for simple complexity or <2 props, 'multi' otherwise."""
    if num_props < 2 or complexity == "simple":
        return "single"
    return "multi"


def basic_sanity_check(formula: str) -> Tuple[bool, str]:
    if not formula or not formula.strip():
        return False, "empty"
    s = normalize_ltl(formula)
    if "prop_" not in s:
        return False, "no_prop"

    tokens = s.replace("(", " ( ").replace(")", " ) ").split()
    for t in tokens:
        if t in ALLOWED_KEYWORDS or re.fullmatch(r"prop_\d+", t):
            continue
        if t in {"&&", "||", "!", "->"}:
            return False, "spot_style_operator_found"
        if re.search(r"[{}\[\]\\]", t):
            return False, "illegal_char_found"
    return True, "ok"


def looks_like_multi_task(formula: str) -> bool:
    s = normalize_ltl(formula)
    return len(set(extract_props(s))) >= 2 and " and " in f" {s} "


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", text.strip())
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.7
    max_tokens: int = 256
    timeout_s: int = 60


class OpenAICompatibleClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._url = cfg.base_url.rstrip("/") + "/v1/chat/completions"

    async def chat(
        self, session: aiohttp.ClientSession, messages: List[Dict[str, str]]
    ) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_s)

        for attempt in range(6):
            async with session.post(
                self._url, headers=headers, json=payload, timeout=timeout
            ) as resp:
                text = await resp.text()
                if resp.status in {429, 500, 502, 503, 504} and attempt < 5:
                    await asyncio.sleep(min(30.0, 1.5**attempt + random.random()))
                    continue
                if resp.status >= 400:
                    raise RuntimeError(
                        f"[{self.cfg.provider}/{self.cfg.model}] HTTP {resp.status}: {text[:500]}"
                    )
                try:
                    response_json = json.loads(text)
                    content = response_json["choices"][0]["message"]["content"]
                    if not content or not content.strip():
                        raise RuntimeError(
                            f"[{self.cfg.provider}/{self.cfg.model}] Empty response content"
                        )
                    return content
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    raise RuntimeError(
                        f"[{self.cfg.provider}/{self.cfg.model}] Invalid response format: {e}. Response: {text[:500]}"
                    )
        raise RuntimeError(
            f"[{self.cfg.provider}/{self.cfg.model}] HTTP retries exhausted"
        )


# ----------------- 核心重构区：提示词与生成逻辑 -----------------


def build_messages(
    original_tl: str,
    complexity: str,
    props: List[str],
    preferred_task_type: str,
    repair_hint: Optional[str] = None,
) -> List[Dict[str, str]]:
    props_str = ", ".join(props) if props else "(none)"

    system = (
        "You are an expert in formal methods and autonomous multi-agent systems.\n"
        "Your task is to take an input LTL formula and EVOLVE it into a high-quality mission specification.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. EVOLUTION LOGIC: If the input is reasonable, do not just rephrase it. ENRICH it by applying multi-agent coordination patterns. You may add realistic constraints (e.g., mutual exclusion, ordering, or liveness) based on the original propositions.\n"
        "2. NO TAUTOLOGIES: NEVER generate formulas that simplify to '1' or '0'. Every operator must serve a purpose in constraining the agent's behavior.\n"
        "3. COMPLEXITY ALIGNMENT: Ensure the generated formula matches the 'Target complexity'. For 'complex' tasks, combine at least two different template styles (e.g., Trigger-Response + Mutual Exclusion).\n"
        "4. SYNTAX: Use only: globally, finally, next, until, release, implies, and, or, not, (). Mandatory full parenthesization for tree-like parsing."
    )

    user_parts = [
        f"Input LTL formula:\n{original_tl}\n",
        f"Available propositions: {props_str}",
        f"Target complexity: {complexity} (simple/medium/complex) | Preferred task type: {preferred_task_type} (single/multi)",
        "Mission Design Goal: Imagine these propositions (prop_x) represent robot states or environment events. "
        "The generated formula should represent a plausible, sophisticated mission for an unmanned cluster, "
        "ensuring safety while pursuing goals.",
        "Use the following common multi-agent LTL template styles to generate a variant or a new formula:\n"
        "1. Sequential Reachability: Visit in order, e.g., globally finally ( prop_1 and finally ( prop_2 and finally prop_3 ) )\n"
        "2. Parallel Goals & Safety: Multi-goal coordination and conflict avoidance, e.g., ( globally finally prop_1 ) and ( globally finally prop_2 ) and ( globally not prop_3 )\n"
        "3. Strict Prerequisite via Until: Prevent an action until another is completed, e.g., ( finally prop_1 ) and ( not prop_2 until prop_1 )\n"
        "4. Sustained Action: Maintain action until a condition triggers, e.g., finally ( prop_1 and next ( prop_1 until prop_2 ) )\n"
        "5. Branching / Choice: Multiple alternative sequences, e.g., ( finally prop_1 ) and ( ( finally ( prop_2 and finally prop_3 ) ) or ( finally ( prop_4 and finally prop_5 ) ) )\n"
        "6. Cyclic / Patrol Loops: Repeatedly visit goals, e.g., globally ( finally ( prop_1 ) and finally ( prop_2 ) and finally ( prop_3 ) )\n"
        "7. Mutual Exclusion / Conflict Avoidance: Prevent simultaneous execution, e.g., globally ( not ( prop_1 and prop_2 ) )\n"
        "8. Trigger-Response / Event Reaction: One event triggers another, e.g., globally ( prop_1 implies finally ( prop_2 ) )\n"
        "9. Priority / Hierarchy: High-priority tasks first, e.g., globally ( ( prop_1 implies finally prop_2 ) and ( not prop_3 until prop_1 ) )\n"
        "10. Nested / Multi-layered Goals: Multi-stage nesting, e.g., globally ( finally ( prop_1 and ( finally ( prop_2 and finally prop_3 ) ) ) )\n"
        "11. Complex Cyclic Coordination: Combine loops, exclusion, triggers, e.g., globally ( ( prop_1 implies finally prop_2 ) and not ( prop_3 and prop_4 ) and finally ( prop_5 ) )\n"
        "12. Random Task: Create any reasonable LTL task",
        "Judgment criteria:\n- single: focus on a single goal or simple sustained state.\n- multi: focus on parallel tasks, branching, or complex coordination.\n- If the original formula is obviously contradictory or meaningless, treat as unreasonable.",
        "Strictly output the following JSON structure:\n"
        "{\n"
        '  "task_type": "single" or "multi",\n'
        '  "action": "keep" (original formula reasonable, variant generated) or "generate" (new formula generated),\n'
        '  "style_used": "template style(s) used",\n'
        '  "generated_tl": "your generated LTL formula here",\n'
        '  "reason": "brief explanation for your decision"\n'
        "}\n",
    ]

    if repair_hint:
        user_parts.append(
            f"⚠️ Previous attempt failed validation, please fix and retry:\n{repair_hint}"
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def fallback_generate_formula(props: List[str], task_type: str) -> str:
    """Generate a simple, satisfiable, fully-parenthesized formula without calling an LLM."""
    uniq_props = [p for p in props if re.fullmatch(r"prop_\d+", p)]
    if not uniq_props:
        uniq_props = ["prop_1"]

    # Ensure enough props for multi
    if task_type == "multi" and len(uniq_props) < 2:
        uniq_props = uniq_props + ["prop_2"]

    if task_type == "single":
        p = uniq_props[0]
        return f"globally ( finally ( {p} ) )"

    p1, p2 = uniq_props[0], uniq_props[1]
    # Multi: two goals eventually, globally
    return f"globally ( ( finally ( {p1} ) ) and ( finally ( {p2} ) ) )"


async def generate_one(
    client: OpenAICompatibleClient,
    session: aiohttp.ClientSession,
    original_tl: str,
    complexity: str,
    verify: bool,
    max_retries: int,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """Returns (action, LLM result dict, Spot error message for original formula if any)."""
    props = extract_props(original_tl)
    preferred_task_type = preferred_task_type_from_complexity(complexity, len(props))
    repair_hint = None

    for attempt in range(max_retries + 1):
        messages = build_messages(
            original_tl, complexity, props, preferred_task_type, repair_hint
        )
        raw = await client.chat(session, messages)
        obj = _extract_json_object(raw)

        if not obj:
            repair_hint = f"No valid JSON detected. Output must be a single JSON object that matches the required schema. Your output: {raw[:200]}"
            continue

        task_type = str(obj.get("task_type", ""))
        action = str(obj.get("action", ""))
        style_used = str(obj.get("style_used", "Unknown Style"))
        generated_tl = normalize_ltl(
            str(obj.get("generate") or obj.get("generated_tl", ""))
        )
        reason = str(obj.get("reason", ""))

        # 1. JSON 基础字段校验
        if task_type not in {"single", "multi"}:
            repair_hint = "Field 'task_type' must be either 'single' or 'multi'."
            continue
        if action not in {"keep", "generate"}:
            repair_hint = "Field 'action' must be either 'keep' or 'generate'."
            continue
        if not generated_tl:
            repair_hint = "Field 'generated_tl' must be non-empty."
            continue

        # 2. 语法与业务逻辑检查
        ok, why = basic_sanity_check(generated_tl)
        if not ok:
            repair_hint = (
                f"Formula failed format/sanity check ({why}). Do NOT use SPOT-style symbols like &&, ||, !, ->. "
                f"Current formula: {generated_tl}"
            )
            continue

        if task_type == "multi" and not looks_like_multi_task(generated_tl):
            repair_hint = "You declared a 'multi' task, but the formula does not reflect coordination: it should include an 'and' relationship and at least two distinct propositions."
            continue

        normalized_original = normalize_ltl(original_tl)
        if action == "keep":
            if generated_tl == normalized_original:
                repair_hint = "When action='keep', the variant formula must not be exactly identical to the original."
                continue

            orig_simple = (
                normalized_original.replace("(", "").replace(")", "").replace(" ", "")
            )
            gen_simple = generated_tl.replace("(", "").replace(")", "").replace(" ", "")
            if orig_simple == gen_simple:
                repair_hint = "The variant is too structurally similar to the original. Use a different combination of LTL operators and/or change how propositions are grouped and nested."
                continue

        # 3. SPOT 可满足性与等价化简校验
        spot_error_msg = None
        if verify and SPOT_AVAILABLE:
            is_sat, _, sim_nat, err = verify_ltl_formula(generated_tl)
            if err:
                repair_hint = f"The generated formula has a SPOT syntax error: {err}"
                continue
            if not is_sat:
                repair_hint = "The generated formula is logically unsatisfiable. Please fix the logical conflict and regenerate."
                continue

            # 拦截点 1：防 "1" 或 "0" 的废话公式 (GPT 崩溃的元凶)
            sim_str = str(sim_nat).strip().lower()
            if sim_str in {"1", "0", "t", "f", "true", "false"}:
                repair_hint = (
                    f"Warning: Your formula simplified to absolute '{sim_str}'. "
                    "DO NOT generate meaningless tautologies like 'not prop_1 release true'. "
                    "You MUST generate a formula that actively limits the propositions."
                )
                continue

            # 拦截点 2：防 "伪变体" (DeepSeek 假装修改的元凶)
            if action == "keep":
                orig_is_sat, _, orig_sim_nat, orig_err = verify_ltl_formula(original_tl)
                if orig_err or not orig_is_sat:
                    spot_error_msg = orig_err or "Unsatisfiable"
                    repair_hint = "The original formula has a SPOT error or is unsatisfiable. You must set action='generate' and produce a brand-new formula."
                    continue

                # 如果两者化简后的形式一模一样，说明大模型在偷懒
                if normalize_ltl(sim_str) == normalize_ltl(str(orig_sim_nat)):
                    repair_hint = (
                        f"Your variant is too trivial. After standard LTL simplification, it reverts exactly to the original simplified form: '{sim_str}'. "
                        "You MUST use a fundamentally different combination of operators so its simplest form is distinct, while maintaining exact semantic equivalence."
                    )
                    continue

            # 校验全部通过，使用SPOT化简后的公式
            generated_tl = normalize_ltl(str(sim_nat))

        # 构建扁平化字典
        result = {
            "llm_task_type": task_type,
            "llm_style": style_used,
            "final_tl": generated_tl,
            "llm_reason": reason,
            "llm_attempts": attempt + 1,
        }

        return action, result, spot_error_msg

    raise RuntimeError(f"Exceeded retries; last hint: {repair_hint}")


# ----------------- 调度器逻辑 -----------------


async def run_async(args: argparse.Namespace) -> None:
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=dotenv_path, override=False)
    random.seed(args.seed)

    if not args.output:
        in_path = str(args.input or DEFAULT_INPUT_PATH)
        in_dir = os.path.dirname(in_path) or "."
        base = os.path.basename(in_path)
        out_name = (
            base[:-6] + "_improved.jsonl"
            if base.endswith(".jsonl")
            else base + "_improved.jsonl"
        )
        args.output = os.path.join(in_dir, out_name)

    items = (
        read_jsonl(args.input)[: args.max_items]
        if args.max_items
        else read_jsonl(args.input)
    )

    KEEP_FIELDS = (
        "id",
        "masked_tl",
        "llm_task_type",
        "llm_style",
        "final_tl",
        "llm_reason",
        "llm_attempts",
        "llm_action",
        "llm_model",
    )

    def _keep_only_fields(record: Dict[str, Any]) -> Dict[str, Any]:
        return {k: record.get(k, "") for k in KEEP_FIELDS}

    # Initialize deepseek and openai clients
    def _create_client(provider: str, model: str) -> OpenAICompatibleClient:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", "https://www.dmxapi.cn")
        else:  # deepseek
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        cfg = LLMConfig(
            provider,
            model,
            api_key,
            base_url,
            args.temperature,
            args.max_tokens,
            args.timeout,
        )
        return OpenAICompatibleClient(cfg)

    deepseek_model = args.deepseek_model
    openai_model = args.openai_model
    deepseek_client = _create_client("deepseek", deepseek_model)
    openai_client = _create_client("openai", openai_model)

    def get_client_and_model(index: int) -> Tuple[OpenAICompatibleClient, str]:
        """Allocate clients based on deepseek_ratio."""
        # deepseek_ratio 为分子，总数为 10
        # 例如 deepseek_ratio=7 表示 7/10 给 deepseek，3/10 给 openai
        if index % 10 < args.deepseek_ratio:
            return deepseek_client, deepseek_model
        else:
            return openai_client, openai_model

    sem = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as session:
        tasks = []

        async def _worker(
            i: int, item: Dict[str, Any]
        ) -> Tuple[int, List[Dict[str, Any]]]:
            """返回一个列表，可能包含 1 条记录 (替换/错误) 或 2 条记录 (原句 + 变体)"""
            async with sem:
                source_field = str(args.source_field)
                source_tl = item.get(source_field, "")

                out: Dict[str, Any] = {
                    "id": item.get("id", i + 1),
                    "masked_tl": str(source_tl)
                    if source_field == "masked_tl"
                    else str(item.get("masked_tl", "")),
                }

                if not source_tl:
                    # Skip items with missing source field
                    return i, []

                # Check if input formula is valid before calling LLM
                input_ok, input_why = basic_sanity_check(str(source_tl))
                if not input_ok:
                    # Skip items with invalid input formula
                    return i, []

                try:
                    selected_client, selected_model = get_client_and_model(i)
                    action, result, spot_err = await generate_one(
                        selected_client,
                        session,
                        str(source_tl),
                        choose_complexity(i),
                        bool(args.verify),
                        args.retries,
                    )
                    result["llm_model"] = selected_model

                    if spot_err:
                        result["spot_original_error"] = spot_err

                    out_records = []

                    if action == "keep":
                        # 记录1：原始公式的 JSON (打上 original_kept 标签)
                        rec1 = out.copy()
                        rec1.update(result)
                        rec1["llm_action"] = "variant_original"
                        rec1["final_tl"] = str(source_tl)
                        rec1["llm_reason"] = (
                            "Retained original formula as it is logically sound."
                        )
                        out_records.append(_keep_only_fields(rec1))

                        # 记录2：新生成的变体公式 JSON (打上 variant_generated 标签)
                        rec2 = out.copy()
                        rec2.update(result)  # final_tl 已经是生成的新公式
                        rec2["llm_action"] = "variant_generated"
                        out_records.append(_keep_only_fields(rec2))

                    else:
                        # action == "generate": 原公式不合理，只有一条替换记录
                        rec = out.copy()
                        rec.update(result)
                        rec["llm_action"] = "variant_generated"
                        out_records.append(_keep_only_fields(rec))

                    return i, out_records

                except Exception as e:
                    # Skip items that failed after all retries
                    return i, []

        for i, item in enumerate(items):
            tasks.append(asyncio.create_task(_worker(i, item)))

        results: List[Dict[str, Any]] = []
        for fut in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="LLM improving"
        ):
            idx, out_items = await fut
            results.extend(out_items)  # 使用 extend 展平所有返回的列表

    write_jsonl(args.output, results)

    # 统计信息更新，匹配新的 llm_action 标签
    ok_count = sum(1 for r in results if r.get("final_tl"))
    variant_original_count = sum(
        1 for r in results if r.get("llm_action") == "variant_original"
    )
    variant_generated_count = sum(
        1 for r in results if r.get("llm_action") == "variant_generated"
    )
    error_count = sum(1 for r in results if r.get("llm_action") == "error")

    # 统计各模型处理的记录数
    model_stats = {}
    for r in results:
        model = r.get("llm_model", "")
        if model:
            model_stats[model] = model_stats.get(model, 0) + 1

    print(
        f"\nDone. total records={len(results)} ok={ok_count} err={len(results) - ok_count}"
    )
    print(
        f"variant_original={variant_original_count} variant_generated={variant_generated_count} error={error_count}"
    )
    print(f"Model distribution: {model_stats}" if model_stats else "")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT_PATH)
    p.add_argument("--output", default="")
    p.add_argument("--source-field", default="masked_tl")
    p.add_argument(
        "--deepseek-model", default="deepseek-chat", help="Deepseek model name"
    )
    p.add_argument(
        "--openai-model",
        default="gpt-4.1",
        help="OpenAI model name (e.g., gpt-4o, gpt-4-turbo, gpt-3.5-turbo)",
    )
    p.add_argument(
        "--deepseek-ratio",
        type=int,
        default=8,
        help="Deepseek data ratio (0-10). Example: 7 means 7/10 for deepseek, 3/10 for openai.",
    )
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument(
        "--max-items", type=int, default=1000, help="Max items to process (0 for all)"
    )
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify", default=True)
    return p


if __name__ == "__main__":
    try:
        asyncio.run(run_async(build_arg_parser().parse_args()))
    except KeyboardInterrupt:
        print("Interrupted.")

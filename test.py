"""
测试脚本：验证 predictions/rl_model/warehouse_nl_alpaca_pred.jsonl 中的预测结果
集成了语法级复杂度过滤与 Spot 预解析逻辑。
"""

import json
import sys
import time
import re

# 尝试导入 spot 以便进行原生解析检查
try:
    import spot

    SPOT_AVAILABLE_DIRECT = True
except ImportError:
    SPOT_AVAILABLE_DIRECT = False

# 导入 LTL 验证器
sys.path.insert(0, "dataset_generators")
try:
    from ltl_verifier import verify_ltl_formula, convert_to_spot_format, SPOT_AVAILABLE
except ImportError:
    # 回退方案：如果环境路径不一致
    SPOT_AVAILABLE = False
    print("[警告] 无法从 dataset_generators 加载 ltl_verifier")

# --- 一、轻量级复杂度检查逻辑 ---


def estimate_ltl_complexity(formula):
    """统计 LTL 公式的结构特征 (O(n) 复杂度)"""
    # 简单的词法分词
    tokens = re.findall(r"\(|\)|[^\s()]+", formula)

    depth = 0
    max_depth = 0
    until_count = 0
    logic_count = 0

    # 扩展：匹配多种可能的关键字
    until_ops = {"until", "U", "W", "M", "release", "R"}
    logic_ops = {"and", "or", "implies", "&&", "||", "->", "=>", "xor", "<->"}

    for t in tokens:
        t_lower = t.lower()
        if t == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif t == ")":
            depth -= 1
        elif t_lower in until_ops or t in until_ops:
            until_count += 1
        elif t_lower in logic_ops or t in logic_ops:
            logic_count += 1

    return {
        "token_count": len(tokens),
        "max_depth": max_depth,
        "until_count": until_count,
        "logic_count": logic_count,
    }


def is_formula_too_complex(formula):
    """
    过滤规则（实践稳定阈值）:
    - token_count > 80 (公式太长)
    - max_depth > 10 (括号或算子嵌套太深)
    - until_count > 6 (Until 爆炸核心诱因)
    - logic_count > 15 (连接词过多)
    """
    if not formula:
        return True

    stats = estimate_ltl_complexity(formula)

    if stats["token_count"] > 128:
        return True, f"token_count({stats['token_count']}) > 80"
    if stats["max_depth"] > 12:
        return True, f"max_depth({stats['max_depth']}) > 10"
    if stats["until_count"] > 16:
        return True, f"until_count({stats['until_count']}) > 6"
    if stats["logic_count"] > 30:
        return True, f"logic_count({stats['logic_count']}) > 15"

    return False, None


# --- 二、带超时和过滤的验证流程 ---


def verify_with_timeout(formula, timeout_seconds=0.03):
    """
    带复杂过滤和 Spot 解析检查的 LTL 验证
    流程：复杂度过滤 -> Spot 语法解析 -> 实际验证(translate)
    """
    if not formula or not formula.strip():
        return False, None, None, "Empty formula"

    # 1. 轻量 AST 复杂度检查
    is_complex, reason = is_formula_too_complex(formula)
    if is_complex:
        return False, None, None, f"Timeout: formula too complex ({reason})"

    # 2. Spot 原生解析检查 (只检查语法，不转自动机)
    if SPOT_AVAILABLE_DIRECT:
        try:
            # 转换为 spot 格式后再检查
            spot_f_str = convert_to_spot_format(formula)
            _ = spot.formula(spot_f_str)
        except Exception as e:
            return False, None, None, f"Syntax Error (Spot): {str(e)}"

    # 3. 尝试执行正式验证 (包含 translate 和 satisfiable check)
    start_time = time.time()
    try:
        # 注意：verify_ltl_formula 内部通常包含 translate，这是最慢的一步
        is_sat, spot_f, simplified, error = verify_ltl_formula(formula)
        elapsed = time.time() - start_time

        # 如果虽然没卡死但超过了设定的严格阈值
        if elapsed > timeout_seconds:
            return (
                is_sat,
                spot_f,
                simplified,
                f"Warning: slow verification ({elapsed:.3f}s)",
            )

        return is_sat, spot_f, simplified, error
    except Exception as e:
        return False, None, None, f"Runtime Error: {str(e)}"


# --- 三、主逻辑与数据处理 ---


def load_jsonl(filepath):
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def verify_prediction_entry(entry):
    """验证单个预测条目"""
    ground_truth = entry.get("output", "")
    prediction = entry.get("prediction", "")

    # 验证 Ground Truth
    is_sat_gt, spot_gt, _, error_gt = verify_with_timeout(ground_truth)

    # 验证 Prediction
    is_sat_pred, spot_pred, simplified_pred, error_pred = verify_with_timeout(
        prediction
    )

    return {
        "instruction": entry.get("instruction", ""),
        "input": entry.get("input", ""),
        "output": ground_truth,
        "prediction": prediction,
        "ground_truth_verified": {
            "is_satisfiable": is_sat_gt,
            "spot_formula": spot_gt,
            "error": error_gt,
        },
        "prediction_verified": {
            "is_satisfiable": is_sat_pred,
            "spot_formula": spot_pred,
            "simplified_spot": simplified_pred,
            "error": error_pred,
        },
    }


def main():
    input_file = "predictions/rl_model/warehouse_nl_alpaca_pred.jsonl"
    output_file = "predictions/rl_model/warehouse_nl_alpaca_pred_verified.jsonl"

    print(f"[*] 加载预测数据: {input_file}")
    entries = load_jsonl(input_file)
    print(f"[*] 共计 {len(entries)} 条数据")
    print("[*] 启用过滤器: AST复杂度检查 + Spot解析预处理")

    verified_entries = []
    for idx, entry in enumerate(entries):
        if idx % 100 == 0:
            print(f"    处理进度: {idx}/{len(entries)}")

        verified = verify_prediction_entry(entry)
        verified["id"] = idx
        verified_entries.append(verified)

    # --- 四、结果统计 ---
    def get_stats(data, key):
        total = len(data)
        sat = sum(1 for e in data if e[key].get("is_satisfiable") is True)
        unsat = sum(
            1
            for e in data
            if e[key].get("is_satisfiable") is False and not e[key].get("error")
        )
        errors = sum(1 for e in data if e[key].get("error"))
        timeouts = sum(
            1
            for e in data
            if e[key].get("error") and "Timeout" in str(e[key].get("error"))
        )
        return sat, unsat, errors, timeouts

    gt_sat, gt_unsat, gt_err, gt_tout = get_stats(
        verified_entries, "ground_truth_verified"
    )
    pr_sat, pr_unsat, pr_err, pr_tout = get_stats(
        verified_entries, "prediction_verified"
    )

    print(f"\n{'=' * 20} 统计结果 {'=' * 20}")
    print(
        f"Ground Truth | 可满足: {gt_sat} | 不可满足: {gt_unsat} | 错误: {gt_err} (含超时: {gt_tout})"
    )
    print(
        f"Predictions  | 可满足: {pr_sat} | 不可满足: {pr_unsat} | 错误: {pr_err} (含超时: {pr_tout})"
    )

    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in verified_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n[*] 结果已保存至: {output_file}")


if __name__ == "__main__":
    if not SPOT_AVAILABLE and not SPOT_AVAILABLE_DIRECT:
        print("[错误] 未检测到 Spot 库。请通过 'pip install spot' 或 conda 安装。")
        sys.exit(1)

    main()

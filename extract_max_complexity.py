#!/usr/bin/env python3
"""
提取Ground Truth中满足各条件的情况
"""

import json
import re
import os


def estimate_ltl_complexity(formula):
    """统计 LTL 公式的结构特征"""
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


def main():
    input_file = "predictions/rl_model/warehouse_nl_alpaca_pred.jsonl"

    # 统计各条件的满足情况
    stats = {
        "total": 0,
        "token_count_gt_80": 0,
        "max_depth_gt_10": 0,
        "until_count_gt_6": 0,
        "logic_count_gt_15": 0,
        "all_conditions": 0,
    }

    # LTL公式长度分布统计
    length_distribution = {
        "0-20": 0,
        "21-40": 0,
        "41-60": 0,
        "61-80": 0,
        "81-100": 0,
        "101-150": 0,
        "151-200": 0,
        "200+": 0,
    }

    # 存储每个指标的最大值
    max_token_count = 0
    max_max_depth = 0
    max_until_count = 0
    max_logic_count = 0

    # 存储满足各个单独条件的条目示例
    examples = {
        "token_count_gt_80": None,
        "max_depth_gt_10": None,
        "until_count_gt_6": None,
        "logic_count_gt_15": None,
    }

    print(f"处理文件: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # 获取LTL公式 - 优先从output字段获取
                tl_formula = entry.get("output", entry.get("tl", []))
                if isinstance(tl_formula, list):
                    tl_str = " ".join(tl_formula)
                else:
                    tl_str = str(tl_formula)

                if tl_str:
                    s = estimate_ltl_complexity(tl_str)
                    stats["total"] += 1

                    # 统计长度分布
                    token_count = s["token_count"]
                    if token_count <= 20:
                        length_distribution["0-20"] += 1
                    elif token_count <= 40:
                        length_distribution["21-40"] += 1
                    elif token_count <= 60:
                        length_distribution["41-60"] += 1
                    elif token_count <= 80:
                        length_distribution["61-80"] += 1
                    elif token_count <= 100:
                        length_distribution["81-100"] += 1
                    elif token_count <= 150:
                        length_distribution["101-150"] += 1
                    elif token_count <= 200:
                        length_distribution["151-200"] += 1
                    else:
                        length_distribution["200+"] += 1

                    # 更新最大值
                    max_token_count = max(max_token_count, s["token_count"])
                    max_max_depth = max(max_max_depth, s["max_depth"])
                    max_until_count = max(max_until_count, s["until_count"])
                    max_logic_count = max(max_logic_count, s["logic_count"])

                    # 检查各条件
                    if s["token_count"] > 80:
                        stats["token_count_gt_80"] += 1
                        if examples["token_count_gt_80"] is None:
                            examples["token_count_gt_80"] = (input_file, s)

                    if s["max_depth"] > 10:
                        stats["max_depth_gt_10"] += 1
                        if examples["max_depth_gt_10"] is None:
                            examples["max_depth_gt_10"] = (input_file, s)

                    if s["until_count"] > 6:
                        stats["until_count_gt_6"] += 1
                        if examples["until_count_gt_6"] is None:
                            examples["until_count_gt_6"] = (input_file, s)

                    if s["logic_count"] > 15:
                        stats["logic_count_gt_15"] += 1
                        if examples["logic_count_gt_15"] is None:
                            examples["logic_count_gt_15"] = (input_file, s)

                    # 检查所有条件
                    if (
                        s["token_count"] > 80
                        and s["max_depth"] > 10
                        and s["until_count"] > 6
                        and s["logic_count"] > 15
                    ):
                        stats["all_conditions"] += 1

    print(f"\n=== 统计结果 ===")
    print(f"总条目数: {stats['total']}")
    print(f"token_count > 80: {stats['token_count_gt_80']}")
    print(f"max_depth > 10: {stats['max_depth_gt_10']}")
    print(f"until_count > 6: {stats['until_count_gt_6']}")
    print(f"logic_count > 15: {stats['logic_count_gt_15']}")
    print(f"同时满足所有条件: {stats['all_conditions']}")

    print(f"\n=== LTL公式长度分布 ===")
    total = stats["total"]
    if total > 0:
        for range_key in [
            "0-20",
            "21-40",
            "41-60",
            "61-80",
            "81-100",
            "101-150",
            "151-200",
            "200+",
        ]:
            count = length_distribution[range_key]
            percentage = (count / total) * 100
            print(f"{range_key:>10} tokens: {count:>4} ({percentage:>5.2f}%)")

    print(f"\n=== 最大值 ===")
    print(f"max_token_count: {max_token_count}")
    print(f"max_max_depth: {max_max_depth}")
    print(f"max_until_count: {max_until_count}")
    print(f"max_logic_count: {max_logic_count}")

    print(f"\n=== 各条件示例 ===")
    for key, val in examples.items():
        if val:
            print(f"{key}: {val[0]} -> {val[1]}")


if __name__ == "__main__":
    main()

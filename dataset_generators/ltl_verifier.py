"""
LTL 公式验证器 - 使用 SPOT 库

功能：
1. 将自然语言格式的 LTL 公式转换为 SPOT 格式
2. 化简公式
3. 检查可满足性
"""

import re
import sys
from typing import Tuple, Optional

# 尝试导入 spot
try:
    import spot

    SPOT_AVAILABLE = True
except ImportError:
    SPOT_AVAILABLE = False
    print("[警告] SPOT 库未安装，验证功能将不可用")
    print("安装方法: pip install spot")


# 关键词映射：自然语言 -> SPOT 格式
# 说明：Spot 在输出/简化时可能使用单字符布尔算子 '&' / '|'，以及释放算子 'R'。
# 为了让“还原 masked_tl”为自然语言格式稳定，需要同时支持这些符号。
KEYWORD_MAP = {
    # 时序算子
    "globally": "G",
    "finally": "F",
    "next": "X",
    "until": "U",
    "release": "R",
    # 逻辑连接词
    "implies": "->",
    "and": "&&",
    "or": "||",
    "not": "!",
    # 括号
    "(": "(",
    ")": ")",
}

# 反向映射：SPOT 格式 -> 自然语言
REVERSE_KEYWORD_MAP = {
    # 时序算子
    "G": "globally",
    "F": "finally",
    "X": "next",
    "U": "until",
    "R": "release",
    "W": "weak until",
    "M": "strong release",
    # 逻辑连接词
    "->": "implies",
    "&&": "and",
    "||": "or",
    "&": "and",
    "|": "or",
    "!": "not",
    "<->": "equivalence",
    "^": "xor",
    # 括号
    "(": "(",
    ")": ")",
}


def _tokenize_spot_formula(spot_formula: str) -> list:
    """将 Spot 风格字符串粗粒度分词。

    目标：稳定处理 Spot 简化/pretty-print 里常见的紧贴写法：
    - Fprop_7, Xp0, G(Fp1 & Fp2)
    - 单字符布尔算子: & |
    - 释放/弱直到: R W

    注意：这不是完整的语法分析器，但足够用于把 Spot 输出“还原”为自然语言关键字格式。
    """
    if not spot_formula:
        return []

    s = spot_formula.strip()
    tokens = []
    i = 0

    multi_ops = ("<->", "->", "&&", "||")
    single_ops = set("()!&|")
    ltl_unary_binary = set("GFUXRWM")

    while i < len(s):
        ch = s[i]

        if ch.isspace():
            i += 1
            continue

        matched_multi = False
        for op in multi_ops:
            if s.startswith(op, i):
                tokens.append(op)
                i += len(op)
                matched_multi = True
                break
        if matched_multi:
            continue

        if ch in single_ops:
            tokens.append(ch)
            i += 1
            continue

        if ch in ltl_unary_binary:
            tokens.append(ch)
            i += 1
            continue

        # 读取原子命题：一直读到空白或遇到运算符起始字符
        # 注意：不要在这里因为遇到 'G/F/X/...' 这种字母就提前截断，
        # 否则会误拆包含大写字母的原子命题名；
        # 像 Fprop_7 这种紧贴写法会在前面的“算子分支”中先消费掉 'F'。
        start = i
        while i < len(s):
            if s[i].isspace():
                break
            if (
                s.startswith("<->", i)
                or s.startswith("->", i)
                or s.startswith("&&", i)
                or s.startswith("||", i)
            ):
                break
            if s[i] in single_ops:
                break
            i += 1

        if i == start:
            # 理论上不该发生；兜底避免死循环
            tokens.append(ch)
            i += 1
        else:
            tokens.append(s[start:i])

    return [t for t in tokens if t != ""]


def convert_to_spot_format(formula: str) -> str:
    """
    将自然语言格式的 LTL 公式转换为 SPOT 格式

    示例:
        "globally ( not p0 )" -> "G(!p0)"
        "finally ( p0 and next p1 )" -> "F(p0 && X(p1))"
    """
    # 1. 分割成 tokens
    tokens = formula.split()

    # 2. 转换每个 token
    spot_tokens = []
    for token in tokens:
        # 去除可能的括号
        clean_token = token.strip()

        # 检查是否是关键词
        if clean_token in KEYWORD_MAP:
            spot_tokens.append(KEYWORD_MAP[clean_token])
        elif clean_token.startswith("("):
            # 处理 "(p0" 这样的情况
            spot_tokens.append("(")
            inner = clean_token[1:]
            if inner:
                spot_tokens.append(inner)
        elif clean_token.endswith(")"):
            # 处理 "p0)" 这样的情况
            inner = clean_token[:-1]
            if inner:
                spot_tokens.append(inner)
            spot_tokens.append(")")
        else:
            # 普通命题或原子
            spot_tokens.append(clean_token)

    # 3. 合并并清理空格
    spot_formula = " ".join(spot_tokens)

    # 4. 清理多余空格（括号紧挨着）
    spot_formula = re.sub(r"\(\s+", "(", spot_formula)
    spot_formula = re.sub(r"\s+\)", ")", spot_formula)
    spot_formula = re.sub(r"\s+", " ", spot_formula).strip()

    return spot_formula


def convert_from_spot_format(spot_formula: str, prop_map: Optional[dict] = None) -> str:
    """
    将 SPOT 格式的 LTL 公式转换回自然语言格式

    使用正则表达式正确识别多字符运算符（如 ||, &&, ->），避免被拆分成单字符。

    Args:
        spot_formula: SPOT 格式的公式字符串（如 "G(!p0)"）
        prop_map: 命题映射字典，用于还原原始命题名称 {简化名: 原始名}
                 如 {"p0": "pick(apple)", "p1": "drop(apple)"}

    Returns:
        自然语言格式的公式字符串（如 "globally ( not pick(apple) )"）

    示例:
        "G(!p0)" -> "globally ( not p0 )"
        "F(p0 && X(p1))" -> "finally ( p0 and next p1 )"
        "!p0 || p1" -> "not p0 or p1"
    """
    if not spot_formula or spot_formula.strip() == "":
        return ""

    # 1. 反向映射表（支持 Spot 的单字符 '&' / '|', 以及 R/W/M）
    reverse_map = dict(REVERSE_KEYWORD_MAP)

    # 2. 分词：不要用 '\\w+' 这种过强假设（会把 assist(printer,storage) 拆碎），
    # 同时要识别 '&' / '|' 等 Spot 常见输出。
    tokens = _tokenize_spot_formula(spot_formula)

    # 3. 转换每个 token
    natural_tokens = []
    for token in tokens:
        if token in reverse_map:
            natural_tokens.append(reverse_map[token])
        else:
            # 命题（可能是简化后的 p0, p1 等，或实际命题如 pick(apple)）
            # 如果有映射，还原为原始命题
            if prop_map and token in prop_map:
                natural_tokens.append(prop_map[token])
            else:
                natural_tokens.append(token)

    # 4. 合并并做轻量格式化（保持与生成器里“token + 空格”的风格一致）
    result = " ".join(natural_tokens)
    result = re.sub(r"\(\s+", "( ", result)
    result = re.sub(r"\s+\)", " )", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def extract_props_from_formula(formula: str) -> list:
    """
    从公式中提取所有命题

    Args:
        formula: 自然语言或 SPOT 格式的公式

    Returns:
        命题列表（去重）
    """
    # 定义所有关键字
    all_keywords = (
        set(KEYWORD_MAP.keys())
        | set(REVERSE_KEYWORD_MAP.keys())
        | {"IMPLIES", "AND", "OR", "(", ")"}
    )

    tokens = formula.replace("(", " ").replace(")", " ").split()
    props = []

    for token in tokens:
        token = token.strip()
        if (
            token
            and token not in all_keywords
            and token not in ["G", "F", "X", "U", "!", "&&", "||", "->"]
        ):
            # 可能是命题
            props.append(token)

    # 去重并保持顺序
    seen = set()
    unique_props = []
    for p in props:
        if p not in seen:
            seen.add(p)
            unique_props.append(p)

    return unique_props


def _normalize_formula_obj(formula_obj) -> str:
    """将公式对象化简为稳定的 Spot 字符串，并展开 W/R/M。"""
    simplified_f = spot.simplify(formula_obj, ltl2tgba_optimize=False)
    simplified_spot_str = str(simplified_f)

    if (
        "W" in simplified_spot_str
        or "R" in simplified_spot_str
        or "M" in simplified_spot_str
    ):
        simplified_f = spot.unabbreviate(simplified_f, "WRM")
        simplified_spot_str = str(simplified_f)

    return simplified_spot_str


def verify_ltl_formula(
    formula: str, original_props: Optional[dict] = None
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    验证 LTL 公式的可满足性

    注意：此函数会确保简化后的公式不包含 W（Weak Until）、R（Release）、M（Strong Release）操作符，
    只保留 U、X、F、G 等基础操作符。

    Args:
        formula: 自然语言格式的 LTL 公式
        original_props: 原始命题映射 {简化名: 原始名}，用于还原简化后的公式

    Returns:
        (is_satisfiable, simplified_spot_formula, simplified_natural_formula, error_message)
        - is_satisfiable: 是否可满足
        - simplified_spot_formula: 化简后的 SPOT 格式公式（不含 W/R/M）
        - simplified_natural_formula: 化简后的自然语言格式公式（不含 W/R/M）
        - error_message: 错误信息（如果有）
    """
    if not SPOT_AVAILABLE:
        # SPOT 不可用时，原样返回
        return True, formula, formula, "SPOT not available"

    try:
        # 1. 转换为 SPOT 格式
        spot_formula_str = convert_to_spot_format(formula)

        # 2. 解析公式
        f = spot.formula(spot_formula_str)

        # 3. 化简公式（禁用自动优化，避免引入 W/R/M 操作符）
        simplified_spot_str = _normalize_formula_obj(f)

        # 4. 将简化后的公式转换回自然语言格式
        simplified_natural_str = convert_from_spot_format(
            simplified_spot_str, original_props
        )

        # 5. 检查可满足性
        # 使用 translate 将 LTL 转为自动机，is_empty() 检查是否有接受路径
        aut = spot.translate(f)
        is_satisfiable = not aut.is_empty()

        return is_satisfiable, simplified_spot_str, simplified_natural_str, None

    except Exception as e:
        return False, None, None, f"验证错误: {str(e)}"


def are_formulas_equivalent(
    formula1: str,
    formula2: str,
    props1: Optional[dict] = None,
    props2: Optional[dict] = None,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """使用 SPOT 判断两条 LTL 公式是否语义等价。"""
    if not SPOT_AVAILABLE:
        return False, None, None, "SPOT not available"

    try:
        spot_formula_1 = convert_to_spot_format(formula1)
        spot_formula_2 = convert_to_spot_format(formula2)

        f1 = spot.formula(spot_formula_1)
        f2 = spot.formula(spot_formula_2)

        normalized_1 = _normalize_formula_obj(f1)
        normalized_2 = _normalize_formula_obj(f2)

        diff_formula = spot.formula(f"!(({spot_formula_1}) <-> ({spot_formula_2}))")
        is_equivalent = spot.translate(diff_formula).is_empty()

        normalized_natural_1 = convert_from_spot_format(normalized_1, props1)
        normalized_natural_2 = convert_from_spot_format(normalized_2, props2)

        return is_equivalent, normalized_natural_1, normalized_natural_2, None

    except Exception as e:
        return False, None, None, f"等价性验证错误: {str(e)}"


def verify_dataset_entries(entries: list, prop_maps: Optional[list] = None) -> list:
    """
    批量验证数据集条目

    Args:
        entries: 数据集条目列表
        prop_maps: 每个条目的命题映射列表 [{简化名: 原始名}, ...]

    Returns:
        添加验证结果的条目列表
    """
    verified_entries = []

    for idx, entry in enumerate(entries):
        tl_formula = entry.get("tl", "")

        # 获取当前条目的命题映射
        prop_map = prop_maps[idx] if prop_maps and idx < len(prop_maps) else None

        # 验证公式
        is_sat, simplified_spot, simplified_natural, error = verify_ltl_formula(
            tl_formula, prop_map
        )

        # 添加验证结果
        verified_entry = entry.copy()
        verified_entry["spot_verified"] = {
            "is_satisfiable": is_sat,
            "spot_formula": convert_to_spot_format(tl_formula) if not error else None,
            "simplified_spot": simplified_spot,
            "simplified_natural": simplified_natural,
            "error": error,
        }

        verified_entries.append(verified_entry)

    # 打印统计信息
    total = len(verified_entries)
    sat_count = sum(1 for e in verified_entries if e["spot_verified"]["is_satisfiable"])
    unsat_count = total - sat_count

    print("\n=== SPOT 验证结果 ===")
    print(f"总计: {total} 条")
    print(f"可满足 (Satisfiable): {sat_count} 条")
    print(f"不可满足 (Unsatisfiable): {unsat_count} 条")

    if unsat_count > 0:
        print("\n不可满足公式示例:")
        for entry in verified_entries:
            if not entry["spot_verified"]["is_satisfiable"]:
                print(f"  ID {entry['id']}: {entry['tl'][:60]}...")

    return verified_entries


def main():
    """主程序 - 用于测试验证功能"""
    import json

    if not SPOT_AVAILABLE:
        print("SPOT 库未安装，无法运行验证")
        sys.exit(1)

    # 测试转换
    test_formulas = [
        "globally ( not p0 )",
        "finally ( p0 and next p1 )",
        "globally ( p0 implies finally p1 )",
        "not p0 until p1",
        "globally finally p0",
    ]

    print("=== LTL 格式转换测试 ===")
    for formula in test_formulas:
        spot_f = convert_to_spot_format(formula)
        print(f"原始: {formula}")
        print(f"SPOT: {spot_f}")

        # 验证
        is_sat, simplified_spot, simplified_natural, error = verify_ltl_formula(formula)
        if error:
            print(f"错误: {error}")
        else:
            print(f"可满足: {is_sat}")
            print(f"SPOT 化简: {simplified_spot}")
            print(f"自然语言化简: {simplified_natural}")
        print()

    # 如果提供了文件路径，验证整个数据集
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"=== 验证数据集: {filepath} ===")

        try:
            with open(filepath, "r") as f:
                entries = [json.loads(line) for line in f]

            verified = verify_dataset_entries(entries)

            # 保存验证结果
            output_path = filepath.replace(".jsonl", "_verified.jsonl")
            with open(output_path, "w") as f:
                for entry in verified:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"\n验证结果已保存: {output_path}")

        except Exception as e:
            print(f"验证失败: {e}")


if __name__ == "__main__":
    main()

"""
LTL 公式生成器 - AST 元蓝图版

基于抽象语法树 (AST) 思想，通过递归解析逻辑树生成任意复杂的 LTL 公式。
支持模板注册表、元蓝图组装、随机逻辑树生成。
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Callable, Optional

import yaml
import re

from __init__ import parse_object_names, build_actions_dict

# 导入 LTL 验证器
try:
    from ltl_verifier import (
        verify_ltl_formula,
        convert_from_spot_format,
        SPOT_AVAILABLE,
    )
except ImportError:
    try:
        from dataset_generators.ltl_verifier import (
            verify_ltl_formula,
            convert_from_spot_format,
            SPOT_AVAILABLE,
        )
    except ImportError:
        SPOT_AVAILABLE = False
        print("[警告] 无法导入 ltl_verifier 模块，验证功能将不可用")


# ============================================================================
# 1. LTL 模板定义与注册表
# ============================================================================

# 基础时序模板
LTL_TEMPLATES_STATE = [
    # F: Finally (最终) - 活性 (Liveness)
    ("F_NOT", 1, lambda P: ["finally", "(", "not", P[0], ")"]),
    ("F_AND", 2, lambda P: ["finally", "(", P[0], "and", P[1], ")"]),
    ("F_OR", 2, lambda P: ["finally", "(", P[0], "or", P[1], ")"]),
    # G: Globally (全局) - 安全 (Safety)
    ("G_NOT", 1, lambda P: ["globally", "(", "not", P[0], ")"]),
    ("G_AND", 2, lambda P: ["globally", "(", P[0], "and", P[1], ")"]),
    ("G_OR", 2, lambda P: ["globally", "(", P[0], "or", P[1], ")"]),
    # X: Next (下一步)
    ("X", 1, lambda P: ["next", P[0]]),
    # U: Until (直到)
    ("U", 2, lambda P: ["(", P[0], "until", P[1], ")"]),
]

# 扩展时序模板
ADDITIONAL_LTL_TEMPLATES = [
    # 蕴含响应: G(a -> Fb)
    ("G_IMPL_F", 2, lambda P: f"globally ( {P[0]} implies finally {P[1]} )".split()),
    # 互斥: G(!(a & b))
    ("G_NOT_AND", 2, lambda P: f"globally ( not ( {P[0]} and {P[1]} ) )".split()),
    # 延迟3步: G(a -> XXXb)
    (
        "G_IMPL_XXX",
        2,
        lambda P: f"globally ( {P[0]} implies next ( next ( next {P[1]} ) ) )".split(),
    ),
    # 直到+无限经常: a U G(Fb)
    ("U_GF", 2, lambda P: f"{P[0]} until ( globally ( finally {P[1]} ) )".split()),
    # 前置条件: Fb -> (¬b U (a & ¬b))
    (
        "F_B_IMPL_A_BEFORE",
        2,
        lambda P: f"( finally {P[1]} ) implies ( not {P[1]} until ( {P[0]} and not {P[1]} ) )".split(),
    ),
    # 简单蕴含: G(a -> b)
    ("G_IMPL", 2, lambda P: f"globally ( {P[0]} implies {P[1]} )".split()),
    # 全局合取: G(a & b)
    ("G_AND_PAIR", 2, lambda P: f"globally ( {P[0]} and {P[1]} )".split()),
    # 组合安全: Ga & G(b -> ¬c)
    (
        "G_A_AND_G_B_IMPL_NOT_C",
        3,
        lambda P: f"globally {P[0]} and globally ( {P[1]} implies not {P[2]} )".split(),
    ),
    # 复杂蕴含: G(a->Fb) -> G(Fc)
    (
        "G_IMPL_F_IMPL_GF",
        3,
        lambda P: f"globally ( {P[0]} implies finally {P[1]} ) implies globally finally {P[2]}".split(),
    ),
    # 无限经常蕴含: GF(a) -> GF(b)
    (
        "GF_IMPL_GF",
        2,
        lambda P: f"globally finally {P[0]} implies globally finally {P[1]}".split(),
    ),
    # 无限经常选择: GF(a) | GF(b)
    (
        "GF_OR_GF",
        2,
        lambda P: f"globally finally {P[0]} or globally finally {P[1]}".split(),
    ),
    # 最终永久停止: FG(¬a)
    ("FG_NOT", 1, lambda P: f"finally globally not {P[0]}".split()),
    # 前置约束: ¬a U b
    ("NOT_A_UNTIL_B", 2, lambda P: f"not {P[0]} until {P[1]}".split()),
    # 触发后永久禁止: G(a -> XG(¬b))
    (
        "G_A_IMPL_XG_NOT_B",
        2,
        lambda P: f"globally ( {P[0]} implies next globally not {P[1]} )".split(),
    ),
    # 释放算子: b R a
    (
        "A_RELEASES_B",
        2,
        lambda P: f"( {P[1]} until ( {P[1]} and not {P[0]} ) ) or globally {P[1]}".split(),
    ),
    # 两步触发: G(a & Xb -> XXc)
    (
        "TWO_STEP_TRIGGER",
        3,
        lambda P: f"globally ( {P[0]} and next {P[1]} implies next next {P[2]} )".split(),
    ),
    # 下一步响应: G(a -> XFb)
    (
        "NEXT_EVENTUAL",
        2,
        lambda P: f"globally ( {P[0]} implies next finally {P[1]} )".split(),
    ),
    # 全局始终: Ga
    ("G_ALWAYS_A", 1, lambda P: f"globally {P[0]}".split()),
    # 1步内响应: G(a -> (b | Xb))
    (
        "A_IMPL_B_WITHIN_1",
        2,
        lambda P: f"globally ( {P[0]} implies ( {P[1]} or next {P[1]} ) )".split(),
    ),
    # 三选一: G(a|b|c)
    ("G_ONE_OF_ABC", 3, lambda P: f"globally ( {P[0]} or {P[1]} or {P[2]} )".split()),
    # 最终响应: G(a -> Fb)
    (
        "A_IMPL_EVENTUAL_B",
        2,
        lambda P: f"globally ( {P[0]} implies finally {P[1]} )".split(),
    ),
    # 下一步跟随: G(a -> Xb)
    ("NEXT_FOLLOW", 2, lambda P: f"globally ( {P[0]} implies next {P[1]} )".split()),
    # 最终同时: F(a&b)
    ("EVENTUALLY_BOTH", 2, lambda P: f"finally ( {P[0]} and {P[1]} )".split()),
    # 分别最终: Fa & Fb
    ("BOTH_EVENTUAL", 2, lambda P: f"finally {P[0]} and finally {P[1]}".split()),
]

# 复杂任务模板
COMPLEX_LTL_TEMPLATES = [
    # 顺序访问 A -> B -> C
    (
        "SEQ_VISIT_3",
        3,
        lambda P: f"finally ( {P[0]} and finally ( {P[1]} and finally {P[2]} ) )".split(),
    ),
    # 顺序访问 + 全局避免
    (
        "SEQ_VISIT_2_AVOID_1",
        3,
        lambda P: f"( finally ( {P[0]} and finally {P[1]} ) ) and globally ( not {P[2]} )".split(),
    ),
    # 严格优先
    ("PRECEDENCE_STRONG", 2, lambda P: f"not {P[1]} until {P[0]}".split()),
    # 双点巡逻
    (
        "PATROL_2_POINTS",
        2,
        lambda P: f"( globally finally {P[0]} ) and ( globally finally {P[1]} )".split(),
    ),
    # 安全响应
    (
        "SAFE_RESPONSE",
        3,
        lambda P: f"globally ( {P[0]} implies ( not {P[1]} until {P[2]} ) )".split(),
    ),
    # 触发稳定
    (
        "STABILIZATION",
        2,
        lambda P: f"globally ( {P[0]} implies globally {P[1]} )".split(),
    ),
]


def _W(p: str, q: str) -> str:
    """弱直到: p W q ≡ (p U q) or (globally p)"""
    return f"( ( ( {p} ) until ( {q} ) ) or ( globally ( {p} ) ) )"


# 机器人任务专用模板（多机器人协作场景）
ROBOTIC_TASK_TEMPLATES = [
    # 1. 基础无限巡逻: [] <> p (Globally Finally)
    # 语义: 无限次访问/执行 p（最基础的巡逻模式）
    ("GF", 1, lambda P: f"globally finally {P[0]}".split()),
    # 2. 嵌套巡逻链: [] <> (p1 && <> p2)
    # 语义: 无限次按顺序访问 p1, p2（p1发生后，最终p2必须发生）
    (
        "GF_NESTED_2",
        2,
        lambda P: f"globally finally ( {P[0]} and finally {P[1]} )".split(),
    ),
    # 3. 任务触发与保持: [](p1 -> X(p1 U p2))
    # 语义: 一旦发生p1，下一步开始必须保持p1直到p2发生（避障/运输核心模式）
    (
        "TRIGGER_KEEP",
        2,
        lambda P: f"globally ( {P[0]} implies next ( {P[0]} until {P[1]} ) )".split(),
    ),
    # 4. 协同到达: <> (p1 && p2)
    # 语义: 最终p1和p2必须同时成立（多机器人同时到达某区域）
    ("SYNC_ARRIVAL", 2, lambda P: f"finally ( {P[0]} and {P[1]} )".split()),
    # 5. 访问并离开: [] <> (p1 && X(<> !p1))
    # 语义: 无限次访问p1，并且每次访问后必须离开（不能一直停留）
    (
        "VISIT_AND_LEAVE",
        1,
        lambda P: f"globally finally ( {P[0]} and next ( finally not {P[0]} ) )".split(),
    ),
    # 6. 强顺序约束（多重禁止）: (!p1 U p3) && (!p2 U p3)
    # 语义: 在完成p3之前，绝对禁止p1和p2发生（严格前置条件）
    (
        "MULTI_PRECEDENCE",
        3,
        lambda P: f"( not {P[0]} until {P[2]} ) and ( not {P[1]} until {P[2]} )".split(),
    ),
    # 7. 无限次顺序巡逻（任意长度链）: [] <> (p1 && <> (p2 && <> p3))
    # 语义: 无限次按p1->p2->p3顺序循环访问（深度嵌套巡逻）
    (
        "PATROL_CHAIN_3",
        3,
        lambda P: f"globally finally ( {P[0]} and finally ( {P[1]} and finally {P[2]} ) )".split(),
    ),
]

# Dwyer 规范模式模板
DWYER_TEMPLATES = [
    # Absence 模式 - P 永不发生
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
    # Existence 模式 - P 最终发生
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
    # Bounded Existence - P 最多发生2次
    (
        "BOUNDED_2_GLOBAL",
        1,
        lambda P: _W(
            f"not {P[0]}", _W(P[0], _W(f"not {P[0]}", _W(P[0], f"globally not {P[0]}")))
        ).split(),
    ),
    # Universality 模式 - P 始终成立
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
    # Precedence 模式 - S 在 P 之前
    ("PRECEDENCE_GLOBAL", 2, lambda P: _W(f"not {P[0]}", P[1]).split()),
    # Response 模式 - S 响应 P
    (
        "RESPONSE_GLOBAL",
        2,
        lambda P: f"globally ( {P[0]} implies finally {P[1]} )".split(),
    ),
    (
        "RESPONSE_AFTER_Q",
        3,
        lambda P: f"globally ( {P[2]} implies globally ( {P[0]} implies finally {P[1]} ) )".split(),
    ),
    # Chain 链式模式
    (
        "PRECEDENCE_CHAIN_21_GLOBAL",
        3,
        lambda P: f"( finally {P[0]} ) implies ( ( not {P[0]} ) until ( {P[1]} and not {P[0]} and next ( ( not {P[0]} ) until {P[2]} ) ) )".split(),
    ),
    (
        "RESPONSE_CHAIN_12_GLOBAL",
        3,
        lambda P: f"globally ( {P[0]} implies finally ( {P[1]} and next finally {P[2]} ) )".split(),
    ),
]

# 合并所有模板
ALL_TEMPLATES = (
    LTL_TEMPLATES_STATE
    + ADDITIONAL_LTL_TEMPLATES
    + COMPLEX_LTL_TEMPLATES
    + ROBOTIC_TASK_TEMPLATES
    + DWYER_TEMPLATES
)

# ============================================================================
# 2. 模板注册表与统一调用接口
# ============================================================================

# 模板数据库: {模板名: (参数数量, 公式生成函数)}
TEMPLATE_DB: Dict[str, Tuple[int, Callable]] = {
    name: (arity, func) for name, arity, func in ALL_TEMPLATES
}

# 逻辑连接词集合
LOGICAL_CONNECTIVES = {"AND", "OR", "IMPLIES", "NOT"}


def apply_ltl(template_name: str, *args) -> str:
    """
    统一引擎：从模板库调用模板，生成 LTL 公式字符串

    Args:
        template_name: 模板名称（如 "G_IMPL_F", "SEQ_VISIT_3"）
        *args: 模板参数（原子命题字符串）

    Returns:
        LTL 公式字符串

    Raises:
        ValueError: 模板不存在或参数数量不匹配
    """
    if template_name not in TEMPLATE_DB:
        raise ValueError(f"未知 LTL 模板: {template_name}")

    arity, func = TEMPLATE_DB[template_name]
    if len(args) != arity:
        raise ValueError(
            f"模板 [{template_name}] 期望 {arity} 个参数，但接收到 {len(args)} 个"
        )

    # 执行模板函数
    result = func(list(args))

    # 兼容处理：模板可能返回 list 或已拼接的字符串
    if isinstance(result, list):
        return " ".join(result)
    return str(result)


# ============================================================================
# 3. 元蓝图生成器 (核心 - AST 递归解析)
# ============================================================================

# 逻辑树节点类型
LogicNode = Union[
    str, Tuple
]  # 叶子节点是字符串（命题ID），内部节点是元组 (op, *children)


def evaluate_logic_tree(node: LogicNode, mapper: Callable[[str], str]) -> str:
    """
    递归求值逻辑树，生成 LTL 公式

    Args:
        node: 逻辑树节点
        mapper: 命题映射函数（prop_id -> atom_string）

    Returns:
        LTL 公式字符串

    示例:
        >>> tree = ("AND", ("SEQ_VISIT_3", "p1", "p2", "p3"), ("G_NOT", "p4"))
        >>> evaluate_logic_tree(tree, lambda x: x)
        '( finally ( p1 and finally ( p2 and finally p3 ) ) ) and ( globally not p4 )'
    """
    # 1. 触底：如果是纯字符串（命题占位符），直接映射
    if isinstance(node, str):
        return mapper(node)

    # 2. 解析节点：第一个元素是操作符，后面是子节点
    op = node[0]
    children = node[1:]

    # 3. 处理基础逻辑连接词
    if op == "AND":
        # (AND, child1, child2, ...) -> ( child1 ) and ( child2 ) ...
        return " and ".join([f"( {evaluate_logic_tree(c, mapper)} )" for c in children])

    if op == "OR":
        # (OR, child1, child2, ...) -> ( child1 ) or ( child2 ) ...
        return " or ".join([f"( {evaluate_logic_tree(c, mapper)} )" for c in children])

    if op == "IMPLIES":
        # (IMPLIES, premise, conclusion) -> ( premise ) implies ( conclusion )
        if len(children) != 2:
            raise ValueError("IMPLIES 需要恰好 2 个子节点")
        return f"( {evaluate_logic_tree(children[0], mapper)} ) implies ( {evaluate_logic_tree(children[1], mapper)} )"

    if op == "NOT":
        # (NOT, child) -> not ( child )
        if len(children) != 1:
            raise ValueError("NOT 需要恰好 1 个子节点")
        return f"not ( {evaluate_logic_tree(children[0], mapper)} )"

    # 4. 处理 LTL 模板（调用 TEMPLATE_DB）
    # 先递归求值所有子节点
    evaluated_children = [evaluate_logic_tree(c, mapper) for c in children]
    return apply_ltl(op, *evaluated_children)


def build_entry_from_tree(
    idx: int,
    logic_tree: LogicNode,
    props: Dict[str, Dict],
    use_simplified: bool = False,
    simplified_tl: Optional[str] = None,
    simplified_masked_tl: Optional[str] = None,
) -> Dict:
    """
    从逻辑树构建数据集条目

    Args:
        idx: 条目ID
        logic_tree: 逻辑树结构
        props: 命题信息字典 {prop_id: prop_info}
        use_simplified: 是否使用简化后的公式
        simplified_tl: 简化后的 tl 公式（使用原始命题）
        simplified_masked_tl: 简化后的 masked_tl 公式（使用 prop_id）

    Returns:
        {"id": idx, "tl": ..., "masked_tl": ..., "props": [...]}
    """

    # 提取使用的所有 prop_id
    def extract_props(node):
        if isinstance(node, str):
            return [node]
        result = []
        for child in node[1:]:
            result.extend(extract_props(child))
        return result

    used_props = list(dict.fromkeys(extract_props(logic_tree)))  # 去重保持顺序

    # 输出时让 props 顺序稳定：按 prop 编号升序
    def _prop_key(p: str) -> int:
        m = re.match(r"^prop_(\d+)$", p)
        return int(m.group(1)) if m else 10**9

    used_props = sorted(used_props, key=_prop_key)

    # 如果使用简化公式
    if use_simplified and simplified_tl and simplified_masked_tl:
        # 注意：Spot 简化可能会消去子公式，导致逻辑树里出现过的 prop 在简化后不再出现。
        # 这里以简化后的 masked_tl 为准重新提取一次 props，保证字段一致。
        extracted = re.findall(r"prop_\d+", simplified_masked_tl)
        used_props_simplified = sorted(list(dict.fromkeys(extracted)), key=_prop_key)
        return {
            "id": idx,
            "tl": simplified_tl,
            "masked_tl": simplified_masked_tl,
            "props": used_props_simplified or used_props,
            "tree_structure": str(logic_tree),
            "simplified": True,
        }

    # 构建映射器
    def real_mapper(prop_id: str) -> str:
        """将 prop_id 映射为实际原子命题"""
        if prop_id not in props:
            return prop_id
        info = props[prop_id]
        args = ",".join(info["args_canon"])
        return f"{info['action_canon']}({args})" if args else info["action_canon"]

    def masked_mapper(prop_id: str) -> str:
        """保持 prop_id 不变（用于 masked_tl）"""
        return prop_id

    # 生成 LTL 公式
    tl = evaluate_logic_tree(logic_tree, real_mapper)
    masked_tl = evaluate_logic_tree(logic_tree, masked_mapper)

    return {
        "id": idx,
        "tl": tl,
        "masked_tl": masked_tl,
        "props": used_props,
        "tree_structure": str(logic_tree),
        "simplified": False,
    }


# ============================================================================
# 4. 随机逻辑树生成器
# ============================================================================


def generate_random_logic_tree(
    available_props: List[str], max_depth: int = 3, current_depth: int = 0
) -> LogicNode:
    """
    随机生成逻辑树

    策略：
    - 叶子节点：直接使用命题ID
    - 内部节点：随机选择模板或逻辑连接词
    - 深度控制：达到一定深度后强制使用叶子节点

    Args:
        available_props: 可用的命题ID列表
        max_depth: 最大递归深度
        current_depth: 当前深度

    Returns:
        逻辑树节点
    """
    # 终止条件：达到最大深度或随机决定停止
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
        return random.choice(available_props)

    # 随机选择节点类型
    node_type = random.choice(
        [
            "template",  # 使用 LTL 模板
            "logical_and",  # 使用 AND 连接
            "logical_or",  # 使用 OR 连接
            "logical_implies",  # 使用 IMPLIES
        ]
    )

    if node_type == "template":
        # 从模板库随机选择模板
        template_name = random.choice(list(TEMPLATE_DB.keys()))
        arity, _ = TEMPLATE_DB[template_name]

        # 为模板参数生成子树或直接使用命题
        children = []
        for _ in range(arity):
            if random.random() < 0.5 and current_depth < max_depth:
                # 递归生成子树
                child = generate_random_logic_tree(
                    available_props, max_depth, current_depth + 1
                )
            else:
                # 使用叶子命题
                child = random.choice(available_props)
            children.append(child)

        return (template_name, *children)

    elif node_type == "logical_and":
        # 生成 AND 节点，连接 2-4 个子树
        num_children = random.randint(2, min(4, len(available_props)))
        children = [
            generate_random_logic_tree(available_props, max_depth, current_depth + 1)
            for _ in range(num_children)
        ]
        return ("AND", *children)

    elif node_type == "logical_or":
        # 生成 OR 节点，连接 2-3 个子树
        num_children = random.randint(2, min(3, len(available_props)))
        children = [
            generate_random_logic_tree(available_props, max_depth, current_depth + 1)
            for _ in range(num_children)
        ]
        return ("OR", *children)

    elif node_type == "logical_implies":
        # 生成 IMPLIES 节点 (premise -> conclusion)
        premise = generate_random_logic_tree(
            available_props, max_depth, current_depth + 1
        )
        conclusion = generate_random_logic_tree(
            available_props, max_depth, current_depth + 1
        )
        return ("IMPLIES", premise, conclusion)

    # 默认回退到叶子节点
    return random.choice(available_props)


def generate_mission_tree(
    available_props: List[str], mission_type: str = "hybrid"
) -> LogicNode:
    """
    生成特定类型的使命逻辑树（根据可用命题数量自适应）
    """
    props = available_props[:]
    random.shuffle(props)
    n = len(props)

    if mission_type == "sequence":
        # 根据命题数量选择顺序模板
        if n >= 3:
            return ("SEQ_VISIT_3", props[0], props[1], props[2])
        elif n >= 2:
            return ("G_IMPL_F", props[0], props[1])  # 回退到简单响应
        else:
            return ("GF", props[0])  # 回退到基础巡逻

    elif mission_type == "patrol":
        return ("GF", props[0])

    elif mission_type == "multi_robot_patrol":
        if n >= 2:
            return ("GF_NESTED_2", props[0], props[1])
        else:
            return ("GF", props[0])

    elif mission_type == "transport_mission":
        if n >= 2:
            return ("TRIGGER_KEEP", props[0], props[1])
        else:
            return ("G_ALWAYS_A", props[0])

    elif mission_type == "sync_arrival":
        if n >= 2:
            return ("SYNC_ARRIVAL", props[0], props[1])
        else:
            return ("EXISTENCE_GLOBAL", props[0])

    elif mission_type == "visit_and_leave":
        return ("VISIT_AND_LEAVE", props[0])

    elif mission_type == "multi_precedence":
        if n >= 3:
            return ("MULTI_PRECEDENCE", props[0], props[1], props[2])
        elif n >= 2:
            return ("NOT_A_UNTIL_B", props[0], props[1])  # 回退到简单优先
        else:
            return ("G_NOT", props[0])

    elif mission_type == "patrol_chain":
        if n >= 3:
            return ("PATROL_CHAIN_3", props[0], props[1], props[2])
        elif n >= 2:
            return ("GF_NESTED_2", props[0], props[1])
        else:
            return ("GF", props[0])

    elif mission_type == "safety_liveness":
        if n >= 3:
            return ("AND", ("G_NOT", props[0]), ("GF", props[1]), ("GF", props[2]))
        elif n >= 2:
            return ("AND", ("G_NOT", props[0]), ("GF", props[1]))
        else:
            return ("G_NOT", props[0])

    elif mission_type == "response":
        if n >= 2:
            return ("G_IMPL_F", props[0], props[1])
        else:
            return ("F_NOT", props[0])

    elif mission_type == "hybrid":
        # 混合：根据可用命题数量动态组合
        if n >= 6:
            return (
                "AND",
                ("PATROL_CHAIN_3", props[0], props[1], props[2]),
                ("SYNC_ARRIVAL", props[3], props[4]),
                ("MULTI_PRECEDENCE", props[0], props[1], props[5]),
            )
        elif n >= 4:
            return (
                "AND",
                ("GF_NESTED_2", props[0], props[1]),
                ("SYNC_ARRIVAL", props[2], props[3]),
            )
        elif n >= 2:
            return ("AND", ("GF", props[0]), ("TRIGGER_KEEP", props[0], props[1]))
        else:
            return ("GF", props[0])

    else:
        return generate_random_logic_tree(props, max_depth=3)


# ============================================================================
# 5. 场景加载与参数采样
# ============================================================================


def load_scenario(
    scenario_name: str, yaml_path: str = "dataset_generators/scenarios.yaml"
) -> Tuple[Dict, Dict, Dict, List[str], Dict]:
    """加载场景配置"""
    cfg_all = yaml.safe_load(Path(yaml_path).read_text())
    if scenario_name not in cfg_all:
        raise ValueError(f"场景 '{scenario_name}' 未在 {yaml_path} 中找到")

    cfg = cfg_all[scenario_name]
    object_dict = parse_object_names("dataset_generators/object_names.txt")
    actions_dict = {
        k: v for k, v in build_actions_dict().items() if k in cfg["actions"]
    }

    return cfg, object_dict, actions_dict, cfg.get("locations", []), cfg["actions"]


def _sample_argument(kind: str, objects: Dict, locs: List[str]) -> Tuple[str, str]:
    """采样参数值，返回 (canonical_name, reference_name)"""
    if kind == "item":
        key = random.choice(list(objects.keys()))
        canon = key.replace(" ", "_")
        ref = random.choice(objects[key]).replace("_", " ")
    elif kind == "location":
        canon = random.choice(locs)
        ref = canon.replace("_", " ")
    elif kind == "ego":
        canon, ref = "ego", "yourself"
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
        pos = random.choice(locs)
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
        canon = f"{dir}_{num}_{road}"
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
        if random.random() < 0.5:
            mod1 = random.choice(["injured", "safe", "unsafe"])
            canon = (
                mod1 + "_" + random.choice(["person", "civilian", "victim", "rescuer"])
            )
            ref = canon.replace("_", " ")
        else:
            canon = random.choice(
                ["gas_leak", "unstable_beam", "fire_source", "debris", "flood"]
            )
            ref = canon.replace("_", " ")
    elif kind == "color":
        canon = random.choice(["red", "yellow", "green"])
        ref = canon
    else:
        raise ValueError(f"未知参数类型: '{kind}'")

    return canon, ref


# ============================================================================
# 6. 数据集构建
# ============================================================================


# Token limit for LTL formulas
MAX_LTL_TOKENS = 80


def count_ltl_tokens(ltl_formula: str) -> int:
    """计算LTL公式的token数量"""
    # LTL公式以空格分隔，每个token就是一个token
    tokens = ltl_formula.split()
    return len(tokens)


def build_dataset_entries(
    object_dict: Dict,
    actions_dict: Dict,
    locations: List[str],
    actions_cfg: Dict,
    num_entries: int,
    max_props: int = 10,
    verify: bool = True,
) -> List[Dict]:
    """
    构建数据集条目 - 使用 AST 元蓝图
    特性：
    1. 命题数量 2-10 均匀分布
    2. 确保每个模板至少被使用一次
    3. 使命类型多样化
    4. 实时验证 LTL 公式，过滤不可满足的公式，保存简化公式
    5. LTL公式不超过80个token

    Args:
        object_dict: 对象字典
        actions_dict: 动作字典
        locations: 位置列表
        actions_cfg: 动作配置
        num_entries: 需要生成的条目数
        max_props: 最大命题数（默认10）
        verify: 是否启用 SPOT 验证（默认True）

    Returns:
        数据集条目列表
    """
    dataset = []
    # 重要：每条样本内部的 prop 编号从 1 开始连续增长（prop_1..prop_k），
    # 不要随机抽样 prop_8/prop_10 这种跳号，避免后续还原/对齐变复杂。

    # 使命类型分布（包含机器人专用任务模式）
    mission_types = [
        "sequence",
        "patrol",
        "safety_liveness",
        "response",
        "hybrid",
        "random",
        "multi_robot_patrol",
        "transport_mission",
        "sync_arrival",
        "visit_and_leave",
        "multi_precedence",
        "patrol_chain",
    ]
    mission_weights = [10, 10, 10, 10, 15, 15, 8, 8, 7, 5, 4, 8]

    # 获取所有模板名称，用于确保全覆盖
    all_template_names = list(TEMPLATE_DB.keys())
    template_usage = {name: 0 for name in all_template_names}  # 使用计数

    # 验证统计
    verified_count = 0
    rejected_count = 0
    token_limit_rejected_count = 0  # 因token数量超限被拒绝的计数

    entry_idx = 0
    while len(dataset) < num_entries:
        # ===== 1. 采样命题数量（2-10均匀分布） =====
        # 策略：循环使用 2-10 保证均匀
        num_props = (entry_idx % 9) + 2  # 2,3,4,5,6,7,8,9,10 循环
        num_props = min(num_props, max_props)  # 不超过最大值
        # 每条样本固定使用 prop_1..prop_k（k=num_props），保证从 1 开始且连续
        want_labels = [f"prop_{i + 1}" for i in range(num_props)]

        # ===== 2. 为每个命题生成动作和参数 =====
        props = {}
        for lbl in want_labels:
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

        # ===== 3. 生成逻辑树（优先使用未覆盖的模板） =====
        # 检查哪些模板还未使用
        unused_templates = [t for t in all_template_names if template_usage[t] == 0]

        if unused_templates and random.random() < 0.7:
            # 70%概率优先使用未使用的模板
            template_name = random.choice(unused_templates)
            arity, _ = TEMPLATE_DB[template_name]

            # 根据模板参数数量选择相应数量的命题
            if arity <= len(want_labels):
                selected_props = want_labels[:arity]
            else:
                # 如果命题不够，重复使用
                selected_props = [random.choice(want_labels) for _ in range(arity)]

            # 构建简单逻辑树
            logic_tree = (template_name, *selected_props)
            mission_type = f"template_{template_name}"
        else:
            # 使用使命模式或随机生成
            mission_type = random.choices(mission_types, weights=mission_weights)[0]

            if mission_type == "random":
                logic_tree = generate_random_logic_tree(
                    want_labels, max_depth=random.randint(1, 4)
                )
            else:
                logic_tree = generate_mission_tree(want_labels, mission_type)

        # ===== 4. 从逻辑树生成条目（先使用原始公式） =====
        entry = build_entry_from_tree(entry_idx, logic_tree, props)
        entry["mission_type"] = mission_type
        entry["num_props"] = num_props

        # ===== 5. 检查LTL公式token数量 =====
        ltl_token_count = count_ltl_tokens(entry["tl"])
        if ltl_token_count > MAX_LTL_TOKENS:
            # 超过80个token，重新生成
            token_limit_rejected_count += 1
            entry_idx += 1
            continue

        # ===== 6. 验证公式（如果启用） =====
        if verify and SPOT_AVAILABLE:
            # 构建命题映射：prop_id -> 实际命题字符串
            # 用于将简化后的公式中的简化命题名还原为实际命题
            prop_map = {}
            for lbl in want_labels:
                if lbl in props:
                    info = props[lbl]
                    args = ",".join(info["args_canon"])
                    prop_map[lbl] = (
                        f"{info['action_canon']}({args})"
                        if args
                        else info["action_canon"]
                    )

            # 验证公式：只验证 masked_tl（避免把带括号/逗号的真实动作名喂给 Spot 解析器）
            # 返回的 simplified_spot 是 Spot 语法（可能包含 & | R 等），需要再“还原”为自然语言关键字格式。
            is_sat, simplified_spot, simplified_natural, error = verify_ltl_formula(
                entry["masked_tl"], prop_map
            )

            if error:
                # 验证出错，跳过此条目
                rejected_count += 1
                entry_idx += 1
                continue

            if not is_sat:
                # 不可满足，丢弃此条目
                rejected_count += 1
                entry_idx += 1
                continue

            # 可满足，使用简化后的公式重建条目
            verified_count += 1

            # 生成“简化后 masked_tl”的自然语言关键字格式（保留 prop_id）
            if simplified_spot:
                simplified_masked_natural = convert_from_spot_format(
                    simplified_spot, prop_map=None
                )
            else:
                simplified_masked_natural = entry["masked_tl"]

            # 由 masked 版本派生 tl：直接把 prop_id 替换为真实原子命题字符串
            simplified_tl_natural = simplified_masked_natural
            for prop_id, real_prop in prop_map.items():
                simplified_tl_natural = simplified_tl_natural.replace(
                    prop_id, real_prop
                )

            # 将简化后的公式替换原始公式
            entry = build_entry_from_tree(
                entry_idx,
                logic_tree,
                props,
                use_simplified=True,
                simplified_tl=simplified_tl_natural,
                simplified_masked_tl=simplified_masked_natural,
            )
            entry["mission_type"] = mission_type
            entry["num_props"] = num_props
            entry["spot_verified"] = True
            entry["spot_simplified"] = simplified_spot
            entry["original_tl"] = evaluate_logic_tree(
                logic_tree, lambda pid: prop_map.get(pid, pid)
            )
            entry["original_masked_tl"] = evaluate_logic_tree(logic_tree, lambda x: x)

        dataset.append(entry)

        # ===== 7. 更新模板使用计数 =====
        def count_template_usage(node):
            """递归统计逻辑树中使用的模板"""
            if isinstance(node, str):
                return
            op = node[0]
            if op not in LOGICAL_CONNECTIVES and op in template_usage:
                template_usage[op] += 1
            for child in node[1:]:
                count_template_usage(child)

        count_template_usage(logic_tree)
        entry_idx += 1

    # 打印统计信息
    print(
        f"[LTL-gen] 模板覆盖情况: {len([t for t in template_usage if template_usage[t] > 0])}/{len(all_template_names)}"
    )
    unused = [t for t in template_usage if template_usage[t] == 0]
    if unused:
        print(f"[LTL-gen] 未使用的模板: {unused}")

    if verify and SPOT_AVAILABLE:
        print(f"[LTL-gen] 验证统计: 通过 {verified_count} 条, 拒绝 {rejected_count} 条")

    print(f"[LTL-gen] Token限制: 超过80个token被拒绝 {token_limit_rejected_count} 条")

    return dataset


# ============================================================================
# 7. 主程序
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="LTL 公式生成器 - AST 元蓝图版")
    parser.add_argument("-s", "--scenario", default="warehouse", help="场景名称")
    parser.add_argument(
        "-n", "--num_entries", type=int, default=3000, help="生成条目数"
    )
    parser.add_argument(
        "-m", "--max_props", type=int, default=10, help="每个条目的最大命题数"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/data/ljr/llm_experiments/ljr_ltl_datasetV1/rawltl.jsonl",
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--show_templates", action="store_true", help="显示所有可用模板"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="启用 SPOT 验证，过滤不可满足公式（默认启用）",
    )
    parser.add_argument(
        "--no-verify", action="store_false", dest="verify", help="禁用 SPOT 验证"
    )
    args = parser.parse_args()

    # 显示模板列表（如果请求）
    if args.show_templates:
        print("=" * 60)
        print("可用 LTL 模板列表:")
        print("=" * 60)
        for name, (arity, _) in sorted(TEMPLATE_DB.items()):
            print(f"  {name:30s} (参数数: {arity})")
        print("=" * 60)
        print(f"总计: {len(TEMPLATE_DB)} 个模板")
        return

    # 加载场景
    cfg, obj_dict, act_dict, locs, act_cfg = load_scenario(args.scenario)

    # 生成数据集
    entries = build_dataset_entries(
        obj_dict,
        act_dict,
        locs,
        act_cfg,
        num_entries=args.num_entries,
        max_props=args.max_props,
        verify=args.verify,
    )

    # 保存输出
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[LTL-gen] 场景: {args.scenario}")
    print(f"[LTL-gen] 生成 {args.num_entries} 条 LTL 公式 -> {out.as_posix()}")

    # # 显示示例
    # print("\n生成示例:")
    # for i, entry in enumerate(entries[:3]):
    #     print(f"\n--- 条目 {i+1} [{entry.get('mission_type', 'unknown')}] ---")
    #     print(f"TL:        {entry['tl'][:100]}...")
    #     print(f"Masked TL: {entry['masked_tl'][:100]}...")


if __name__ == "__main__":
    main()

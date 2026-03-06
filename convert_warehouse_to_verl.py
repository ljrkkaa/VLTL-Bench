#!/usr/bin/env python3
"""
将 warehouse_nl.jsonl 转换为 VERL parquet 格式，用于策略训练。

VERL 所需的 5 个字段:
- data_source: 数据源名称
- prompt: 符合 HF chat_template 格式的消息列表
- ability: 任务能力标签
- reward_model: 奖励模型配置 (包含 ground_truth 用于评估)
- extra_info: 元数据

输入文件格式 (warehouse_nl.jsonl):
- id: 记录ID
- original_id: 原始ID
- tl: LTL公式 (ground truth)
- masked_tl: 带占位符的LTL公式
- nl: 自然语言描述
- sentence_idx: 句子索引
"""

import json
from pathlib import Path
from datetime import datetime

# 统一的数据源标识符
DATA_SOURCE = "vltl-bench-warehouse"
# 任务能力标签
ABILITY = "logic"


def get_parquet_writer():
    """
    自动检测可用的 parquet 写入后端，按优先级尝试:
    1. pyarrow (优先)
    2. datasets (HuggingFace)
    3. pandas
    返回 (backend_name, backend_module) 元组。
    """
    try:
        import pyarrow
        import pyarrow.parquet

        return "pyarrow", pyarrow
    except ImportError:
        pass

    try:
        import datasets

        return "datasets", datasets
    except ImportError:
        pass

    try:
        import pandas as pd

        return "pandas", pd
    except ImportError:
        pass

    raise ImportError(
        "未找到可用的 parquet 写入库，请安装: pyarrow, datasets 或 pandas"
    )


def write_parquet(rows: list, output_path: Path, backend_name: str, backend_module):
    """
    使用指定后端将 Python 列表写入 parquet 文件。
    """
    if backend_name == "datasets":
        ds = backend_module.Dataset.from_list(rows)
        ds.to_parquet(str(output_path))
    elif backend_name == "pyarrow":
        import pyarrow.parquet as pq

        table = backend_module.Table.from_pylist(rows)
        pq.write_table(table, str(output_path))
    elif backend_name == "pandas":
        df = backend_module.DataFrame(rows)
        df.to_parquet(str(output_path), index=False)


def convert_jsonl(input_path: Path, split: str = "test") -> tuple[list, int, list, list]:
    """
    将 warehouse_nl.jsonl 转换为 VERL 格式行。
    
    Args:
        input_path: 输入文件路径
        split: "train" 或 "test"，用于筛选数据
        
    Returns:
        (rows, total_lines, skipped, original_ids) 元组
        - rows: VERL 格式的行列表
        - total_lines: 总行数
        - skipped: 跳过的行信息列表
        - original_ids: 每行对应的 original_id 列表（用于分割）
    """
    rows = []
    total_lines = 0
    skipped = []
    original_ids = []  # 用于分割
    global_index = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()

            # 跳过空行
            if not line:
                skipped.append((input_path.name, line_num, "empty line"))
                continue

            # 解析 JSON
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                skipped.append((input_path.name, line_num, f"JSON parse error: {e}"))
                continue

            # 验证 nl 字段 (作为 question/prompt)
            nl = record.get("nl")
            if nl is None:
                skipped.append((input_path.name, line_num, "missing 'nl'"))
                continue
            if not isinstance(nl, str) or not nl.strip():
                skipped.append((input_path.name, line_num, "invalid 'nl'"))
                continue

            # 验证 tl 字段 (完整 LTL 公式)
            tl = record.get("tl")
            if tl is None:
                skipped.append((input_path.name, line_num, "missing 'tl'"))
                continue
            if not isinstance(tl, str) or not tl.strip():
                skipped.append((input_path.name, line_num, "invalid 'tl'"))
                continue

            # 验证 masked_tl 字段 (带占位符的LTL公式，作为训练目标)
            masked_tl = record.get("masked_tl")
            if masked_tl is None:
                skipped.append((input_path.name, line_num, "missing 'masked_tl'"))
                continue
            if not isinstance(masked_tl, str) or not masked_tl.strip():
                skipped.append((input_path.name, line_num, "invalid 'masked_tl'"))
                continue

            # 获取其他可选字段
            record_id = record.get("id", "")
            original_id = record.get("original_id", 0)
            sentence_idx = record.get("sentence_idx", 0)

            # 构建 VERL 格式的行
            # 注意: ground_truth 使用 masked_tl（带占位符的LTL公式），因为训练时模型需要预测带占位符的公式
            verl_row = {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": nl}],
                "ability": ABILITY,
                "reward_model": {"style": "rule", "ground_truth": masked_tl},
                "extra_info": {
                    "index": global_index,
                    "sentence_idx": sentence_idx
                },
            }

            rows.append(verl_row)
            original_ids.append(original_id)
            global_index += 1

    return rows, total_lines, skipped, original_ids


def main():
    """
    主函数: 将 warehouse_nl.jsonl 转换为 VERL parquet 格式，
    按 original_id 分割为 train (80%) 和 test (20%) 两部分。
    """
    base_dir = Path(__file__).parent
    input_path = base_dir / "new_generated_datasets" / "warehouse_nl.jsonl"
    output_dir = base_dir / "new_generated_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检测可用的 parquet 写入后端
    backend_name, backend_module = get_parquet_writer()

    log_path = output_dir / "conversion.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"VERL Parquet Conversion Log - {datetime.now().isoformat()}\n")
        log_file.write(f"Input: {input_path}\n")
        log_file.write(f"Backend: {backend_name}\n")
        log_file.write("=" * 60 + "\n\n")

        if not input_path.exists():
            log_file.write(f"ERROR: Input file not found: {input_path}\n")
            print(f"ERROR: Input file not found: {input_path}")
            return

        log_file.write(f"Processing: {input_path}\n")

        # 读取所有数据（同时获取 original_ids 用于分割）
        all_rows, total, skipped, original_ids_list = convert_jsonl(input_path, split="all")

        # 按 original_id 分割数据
        # 收集所有唯一的 original_id
        original_ids = set(original_ids_list)
        original_ids = sorted(list(original_ids))
        total_unique = len(original_ids)
        
        # 80% train, 20% test (按 original_id 分割)
        train_ratio = 0.8
        split_idx = int(total_unique * train_ratio)
        train_original_ids = set(original_ids[:split_idx])
        test_original_ids = set(original_ids[split_idx:])
        
        train_rows = []
        test_rows = []
        
        for row, orig_id in zip(all_rows, original_ids_list):
            if orig_id in train_original_ids:
                train_rows.append(row)
            else:
                test_rows.append(row)
        
        # 重新编号 index
        for i, row in enumerate(train_rows):
            row["extra_info"]["index"] = i
        for i, row in enumerate(test_rows):
            row["extra_info"]["index"] = i

        # 写入 train parquet 文件
        train_output_path = output_dir / "warehouse_nl_train.parquet"
        write_parquet(train_rows, train_output_path, backend_name, backend_module)

        # 写入 test parquet 文件
        test_output_path = output_dir / "warehouse_nl_test.parquet"
        write_parquet(test_rows, test_output_path, backend_name, backend_module)

        log_file.write(f"\n  Total lines: {total}\n")
        log_file.write(f"  Total unique original_ids: {total_unique}\n")
        log_file.write(f"  Train original_ids: {len(train_original_ids)}\n")
        log_file.write(f"  Test original_ids: {len(test_original_ids)}\n")
        log_file.write(f"  Written train rows: {len(train_rows)}\n")
        log_file.write(f"  Written test rows: {len(test_rows)}\n")
        log_file.write(f"  Skipped: {len(skipped)}\n")

        # 记录跳过的行
        if skipped:
            log_file.write("  Skipped details:\n")
            for fname, line_num, reason in skipped[:20]:
                log_file.write(f"    {fname}:{line_num} - {reason}\n")
            if len(skipped) > 20:
                log_file.write(f"    ... and {len(skipped) - 20} more\n")

        log_file.write(f"\n  Train output: {train_output_path}\n")
        log_file.write(f"  Test output: {test_output_path}\n")
        log_file.write("=" * 60 + "\n")
        log_file.write(f"TOTAL: Train {len(train_rows)} / Test {len(test_rows)} / Skipped {len(skipped)}\n")

        print(f"Converted: {len(train_rows)} train rows -> {train_output_path}")
        print(f"Converted: {len(test_rows)} test rows -> {test_output_path}")
        print(f"Backend used: {backend_name}")
        print(f"Log: {log_path}")


if __name__ == "__main__":
    main()

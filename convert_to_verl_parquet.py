#!/usr/bin/env python3
"""
将 VLTL JSONL 文件转换为 VERL parquet 格式，用于策略训练。

VERL 所需的 5 个字段:
- data_source: 数据源名称
- prompt: 符合 HF chat_template 格式的消息列表
- ability: 任务能力标签
- reward_model: 奖励模型配置 (包含 ground_truth 用于评估)
- extra_info: 元数据 + 语义评估所需字段
  - split, index
  - good_trace, bad_trace (raw traces)
  - prop_dict (命题字典)
  - grounded_tl (prop-space LTL 公式字符串)
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter

# 固定指令: 将自然语言规范转换为 LTL 公式
INSTRUCTION = "Translate the following natural language specification into a Linear Temporal Logic (LTL) formula."
# 统一的数据源标识符
DATA_SOURCE = "vltl-bench"
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


def convert_split(
    split: str, input_dir: Path, backend_name: str, backend_module, log_file
) -> tuple[list, int, list]:
    """
    转换一个 split (train/test) 下的所有 JSONL 文件为 VERL 格式行。
    """
    # 存储转换后的 VERL 格式行
    rows = []
    total_lines = 0
    skipped = []
    global_index = 0  # 全局递增索引，用于唯一标识每个样本
    missing_fields_count = Counter()  # 统计缺失字段

    # 获取该目录下所有 JSONL 文件并按文件名排序
    input_files = sorted(input_dir.glob("*.jsonl"))

    for input_path in input_files:
        file_total = 0
        file_written = 0

        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                file_total += 1
                line = line.strip()

                # 跳过空行
                if not line:
                    skipped.append((input_path.name, line_num, "empty line"))
                    continue

                # 解析 JSON
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    skipped.append(
                        (input_path.name, line_num, f"JSON parse error: {e}")
                    )
                    continue

                # 验证 sentence 字段
                sentence = record.get("sentence")
                if sentence is None:
                    skipped.append((input_path.name, line_num, "missing 'sentence'"))
                    continue
                if not isinstance(sentence, list) or len(sentence) == 0:
                    skipped.append((input_path.name, line_num, "invalid 'sentence'"))
                    continue
                if not all(isinstance(t, str) for t in sentence):
                    skipped.append(
                        (
                            input_path.name,
                            line_num,
                            "'sentence' has non-string elements",
                        )
                    )
                    continue

                # 验证 tl 字段
                tl = record.get("tl")
                if tl is None:
                    skipped.append((input_path.name, line_num, "missing 'tl'"))
                    continue
                if not isinstance(tl, list) or len(tl) == 0:
                    skipped.append((input_path.name, line_num, "invalid 'tl'"))
                    continue
                if not all(isinstance(t, str) for t in tl):
                    skipped.append(
                        (input_path.name, line_num, "'tl' has non-string elements")
                    )
                    continue

                # 将 token 列表还原为字符串
                sentence_str = " ".join(sentence)
                tl_str = " ".join(tl)

                # === 新增：提取语义评估所需字段 ===

                # grounded_tl: prop-space LTL 公式
                grounded_tl_tokens = record.get("grounded_tl")
                if grounded_tl_tokens and isinstance(grounded_tl_tokens, list):
                    grounded_tl_str = " ".join(grounded_tl_tokens)
                else:
                    grounded_tl_str = ""
                    missing_fields_count["grounded_tl"] += 1

                # prop_dict: 原子命题字典
                prop_dict = record.get("prop_dict")
                if not prop_dict or not isinstance(prop_dict, dict):
                    prop_dict = {}
                    missing_fields_count["prop_dict"] += 1

                # good_trace 和 bad_trace
                good_trace = record.get("good_trace")
                if not good_trace or not isinstance(good_trace, list):
                    good_trace = []
                    missing_fields_count["good_trace"] += 1

                bad_trace = record.get("bad_trace")
                if not bad_trace or not isinstance(bad_trace, list):
                    bad_trace = []
                    missing_fields_count["bad_trace"] += 1

                # === 构建 VERL 格式的行 ===
                verl_row = {
                    "data_source": DATA_SOURCE,
                    "prompt": [{"role": "user", "content": sentence_str}],
                    "ability": ABILITY,
                    "reward_model": {"style": "rule", "ground_truth": tl_str},
                    "extra_info": {
                        # 原有字段
                        "split": split,
                        "index": global_index,
                        # 新增语义评估字段
                        "good_trace": good_trace,
                        "bad_trace": bad_trace,
                        "prop_dict": prop_dict,
                        "grounded_tl": grounded_tl_str,
                    },
                }

                rows.append(verl_row)
                global_index += 1
                file_written += 1

        log_file.write(f"  {input_path.name}: {file_written}/{file_total} lines\n")

    # 记录统计信息
    log_file.write(f"\n  Semantic fields statistics:\n")
    if missing_fields_count:
        for field, count in missing_fields_count.items():
            log_file.write(f"    Missing '{field}': {count} records\n")
    else:
        log_file.write(f"    All semantic fields present in all records\n")

    return rows, total_lines, skipped


def main():
    """
    主函数: 遍历 train 和 test 目录，将所有 JSONL 合并为 VERL parquet 格式。
    """
    base_dir = Path("VLTL-Bench")
    output_dir = base_dir / "verl_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检测可用的 parquet 写入后端
    backend_name, backend_module = get_parquet_writer()

    log_path = output_dir / "conversion.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"VERL Parquet Conversion Log - {datetime.now().isoformat()}\n")
        log_file.write(f"Backend: {backend_name}\n")
        log_file.write("=" * 60 + "\n\n")

        total_all = 0
        written_all = 0

        # 分别处理 train 和 test
        for split in ["train", "test"]:
            input_dir = base_dir / split
            if not input_dir.exists():
                log_file.write(f"WARNING: {split}/ directory not found, skipping\n\n")
                continue

            log_file.write(f"Split: {split}\n")

            rows, total, skipped = convert_split(
                split, input_dir, backend_name, backend_module, log_file
            )

            # 写入 parquet 文件
            output_path = output_dir / f"{split}.parquet"
            write_parquet(rows, output_path, backend_name, backend_module)

            total_all += total
            written_all += len(rows)

            log_file.write(f"  Total lines: {total}\n")
            log_file.write(f"  Written rows: {len(rows)}\n")
            log_file.write(f"  Skipped: {len(skipped)}\n")

            # 记录跳过的行
            if skipped:
                log_file.write("  Skipped details:\n")
                for fname, line_num, reason in skipped[:20]:
                    log_file.write(f"    {fname}:{line_num} - {reason}\n")
                if len(skipped) > 20:
                    log_file.write(f"    ... and {len(skipped) - 20} more\n")

            log_file.write(f"  Output: {output_path}\n\n")

            print(f"{split}: {len(rows)}/{total} rows -> {output_path}")

        log_file.write("=" * 60 + "\n")
        log_file.write(f"TOTAL: {written_all}/{total_all} rows converted\n")

        print(f"\nSummary: {written_all}/{total_all} rows converted")
        print(f"Backend used: {backend_name}")
        print(f"Log: {log_path}")


if __name__ == "__main__":
    main()

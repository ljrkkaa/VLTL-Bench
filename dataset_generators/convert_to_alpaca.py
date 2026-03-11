#!/usr/bin/env python3
"""
Convert VLTL JSONL files to LlamaFactory Alpaca format.

For each record:
- input = nl field
- output = masked_tl field
- instruction = fixed string
"""

import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Optional

INSTRUCTION = "Translate the following natural language specification into a Linear Temporal Logic (LTL) formula."

def _extract_output_formula(record: dict) -> Optional[str]:
    """从记录中提取可用的LTL输出字段，按优先级尝试。"""
    for key in ("masked_tl", "tl", "final_tl"):
        value = record.get(key)
        if isinstance(value, str) and len(value) > 0:
            return value
    return None


def convert_file(
    input_path: Path,
    output_path: Path,
    log_file,
    sample_size: int = 2000,
    seed: int = 42,
) -> tuple[int, int, int, list]:
    """Convert a JSONL file to Alpaca format and randomly sample records."""
    total_lines = 0
    valid_records = []
    skipped = []
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                skipped.append((line_num, "empty line"))
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                skipped.append((line_num, f"JSON parse error: {e}"))
                continue
            
            # Validate nl field (input)
            nl = record.get('nl')
            if nl is None:
                skipped.append((line_num, "missing 'nl' field"))
                continue
            if not isinstance(nl, str):
                skipped.append((line_num, f"'nl' is not a string: {type(nl).__name__}"))
                continue
            if len(nl) == 0:
                skipped.append((line_num, "'nl' is empty string"))
                continue
            
            # Validate output field (masked_tl / tl / final_tl)
            output_formula = _extract_output_formula(record)
            if output_formula is None:
                skipped.append((line_num, "missing valid output field: expected one of 'masked_tl' / 'tl' / 'final_tl'"))
                continue
            
            # Create Alpaca record
            alpaca_record = {
                "instruction": INSTRUCTION,
                "input": nl,
                "output": output_formula
            }
            valid_records.append(alpaca_record)

        valid_count = len(valid_records)
        if sample_size > 0 and valid_count > sample_size:
            rng = random.Random(seed)
            selected_records = rng.sample(valid_records, sample_size)
        else:
            selected_records = valid_records

        for rec in selected_records:
            outfile.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    return total_lines, len(selected_records), valid_count, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to Alpaca format and randomly sample records"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="/data/ljr/llm_experiments/ljr_ltl_datasetV1/nl2ltl.jsonl",
        help="输入JSONL文件路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/data/ljr/llm_experiments/ljr_ltl_datasetV1/sft/nl2ltl_alpaca_sample2000.jsonl",
        help="输出Alpaca JSONL文件路径",
    )
    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=2000,
        help="随机采样数量（默认2000）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认42）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open log file
    log_path = output_path.parent / 'conversion.log'
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Conversion Log - {datetime.now().isoformat()}\n")
        log_file.write("=" * 60 + "\n\n")
        
        total, written, valid_count, skipped = convert_file(
            input_path,
            output_path,
            log_file,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        
        # Write to log
        log_file.write(f"Input File: {input_path}\n")
        log_file.write(f"Output File: {output_path}\n")
        log_file.write(f"Sample size requested: {args.sample_size}\n")
        log_file.write(f"Random seed: {args.seed}\n")
        log_file.write(f"Total lines: {total}\n")
        log_file.write(f"Valid records: {valid_count}\n")
        log_file.write(f"Written: {written}\n")
        log_file.write(f"Skipped: {len(skipped)}\n")
        
        if skipped:
            log_file.write("Skipped lines:\n")
            for line_num, reason in skipped:
                log_file.write(f"  Line {line_num}: {reason}\n")
        
        log_file.write("\n" + "=" * 60 + "\n")
        log_file.write(f"TOTAL: {written}/{total} lines converted (from {valid_count} valid), {len(skipped)} skipped\n")
        
        print(f"Converted {input_path} -> {output_path} ({written}/{total} lines)")
        print(f"\nSummary: {written}/{total} lines converted (from {valid_count} valid), {len(skipped)} skipped")
        print(f"Log written to: {log_path}")


if __name__ == '__main__':
    main()

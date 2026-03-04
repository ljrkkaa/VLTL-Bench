#!/usr/bin/env python3
"""
Convert VLTL JSONL files to LlamaFactory Alpaca format.

For each record:
- input = nl field
- output = masked_tl field
- instruction = fixed string
"""

import json
import os
from pathlib import Path
from datetime import datetime

INSTRUCTION = "Translate the following natural language specification into a Linear Temporal Logic (LTL) formula."

def convert_file(input_path: Path, output_path: Path, log_file) -> tuple[int, int, list]:
    """Convert a single JSONL file to Alpaca format."""
    total_lines = 0
    written_lines = 0
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
            
            # Validate masked_tl field (output)
            masked_tl = record.get('masked_tl')
            if masked_tl is None:
                skipped.append((line_num, "missing 'masked_tl' field"))
                continue
            if not isinstance(masked_tl, str):
                skipped.append((line_num, f"'masked_tl' is not a string: {type(masked_tl).__name__}"))
                continue
            if len(masked_tl) == 0:
                skipped.append((line_num, "'masked_tl' is empty string"))
                continue
            
            # Create Alpaca record
            alpaca_record = {
                "instruction": INSTRUCTION,
                "input": nl,
                "output": masked_tl
            }
            
            outfile.write(json.dumps(alpaca_record, ensure_ascii=False) + '\n')
            written_lines += 1
    
    return total_lines, written_lines, skipped


def main():
    # Direct input and output file paths
    input_path = Path('new_generated_datasets/warehouse_nl.jsonl')
    output_path = Path('new_generated_datasets/warehouse_nl_alpaca.jsonl')
    
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
        
        total, written, skipped = convert_file(input_path, output_path, log_file)
        
        # Write to log
        log_file.write(f"Input File: {input_path}\n")
        log_file.write(f"Output File: {output_path}\n")
        log_file.write(f"Total lines: {total}\n")
        log_file.write(f"Written: {written}\n")
        log_file.write(f"Skipped: {len(skipped)}\n")
        
        if skipped:
            log_file.write("Skipped lines:\n")
            for line_num, reason in skipped:
                log_file.write(f"  Line {line_num}: {reason}\n")
        
        log_file.write("\n" + "=" * 60 + "\n")
        log_file.write(f"TOTAL: {written}/{total} lines converted, {len(skipped)} skipped\n")
        
        print(f"Converted {input_path} -> {output_path} ({written}/{total} lines)")
        print(f"\nSummary: {written}/{total} lines converted, {len(skipped)} skipped")
        print(f"Log written to: {log_path}")


if __name__ == '__main__':
    main()

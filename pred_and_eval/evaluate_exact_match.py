#!/usr/bin/env python3
"""
计算预测完全正确的比例 (exact match accuracy)
"""

import json
import os
from datetime import datetime


def normalize_formula(formula):
    """规范化公式以便比较"""
    if formula is None:
        return ""
    # 移除多余空格，转为小写
    return formula.strip().lower()


def save_results_to_log(pred_file, accuracy, correct, total):
    """保存评估结果到日志文件"""
    # 获取 pred_file 所在目录
    log_dir = os.path.dirname(pred_file)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = os.path.basename(pred_file)
    log_filename = f"eval_{basename}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 写入日志内容
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("VLTL-Bench 预测结果评估\n")
        f.write("=" * 50 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"文件: {pred_file}\n")
        f.write(f"总样本数: {total}\n")
        f.write(f"完全正确数: {correct}\n")
        f.write(f"完全正确比例: {accuracy:.2f}%\n")
        f.write("=" * 50 + "\n")
    
    return log_path


def calculate_exact_match_accuracy(pred_file):
    """计算预测完全正确的比例"""
    total = 0
    correct = 0

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                output = data.get("output", "")
                prediction = data.get("prediction", "")

                total += 1

                # 直接比较字符串是否完全相同
                if output == prediction:
                    correct += 1
                else:
                    # 也尝试规范化后比较
                    if normalize_formula(output) == normalize_formula(prediction):
                        correct += 1

            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON line")
                continue

    if total == 0:
        return 0.0, 0, 0

    accuracy = correct / total * 100
    return accuracy, correct, total


if __name__ == "__main__":
    # pred_file = "predictions/lora_model/warehouse_nl_alpaca_pred.jsonl"
    pred_file = "/data/ljr/llm_experiments/ljr_ltl_datasetV1/predictions/nl2tl_alpca_pred.jsonl"

    accuracy, correct, total = calculate_exact_match_accuracy(pred_file)

    print(f"=" * 50)
    print(f"VLTL-Bench 预测结果评估")
    print(f"=" * 50)
    print(f"文件: {pred_file}")
    print(f"总样本数: {total}")
    print(f"完全正确数: {correct}")
    print(f"完全正确比例: {accuracy:.2f}%")
    print(f"=" * 50)

    # 保存结果到日志文件
    log_path = save_results_to_log(pred_file, accuracy, correct, total)
    print(f"\n结果已保存到: {log_path}")

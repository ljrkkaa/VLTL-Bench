"""
测试脚本：验证 predictions/rl_model/warehouse_nl_alpaca_pred.jsonl 中的预测结果
"""

import json
import sys

# 导入 LTL 验证器
sys.path.insert(0, 'dataset_generators')
from ltl_verifier import (
    verify_ltl_formula,
    convert_to_spot_format,
    SPOT_AVAILABLE
)


def load_jsonl(filepath):
    """加载 JSONL 文件"""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def verify_prediction_entry(entry):
    """验证单个预测条目"""
    result = {
        "instruction": entry.get("instruction", ""),
        "input": entry.get("input", ""),
        "output": entry.get("output", ""),
        "prediction": entry.get("prediction", ""),
    }
    
    ground_truth = entry.get("output", "")
    prediction = entry.get("prediction", "")
    
    # 验证 ground truth
    if ground_truth:
        is_sat_gt, spot_gt, simplified_gt, error_gt = verify_ltl_formula(ground_truth)
        result["ground_truth_verified"] = {
            "is_satisfiable": is_sat_gt,
            "spot_formula": spot_gt,
            "error": error_gt
        }
    else:
        result["ground_truth_verified"] = {"error": "Empty ground truth"}
    
    # 验证 prediction
    if prediction:
        is_sat_pred, spot_pred, simplified_pred, error_pred = verify_ltl_formula(prediction)
        result["prediction_verified"] = {
            "is_satisfiable": is_sat_pred,
            "spot_formula": spot_pred,
            "simplified_spot": simplified_pred,
            "error": error_pred
        }
    else:
        result["prediction_verified"] = {"error": "Empty prediction"}
    
    return result


def main():
    # 数据文件路径
    input_file = "predictions/rl_model/warehouse_nl_alpaca_pred.jsonl"
    output_file = "predictions/rl_model/warehouse_nl_alpaca_pred_verified.jsonl"
    
    print(f"正在加载预测数据: {input_file}")
    
    # 加载数据
    entries = load_jsonl(input_file)
    print(f"加载了 {len(entries)} 条预测数据")
    
    # 验证所有预测
    print("\n开始验证预测结果...")
    
    verified_entries = []
    for idx, entry in enumerate(entries):
        if idx % 100 == 0:
            print(f"  处理进度: {idx}/{len(entries)}")
        
        verified = verify_prediction_entry(entry)
        verified["id"] = idx
        verified_entries.append(verified)
    
    # 统计结果
    total = len(verified_entries)
    
    # Ground truth 统计
    gt_sat_count = sum(1 for e in verified_entries 
                       if e.get("ground_truth_verified", {}).get("is_satisfiable", False))
    gt_unsat_count = sum(1 for e in verified_entries 
                        if e.get("ground_truth_verified", {}).get("is_satisfiable") == False)
    gt_error_count = sum(1 for e in verified_entries 
                        if e.get("ground_truth_verified", {}).get("error"))
    
    # Prediction 统计
    pred_sat_count = sum(1 for e in verified_entries 
                        if e.get("prediction_verified", {}).get("is_satisfiable", False))
    pred_unsat_count = sum(1 for e in verified_entries 
                           if e.get("prediction_verified", {}).get("is_satisfiable") == False)
    pred_error_count = sum(1 for e in verified_entries 
                           if e.get("prediction_verified", {}).get("error"))
    
    print(f"\n=== 验证完成 ===")
    print(f"\nGround Truth 验证:")
    print(f"  总计: {total} 条")
    print(f"  可满足: {gt_sat_count} 条")
    print(f"  不可满足: {gt_unsat_count} 条")
    print(f"  错误: {gt_error_count} 条")
    
    print(f"\nPrediction 验证:")
    print(f"  总计: {total} 条")
    print(f"  可满足: {pred_sat_count} 条")
    print(f"  不可满足: {pred_unsat_count} 条")
    print(f"  错误: {pred_error_count} 条")
    
    # 保存验证结果
    print(f"\n正在保存验证结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in verified_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print("验证完成！")
    
    # 打印一些预测不可满足的例子
    if pred_unsat_count > 0:
        print(f"\n=== 预测不可满足示例 (共 {pred_unsat_count} 条) ===")
        count = 0
        for entry in verified_entries:
            if entry.get("prediction_verified", {}).get("is_satisfiable") == False:
                print(f"\n[ID {entry['id']}]")
                print(f"  输入: {entry.get('input', '')[:60]}...")
                print(f"  Ground Truth: {entry.get('output', '')[:60]}...")
                print(f"  Prediction: {entry.get('prediction', '')[:60]}...")
                count += 1
                if count >= 5:
                    break
    
    # 打印一些预测有转换错误的例子
    if pred_error_count > 0:
        print(f"\n=== 预测转换错误示例 (共 {pred_error_count} 条) ===")
        count = 0
        for entry in verified_entries:
            error = entry.get("prediction_verified", {}).get("error", "")
            if error and error != "SPOT not available":
                print(f"\n[ID {entry['id']}]")
                print(f"  Prediction: {entry.get('prediction', '')[:60]}...")
                print(f"  错误: {error}")
                count += 1
                if count >= 5:
                    break


if __name__ == "__main__":
    if not SPOT_AVAILABLE:
        print("[错误] SPOT 库未安装，无法进行验证")
        print("安装方法: pip install spot")
        sys.exit(1)
    
    main()

#!/usr/bin/env python3
"""
读取VLTL-Bench/test下的jsonl文件，使用大模型生成LTL预测，并保存结果。

使用方法:
    python predict_vltl_test.py --lora-path <lora_path> --test-dir VLTL-Bench/test --output-dir predictions
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging as hf_logging


hf_logging.set_verbosity_error()


def load_model_from_lora(lora_path: str):
    """从 LoRA 路径加载模型：读取 adapter_config.json 找到基础模型，合并后返回完整模型。

    Args:
        lora_path: LoRA adapter 所在目录路径（包含 adapter_config.json）

    Returns:
        (model, tokenizer): 合并后的完整模型和分词器
    """
    lora_path = Path(lora_path).resolve()

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA 路径不存在: {lora_path}")

    adapter_config_path = lora_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"未找到 adapter_config.json: {adapter_config_path}")

    print(f"正在从 LoRA 路径加载: {lora_path}")
    print("正在读取 adapter_config.json...")

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_path = adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(f"adapter_config.json 中未找到 base_model_name_or_path: {adapter_config_path}")

    print(f"基础模型路径: {base_model_path}")
    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("正在加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # Make generation behavior explicit (avoids model-specific defaults surprises).
    try:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
    except Exception:
        pass

    print(f"正在加载 LoRA 适配器: {lora_path}")
    model = PeftModel.from_pretrained(model, str(lora_path))

    print("正在合并 LoRA 适配器到基础模型...")
    model = model.merge_and_unload()
    print("✓ LoRA 适配器已成功合并，模型准备就绪")

    return model, tokenizer


def generate_ltl_translation(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int = 128,
    do_sample: bool = True,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    enable_thinking: bool = True,
) -> tuple[str, str]:
    """Generate LTL translation using the model.

    Args:
        model: 模型实例
        tokenizer: 分词器实例
        instruction: 指令
        input_text: 输入文本
        max_new_tokens: 最大生成 token 数
        do_sample: 是否使用采样
        temperature: 采样温度
        top_p: top-p 采样参数
        repetition_penalty: 重复惩罚
        enable_thinking: 是否启用 thinking 模式（默认 False）

    Returns:
        tuple[str, str]: (thinking_content, response_content)
    """
    messages = [
        {
            "role": "system",
            "content": "You are an expert in translating natural language specifications into Linear Temporal Logic (LTL) formulas.",
        },
        {"role": "user", "content": f"{instruction}\n\n{input_text}".strip()},
    ]

    # 尝试使用 chat template，如果没有则手动构造 Llama 3.2 格式
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,  # Qwen3 特有参数
            return_tensors="pt",
        ).to(model.device)
    else:
        # 手动构造 Llama 3.2 格式的 prompt
        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # 直接用 transformers 的标准生成方式
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 解析 thinking 内容（Qwen3 特有逻辑）
    output_ids = outputs[0][inputs.shape[-1] :].tolist()

    # Qwen3 thinking token ID 为 151668
    thinking_token_id = 151668
    try:
        # rindex finding 151668 (
        index = len(output_ids) - output_ids[::-1].index(thinking_token_id)
    except ValueError:
        index = 0

    # 解码 thinking 内容
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    # 解码实际回复内容
    response_content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return thinking_content, response_content


def process_jsonl_file(
    model,
    tokenizer,
    input_path: Path,
    output_path: Path,
    instruction: str = "Translate the following natural language specification into a Linear Temporal Logic (LTL) formula.",
):
    """处理单个 jsonl 文件，为每个样本生成预测并保存。

    Args:
        model: 模型实例
        tokenizer: 分词器实例
        input_path: 输入 jsonl 文件路径
        output_path: 输出 jsonl 文件路径
        instruction: 指令文本
    """
    print(f"\n处理文件: {input_path}")
    print(f"输出路径: {output_path}")

    results = []
    total = 0
    success = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: 跳过无效的 JSON 行: {e}")
                continue

            total += 1

            # 将 sentence 列表转换为字符串
            sentence_list = data.get("lifted_sentence", [])
            sentence_str = " ".join(sentence_list)

            print(f"  [{total}] ID={data.get('id', 'N/A')}: {sentence_str[:60]}...")

            try:
                # 生成预测
                thinking, prediction = generate_ltl_translation(
                    model, tokenizer, instruction, sentence_str
                )

                # 添加 prediction 字段到数据
                data["prediction"] = prediction

                success += 1
                print(f"      -> 预测: {prediction}...")

            except Exception as e:
                print(f"      -> 错误: {e}")
                data["prediction"] = ""
                data["error"] = str(e)

            results.append(data)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        for data in results:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"\n完成: {success}/{total} 个样本成功生成预测")
    return success, total


def main():
    parser = argparse.ArgumentParser(
        description="使用大模型为 VLTL-Bench 测试集生成 LTL 预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python predict_vltl_test.py --lora-path saves/qwen_sft_lora/Qwen3-4B_r16_a32_lr5e-5
  python predict_vltl_test.py --lora-path <path> --test-dir VLTL-Bench/test --output-dir predictions
  python predict_vltl_test.py --lora-path <path> --test-dir VLTL-Bench/test --output-dir predictions --temperature 0.5
        """,
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="/data/ljr/llm_experiments/LLaMA-Factory/saves/qwen_sft_lora/Qwen3-4B_r16_a32_lr5e-5",
        help="LoRA adapter 路径（必须包含 adapter_config.json）",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="VLTL-Bench/test",
        help="测试数据目录，包含 jsonl 文件（默认: VLTL-Bench/test）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Qwen3-lifted",
        help="输出目录（默认: Qwen3-4B_r16_a32_lr5e-5）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="采样温度（默认: 0.1）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="最大生成 token 数（默认: 128）",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_false",
        help="启用 thinking 模式（仅适用于 Qwen3 模型）",
    )
    args = parser.parse_args()

    # 加载模型
    print("=" * 80)
    print("开始加载模型...")
    print("=" * 80)

    try:
        model, tokenizer = load_model_from_lora(args.lora_path)
        model.eval()
        print(f"\n模型加载成功！参数量: {model.num_parameters() / 1e9:.2f}B")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 查找测试文件
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"错误: 测试目录不存在: {test_dir}")
        return

    jsonl_files = list(test_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"错误: 在 {test_dir} 中未找到 jsonl 文件")
        return

    print(f"\n找到 {len(jsonl_files)} 个测试文件:")
    for f in jsonl_files:
        print(f"  - {f.name}")

    # 处理每个文件
    print("\n" + "=" * 80)
    print("开始生成预测...")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    total_success = 0
    total_samples = 0

    for jsonl_file in sorted(jsonl_files):
        output_path = output_dir / jsonl_file.name
        success, total = process_jsonl_file(
            model,
            tokenizer,
            jsonl_file,
            output_path,
        )
        total_success += success
        total_samples += total

    print("\n" + "=" * 80)
    print("全部完成!")
    print(f"总计: {total_success}/{total_samples} 个样本成功生成预测")
    print(f"输出目录: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

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


def _postprocess_ltl_output(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s

    # Some models may generate multiple candidates separated by blank lines.
    blocks = [b.strip() for b in s.split("\n\n") if b.strip()]
    s = blocks[0] if blocks else s

    # If still multiline, keep the first non-empty line.
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[0] if lines else s


def _get_terminator_token_ids(tokenizer) -> list[int]:
    terminators: list[int] = []
    if tokenizer.eos_token_id is not None:
        terminators.append(int(tokenizer.eos_token_id))

    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id != getattr(tokenizer, "unk_token_id", None):
            terminators.append(int(eot_id))
    except Exception:
        pass

    seen: set[int] = set()
    deduped: list[int] = []
    for t in terminators:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


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
) -> str:
    """生成 LTL 翻译结果（适配 Llama 3.2 的 chat template）。"""

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

    # 只解码新生成的部分
    response = tokenizer.decode(outputs[0][inputs.shape[-1] :], skip_special_tokens=True).strip()
    return response


def test_model(lora_path: str = "saves/llama_sft_lora/test1"):
    """测试微调后的模型

    Args:
        lora_path: LoRA adapter 路径，默认为 saves/llama_sft_lora/test1
    """
    print("=" * 80)
    print("开始加载模型...")
    print("=" * 80)

    try:
        model, tokenizer = load_model_from_lora(lora_path)
        model.eval()

        print("\n模型加载成功！")
        print(f"模型参数量: {model.num_parameters() / 1e9:.2f}B")

        test_cases = [
            {
                "instruction": "Translate the following natural language specification into a Linear Temporal Logic (LTL) formula.",
                "input": "If every establish communication with the injured victim is eventually followed by establishing communication with the injured rescuer, then take a picture of safe victim must occur infinitely often.",
                "expected_output": "globally ( communicate(injured_victim) implies finally communicate(injured_rescuer) ) implies globally finally photo(safe_victim)",
            }
        ]

        print("\n" + "=" * 80)
        print("开始测试模型...")
        print("=" * 80)

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n测试用例 {i}:")
            print("-" * 40)
            print(f"指令: {test_case['instruction']}")
            print(f"输入: {test_case['input']}")
            print(f"期望输出: {test_case['expected_output']}")
            print("\n模型输出:")
            print("-" * 40)

            output = generate_ltl_translation(model, tokenizer, test_case["instruction"], test_case["input"])

            print(output)
            print("\n" + "=" * 80)

        print("\n交互式测试模式 (输入 'quit' 退出)")
        print("-" * 40)

        default_instruction = (
            "Translate the following natural language specification into a Linear Temporal Logic (LTL) formula."
        )

        while True:
            user_input = input("\n请输入自然语言规格 (或 'quit' 退出): ").strip()

            if user_input.lower() == "quit":
                print("测试结束！")
                break

            if not user_input:
                print("输入为空，请重试")
                continue

            print("\n正在生成 LTL 公式...")
            output = generate_ltl_translation(model, tokenizer, default_instruction, user_input)
            print(f"\n生成的 LTL 公式:\n{output}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="测试 LoRA 微调后的 LTL 翻译模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_lora_model.py
  python test_lora_model.py --lora-path saves/llama_sft_lora/checkpoint-100
  python test_lora_model.py --lora-path /path/to/custom/lora
        """,
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="/data/ljr/llm_experiments/LLaMA-Factory/saves/qwen_sft_lora/Qwen3-4B_r16_a32_lr5e-5",
        help="LoRA adapter 路径（默认: saves/llama_sft_lora/test1）",
    )
    args = parser.parse_args()

    test_model(lora_path=args.lora_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate LTL predictions for VLTL-Bench test sets using vLLM (offline, GPU 2/3).

This script mirrors predict_vltl_test.py but swaps the generation backend to vLLM
for faster offline inference. By default it binds to CUDA devices 2 and 3 and uses
LoRA adapters if provided.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


DEFAULT_LORA_PATH = "/data/ljr/llm_experiments/LLaMA-Factory/saves/qwen_sft_lora_lifted/Qwen3-1.7B_r16_a32_lr5e-5"
DEFAULT_TEST_DIR = "VLTL-Bench/test"
DEFAULT_OUTPUT_DIR = "Qwen3-lifted-vllm"
THINKING_TOKEN_ID = 151668  # Qwen3 thinking token


def read_adapter_config(lora_path: Path) -> Tuple[str, int]:
    """Return (base_model_path, max_lora_rank) from adapter_config.json."""
    adapter_config_path = lora_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found under {lora_path}")

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_path = adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError("base_model_name_or_path missing in adapter_config.json")

    rank = adapter_config.get("r", 64)
    return base_model_path, int(rank)


def build_prompt(
    tokenizer, instruction: str, input_text: str, enable_thinking: bool
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an expert in translating natural language specifications into Linear Temporal Logic (LTL) formulas.",
        },
        {"role": "user", "content": f"{instruction}\n\n{input_text}".strip()},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def split_thinking(
    tokenizer, token_ids: List[int], fallback_text: str
) -> Tuple[str, str]:
    """Split thinking content from the final response if the thinking token is present."""
    if not token_ids:
        return "", fallback_text.strip()

    try:
        index = len(token_ids) - token_ids[::-1].index(THINKING_TOKEN_ID)
    except ValueError:
        index = 0

    thinking = tokenizer.decode(token_ids[:index], skip_special_tokens=True).strip()
    response = tokenizer.decode(token_ids[index:], skip_special_tokens=True).strip()

    if not response:
        response = fallback_text.strip()

    return thinking, response


def load_inputs(input_path: Path) -> List[dict]:
    items: List[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(data)
            except json.JSONDecodeError:
                # Skip invalid JSON lines but keep processing others.
                continue
    return items


def write_outputs(output_path: Path, items: Iterable[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for data in items:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def process_file(
    llm: LLM,
    tokenizer,
    input_path: Path,
    output_path: Path,
    instruction: str,
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None,
    enable_thinking: bool,
) -> Tuple[int, int]:
    print(f"\nProcessing file: {input_path}")
    items = load_inputs(input_path)

    if not items:
        print("  -> No valid samples found; skipping.")
        return 0, 0

    prompts: List[str] = []
    for idx, sample in enumerate(items, start=1):
        sentence_list = sample.get("lifted_sentence", [])
        sentence_str = (
            " ".join(sentence_list)
            if isinstance(sentence_list, list)
            else str(sentence_list)
        )
        prompts.append(
            build_prompt(tokenizer, instruction, sentence_str, enable_thinking)
        )
        if idx <= 3:
            print(f"  [{idx}] ID={sample.get('id', 'N/A')}: {sentence_str[:60]}...")
    if len(items) > 3:
        print(f"  ... {len(items)} samples total")

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
        use_tqdm=True,
    )

    success = 0
    for sample, output in zip(items, outputs):
        candidate = output.outputs[0]
        thinking, prediction = split_thinking(
            tokenizer, candidate.token_ids, candidate.text
        )
        sample["prediction"] = prediction
        if thinking:
            sample["thinking"] = thinking
        success += 1 if prediction else 0

    write_outputs(output_path, items)
    print(
        f"  -> Wrote predictions to {output_path} ({success}/{len(items)} successful)"
    )
    return success, len(items)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate VLTL-Bench predictions with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=DEFAULT_LORA_PATH,
        help="Path to LoRA adapter (optional)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model path; otherwise inferred from LoRA",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=DEFAULT_TEST_DIR,
        help="Directory with test jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write predictions",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Translate the following natural language specification into a Linear Temporal Logic (LTL) formula.",
        help="Instruction prepended to each prompt",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.2, help="Repetition penalty"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="vLLM tensor parallel degree",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.8,
        help="GPU memory utilization hint for vLLM",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum context length for vLLM",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="2,3",
        help="CUDA devices to bind for vLLM",
    )
    parser.add_argument(
        "--disable-thinking", action="store_true", help="Disable Qwen thinking mode"
    )
    parser.add_argument(
        "--use-base-only", action="store_true", help="Ignore LoRA and run base model"
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Pin GPUs before initializing vLLM.
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    lora_path = Path(args.lora_path).resolve() if args.lora_path else None
    base_model_path: str | None = args.base_model
    max_lora_rank = None

    if not args.use_base_only and lora_path:
        base_model_path, max_lora_rank = read_adapter_config(lora_path)
    elif not base_model_path:
        raise ValueError(
            "Provide --base-model or --lora-path (without --use-base-only)."
        )

    print("=" * 80)
    print(f"Using base model: {base_model_path}")
    if not args.use_base_only and lora_path:
        print(f"Attaching LoRA adapter: {lora_path}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, use_fast=False
    )

    enable_lora = bool(lora_path) and not args.use_base_only
    llm = LLM(
        model=base_model_path,
        tokenizer=base_model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_lora=enable_lora,
        max_loras=1 if enable_lora else 0,
        max_lora_rank=max_lora_rank,
        max_model_len=args.max_model_len,
    )

    lora_request = None
    if enable_lora:
        lora_request = LoRARequest("vltl_lora", 1, str(lora_path))

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    jsonl_files = sorted(test_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No jsonl files found under {test_dir}")

    output_dir = Path(args.output_dir)
    total_success = 0
    total_samples = 0

    for jsonl_file in jsonl_files:
        output_path = output_dir / jsonl_file.name
        success, total = process_file(
            llm=llm,
            tokenizer=tokenizer,
            input_path=jsonl_file,
            output_path=output_path,
            instruction=args.instruction,
            sampling_params=sampling_params,
            lora_request=lora_request,
            enable_thinking=not args.disable_thinking,
        )
        total_success += success
        total_samples += total

    print("\n" + "=" * 80)
    print(f"All done: {total_success}/{total_samples} samples processed")
    print(f"Outputs written to: {output_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

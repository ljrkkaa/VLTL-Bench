#!/usr/bin/env python3
"""Generate LTL predictions for VLTL-Bench new generated datasets using vLLM (offline, GPU 2/3).

This script is adapted from predict_vltl_vllm.py to handle the new Alpaca format
with 'input' field instead of 'lifted_sentence'. It processes warehouse_nl_alpaca.jsonl
and generates LTL predictions.
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
DEFAULT_INPUT_FILE = "VLTL-Bench/new_generated_datasets/warehouse_nl_alpaca.jsonl"
DEFAULT_OUTPUT_DIR = "VLTL-Bench/predictions"
DEFAULT_OUTPUT_FILE = "warehouse_nl_alpaca_pred.jsonl"
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
        # The Alpaca format has 'input' field containing the natural language specification
        # to translate into LTL. The 'instruction' field contains the task description.
        input_text = sample.get("input", "")
        
        # If input is empty, fall back to using the instruction field
        if not input_text:
            input_text = sample.get("instruction", "")
        
        # Build prompt using just the input text (the instruction is added separately)
        prompts.append(
            build_prompt(tokenizer, instruction, input_text, enable_thinking)
        )
        if idx <= 3:
            print(f"  [{idx}] ID={sample.get('id', idx)}: {input_text[:60]}...")
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
        description="Generate VLTL-Bench predictions with vLLM for new datasets",
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
        "--input-file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Path to input jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write predictions",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Output filename",
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

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_file

    success, total = process_file(
        llm=llm,
        tokenizer=tokenizer,
        input_path=input_path,
        output_path=output_path,
        instruction=args.instruction,
        sampling_params=sampling_params,
        lora_request=lora_request,
        enable_thinking=not args.disable_thinking,
    )

    print("\n" + "=" * 80)
    print(f"All done: {success}/{total} samples processed")
    print(f"Outputs written to: {output_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

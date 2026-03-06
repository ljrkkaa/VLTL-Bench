#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./predict_lora.bash [lora_adapter_path] [/path/to/base_model] [test_dir] [output_dir]
#
# Examples:
#   ./predict_lora.bash
#   ./predict_lora.bash /custom/lora
#   ./predict_lora.bash /custom/lora /custom/base_model VLTL-Bench/test predictions/lora_vllm
#
# Notes:
# - If base_model is omitted, predict_vltl_vllm.py will infer it from adapter_config.json.

DEFAULT_LORA_PATH="/data/ljr/llm_experiments/LLaMA-Factory/saves/qwen_sft_lora_lifted/Qwen3-1.7B_r16_a32_lr5e-5"
DEFAULT_INPUT_FILE="new_generated_datasets/warehouse_nl_alpaca.jsonl"
DEFAULT_OUTPUT_DIR="predictions"

# All args are optional now; pass them to override defaults.
LORA_PATH="${1:-$DEFAULT_LORA_PATH}"
BASE_MODEL="${2:-}"
INPUT_FILE="${3:-$DEFAULT_INPUT_FILE}"
OUTPUT_DIR="${4:-$DEFAULT_OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

cmd=(
  python3 predict_vltl_vllm.py
  --lora-path "$LORA_PATH"
  --input-file "$INPUT_FILE"
  --output-dir "$OUTPUT_DIR"
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES"
)

if [[ -n "$BASE_MODEL" ]]; then
  cmd+=(--base-model "$BASE_MODEL")
fi

"${cmd[@]}"

#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./predict_base.bash [/path/to/base_model] [test_dir] [output_dir]
#
# Examples:
#   ./predict_base.bash
#   ./predict_base.bash /custom/base_model

# DEFAULT_BASE_MODEL="/data/ljr/llm_experiments/model/qwen3-1.7B_lifted"
DEFAULT_BASE_MODEL="/data/ljr/llm_experiments/model/qwen3-1.7B_lifted_reward1_step561_merged"
DEFAULT_INPUT_FILE="new_generated_datasets/warehouse_nl_alpaca.jsonl"
DEFAULT_OUTPUT_DIR="predictions/rl_model"

# All args are optional now; pass them to override defaults.
BASE_MODEL="${1:-$DEFAULT_BASE_MODEL}"
INPUT_FILE="${2:-$DEFAULT_INPUT_FILE}"
OUTPUT_DIR="${3:-$DEFAULT_OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

python3 predict_vltl_vllm.py \
	--base-model "$BASE_MODEL" \
	--use-base-only \
	--input-file "$INPUT_FILE" \
	--output-dir "$OUTPUT_DIR" \
	--cuda-visible-devices "$CUDA_VISIBLE_DEVICES"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

: "${INPUT_PATH:?Please set INPUT_PATH to your original openbmb/MiniCPM-V-4.6 Hugging Face model path}"

if [ -n "${CONDA_SH:-}" ]; then
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-npu}"
fi

if ! command -v pulsar2 >/dev/null 2>&1; then
  echo "pulsar2 not found in PATH. Please activate AXERA NPU build environment first." >&2
  exit 1
fi

OUTPUT_PATH="${1:-${OUTPUT_PATH:-$REPO_ROOT/python/MiniCPM-V-4.6_axmodel}}"

FLOAT_MATMUL_USE_CONV_EU=1 pulsar2 llm_build \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --model_type qwen3_5_text \
  --hidden_state_type bf16 \
  --prefill_len 128 \
  --kv_cache_len 2047 \
  --last_kv_cache_len 128 \
  --last_kv_cache_len 256 \
  --last_kv_cache_len 384 \
  --last_kv_cache_len 512 \
  --last_kv_cache_len 640 \
  --last_kv_cache_len 768 \
  --last_kv_cache_len 896 \
  --last_kv_cache_len 1024 \
  --last_kv_cache_len 1152 \
  --chip AX650 \
  -c 0 \
  --parallel 32

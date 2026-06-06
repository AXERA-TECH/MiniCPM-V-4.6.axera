#!/usr/bin/env bash
set -euo pipefail

: "${CODEBASE_ROOT:=/home/baiyongqiang/local_space/npu-codebase}"
: "${DEPLOY_ROOT:=/data/tmp/yongqiang/nfs/auto_model_deployment}"
: "${INPUT_PATH:=$DEPLOY_ROOT/Minicpm-V-4.6-hf-original/MiniCPM-V-4.6}"
: "${OUTPUT_PATH:=$DEPLOY_ROOT/MiniCPM-V-4.6.axera/python/MiniCPM-V-4.6_axmodel}"
: "${CONDA_SH:=/home/baiyongqiang/miniforge-pypy3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=npu}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODEBASE_ROOT"
source script/npu_dev

if [ "${1:-}" != "" ]; then
  OUTPUT_PATH="$1"
fi

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

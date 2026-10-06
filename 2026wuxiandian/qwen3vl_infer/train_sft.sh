#!/usr/bin/env bash
# Qwen3-VL-8B LoRA SFT（相对路径，cd 到本目录后可复现）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:/root/miniconda/bin:/opt/conda/bin:$PATH"

MODEL_PATH="${MODEL_PATH:-../../pretrain_model/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/sft_lora}"
DATASET="${DATASET:-./train.jsonl}"
VAL_DATASET="${VAL_DATASET:-./valid.jsonl}"

if ! command -v swift >/dev/null 2>&1; then
  echo "[error] swift not found in PATH=$PATH"
  echo "  install: pip install -r requirements-train.txt"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[error] model not found: ${MODEL_PATH}"
  echo "  set MODEL_PATH or place model at ../../pretrain_model/Qwen3-VL-8B-Instruct"
  exit 1
fi
if [[ ! -f "$DATASET" ]]; then
  echo "[error] dataset missing: $DATASET"
  echo "  first: python build_official.py train --official-dir ... --spec-dir ..."
  exit 1
fi

mkdir -p "$OUTPUT_DIR" ./logs
ts="$(date +%Y%m%d_%H%M%S)"
log="./logs/sft_lora_${ts}.log"

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-256}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-./.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-./.cache/modelscope}"

echo "[train] root=$ROOT"
echo "[train] model=$MODEL_PATH"
echo "[train] dataset=$DATASET val=$VAL_DATASET"
echo "[train] output=$OUTPUT_DIR"
echo "[train] IMAGE_MAX_TOKEN_NUM=$IMAGE_MAX_TOKEN_NUM"
echo "[train] log=$log"

# shellcheck disable=SC2086
nohup swift sft \
  --model "$MODEL_PATH" \
  --dataset "$DATASET" \
  --val_dataset "$VAL_DATASET" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --num_train_epochs "${EPOCHS:-10}" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --learning_rate "${LR:-1e-4}" \
  --eval_steps "${EVAL_STEPS:-200}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --save_total_limit 5 \
  --logging_steps 20 \
  --max_length "${MAX_LENGTH:-768}" \
  --output_dir "$OUTPUT_DIR" \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  ${EXTRA_ARGS:-} \
  >"$log" 2>&1 &

pid=$!
echo "$pid" > ./logs/sft_lora.pid
echo "[train] started pid=$pid"
echo "[train] tail -f $log"

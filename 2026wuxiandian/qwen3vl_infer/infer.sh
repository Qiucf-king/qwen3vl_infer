#!/usr/bin/env bash
# 纯推理：base + LoRA ckpt -> 提交 jsonl（ENU 4 位小数）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PATH="/root/miniconda3/bin:/root/miniconda/bin:/opt/conda/bin:$PATH"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-256}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL_PATH:-../../pretrain_model/Qwen3-VL-8B-Instruct}"
CKPT="${CKPT:?set CKPT=/path/to/lora_checkpoint}"
DATASET="${DATASET:-./infer_public.jsonl}"
OUT="${OUT:-./submits/infer_submit.jsonl}"
RAW="${RAW:-./submits/infer_raw.jsonl}"
LIMIT="${LIMIT:-0}"

mkdir -p "$(dirname "$OUT")" ./logs
ts="$(date +%Y%m%d_%H%M%S)"
log="./logs/infer_${ts}.log"

cmd=(python -u run_infer_submit.py
  --ckpt "$CKPT"
  --model "$MODEL"
  --dataset "$DATASET"
  --out "$OUT"
  --raw-out "$RAW"
  --max-new-tokens "${MAX_NEW_TOKENS:-256}"
)
if [[ "$LIMIT" != "0" ]]; then
  cmd+=(--limit "$LIMIT")
fi

echo "[infer] ${cmd[*]}"
echo "[infer] log=$log"
"${cmd[@]}" 2>&1 | tee "$log"

#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./experiments/run_generalization.sh <checkpoint> <dataset> [device]
# Example:
#   ./experiments/run_generalization.sh ./components/diffusion_smoketest/checkpoints/diffusion-epoch=00-val/loss=1.3679.ckpt cifar10 mps

ckpt="${1:?diffusion checkpoint required}"
dataset="${2:?dataset required (cifar10|stl10)}"
device="${3:-mps}"
pool_split="${POOL_SPLIT:-test}"
save_weights_dir="${SAVE_WEIGHTS_DIR:-}"

# Quick/default settings (override by exporting these env vars before running)
TASKS="${TASKS:-10}"
SUPPORT="${SUPPORT:-5}"
QUERY="${QUERY:-50}"
DDIM_STEPS="${DDIM_STEPS:-50}"
GD_STEPS="${GD_STEPS:-200}"
GD_LR="${GD_LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_CSV="${OUT_CSV:-./results/generalization_${dataset}.csv}"

mkdir -p "$(dirname "$OUT_CSV")"

extra_args=()
if [[ -n "$save_weights_dir" ]]; then
  extra_args+=(--save_weights_dir "$save_weights_dir")
fi

conda run -n param_diffusion python experiments/eval_generalization.py \
  --diffusion_checkpoint "$ckpt" \
  --dataset "$dataset" --data_root "$DATA_ROOT" \
  --tasks "$TASKS" --support "$SUPPORT" --query "$QUERY" \
  --ddim_steps "$DDIM_STEPS" \
  --gd_steps "$GD_STEPS" --gd_lr "$GD_LR" \
  --batch_size "$BATCH_SIZE" \
  --device "$device" \
  --pool_split "$pool_split" \
  --out_csv "$OUT_CSV" \
  "${extra_args[@]}"
echo "Wrote: $OUT_CSV"
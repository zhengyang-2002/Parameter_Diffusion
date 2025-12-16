#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./experiments/run_supervised_gd.sh <dataset> [device]
# Example:
#   ./experiments/run_supervised_gd.sh cifar10 mps


dataset="${1:?dataset required (cifar10|stl10)}"
device="${2:-mps}"

EPOCHS="${EPOCHS:-10}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_CSV="${OUT_CSV:-./results/supervised_gd_${dataset}.csv}"
SAVE_WEIGHTS_DIR="${SAVE_WEIGHTS_DIR:-}"

mkdir -p "$(dirname "$OUT_CSV")"

extra_args=()
if [[ -n "$SAVE_WEIGHTS_DIR" ]]; then
  extra_args+=(--save_weights_dir "$SAVE_WEIGHTS_DIR")
fi

conda run -n param_diffusion python experiments/eval_supervised_gd.py \
  --dataset "$dataset" \
  --data_root "$DATA_ROOT" \
  --image_size "$IMAGE_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --device "$device" \
  --out_csv "$OUT_CSV" \
  "${extra_args[@]}"

echo "Wrote: $OUT_CSV"

#!/usr/bin/env bash
set -euo pipefail

# Runs both experiments for CIFAR-10, STL10, CIFAR-100, and Tiny-ImageNet.
#
# Usage:
#   ./experiments/run_all_results.sh <checkpoint> [device]
#
# Notes:
# - Override defaults via env vars, e.g.:
#     TASKS=50 SUPPORT=5 QUERY=50 DDIM_STEPS=50 GD_STEPS=200 STEPS=5,10,25,50,100 \
#       ./experiments/run_all_results.sh <ckpt> mps

ckpt="${1:?diffusion checkpoint required}"
device="${2:-mps}"

./experiments/run_generalization.sh "$ckpt" cifar10 "$device"
./experiments/run_generalization.sh "$ckpt" stl10 "$device"
./experiments/run_generalization.sh "$ckpt" cifar100 "$device"

# Tiny-ImageNet pool is typically only folder-organized for train.
DATA_ROOT=./tiny-imagenet-data POOL_SPLIT=train ./experiments/run_generalization.sh "$ckpt" tinyimagenet "$device"

./experiments/run_ablation.sh "$ckpt" cifar10 "$device"
./experiments/run_ablation.sh "$ckpt" stl10 "$device"
./experiments/run_ablation.sh "$ckpt" cifar100 "$device"
DATA_ROOT=./tiny-imagenet-data POOL_SPLIT=train ./experiments/run_ablation.sh "$ckpt" tinyimagenet "$device"

# Upper-bound supervised baselines (train on full train split, evaluate on test)
./experiments/run_supervised_gd.sh cifar10 "$device"
./experiments/run_supervised_gd.sh stl10 "$device"
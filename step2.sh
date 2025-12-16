#!/bin/bash

# --- 实验配置 ---
# DNNWG 库路径
dnnwg_path="./External/DNNWG"

# 上一阶段生成的权重目录
weights_source_dir="./Model_Zoo/Resnet18_TinyImageNet_HC"

# VAE 输出目录
output_dir="./components/resnet18_TinyImagenet_HC_VAE_l2"

# 训练参数
num_subsets=500       # 需要与上一阶段生成的子集数量一致
num_classes=10        # 每个子集的类别数
epochs=100
batch_size=16
lr=0.0001             # 1e-4
num_workers=0
seed=42
val_split=0.1
device="mps"          # auto/cpu/mps/cuda
devices=1

# Loss weights (optimize cosine/relative quality)
kl_weight=0.0
mse_weight=1.0
cos_weight=5.0
rel_weight=1.0

# Weight normalization (per-vector)
weight_normalization="l2"
norm_eps=1e-8

# --- 执行 ---
echo "开始执行 train_vae_weight_encoder.py"
echo "-- 加载DNNWG库: $dnnwg_path"
echo "-- 设定随机种子为 $seed"
echo "-- 从 $weights_source_dir 读取 $num_subsets 个子集的 ResNet18 分类头权重"
echo "-- 训练 VAE 模型: Batch=$batch_size, LR=$lr, Epochs=$epochs, Val Split=$val_split"
echo "-- 目标输出目录: $output_dir"

python train_vae_weight_encoder.py \
    --dnnwg_path "$dnnwg_path" \
    --weights_dir "$weights_source_dir" \
    --output_dir "$output_dir" \
    --num_subsets $num_subsets \
    --num_classes $num_classes \
    --epochs $epochs \
    --batch_size $batch_size \
    --lr $lr \
    --weight_normalization $weight_normalization \
    --norm_eps $norm_eps \
    --kl_weight $kl_weight \
    --mse_weight $mse_weight \
    --cos_weight $cos_weight \
    --rel_weight $rel_weight \
    --val_split $val_split \
    --num_workers $num_workers \
    --seed $seed \
    --device $device \
    --devices $devices
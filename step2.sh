#!/bin/bash

# --- 实验配置 ---
# DNNWG 库的绝对路径 (请根据实际情况修改)
dnnwg_path="/root/External/DNNWG"

# 上一阶段生成的权重目录
weights_source_dir="./model_zoo/TinyImagenet_Resnet18"

# VAE 输出目录
output_dir="./components/resnet18_TinyImagenet_HC_VAE"

# 训练参数
num_subsets=500       # 需要与上一阶段生成的子集数量一致
num_classes=10        # 每个子集的类别数
epochs=100
batch_size=16
lr=0.0000045          # 4.5e-6
num_workers=4
seed=42
val_split=0.1
gpus=1                # 使用的GPU数量

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
    --val_split $val_split \
    --num_workers $num_workers \
    --seed $seed \
    --gpus $gpus
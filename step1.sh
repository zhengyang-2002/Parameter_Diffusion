#!/bin/bash

# --- 实验配置 ---
num_subsets=500
classes_per_subset=10
epochs=10 
batch_size=64 
lr=0.001 
num_workers=8 
data_dir="./tiny-imagenet-dataset"
output_dir="./model_zoo/TinyImagenet_Resnet18"
seed=42
val_split=0.1

# --- 执行 ---
echo "开始执行train_model_zoo.py"
echo "-- 设定随机种子为$seed"
echo "-- 下载并解压TinyImageNet数据集，使用$num_workers进行并行，在$data_dir中完成"
echo "-- 随机划分数据集，number of subset = $num_subsets, classes per subset = $classes_per_subset"
echo "-- 使用subsets只训练Resnet18的Classified Head，使用lr=$lr共训练$epochs个epochs，验证集比例为$val_split"
echo "-- 将训练完成的Classified Head存储至$output_dir"

python train_model_zoo.py \
    --num_subsets $num_subsets \
    --classes_per_subset $classes_per_subset \
    --epochs $epochs \
    --batch_size $batch_size \
    --lr $lr \
    --num_workers $num_workers \
    --data_dir "$data_dir" \
    --output_dir "$output_dir" \
    --seed $seed \
    --val_split $val_split

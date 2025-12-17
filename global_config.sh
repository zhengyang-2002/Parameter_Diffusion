#!/bin/bash

# ==============================================================================
# Global Configuration for Parameter Diffusion
# ==============================================================================

# --- 1. 基础路径配置 ---
export GLOBAL_DATA_DIR="./tiny-imagenet-data/tiny-imagenet-200"
export GLOBAL_DNNWG_PATH="./External/DNNWG"

# --- 2. 核心数据规格 ---
export GLOBAL_NUM_SUBSETS=500
export GLOBAL_NUM_CLASSES=10
export GLOBAL_SEED=42
export GLOBAL_VAL_SPLIT=0.1

# --- 3. 模块输出目录 ---
export DIR_MODEL_ZOO="./Model_Zoo/Resnet18_TinyImageNet_HC"
export DIR_VAE_OUTPUT="./components/vae"
export DIR_TASK_ENCODER_OUTPUT="./components/task_encoder"
export DIR_DIFFUSION_OUTPUT="./components/diffusion"

# --- 4. 潜在空间规格 ---
export GLOBAL_LATENT_CHANNELS=4
export GLOBAL_LATENT_SIZE=16

# --- 5. [新增] 推理与测试配置 (Inference & Testing) ---
# 指定要生成和测试的类别 (TinyImageNet Class IDs)
# 格式：逗号分隔，无空格
# 这里的10个类别必须是 TinyImageNet 中真实存在的文件夹名
export GLOBAL_TARGET_CLASSES="n02124075,n02364673,n02814533,n03160309,n03770439,n04070727,n04074963,n04146614,n04371430,n03100240"
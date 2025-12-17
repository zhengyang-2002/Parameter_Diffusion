#!/bin/bash

# ==============================================================================
# Global Configuration (No MPS Mode)
# ==============================================================================

# --- 1. 设备自动检测 (Auto-detect: CUDA or CPU only) ---
# 逻辑：检查 nvidia-smi 是否可用。如果可用，认为有 N 卡，使用 cuda；否则一律 cpu。
if command -v nvidia-smi &> /dev/null; then
    echo "[Config] Detected NVIDIA GPU. Using CUDA."
    export GLOBAL_DEVICE="cuda"
    
    # PyTorch Lightning 配置
    export PL_ACCELERATOR="gpu"
    export PL_DEVICES=1
    export USE_GPU_FLAG=1    # 给部分脚本传参用 (1=启用)
else
    echo "[Config] No NVIDIA GPU detected. Using CPU (MPS disabled)."
    export GLOBAL_DEVICE="cpu"
    
    # PyTorch Lightning 配置
    export PL_ACCELERATOR="cpu"
    export PL_DEVICES=1
    export USE_GPU_FLAG=0    # 给部分脚本传参用 (0=CPU)
fi

# --- 2. 基础路径配置 ---
export GLOBAL_DATA_DIR="./tiny-imagenet-data/tiny-imagenet-200"
export GLOBAL_DNNWG_PATH="./External/DNNWG"

# --- 3. 核心数据规格 ---
export GLOBAL_NUM_SUBSETS=500
export GLOBAL_NUM_CLASSES=10
export GLOBAL_SEED=42
export GLOBAL_VAL_SPLIT=0.1

# --- 4. 模块输出目录 ---
export DIR_MODEL_ZOO="./Model_Zoo/Resnet18_TinyImageNet_HC"
export DIR_VAE_OUTPUT="./components/vae"
export DIR_TASK_ENCODER_OUTPUT="./components/task_encoder"
export DIR_DIFFUSION_OUTPUT="./components/diffusion"

# --- 5. 潜在空间规格 ---
export GLOBAL_LATENT_CHANNELS=4
export GLOBAL_LATENT_SIZE=16

# --- 6. 测试目标类别 ---
export GLOBAL_TARGET_CLASSES="n02124075,n02364673,n02814533,n03160309,n03770439,n04070727,n04074963,n04146614,n04371430,n03100240"
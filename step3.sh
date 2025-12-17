#!/bin/bash
source ./global_config.sh

# --- 本步骤独立的训练超参数 ---
current_epochs=100
current_batch_size=16
current_lr=0.0001
embed_dim=1024
temperature=0.07
num_samples=5            # 每类采样几张图 (Task Encoder 特有参数)
num_workers=4

# 设备配置
accelerator="mps"        # Mac M系列芯片
devices=1

# 自动推断 VAE Checkpoint 路径 (取 last.ckpt)
vae_ckpt_path="${DIR_VAE_OUTPUT}/checkpoints/last.ckpt"

# --- 执行 ---
echo "Step 3: 训练 Task Encoder"
echo "  - TinyImageNet: $GLOBAL_DATA_DIR"
echo "  - VAE Checkpoint: $vae_ckpt_path"

# 检查 VAE 是否存在 (可选)
if [ ! -f "$vae_ckpt_path" ]; then
    echo "Warning: VAE checkpoint not found at $vae_ckpt_path, using simple encoder mode."
    vae_ckpt_path="" # 置空以触发 simple 模式
fi

python train_task_encoder.py \
    --weights_dir "$DIR_MODEL_ZOO" \
    --tinyimagenet_dir "$GLOBAL_DATA_DIR" \
    --output_dir "$DIR_TASK_ENCODER_OUTPUT" \
    --dnnwg_path "$GLOBAL_DNNWG_PATH" \
    --num_subsets $GLOBAL_NUM_SUBSETS \
    --num_classes $GLOBAL_NUM_CLASSES \
    --num_samples $num_samples \
    --epochs $current_epochs \
    --batch_size $current_batch_size \
    --lr $current_lr \
    --embed_dim $embed_dim \
    --temperature $temperature \
    --num_workers $num_workers \
    --seed $GLOBAL_SEED \
    --accelerator $accelerator \
    --devices $devices \
    --val_split $GLOBAL_VAL_SPLIT \
    --vae_checkpoint "$vae_ckpt_path"
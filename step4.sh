#!/bin/bash
source ./global_config.sh

# --- 本步骤独立的训练超参数 ---
dry_run=0
log_every_n_steps=10

current_epochs=500
current_batch_size=8      # Diffusion 显存占用大，通常较小
current_lr=0.0001
num_timesteps=1000
beta_schedule="linear"
num_samples=5             # Conditioning 采样数
num_workers=2

accelerator="mps"
devices=1

# 自动推断上一阶段的 Checkpoints
vae_ckpt_path="${DIR_VAE_OUTPUT}/checkpoints/last.ckpt"
task_encoder_ckpt_path="${DIR_TASK_ENCODER_OUTPUT}/checkpoints/last.ckpt"

# --- 执行 ---
echo "Step 4: 训练 Latent Diffusion"
echo "  - Latent Spec: ${GLOBAL_LATENT_CHANNELS} x ${GLOBAL_LATENT_SIZE} x ${GLOBAL_LATENT_SIZE}"

# 简易模式参数调整
if [ "$dry_run" -eq 1 ]; then
    echo "!!! DRY RUN MODE !!!"
    current_epochs=1
    GLOBAL_NUM_SUBSETS=8  # 临时覆盖全局变量仅用于 dry run
fi

python train_diffusion.py \
    --weights_dir "$DIR_MODEL_ZOO" \
    --tinyimagenet_dir "$GLOBAL_DATA_DIR" \
    --dnnwg_path "$GLOBAL_DNNWG_PATH" \
    --output_dir "$DIR_DIFFUSION_OUTPUT" \
    --vae_checkpoint "$vae_ckpt_path" \
    --task_encoder_checkpoint "$task_encoder_ckpt_path" \
    --num_subsets $GLOBAL_NUM_SUBSETS \
    --num_classes $GLOBAL_NUM_CLASSES \
    --num_samples $num_samples \
    --num_timesteps $num_timesteps \
    --beta_schedule $beta_schedule \
    --latent_channels $GLOBAL_LATENT_CHANNELS \
    --latent_size $GLOBAL_LATENT_SIZE \
    --epochs $current_epochs \
    --batch_size $current_batch_size \
    --lr $current_lr \
    --num_workers $num_workers \
    --seed $GLOBAL_SEED \
    --log_every_n_steps $log_every_n_steps \
    --accelerator $accelerator \
    --devices $devices \
    --val_split $GLOBAL_VAL_SPLIT
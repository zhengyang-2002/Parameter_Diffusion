#!/bin/bash
source ./global_config.sh

# --- 独立训练参数 ---
current_epochs=500
current_batch_size=8
current_lr=0.0001
num_timesteps=1000
beta_schedule="linear"
num_samples=5
num_workers=2
log_every_n_steps=10

vae_ckpt_path="${DIR_VAE_OUTPUT}/checkpoints/last.ckpt"
task_encoder_ckpt_path="${DIR_TASK_ENCODER_OUTPUT}/checkpoints/last.ckpt"

echo "Step 4: 训练 Diffusion (Accel: $PL_ACCELERATOR)"

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
    --accelerator $PL_ACCELERATOR \
    --devices $PL_DEVICES \
    --val_split $GLOBAL_VAL_SPLIT
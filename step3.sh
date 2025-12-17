#!/bin/bash
source ./global_config.sh

# --- 独立训练参数 ---
current_epochs=100
current_batch_size=16
current_lr=0.0001
embed_dim=1024
temperature=0.07
num_samples=5
num_workers=4

vae_ckpt_path="${DIR_VAE_OUTPUT}/checkpoints/last.ckpt"

echo "Step 3: 训练 Task Encoder (Accel: $PL_ACCELERATOR)"

if [ ! -f "$vae_ckpt_path" ]; then
    echo "Warning: VAE checkpoint not found, using simple mode."
    vae_ckpt_path=""
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
    --accelerator $PL_ACCELERATOR \
    --devices $PL_DEVICES \
    --val_split $GLOBAL_VAL_SPLIT \
    --vae_checkpoint "$vae_ckpt_path"
#!/bin/bash
source ./global_config.sh

# --- 独立训练参数 ---
current_epochs=100
current_batch_size=16
current_lr=0.0000045
num_workers=4

echo "Step 2: 训练 VAE (Device: $GLOBAL_DEVICE)"

python train_vae_weight_encoder.py \
    --dnnwg_path "$GLOBAL_DNNWG_PATH" \
    --weights_dir "$DIR_MODEL_ZOO" \
    --output_dir "$DIR_VAE_OUTPUT" \
    --num_subsets $GLOBAL_NUM_SUBSETS \
    --num_classes $GLOBAL_NUM_CLASSES \
    --epochs $current_epochs \
    --batch_size $current_batch_size \
    --lr $current_lr \
    --val_split $GLOBAL_VAL_SPLIT \
    --num_workers $num_workers \
    --seed $GLOBAL_SEED \
    --gpus $USE_GPU_FLAG
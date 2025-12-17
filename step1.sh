#!/bin/bash
source ./global_config.sh

# --- 独立训练参数 ---
current_epochs=10 
current_batch_size=64 
current_lr=0.001 
num_workers=8 

echo "Step 1: 生成 Model Zoo (Device: $GLOBAL_DEVICE)"

python train_model_zoo.py \
    --num_subsets $GLOBAL_NUM_SUBSETS \
    --classes_per_subset $GLOBAL_NUM_CLASSES \
    --epochs $current_epochs \
    --batch_size $current_batch_size \
    --lr $current_lr \
    --num_workers $num_workers \
    --data_dir "$GLOBAL_DATA_DIR" \
    --output_dir "$DIR_MODEL_ZOO" \
    --seed $GLOBAL_SEED \
    --val_split $GLOBAL_VAL_SPLIT \
    --device "$GLOBAL_DEVICE"
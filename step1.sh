#!/bin/bash
source ./global_config.sh

# --- 本步骤独立的训练超参数 (Independent Hyperparameters) ---
# 这些参数只影响当前步骤的训练效果，不需要与其他步骤一致
current_epochs=10 
current_batch_size=64 
current_lr=0.001 
num_workers=8 

# --- 执行 ---
echo "Step 1: 生成 Model Zoo"
echo "  - Subsets: $GLOBAL_NUM_SUBSETS"
echo "  - Classes: $GLOBAL_NUM_CLASSES"
echo "  - Output:  $DIR_MODEL_ZOO"

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
    --val_split $GLOBAL_VAL_SPLIT
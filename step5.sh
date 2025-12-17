#!/bin/bash
source ./global_config.sh

num_weights_to_generate=1
samples_per_class_for_cond=5
output_file="./generated_weights.pth"
diffusion_ckpt_path="${DIR_DIFFUSION_OUTPUT}/checkpoints/last.ckpt"

echo "Step 5: 生成权重 (Device: $GLOBAL_DEVICE)"

if [ ! -f "$diffusion_ckpt_path" ]; then
    echo "Error: Checkpoint not found at $diffusion_ckpt_path"
    exit 1
fi

python sample_weights.py \
    --diffusion_checkpoint "$diffusion_ckpt_path" \
    --tinyimagenet_dir "$GLOBAL_DATA_DIR" \
    --classes "$GLOBAL_TARGET_CLASSES" \
    --num_samples_per_class $samples_per_class_for_cond \
    --num_weights $num_weights_to_generate \
    --num_classes $GLOBAL_NUM_CLASSES \
    --output_file "$output_file" \
    --device "$GLOBAL_DEVICE" \
    --dnnwg_path "$GLOBAL_DNNWG_PATH"
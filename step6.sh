#!/bin/bash
source ./global_config.sh

weights_file="./generated_weights.pth"
batch_size=64

echo "Step 6: 测试权重 (Device: $GLOBAL_DEVICE)"

if [ ! -f "$weights_file" ]; then
    echo "Error: 找不到权重文件 $weights_file"
    exit 1
fi

python test_generated_weights.py \
    --weights_file "$weights_file" \
    --tinyimagenet_dir "$GLOBAL_DATA_DIR" \
    --classes "$GLOBAL_TARGET_CLASSES" \
    --batch_size $batch_size \
    --device "$GLOBAL_DEVICE"
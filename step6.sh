#!/bin/bash
source ./global_config.sh

# --- 测试参数 ---
weights_file="./generated_weights.pth"  # 对应 Step 5 的输出
batch_size=64

# 设备配置
device="mps"  # Mac 用户使用 mps

# --- 执行 ---
echo "Step 6: 测试生成的权重"
echo "  - Weights File: $weights_file"
echo "  - Dataset Dir:  $GLOBAL_DATA_DIR"
echo "  - Classes:      $GLOBAL_TARGET_CLASSES"

if [ ! -f "$weights_file" ]; then
    echo "Error: 找不到权重文件 $weights_file"
    echo "请先运行 step5.sh 生成权重。"
    exit 1
fi

python test_generated_weights.py \
    --weights_file "$weights_file" \
    --tinyimagenet_dir "$GLOBAL_DATA_DIR" \
    --classes "$GLOBAL_TARGET_CLASSES" \
    --batch_size $batch_size \
    --device "$device"
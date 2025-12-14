#!/bin/bash

# =============================================================================
# Step 3: Train Task Encoder (Dataset Conditioning Model)
# =============================================================================
#
# This script trains a "Task Encoder" that learns to embed image datasets into
# a conditioning vector that guides the diffusion model to generate appropriate
# neural network weights.
#
# The Task Encoder uses CLIP-style contrastive learning to align:
# - Weight embeddings (from trained classifier heads)
# - Dataset embeddings (from sample images of the training dataset)
#
# Prerequisites:
# - Model Zoo weights from Step 1 (in ./Model_Zoo/Resnet18_TinyImageNet_HC/)
# - TinyImageNet dataset downloaded
# - (Optional) VAE checkpoint from Step 2
# =============================================================================

# --- Configuration ---

# Path to trained classifier heads from Step 1 (or from HuggingFace)
weights_dir="./Model_Zoo/Resnet18_TinyImageNet_HC"

# Path to TinyImageNet dataset (will be downloaded if not present)
tinyimagenet_dir="./tiny-imagenet-data/tiny-imagenet-200"

# Output directory for Task Encoder checkpoints
output_dir="./components/task_encoder"

# (Optional) VAE checkpoint from Step 2 - set to empty string to use simple encoder
# vae_checkpoint=""
vae_checkpoint="./Pretrained_Components/VAE"

# Path to DNNWG library
dnnwg_path="./External/DNNWG"

# Training parameters
num_subsets=500          # Number of weight-dataset pairs to use
num_classes=10           # Number of classes per subset
num_samples=5            # Sample images per class for encoding
epochs=100               # Training epochs
batch_size=16            # Batch size (reduce if OOM)
lr=0.0001                # Learning rate
embed_dim=1024           # Embedding dimension
temperature=0.07         # Contrastive loss temperature
num_workers=4            # DataLoader workers
seed=42                  # Random seed
gpus=0                   # Keep 0 on macOS; use --accelerator mps below

# Device settings (Apple Silicon)
accelerator="mps"
devices=1

val_split=0.1            # Validation split ratio

# --- Download TinyImageNet if needed ---
if [ ! -d "$tinyimagenet_dir" ]; then
    echo "TinyImageNet not found. Downloading..."
    mkdir -p "./tiny-imagenet-data"
    cd "./tiny-imagenet-data"
    
    if [ ! -f "tiny-imagenet-200.zip" ]; then
        echo "Downloading TinyImageNet (~240MB)..."
        curl -L -o tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
    fi
    
    echo "Extracting..."
    unzip -q tiny-imagenet-200.zip
    cd ..
    echo "TinyImageNet downloaded and extracted."
fi

# --- Execution ---
echo "=============================================="
echo "Step 3: Train Task Encoder"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - Weights directory: $weights_dir"
echo "  - TinyImageNet directory: $tinyimagenet_dir"
echo "  - Output directory: $output_dir"
echo "  - VAE checkpoint: ${vae_checkpoint:-'Not using VAE (simple encoder)'}"
echo ""
echo "Training parameters:"
echo "  - Num subsets: $num_subsets"
echo "  - Num classes: $num_classes"
echo "  - Num samples per class: $num_samples"
echo "  - Epochs: $epochs"
echo "  - Batch size: $batch_size"
echo "  - Learning rate: $lr"
echo "  - Embedding dim: $embed_dim"
echo "  - Temperature: $temperature"
echo ""

# Build command
cmd="python train_task_encoder.py \
    --weights_dir \"$weights_dir\" \
    --tinyimagenet_dir \"$tinyimagenet_dir\" \
    --output_dir \"$output_dir\" \
    --dnnwg_path \"$dnnwg_path\" \
    --num_subsets $num_subsets \
    --num_classes $num_classes \
    --num_samples $num_samples \
    --epochs $epochs \
    --batch_size $batch_size \
    --lr $lr \
    --embed_dim $embed_dim \
    --temperature $temperature \
    --num_workers $num_workers \
    --seed $seed \
    --gpus $gpus \
    --accelerator $accelerator \
    --devices $devices \
    --val_split $val_split"

# Add VAE checkpoint if specified
if [ -n "$vae_checkpoint" ] && [ -f "$vae_checkpoint" ]; then
    cmd="$cmd --vae_checkpoint \"$vae_checkpoint\""
fi

echo "Running command:"
echo "$cmd"
echo ""

eval $cmd

#!/bin/bash

# =============================================================================
# Step 4: Train Latent Diffusion Model for Weight Generation
# =============================================================================
#
# This script trains a latent diffusion model that generates neural network
# weights conditioned on dataset samples.
#
# The diffusion model operates in the latent space of the VAE (from Step 2),
# and uses a Task Encoder to condition generation on dataset samples.
#
# Pipeline:
#   Dataset Samples → Task Encoder → Condition
#                                       ↓
#   Noise → Diffusion UNet → Latent → VAE Decoder → Generated Weights
#
# Prerequisites:
# - Model Zoo weights from Step 1 (in ./Model_Zoo/Resnet18_TinyImageNet_HC/)
# - (Recommended) VAE checkpoint from Step 2
# - (Optional) Task Encoder checkpoint from Step 3
# - TinyImageNet dataset
# =============================================================================

# --- Configuration ---

# Path to trained classifier heads from Step 1
weights_dir="./Model_Zoo/Resnet18_TinyImageNet_HC"

# Path to TinyImageNet dataset
tinyimagenet_dir="./tiny-imagenet-data/tiny-imagenet-200"

# (Optional) VAE checkpoint from Step 2
# If not provided, a simple VAE will be trained jointly
# vae_checkpoint=""
vae_checkpoint="./Pretrained_Components/VAE"

# (Optional) Task Encoder checkpoint from Step 3
# If not provided, task encoder is trained jointly with diffusion
task_encoder_checkpoint=""
# task_encoder_checkpoint="./components/task_encoder/checkpoints/last.ckpt"

# Path to DNNWG library
dnnwg_path="./External/DNNWG"

# Output directory
output_dir="./components/diffusion"

# Data parameters
num_subsets=500          # Number of weight-dataset pairs
num_classes=10           # Classes per subset
num_samples=5            # Sample images per class

# Model parameters
num_timesteps=1000       # Diffusion timesteps
beta_schedule="linear"   # Noise schedule: linear or cosine
latent_channels=4        # Latent space channels
latent_size=16           # Latent spatial size

# Training parameters
epochs=50               # Training epochs (staryt with 50 for first exp)
batch_size=8             # Batch size (optimized for M4 Pro, increase to 16 if memory allows)
lr=0.0001                # Learning rate
num_workers=2            # DataLoader workers (lower for macOS)
seed=42                  # Random seed

# Device configuration
# For Apple Silicon (M1/M2/M3/M4), we use MPS backend
# Set to "mps" for Mac, "cuda" for NVIDIA GPU, "cpu" for CPU only
accelerator="mps"        # Options: "mps", "cuda", "cpu", "auto"
devices=1                # Number of devices

val_split=0.1            # Validation split

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
echo "Step 4: Train Latent Diffusion Model"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - Weights directory: $weights_dir"
echo "  - TinyImageNet directory: $tinyimagenet_dir"
echo "  - Output directory: $output_dir"
echo "  - VAE checkpoint: ${vae_checkpoint:-'None (will train simple VAE)'}"
echo "  - Task Encoder checkpoint: ${task_encoder_checkpoint:-'None (will train jointly)'}"
echo ""
echo "Model parameters:"
echo "  - Diffusion timesteps: $num_timesteps"
echo "  - Beta schedule: $beta_schedule"
echo "  - Latent size: ${latent_channels}x${latent_size}x${latent_size}"
echo ""
echo "Training parameters:"
echo "  - Epochs: $epochs"
echo "  - Batch size: $batch_size"
echo "  - Learning rate: $lr"
echo "  - Accelerator: $accelerator"
echo ""

# Build command
cmd="python train_diffusion.py \
    --weights_dir \"$weights_dir\" \
    --tinyimagenet_dir \"$tinyimagenet_dir\" \
    --dnnwg_path \"$dnnwg_path\" \
    --output_dir \"$output_dir\" \
    --num_subsets $num_subsets \
    --num_classes $num_classes \
    --num_samples $num_samples \
    --num_timesteps $num_timesteps \
    --beta_schedule $beta_schedule \
    --latent_channels $latent_channels \
    --latent_size $latent_size \
    --epochs $epochs \
    --batch_size $batch_size \
    --lr $lr \
    --num_workers $num_workers \
    --seed $seed \
    --accelerator $accelerator \
    --devices $devices \
    --val_split $val_split"

# Add VAE checkpoint if specified
if [ -n "$vae_checkpoint" ] && [ -f "$vae_checkpoint" ]; then
    cmd="$cmd --vae_checkpoint \"$vae_checkpoint\""
fi

# Add Task Encoder checkpoint if specified
if [ -n "$task_encoder_checkpoint" ] && [ -f "$task_encoder_checkpoint" ]; then
    cmd="$cmd --task_encoder_checkpoint \"$task_encoder_checkpoint\""
fi

echo "Running command:"
echo "$cmd"
echo ""

eval $cmd

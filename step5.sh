#!/bin/bash

# =============================================================================
# Step 5: Sample/Generate Neural Network Weights
# =============================================================================
#
# This script uses the trained diffusion model to generate new neural network
# weights conditioned on sample images from a target dataset.
#
# Prerequisites:
# - Trained diffusion model from Step 4
# - Sample images from target dataset (or TinyImageNet)
# =============================================================================

# --- Configuration ---

# Path to trained diffusion model checkpoint
diffusion_checkpoint="./components/diffusion/checkpoints/last.ckpt"

# Option 1: Use TinyImageNet classes
tinyimagenet_dir="./tiny-imagenet-data/tiny-imagenet-200"
# Comma-separated class IDs (10 classes for ResNet18 head)
classes="n02124075,n02364673,n02814533,n03160309,n03770439,n04070727,n04074963,n04146614,n04371430,n03100240"

# Option 2: Use custom sample images directory
# sample_images_dir="./my_samples"  # Directory with subdirectories for each class

# Sampling parameters
num_samples_per_class=5   # Images per class for conditioning
num_weights=1             # Number of weight sets to generate
num_classes=10            # Number of classes

# Output
output_file="./generated_weights.pth"

# Device
device="cuda"  # "cuda" | "mps" | "cpu"
if [ "$(uname)" = "Darwin" ]; then
    device="mps"
fi

# --- Execution ---
echo "=============================================="
echo "Step 5: Generate Neural Network Weights"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - Diffusion checkpoint: $diffusion_checkpoint"
echo "  - Classes: $classes"
echo "  - Samples per class: $num_samples_per_class"
echo "  - Number of weights to generate: $num_weights"
echo "  - Output file: $output_file"
echo ""

# Resolve checkpoint if a directory is provided
if [ -d "$diffusion_checkpoint" ]; then
    if [ -f "$diffusion_checkpoint/checkpoints/last.ckpt" ]; then
        diffusion_checkpoint="$diffusion_checkpoint/checkpoints/last.ckpt"
    fi
fi

# Check if checkpoint exists
if [ ! -f "$diffusion_checkpoint" ]; then
    echo "Error: Diffusion checkpoint not found at $diffusion_checkpoint"
    echo "Please train the diffusion model first (Step 4)"
    exit 1
fi

# Run sampling
python sample_weights.py \
    --diffusion_checkpoint "$diffusion_checkpoint" \
    --tinyimagenet_dir "$tinyimagenet_dir" \
    --classes "$classes" \
    --num_samples_per_class $num_samples_per_class \
    --num_weights $num_weights \
    --num_classes $num_classes \
    --output_file "$output_file" \
    --device "$device"

echo ""
echo "=============================================="
echo "Weight generation complete!"
echo "=============================================="
echo ""
echo "Generated weights saved to: $output_file"
echo ""
echo "To use the generated weights in a ResNet18 classifier:"
echo ""
echo "  import torch"
echo "  import torchvision.models as models"
echo "  "
echo "  # Load generated weights"
echo "  data = torch.load('$output_file')"
echo "  "
echo "  # Create ResNet18"
echo "  model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)"
echo "  model.fc = torch.nn.Linear(512, 10)"
echo "  model.fc.weight.data = data['weight']"
echo "  model.fc.bias.data = data['bias']"
echo "  "
echo "  # Now model is ready for inference!"

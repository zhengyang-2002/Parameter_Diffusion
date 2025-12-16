#!/usr/bin/env python3
"""
Step 5: Sample/Generate Neural Network Weights using Trained Diffusion Model

This script uses the trained diffusion model to generate new neural network
weights conditioned on sample images from a target dataset.

Usage:
    python sample_weights.py \
        --diffusion_checkpoint ./components/diffusion/checkpoints/last.ckpt \
        --sample_images_dir ./my_dataset_samples \
        --output_file ./generated_weights.pth
    
    # Or generate from TinyImageNet classes:
    python sample_weights.py \
        --diffusion_checkpoint ./components/diffusion/checkpoints/last.ckpt \
        --tinyimagenet_dir ./tiny-imagenet-data/tiny-imagenet-200 \
        --classes n02124075,n02364673,n02814533 \
        --num_samples 10 \
        --output_file ./generated_weights.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from train_diffusion import LatentDiffusionModule, SimpleTaskEncoder


def load_images_from_dir(
    image_dir: str,
    num_samples: int = 5,
    image_size: int = 64,
) -> torch.Tensor:
    """Load and transform images from a directory."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + \
                  list(image_dir.glob("*.JPEG")) + \
                  list(image_dir.glob("*.png"))

    if len(image_files) == 0:
        return torch.zeros(num_samples, 3, image_size, image_size)
    
    if len(image_files) < num_samples:
        image_files = image_files * (num_samples // len(image_files) + 1)
    
    image_files = image_files[:num_samples]
    
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            images.append(torch.zeros(3, image_size, image_size))
    
    return torch.stack(images)


def load_tinyimagenet_classes(
    tinyimagenet_dir: str,
    class_names: List[str],
    num_samples: int = 5,
    image_size: int = 64,
) -> torch.Tensor:
    """Load sample images from TinyImageNet classes."""
    tinyimagenet_dir = Path(tinyimagenet_dir)
    train_dir = tinyimagenet_dir / "train"
    if not train_dir.exists():
        train_dir = tinyimagenet_dir
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    all_class_samples = []
    
    for class_name in class_names:
        class_dir = train_dir / class_name / "images"
        if not class_dir.exists():
            class_dir = train_dir / class_name
        
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            all_class_samples.append(torch.zeros(num_samples, 3, image_size, image_size))
            continue
        
        image_files = list(class_dir.glob("*.JPEG")) + \
                     list(class_dir.glob("*.jpg")) + \
                     list(class_dir.glob("*.png"))

        if len(image_files) == 0:
            print(f"Warning: No images found in: {class_dir}")
            all_class_samples.append(torch.zeros(num_samples, 3, image_size, image_size))
            continue
        
        if len(image_files) < num_samples:
            image_files = image_files * (num_samples // len(image_files) + 1)
        
        # Random sample
        indices = torch.randperm(len(image_files))[:num_samples].tolist()
        
        class_images = []
        for idx in indices:
            try:
                img = Image.open(image_files[int(idx)]).convert('RGB')
                img_tensor = transform(img)
                class_images.append(img_tensor)
            except:
                class_images.append(torch.zeros(3, image_size, image_size))
        
        all_class_samples.append(torch.stack(class_images))
    
    # (num_classes, num_samples, 3, H, W)
    return torch.stack(all_class_samples)


def evaluate_generated_weights(
    weights: torch.Tensor,
    test_loader,
    device: str = 'cuda',
) -> float:
    """Evaluate generated weights on a test set."""
    import torchvision.models as models
    
    # Create ResNet18 with generated head
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Extract weight and bias from generated weights
    num_classes = 10
    w = weights[:num_classes * 512].view(num_classes, 512)
    b = weights[num_classes * 512:num_classes * 512 + num_classes]
    
    # Set the classifier head
    model.fc = torch.nn.Linear(512, num_classes)
    model.fc.weight.data = w
    model.fc.bias.data = b
    
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Step 5: Sample Neural Network Weights")
    
    # Model paths
    parser.add_argument("--diffusion_checkpoint", type=str, required=True,
                        help="Path to trained diffusion model checkpoint")
    parser.add_argument("--dnnwg_path", type=str, default="./External/DNNWG",
                        help="Path to DNNWG library")
    
    # Input options (choose one)
    parser.add_argument("--sample_images_dir", type=str, default=None,
                        help="Directory containing sample images (organized by class)")
    parser.add_argument("--tinyimagenet_dir", type=str, default=None,
                        help="TinyImageNet directory")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated list of class names (e.g., n02124075,n02364673)")
    
    # Sampling parameters
    parser.add_argument("--num_samples_per_class", type=int, default=5,
                        help="Number of sample images per class")
    parser.add_argument("--num_weights", type=int, default=1,
                        help="Number of weight sets to generate")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes")
    
    # DDIM sampling (faster)
    parser.add_argument("--ddim_steps", type=int, default=None,
                        help="DDIM sampling steps (None = full DDPM)")
    
    # Output
    parser.add_argument("--output_file", type=str, default="./generated_weights.pth",
                        help="Output file for generated weights")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()

    # Resolve device preference (supports cuda/mps/cpu)
    requested = (args.device or "").lower()
    if requested == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif requested == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    elif requested == "cpu":
        device = "cpu"
    else:
        # Auto-select best available if request isn't usable
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        if requested and requested != device:
            print(f"Warning: requested device '{requested}' not available; using '{device}'.")

    print(f"Using device: {device}")
    
    # Load model
    # Accept either a direct ckpt path, or an output dir containing checkpoints/last.ckpt
    ckpt_path = Path(args.diffusion_checkpoint)
    if ckpt_path.is_dir():
        candidate = ckpt_path / "checkpoints" / "last.ckpt"
        if candidate.exists():
            ckpt_path = candidate
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {ckpt_path}")

    print(f"Loading diffusion model from {ckpt_path}...")
    
    model = LatentDiffusionModule.load_from_checkpoint(
        str(ckpt_path),
        map_location=device
    )
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Prepare conditioning images
    if args.sample_images_dir:
        # Load from custom directory (expects subdirectories for each class)
        sample_dir = Path(args.sample_images_dir)
        class_dirs = sorted([d for d in sample_dir.iterdir() if d.is_dir()])[:args.num_classes]
        
        all_samples = []
        for class_dir in class_dirs:
            samples = load_images_from_dir(
                str(class_dir),
                num_samples=args.num_samples_per_class,
            )
            all_samples.append(samples)
        
        dataset_samples = torch.stack(all_samples)  # (num_classes, num_samples, 3, H, W)
        
    elif args.tinyimagenet_dir and args.classes:
        # Load from TinyImageNet
        class_names = args.classes.split(',')
        dataset_samples = load_tinyimagenet_classes(
            args.tinyimagenet_dir,
            class_names,
            num_samples=args.num_samples_per_class,
        )
    else:
        raise ValueError("Must provide either --sample_images_dir or both --tinyimagenet_dir and --classes")
    
    print(f"Loaded conditioning images: {dataset_samples.shape}")
    
    # Generate conditioning
    print("Encoding dataset condition...")
    with torch.no_grad():
        # Add batch dimension
        dataset_samples_batch = [dataset_samples.to(device)]
        cond = model.task_encoder(dataset_samples_batch)
        
        # Repeat for multiple samples
        if args.num_weights > 1:
            cond = cond.repeat(args.num_weights, 1, 1, 1)
    
    print(f"Condition shape: {cond.shape}")
    
    # Sample weights
    print(f"\nGenerating {args.num_weights} weight set(s)...")
    with torch.no_grad():
        generated_weights = model.sample(cond, num_samples=args.num_weights)
    
    print(f"Generated weights shape: {generated_weights.shape}")
    
    # Save weights
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to state dict format (compatible with ResNet18)
    saved_data = {
        'generated_weights': generated_weights.cpu(),
        'num_weights': args.num_weights,
        'num_classes': args.num_classes,
    }
    
    # Also save in classifier head format
    if args.num_weights == 1:
        w = generated_weights[0, :args.num_classes * 512].view(args.num_classes, 512)
        b = generated_weights[0, args.num_classes * 512:args.num_classes * 512 + args.num_classes]
        saved_data['weight'] = w
        saved_data['bias'] = b
    
    torch.save(saved_data, output_path)
    print(f"\nSaved generated weights to: {output_path}")
    
    # Print weight statistics
    print("\nWeight statistics:")
    print(f"  Mean: {generated_weights.mean().item():.4f}")
    print(f"  Std:  {generated_weights.std().item():.4f}")
    print(f"  Min:  {generated_weights.min().item():.4f}")
    print(f"  Max:  {generated_weights.max().item():.4f}")
    
    print("\nDone! Use the generated weights in a ResNet18 classifier.")


if __name__ == "__main__":
    main()

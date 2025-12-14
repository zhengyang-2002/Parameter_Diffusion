#!/usr/bin/env python3
"""
Step 3: Train Task Encoder (Dataset Conditioning Model)

This script trains a "Task Encoder" that learns to embed image datasets into a 
conditioning space that can guide the diffusion model to generate appropriate weights.

The Task Encoder uses CLIP-style contrastive learning to align:
- Weight embeddings (from the VAE encoder trained in Step 2)
- Dataset embeddings (from a Set Transformer that processes sample images)

Two modes are supported:
1. "clip" mode: Full CLIP-style contrastive training (requires VAE checkpoint)
2. "simple" mode: Just train the Set Transformer with reconstruction objective

Usage:
    python train_task_encoder.py \
        --mode simple \
        --weights_dir ./Model_Zoo/Resnet18_TinyImageNet_HC \
        --tinyimagenet_dir ./tiny-imagenet-data/tiny-imagenet-200 \
        --output_dir ./components/task_encoder
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from tqdm import tqdm
import yaml
from einops import rearrange

torch.set_float32_matmul_precision('medium')


# ==============================================================================
# Dataset: Pairs of (trained weights, corresponding dataset samples)
# ==============================================================================

class WeightDatasetPairDataset(Dataset):
    """
    Dataset that pairs trained classifier weights with sample images from the 
    corresponding dataset subset.
    
    Each item returns:
    - weight: Flattened weight vector from trained classifier head
    - dataset: List of image tensors sampled from each class used to train that head
    """
    
    def __init__(
        self,
        weights_dir: str,
        tinyimagenet_train_dir: str,
        mapping_file: str,
        num_subsets: int = 500,
        num_samples_per_class: int = 5,
        num_classes: int = 10,
        image_size: int = 64,
        weight_target_size: int = 5130,  # 10 classes * 512 features + 10 bias
    ):
        super().__init__()
        self.weights_dir = Path(weights_dir)
        self.train_dir = Path(tinyimagenet_train_dir)
        self.num_samples = num_samples_per_class
        self.num_classes = num_classes
        self.weight_target_size = weight_target_size
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load mapping
        with open(mapping_file, 'r') as f:
            self.mapping = json.load(f)
        
        # Filter to valid subsets
        self.valid_subsets = []
        print(f"Loading {num_subsets} weight-dataset pairs...")
        
        for i in tqdm(range(1, num_subsets + 1), desc="Validating subsets"):
            subset_key = f"subset_{i}"
            weight_file = self.weights_dir / f"resnet18_head_{subset_key}.pth"
            
            if subset_key not in self.mapping:
                continue
            if not weight_file.exists():
                continue
                
            # Check if all classes exist
            classes = self.mapping[subset_key]["classes"]
            all_exist = all((self.train_dir / c).exists() for c in classes[:num_classes])
            
            if all_exist:
                self.valid_subsets.append({
                    'subset_key': subset_key,
                    'weight_file': weight_file,
                    'classes': classes[:num_classes]
                })
        
        print(f"Found {len(self.valid_subsets)} valid weight-dataset pairs")
        
        if len(self.valid_subsets) == 0:
            raise ValueError("No valid subsets found! Check paths.")
    
    def __len__(self):
        return len(self.valid_subsets)
    
    def _load_weight(self, weight_file: Path) -> torch.Tensor:
        """Load and flatten weight checkpoint."""
        checkpoint = torch.load(weight_file, map_location='cpu')
        w = checkpoint['weight'].flatten()
        b = checkpoint['bias'].flatten()
        full_vector = torch.cat([w, b])
        
        # Pad or truncate to target size
        current_size = full_vector.shape[0]
        if current_size < self.weight_target_size:
            pad_size = self.weight_target_size - current_size
            full_vector = F.pad(full_vector, (0, pad_size), value=0)
        elif current_size > self.weight_target_size:
            full_vector = full_vector[:self.weight_target_size]
        
        return full_vector
    
    def _load_dataset_samples(self, classes: List[str]) -> torch.Tensor:
        """Load sample images from each class."""
        all_samples = []
        
        for class_name in classes:
            class_dir = self.train_dir / class_name / "images"
            if not class_dir.exists():
                class_dir = self.train_dir / class_name  # Fallback
            
            # Get image files
            image_files = list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            if len(image_files) < self.num_samples:
                # Repeat if not enough samples
                image_files = image_files * (self.num_samples // len(image_files) + 1)
            
            # Random sample
            indices = torch.randperm(len(image_files))[:self.num_samples]
            
            class_samples = []
            for idx in indices:
                try:
                    from PIL import Image
                    img = Image.open(image_files[idx]).convert('RGB')
                    img_tensor = self.transform(img)
                    class_samples.append(img_tensor)
                except Exception as e:
                    # Use zeros as fallback
                    class_samples.append(torch.zeros(3, 64, 64))
            
            # Stack samples for this class: (num_samples, 3, H, W)
            class_tensor = torch.stack(class_samples)
            all_samples.append(class_tensor)
        
        # Stack all classes: (num_classes, num_samples, 3, H, W)
        return torch.stack(all_samples)
    
    def __getitem__(self, idx):
        subset_info = self.valid_subsets[idx]
        
        weight = self._load_weight(subset_info['weight_file'])
        dataset_samples = self._load_dataset_samples(subset_info['classes'])
        
        return {
            'weight': weight,
            'dataset': dataset_samples,  # (num_classes, num_samples, 3, H, W)
            'subset_key': subset_info['subset_key']
        }


# ==============================================================================
# Task Encoder Model: Set Transformer for dataset embedding
# ==============================================================================

class MultiheadAttentionBlock(nn.Module):
    """Multi-head attention block with optional layer norm."""
    
    def __init__(self, dim: int, num_heads: int = 4, ln: bool = True):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.ln2 = nn.LayerNorm(dim) if ln else nn.Identity()
    
    def forward(self, q, k, v):
        attn_out, _ = self.attention(q, k, v)
        x = self.ln(q + attn_out)
        ff_out = self.ff(x)
        return self.ln2(x + ff_out)


class ISAB(nn.Module):
    """Induced Set Attention Block - efficient attention for sets."""
    
    def __init__(self, dim: int, num_heads: int = 4, num_inducing: int = 32, ln: bool = True):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing, dim))
        self.mab1 = MultiheadAttentionBlock(dim, num_heads, ln)
        self.mab2 = MultiheadAttentionBlock(dim, num_heads, ln)
    
    def forward(self, x):
        # x: (batch, set_size, dim)
        batch_size = x.shape[0]
        inducing = self.inducing_points.expand(batch_size, -1, -1)
        
        # Inducing points attend to input
        h = self.mab1(inducing, x, x)
        # Input attends to inducing points
        return self.mab2(x, h, h)


class PMA(nn.Module):
    """Pooling by Multihead Attention - aggregates set to fixed size output."""
    
    def __init__(self, dim: int, num_heads: int = 4, num_outputs: int = 1, ln: bool = True):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_outputs, dim))
        self.mab = MultiheadAttentionBlock(dim, num_heads, ln)
    
    def forward(self, x):
        batch_size = x.shape[0]
        seeds = self.seeds.expand(batch_size, -1, -1)
        return self.mab(seeds, x, x)


class SetTransformerEncoder(nn.Module):
    """
    Set Transformer that encodes a set of images into a fixed-size embedding.
    Used as the "Task Encoder" / "Dataset Encoder" for conditioning.
    """
    
    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 1024,
        num_heads: int = 4,
        num_inducing: int = 32,
        num_classes: int = 10,
        num_samples: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        
        # Image encoder (simple CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Intra-class aggregation (aggregate samples within each class)
        self.intra_class = nn.Sequential(
            ISAB(embed_dim, num_heads, num_inducing),
            ISAB(embed_dim, num_heads, num_inducing),
        )
        self.intra_pool = PMA(embed_dim, num_heads, num_outputs=1)
        
        # Inter-class aggregation (aggregate class prototypes)
        self.inter_class = nn.Sequential(
            ISAB(embed_dim, num_heads, num_inducing),
            ISAB(embed_dim, num_heads, num_inducing),
        )
        self.inter_pool = PMA(embed_dim, num_heads, num_outputs=1)
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_classes, num_samples, 3, H, W)
        
        Returns:
            embedding: (batch, output_dim)
        """
        batch_size = x.shape[0]
        num_classes = x.shape[1]
        num_samples = x.shape[2]
        
        # Reshape for image encoder: (batch * num_classes * num_samples, 3, H, W)
        x_flat = x.view(-1, x.shape[3], x.shape[4], x.shape[5])
        
        # Encode all images: (batch * num_classes * num_samples, embed_dim)
        img_features = self.image_encoder(x_flat)
        
        # Reshape to (batch * num_classes, num_samples, embed_dim)
        img_features = img_features.view(batch_size * num_classes, num_samples, -1)
        
        # Intra-class: aggregate samples within each class
        class_features = self.intra_class(img_features)
        class_prototypes = self.intra_pool(class_features)  # (batch * num_classes, 1, embed_dim)
        class_prototypes = class_prototypes.squeeze(1)  # (batch * num_classes, embed_dim)
        
        # Reshape to (batch, num_classes, embed_dim)
        class_prototypes = class_prototypes.view(batch_size, num_classes, -1)
        
        # Inter-class: aggregate class prototypes
        dataset_features = self.inter_class(class_prototypes)
        dataset_embedding = self.inter_pool(dataset_features)  # (batch, 1, embed_dim)
        dataset_embedding = dataset_embedding.squeeze(1)  # (batch, embed_dim)
        
        # Project to output dimension
        return self.proj(dataset_embedding)


# ==============================================================================
# Weight Encoder (frozen VAE encoder from Step 2)
# ==============================================================================

class SimpleWeightEncoder(nn.Module):
    """
    Simple MLP-based weight encoder for when VAE is not available.
    Projects weight vectors to embedding space.
    """
    
    def __init__(self, weight_dim: int = 5130, hidden_dim: int = 1024, output_dim: int = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(weight_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


# ==============================================================================
# CLIP-style Task Encoder Training Module
# ==============================================================================

class TaskEncoderModule(pl.LightningModule):
    """
    PyTorch Lightning module for training the Task Encoder.
    
    Uses CLIP-style contrastive learning to align weight embeddings 
    with dataset embeddings.
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        temperature: float = 0.07,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        image_size: int = 64,
        num_classes: int = 10,
        num_samples: int = 5,
        weight_dim: int = 5130,
        vae_checkpoint: Optional[str] = None,
        dnnwg_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Dataset encoder (Task Encoder)
        self.dataset_encoder = SetTransformerEncoder(
            image_size=image_size,
            embed_dim=256,
            hidden_dim=512,
            output_dim=embed_dim,
            num_classes=num_classes,
            num_samples=num_samples,
        )
        
        # Weight encoder
        if vae_checkpoint and dnnwg_path and os.path.exists(vae_checkpoint):
            print(f"Loading VAE encoder from {vae_checkpoint}")
            self.weight_encoder = self._load_vae_encoder(vae_checkpoint, dnnwg_path)
            self.use_vae = True
        else:
            print("Using simple MLP weight encoder")
            self.weight_encoder = SimpleWeightEncoder(
                weight_dim=weight_dim,
                hidden_dim=1024,
                output_dim=embed_dim
            )
            self.use_vae = False
        
        # Projection heads
        self.weight_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.dataset_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def _load_vae_encoder(self, ckpt_path: str, dnnwg_path: str):
        """Load the encoder part of a trained VAE."""
        sys.path.insert(0, dnnwg_path)
        from stage1.models.autoencoder import VAENoDiscModel
        
        # Load config (you may need to adjust this based on your VAE)
        ddconfig = {
            "double_z": True,
            "z_channels": 4,
            "resolution": 64,
            "in_channels": 10,
            "my_channels": 10,
            "out_ch": 10,
            "ch": 64,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
            "in_dim": 513,
            "fdim": 2048
        }
        
        lossconfig = {
            "target": "stage1.modules.losses.CustomLosses.Myloss",
            "params": {"logvar_init": 0.0, "kl_weight": 1e-6}
        }
        
        # Create and load model
        vae = VAENoDiscModel(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=4,
            learning_rate=1e-4,
            input_key="weight"
        )
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            vae.load_state_dict(checkpoint, strict=False)
        
        # Freeze encoder
        for param in vae.encoder.parameters():
            param.requires_grad = False
        for param in vae.quant_conv.parameters():
            param.requires_grad = False
        
        return vae.encoder
    
    def encode_weights(self, weights):
        """Encode weights to embedding space."""
        if self.use_vae:
            # VAE encoder expects (batch, channels, dim)
            batch_size = weights.shape[0]
            w = weights.view(batch_size, 10, 513)  # Reshape for VAE
            h = self.weight_encoder(w)
            return h.view(batch_size, -1)[:, :self.hparams.embed_dim]
        else:
            return self.weight_encoder(weights)
    
    def forward(self, batch):
        weights = batch['weight']  # (batch, weight_dim)
        datasets = batch['dataset']  # (batch, num_classes, num_samples, 3, H, W)
        
        # Encode both modalities
        weight_embeds = self.encode_weights(weights)
        dataset_embeds = self.dataset_encoder(datasets)
        
        # Project
        weight_embeds = self.weight_proj(weight_embeds)
        dataset_embeds = self.dataset_proj(dataset_embeds)
        
        # Normalize
        weight_embeds = F.normalize(weight_embeds, dim=-1)
        dataset_embeds = F.normalize(dataset_embeds, dim=-1)
        
        return weight_embeds, dataset_embeds
    
    def contrastive_loss(self, weight_embeds, dataset_embeds):
        """Compute CLIP-style contrastive loss."""
        # Compute similarity matrix
        logits = (weight_embeds @ dataset_embeds.T) / self.temperature
        
        # Labels are diagonal (each weight matches its corresponding dataset)
        batch_size = weight_embeds.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross-entropy loss in both directions
        loss_w2d = F.cross_entropy(logits, labels)
        loss_d2w = F.cross_entropy(logits.T, labels)
        
        return (loss_w2d + loss_d2w) / 2
    
    def training_step(self, batch, batch_idx):
        weight_embeds, dataset_embeds = self(batch)
        loss = self.contrastive_loss(weight_embeds, dataset_embeds)
        
        # Compute accuracy
        with torch.no_grad():
            logits = weight_embeds @ dataset_embeds.T
            preds = logits.argmax(dim=1)
            labels = torch.arange(weight_embeds.shape[0], device=self.device)
            acc = (preds == labels).float().mean()
        
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/acc', acc, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        weight_embeds, dataset_embeds = self(batch)
        loss = self.contrastive_loss(weight_embeds, dataset_embeds)
        
        with torch.no_grad():
            logits = weight_embeds @ dataset_embeds.T
            preds = logits.argmax(dim=1)
            labels = torch.arange(weight_embeds.shape[0], device=self.device)
            acc = (preds == labels).float().mean()
        
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/acc', acc, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        # Only train dataset encoder and projections
        params = list(self.dataset_encoder.parameters())
        params += list(self.dataset_proj.parameters())
        
        if not self.use_vae:
            params += list(self.weight_encoder.parameters())
            params += list(self.weight_proj.parameters())
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


# ==============================================================================
# Data Module
# ==============================================================================

class TaskEncoderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        weights_dir: str,
        tinyimagenet_train_dir: str,
        mapping_file: str,
        num_subsets: int = 500,
        num_samples_per_class: int = 5,
        num_classes: int = 10,
        batch_size: int = 16,
        num_workers: int = 4,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage=None):
        full_dataset = WeightDatasetPairDataset(
            weights_dir=self.hparams.weights_dir,
            tinyimagenet_train_dir=self.hparams.tinyimagenet_train_dir,
            mapping_file=self.hparams.mapping_file,
            num_subsets=self.hparams.num_subsets,
            num_samples_per_class=self.hparams.num_samples_per_class,
            num_classes=self.hparams.num_classes,
        )
        
        val_size = int(len(full_dataset) * self.hparams.val_split)
        train_size = len(full_dataset) - val_size
        
        self.train_ds, self.val_ds = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Dataset split: Train {train_size}, Val {val_size}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Step 3: Train Task Encoder")
    
    # Paths
    parser.add_argument("--weights_dir", type=str, required=True,
                        help="Directory containing trained ResNet18 heads from Step 1")
    parser.add_argument("--tinyimagenet_dir", type=str, required=True,
                        help="Path to Tiny-ImageNet train directory")
    parser.add_argument("--output_dir", type=str, default="./components/task_encoder",
                        help="Output directory for checkpoints")
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to VAE checkpoint from Step 2 (optional)")
    parser.add_argument("--dnnwg_path", type=str, default="./External/DNNWG",
                        help="Path to DNNWG library")
    
    # Data params
    parser.add_argument("--num_subsets", type=int, default=500,
                        help="Number of weight subsets to use")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes per subset")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of sample images per class")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=1024,
                        help="Embedding dimension")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Contrastive loss temperature")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine TinyImageNet train path
    tinyimagenet_train = Path(args.tinyimagenet_dir) / "train"
    if not tinyimagenet_train.exists():
        tinyimagenet_train = Path(args.tinyimagenet_dir)
    
    # Mapping file
    mapping_file = Path(args.weights_dir) / "subset_class_mapping.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    print(f"Weights directory: {args.weights_dir}")
    print(f"TinyImageNet train: {tinyimagenet_train}")
    print(f"Mapping file: {mapping_file}")
    
    # Save config
    config = vars(args)
    with open(output_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Data module
    dm = TaskEncoderDataModule(
        weights_dir=args.weights_dir,
        tinyimagenet_train_dir=str(tinyimagenet_train),
        mapping_file=str(mapping_file),
        num_subsets=args.num_subsets,
        num_samples_per_class=args.num_samples,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )
    
    # Model
    model = TaskEncoderModule(
        embed_dim=args.embed_dim,
        temperature=args.temperature,
        learning_rate=args.lr,
        image_size=64,
        num_classes=args.num_classes,
        num_samples=args.num_samples,
        vae_checkpoint=args.vae_checkpoint,
        dnnwg_path=args.dnnwg_path,
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_path / 'checkpoints',
            filename='task_encoder-{epoch:02d}-{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=20,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=output_path,
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 'auto',
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    
    print(f"\nStarting Task Encoder training...")
    print(f"Output directory: {args.output_dir}")
    trainer.fit(model, dm)
    
    # Save final model
    print("\nSaving final model...")
    final_ckpt = trainer.checkpoint_callback.best_model_path
    if final_ckpt:
        print(f"Best checkpoint: {final_ckpt}")
        
        # Also save just the dataset encoder for use in diffusion
        model = TaskEncoderModule.load_from_checkpoint(final_ckpt)
        torch.save(
            model.dataset_encoder.state_dict(),
            output_path / 'dataset_encoder.pth'
        )
        print(f"Saved dataset encoder to: {output_path / 'dataset_encoder.pth'}")
    
    print("\nTask Encoder training complete!")


if __name__ == "__main__":
    main()

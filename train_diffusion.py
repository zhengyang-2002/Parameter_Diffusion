#!/usr/bin/env python3
"""
Step 4: Train Latent Diffusion Model for Weight Generation

This script trains a latent diffusion model that generates neural network weights
conditioned on dataset samples. It operates in the latent space of the VAE trained
in Step 2, and uses the Task Encoder (optional, from Step 3) for conditioning.

Pipeline:
    Dataset Samples → Task Encoder → Condition
                                        ↓
    Noise → Diffusion UNet → Latent → VAE Decoder → Generated Weights

Usage:
    python train_diffusion.py \
        --weights_dir ./Model_Zoo/Resnet18_TinyImageNet_HC \
        --tinyimagenet_dir ./tiny-imagenet-data/tiny-imagenet-200 \
        --vae_checkpoint ./components/vae/checkpoints/last.ckpt \
        --output_dir ./components/diffusion
"""

import os
import sys
import json
import argparse
import datetime
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from tqdm import tqdm
import yaml
from einops import rearrange, repeat
from functools import partial
from contextlib import contextmanager

torch.set_float32_matmul_precision('medium')


# ==============================================================================
# Dataset: Pairs of (trained weights, corresponding dataset samples)
# ==============================================================================

class DiffusionDataset(Dataset):
    """
    Dataset for training diffusion model on weight-dataset pairs.
    Returns flattened weights and corresponding dataset sample images.
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
        weight_target_size: int = 5130,
    ):
        super().__init__()
        self.weights_dir = Path(weights_dir)
        self.train_dir = Path(tinyimagenet_train_dir)
        self.num_samples = num_samples_per_class
        self.num_classes = num_classes
        self.weight_target_size = weight_target_size
        self.image_size = image_size
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load mapping
        with open(mapping_file, 'r') as f:
            self.mapping = json.load(f)
        
        # Preload weights (they're small enough)
        self.data = []
        print(f"Loading {num_subsets} weight-dataset pairs...")
        
        for i in tqdm(range(1, num_subsets + 1), desc="Loading data"):
            subset_key = f"subset_{i}"
            weight_file = self.weights_dir / f"resnet18_head_{subset_key}.pth"
            
            if subset_key not in self.mapping:
                continue
            if not weight_file.exists():
                continue
            
            classes = self.mapping[subset_key]["classes"][:num_classes]
            
            # Check if all classes exist
            all_exist = all((self.train_dir / c).exists() or 
                           (self.train_dir / c / "images").exists() for c in classes)
            
            if not all_exist:
                continue
            
            # Load weight
            try:
                checkpoint = torch.load(weight_file, map_location='cpu')
                w = checkpoint['weight'].flatten()
                b = checkpoint['bias'].flatten()
                weight = torch.cat([w, b])
                
                # Pad/truncate
                if weight.shape[0] < self.weight_target_size:
                    weight = F.pad(weight, (0, self.weight_target_size - weight.shape[0]))
                elif weight.shape[0] > self.weight_target_size:
                    weight = weight[:self.weight_target_size]
                
                self.data.append({
                    'weight': weight,
                    'classes': classes,
                    'subset_key': subset_key
                })
            except Exception as e:
                print(f"Error loading {weight_file}: {e}")
                continue
        
        print(f"Loaded {len(self.data)} valid samples")
        
        if len(self.data) == 0:
            raise ValueError("No valid data found!")
    
    def __len__(self):
        return len(self.data)
    
    def _load_dataset_samples(self, classes: List[str]) -> List[torch.Tensor]:
        """Load sample images from each class."""
        all_class_samples = []
        
        for class_name in classes:
            class_dir = self.train_dir / class_name / "images"
            if not class_dir.exists():
                class_dir = self.train_dir / class_name
            
            image_files = list(class_dir.glob("*.JPEG")) + \
                         list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.png"))
            
            if len(image_files) < self.num_samples:
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
                except:
                    class_samples.append(torch.zeros(3, self.image_size, self.image_size))
            
            # (num_samples, 3, H, W)
            all_class_samples.append(torch.stack(class_samples))
        
        # (num_classes, num_samples, 3, H, W)
        return torch.stack(all_class_samples)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        dataset_samples = self._load_dataset_samples(item['classes'])
        
        return {
            'weight': item['weight'].float(),
            'dataset': dataset_samples,  # (num_classes, num_samples, 3, H, W)
        }


def collate_fn(batch):
    """Custom collate function for the diffusion dataset."""
    weights = torch.stack([item['weight'] for item in batch])
    datasets = [item['dataset'] for item in batch]  # List of tensors
    
    return {
        'weight': weights,
        'dataset': datasets
    }


# ==============================================================================
# VAE Encoder/Decoder (from Step 2)
# ==============================================================================

class VAEWrapper(nn.Module):
    """Wrapper for the VAE from Step 2."""
    
    def __init__(self, vae_checkpoint: str, dnnwg_path: str, device: str = 'cuda'):
        super().__init__()
        
        sys.path.insert(0, dnnwg_path)
        from stage1.models.autoencoder import VAENoDiscModel
        from stage1.modules.distributions import DiagonalGaussianDistribution
        
        self.DiagonalGaussianDistribution = DiagonalGaussianDistribution
        
        # VAE config for ResNet18 heads (10 classes * 512 features + 10 bias = 5130)
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
        
        self.vae = VAENoDiscModel(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=4,
            learning_rate=1e-4,
            input_key="weight",
            device=device
        )
        
        # Load checkpoint
        if os.path.exists(vae_checkpoint):
            checkpoint = torch.load(vae_checkpoint, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.vae.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.vae.load_state_dict(checkpoint, strict=False)
            print(f"Loaded VAE from {vae_checkpoint}")
        else:
            print(f"Warning: VAE checkpoint not found at {vae_checkpoint}")
        
        # Freeze VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def encode(self, x):
        """Encode weights to latent space."""
        out_device = x.device
        try:
            vae_device = next(self.vae.parameters()).device
        except StopIteration:
            vae_device = torch.device('cpu')
        x_in = x.detach().to(vae_device)
        with torch.no_grad():
            posterior = self.vae.encode(x_in)
            z = posterior.sample()
        return z.to(out_device)
    
    def decode(self, z):
        """Decode latent to weights."""
        out_device = z.device
        try:
            vae_device = next(self.vae.parameters()).device
        except StopIteration:
            vae_device = torch.device('cpu')
        z_in = z.detach().to(vae_device)
        with torch.no_grad():
            x = self.vae.decode(z_in)
        return x.to(out_device)


# ==============================================================================
# Simple VAE (fallback when no checkpoint available)
# ==============================================================================

class SimpleVAE(nn.Module):
    """Simple VAE for when trained VAE is not available."""
    
    def __init__(self, input_dim: int = 5130, latent_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)


# ==============================================================================
# Task Encoder (Dataset Conditioning)
# ==============================================================================

class SimpleTaskEncoder(nn.Module):
    """
    MLP-based task encoder that processes dataset samples.
    This is trained jointly with the diffusion model.
    """
    
    def __init__(
        self,
        image_size: int = 64,
        num_classes: int = 10,
        num_samples: int = 5,
        embed_dim: int = 256,
        latent_channels: int = 4,
        latent_h: int = 16,
        latent_w: int = 16,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        
        # Simple CNN to encode each image
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16->8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # Aggregate samples
        self.sample_agg = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )
        
        # Aggregate classes
        self.class_agg = nn.Sequential(
            nn.Linear(embed_dim * num_classes, embed_dim),
            nn.ReLU(),
        )
        
        # Project to condition size (matches latent size)
        self.proj = nn.Linear(embed_dim, latent_channels * latent_h * latent_w)
    
    def forward(self, dataset_samples):
        """
        Args:
            dataset_samples: List of (num_classes, num_samples, 3, H, W) tensors
        
        Returns:
            condition: (batch, latent_channels, latent_h, latent_w)
        """
        batch_outputs = []
        
        for x in dataset_samples:
            # x: (num_classes, num_samples, 3, H, W)
            x = x.to(next(self.parameters()).device)
            nc, ns, c, h, w = x.shape
            
            # Encode all images
            x_flat = x.view(nc * ns, c, h, w)
            img_features = self.image_encoder(x_flat)  # (nc*ns, 128)
            img_features = img_features.view(nc, ns, -1)  # (nc, ns, 128)
            
            # Average pool over samples
            class_features = img_features.mean(dim=1)  # (nc, 128)
            class_features = self.sample_agg(class_features)  # (nc, embed_dim)
            
            # Flatten and aggregate classes
            class_features = class_features.flatten()  # (nc * embed_dim,)
            dataset_feature = self.class_agg(class_features)  # (embed_dim,)
            
            batch_outputs.append(dataset_feature)
        
        # Stack batch
        batch_features = torch.stack(batch_outputs)  # (batch, embed_dim)
        
        # Project to latent size
        cond = self.proj(batch_features)  # (batch, latent_channels * latent_h * latent_w)
        cond = cond.view(-1, self.latent_channels, self.latent_h, self.latent_w)
        
        return cond


# ==============================================================================
# Diffusion Utilities
# ==============================================================================

def make_beta_schedule(schedule: str, n_timestep: int, linear_start: float = 1e-4, linear_end: float = 2e-2):
    """Create noise schedule."""
    if schedule == "linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep)
    elif schedule == "cosine":
        steps = n_timestep + 1
        x = torch.linspace(0, n_timestep, steps)
        alphas_cumprod = torch.cos(((x / n_timestep) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return betas


def extract(a, t, x_shape):
    """Extract values from a based on indices t."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# ==============================================================================
# UNet for Diffusion
# ==============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = F.silu(h)
        
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    """Simple UNet for diffusion."""
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        cond_channels: int = 4,
        base_channels: int = 64,
        channel_mult: List[int] = [1, 2, 4],
        time_dim: int = 256,
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Input includes condition (concatenated)
        total_in_ch = in_channels + cond_channels
        
        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        ch = base_channels
        self.conv_in = nn.Conv2d(total_in_ch, ch, 3, padding=1)
        
        chs = [ch]
        for mult in channel_mult:
            out_ch = base_channels * mult
            self.enc_blocks.append(ResBlock(ch, out_ch, time_dim))
            self.enc_blocks.append(ResBlock(out_ch, out_ch, time_dim))
            self.downs.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            ch = out_ch
            chs.append(ch)
        
        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_dim)
        self.mid_block2 = ResBlock(ch, ch, time_dim)
        
        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            self.ups.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            self.dec_blocks.append(ResBlock(ch + chs.pop(), out_ch, time_dim))
            self.dec_blocks.append(ResBlock(out_ch, out_ch, time_dim))
            ch = out_ch
        
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, t, cond):
        """
        Args:
            x: (batch, in_channels, H, W) noisy latent
            t: (batch,) timesteps
            cond: (batch, cond_channels, H, W) condition
        
        Returns:
            (batch, out_channels, H, W) predicted noise
        """
        # Concatenate condition
        x = torch.cat([x, cond], dim=1)
        
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        h = self.conv_in(x)
        hs = [h]
        
        block_idx = 0
        for down in self.downs:
            h = self.enc_blocks[block_idx](h, t_emb)
            block_idx += 1
            h = self.enc_blocks[block_idx](h, t_emb)
            block_idx += 1
            hs.append(h)
            h = down(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        block_idx = 0
        for up in self.ups:
            h = up(h)
            h = torch.cat([h, hs.pop()], dim=1)
            h = self.dec_blocks[block_idx](h, t_emb)
            block_idx += 1
            h = self.dec_blocks[block_idx](h, t_emb)
            block_idx += 1
        
        return self.conv_out(h)


# ==============================================================================
# Latent Diffusion Model
# ==============================================================================

class LatentDiffusionModule(pl.LightningModule):
    """
    Latent Diffusion Model for neural network weight generation.
    """
    
    def __init__(
        self,
        vae_checkpoint: Optional[str] = None,
        dnnwg_path: str = "./External/DNNWG",
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        latent_channels: int = 4,
        latent_size: int = 16,
        latent_h: Optional[int] = None,
        latent_w: Optional[int] = None,
        num_classes: int = 10,
        num_samples: int = 5,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_timesteps = num_timesteps
        self.latent_channels = latent_channels
        if latent_h is None:
            latent_h = latent_size
        if latent_w is None:
            latent_w = latent_size
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)

        # Keep Lightning hparams in sync (useful for checkpoints/logs)
        self.hparams["latent_h"] = self.latent_h
        self.hparams["latent_w"] = self.latent_w

        # VAE (for encoding/decoding weights)
        if vae_checkpoint and os.path.exists(vae_checkpoint):
            # Keep the trained VAE off the Lightning device management path.
            # (It may not support MPS/CUDA reliably; we move tensors in/out in the wrapper.)
            object.__setattr__(self, 'trained_vae', VAEWrapper(vae_checkpoint, dnnwg_path, device='cpu'))
            self.use_trained_vae = True

            inferred = self._infer_trained_vae_latent_spec()
            if inferred is None:
                print(
                    "[Warning] Provided VAE checkpoint is incompatible (could not infer a stable 4D latent while preserving batch). "
                    "Falling back to SimpleVAE."
                )
                self.vae = SimpleVAE(
                    input_dim=5130,
                    latent_dim=self.latent_channels * self.latent_h * self.latent_w,
                )
                self.use_trained_vae = False
            else:
                inferred_c, inferred_h, inferred_w = inferred
                if inferred_c != self.latent_channels or inferred_h != self.latent_h or inferred_w != self.latent_w:
                    print(
                        "[Info] Inferred latent spec from trained VAE: "
                        f"(C,H,W)=({inferred_c},{inferred_h},{inferred_w}) "
                        "(overriding provided --latent_channels/--latent_size/--latent_h/--latent_w)."
                    )
                self.latent_channels = int(inferred_c)
                self.latent_h = int(inferred_h)
                self.latent_w = int(inferred_w)
                self.hparams["latent_channels"] = self.latent_channels
                self.hparams["latent_h"] = self.latent_h
                self.hparams["latent_w"] = self.latent_w
        else:
            print("Using simple VAE (no checkpoint provided)")
            self.vae = SimpleVAE(
                input_dim=5130,
                latent_dim=self.latent_channels * self.latent_h * self.latent_w,
            )
            self.use_trained_vae = False
        
        # Task encoder (dataset conditioning)
        self.task_encoder = SimpleTaskEncoder(
            num_classes=num_classes,
            num_samples=num_samples,
            latent_channels=self.latent_channels,
            latent_h=self.latent_h,
            latent_w=self.latent_w,
        )
        
        # UNet
        self.unet = SimpleUNet(
            in_channels=self.latent_channels,
            out_channels=self.latent_channels,
            cond_channels=self.latent_channels,
            base_channels=128,
            channel_mult=[1, 2, 4],
        )
        
        # Noise schedule
        betas = make_beta_schedule(beta_schedule, num_timesteps, linear_start, linear_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # For sampling
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _infer_trained_vae_latent_spec(self) -> Optional[tuple[int, int, int]]:
        """Infer (C,H,W) latent spec from the trained VAE.

        Requirements:
        - encode() returns a 4D tensor
        - preserves batch size
        """
        try:
            dummy_bs = 2
            dummy_weights = torch.zeros(dummy_bs, 5130)
            with torch.no_grad():
                z = self.trained_vae.encode(dummy_weights)

            if not isinstance(z, torch.Tensor):
                print(f"[VAE check] Unexpected encode() return type: {type(z)}")
                return None
            if z.dim() != 4:
                print(f"[VAE check] Expected 4D latent, got shape {tuple(z.shape)}")
                return None
            if z.shape[0] != dummy_bs:
                print(
                    f"[VAE check] Batch size changed during encode(). Expected B={dummy_bs}, got B={z.shape[0]} (shape={tuple(z.shape)})"
                )
                return None
            c, h, w = int(z.shape[1]), int(z.shape[2]), int(z.shape[3])
            return (c, h, w)
        except Exception as e:
            print(f"[VAE check] Failed to encode dummy batch: {e}")
            return None
    
    def encode_weights(self, weights):
        """Encode weights to latent space."""
        if self.use_trained_vae:
            z = self.trained_vae.encode(weights)
        else:
            z = self.vae.encode(weights)

        if self.use_trained_vae:
            if z.dim() != 4:
                raise RuntimeError(f"Trained VAE returned non-4D latent: shape={tuple(z.shape)}")
            if tuple(z.shape[1:]) != (self.latent_channels, self.latent_h, self.latent_w):
                raise RuntimeError(
                    "Trained VAE returned unexpected latent spatial shape: "
                    f"got={tuple(z.shape)}, expected=(B,{self.latent_channels},{self.latent_h},{self.latent_w})"
                )
            return z

        # SimpleVAE path: returns flattened latent
        if z.dim() != 2:
            z = z.view(z.shape[0], -1)
        return z.view(-1, self.latent_channels, self.latent_h, self.latent_w)
    
    def decode_latent(self, z):
        """Decode latent to weights."""
        batch_size = z.shape[0]
        z_flat = z.view(batch_size, -1)
        
        if self.use_trained_vae:
            return self.trained_vae.decode(z)
        else:
            return self.vae.decode(z_flat)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def p_losses(self, x_start, t, cond, noise=None):
        """Compute diffusion training loss."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.unet(x_noisy, t, cond)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def forward(self, batch):
        weights = batch['weight'].to(self.device)
        dataset_samples = batch['dataset']
        
        # Encode weights to latent
        z = self.encode_weights(weights)
        
        # Get conditioning
        cond = self.task_encoder(dataset_samples)
        
        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        
        # Compute loss
        loss = self.p_losses(z, t, cond)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, cond):
        """Single denoising step."""
        # Predict noise
        predicted_noise = self.unet(x, t, cond)
        
        # Predict x_0
        x_recon = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * predicted_noise
        )
        x_recon = torch.clamp(x_recon, -1, 1)
        
        # Posterior mean
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_recon +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        
        # Add noise (except at t=0)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        return posterior_mean + nonzero_mask * torch.exp(0.5 * extract(self.posterior_log_variance_clipped, t, x.shape)) * noise
    
    @torch.no_grad()
    def sample(self, cond, num_samples: int = 1):
        """Generate samples from the model."""
        device = self.device
        
        # Start from noise
        shape = (num_samples, self.latent_channels, self.latent_h, self.latent_w)
        x = torch.randn(shape, device=device)
        
        # Denoise
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond)
        
        # Decode to weights
        weights = self.decode_latent(x)
        
        return weights
    
    def configure_optimizers(self):
        params = list(self.unet.parameters()) + list(self.task_encoder.parameters())
        
        if not self.use_trained_vae:
            params += list(self.vae.parameters())
        
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

class DiffusionDataModule(pl.LightningDataModule):
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
        full_dataset = DiffusionDataset(
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
            collate_fn=collate_fn,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Step 4: Train Latent Diffusion Model")
    
    # Paths
    parser.add_argument("--weights_dir", type=str, required=True,
                        help="Directory containing trained ResNet18 heads")
    parser.add_argument("--tinyimagenet_dir", type=str, required=True,
                        help="Path to Tiny-ImageNet train directory")
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to VAE checkpoint from Step 2")
    parser.add_argument("--task_encoder_checkpoint", type=str, default=None,
                        help="Path to Task Encoder checkpoint from Step 3 (optional)")
    parser.add_argument("--dnnwg_path", type=str, default="./External/DNNWG",
                        help="Path to DNNWG library")
    parser.add_argument("--output_dir", type=str, default="./components/diffusion",
                        help="Output directory for checkpoints")
    
    # Data params
    parser.add_argument("--num_subsets", type=int, default=500)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--val_split", type=float, default=0.1)
    
    # Model params
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--latent_h", type=int, default=None,
                        help="Optional latent height. If unset, defaults to --latent_size.")
    parser.add_argument("--latent_w", type=int, default=None,
                        help="Optional latent width. If unset, defaults to --latent_size.")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every_n_steps", type=int, default=10,
                        help="Log interval in steps. Use 1 for immediate feedback.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Quick smoke test: uses tiny training (1 train + 1 val batch) and exits. Disables checkpointing/logging.")
    parser.add_argument("--accelerator", type=str, default="auto",
                        help="Accelerator: 'mps' for Mac, 'cuda' for NVIDIA, 'cpu', or 'auto'")
    parser.add_argument("--devices", type=int, default=1)
    
    args = parser.parse_args()

    if args.dry_run:
        # Force a tiny, fast run (mainly for verifying shapes + device wiring)
        args.epochs = 1
        args.num_subsets = min(args.num_subsets, 8)
        args.batch_size = min(args.batch_size, 2)
        args.num_workers = 0
        args.val_split = 0.5
        args.log_every_n_steps = 1
        # Avoid polluting the main output directory
        args.output_dir = tempfile.mkdtemp(prefix="diffusion_dry_run_")
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Paths
    tinyimagenet_train = Path(args.tinyimagenet_dir) / "train"
    if not tinyimagenet_train.exists():
        tinyimagenet_train = Path(args.tinyimagenet_dir)
    
    mapping_file = Path(args.weights_dir) / "subset_class_mapping.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    print("=" * 60)
    print("Step 4: Train Latent Diffusion Model")
    print("=" * 60)
    print(f"Weights directory: {args.weights_dir}")
    print(f"TinyImageNet train: {tinyimagenet_train}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    if args.dry_run:
        print("[Dry run] Running 1 train batch + 1 val batch, then exiting.")
    print("=" * 60)
    
    # Save config
    if not args.dry_run:
        config = vars(args)
        with open(output_path / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
    
    # Data module
    dm = DiffusionDataModule(
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
    model = LatentDiffusionModule(
        vae_checkpoint=args.vae_checkpoint,
        dnnwg_path=args.dnnwg_path,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        learning_rate=args.lr,
        latent_channels=args.latent_channels,
        latent_size=args.latent_size,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        num_classes=args.num_classes,
        num_samples=args.num_samples,
    )
    
    # Callbacks (disabled in dry_run)
    callbacks = []
    if not args.dry_run:
        callbacks = [
            ModelCheckpoint(
                dirpath=output_path / 'checkpoints',
                filename='diffusion-{epoch:02d}-{val/loss:.4f}',
                monitor='val/loss',
                mode='min',
                save_top_k=3,
                save_last=True
            ),
            EarlyStopping(
                monitor='val/loss',
                patience=50,
                mode='min'
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=output_path,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=1.0,
        enable_checkpointing=(not args.dry_run),
        logger=(False if args.dry_run else True),
        limit_train_batches=(1 if args.dry_run else 1.0),
        limit_val_batches=(1 if args.dry_run else 1.0),
        num_sanity_val_steps=(0 if args.dry_run else 2),
    )
    
    print("\nStarting Diffusion Model training...")
    trainer.fit(model, dm)
    
    print("\nTraining complete!")
    if getattr(trainer, "checkpoint_callback", None) is not None:
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        print("Checkpointing disabled (dry_run).")


if __name__ == "__main__":
    main()

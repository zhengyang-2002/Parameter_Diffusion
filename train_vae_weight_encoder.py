#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
import json

torch.set_float32_matmul_precision('medium')

# --- 1. 数据集定义 ---
class WeightDataset(Dataset):
    def __init__(self, weights_dir, my_channels, in_dim, num_subsets, weight_normalization: str = "none", eps: float = 1e-8):
        self.weights_dir = Path(weights_dir)
        self.target_size = my_channels * in_dim
        self.weights = []
        self.scales = []
        self.weight_normalization = (weight_normalization or "none").lower()
        self.eps = float(eps)
        
        print(f"正在加载 {num_subsets} 个子集的权重...")
        print(f"目标维度: {my_channels} channels × {in_dim} dim = {self.target_size} params")
        
        head_files = sorted(self.weights_dir.glob("resnet18_head_subset_*.pth"))
        if num_subsets is not None and num_subsets > 0:
            head_files = head_files[:num_subsets]

        for weight_file in tqdm(head_files, desc="加载进度"):
            
            try:
                checkpoint = torch.load(weight_file, map_location='cpu')
                # 提取权重和偏置并展平
                w = checkpoint['weight'].flatten()
                b = checkpoint['bias'].flatten()
                full_vector = torch.cat([w, b]) 
                
                # Padding
                current_size = full_vector.shape[0]
                if current_size < self.target_size:
                    pad_size = self.target_size - current_size
                    full_vector = torch.nn.functional.pad(full_vector, (0, pad_size), value=0)
                elif current_size > self.target_size:
                    full_vector = full_vector[:self.target_size]

                if self.weight_normalization == "l2":
                    scale = torch.norm(full_vector).clamp_min(self.eps)
                    full_vector = full_vector / scale
                    self.scales.append(scale)
                else:
                    self.scales.append(torch.tensor(1.0))
                
                self.weights.append(full_vector)
            except Exception as e:
                print(f"加载错误 {weight_file}: {e}")
        
        if not self.weights:
            raise ValueError("未找到任何有效的权重文件！")
            
        self.weights = torch.stack(self.weights)
        self.scales = torch.stack(self.scales).float()
        print(f"成功加载 {len(self.weights)} 个样本，形状: {self.weights.shape}")

    def __len__(self):
        return len(self.weights)
    
    def __getitem__(self, idx):
        return {'weight': self.weights[idx], 'scale': self.scales[idx]}

    @property
    def mean_scale(self) -> float:
        return float(self.scales.mean().item()) if len(self.scales) else 1.0

class WeightDataModule(pl.LightningDataModule):
    def __init__(self, weights_dir, my_channels, in_dim, num_subsets, val_split, batch_size, num_workers, weight_normalization: str = "none", eps: float = 1e-8):
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage=None):
        full_dataset = WeightDataset(
            self.hparams.weights_dir,
            self.hparams.my_channels,
            self.hparams.in_dim,
            self.hparams.num_subsets,
            weight_normalization=self.hparams.weight_normalization,
            eps=self.hparams.eps,
        )
        self.mean_scale = getattr(full_dataset, "mean_scale", 1.0)
        val_size = int(len(full_dataset) * self.hparams.val_split)
        train_size = len(full_dataset) - val_size
        
        self.train_ds, self.val_ds = random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        print(f"数据集划分: 训练集 {train_size}, 验证集 {val_size}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, 
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False, 
                          num_workers=self.hparams.num_workers, pin_memory=True)

# --- 2. 配置与模型构建 ---
def create_vae_config(num_classes):
    """
    硬编码区域：VAE的模型架构配置
    此处根据分类头大小(num_classes * 512)自动计算最接近的 reshape 维度
    """
    # 策略: my_channels = 类别数, in_dim = 512 + 1(bias)
    my_channels = num_classes
    in_dim = 513 
    
    ddconfig = {
        "double_z": True,
        "z_channels": 4,
        "resolution": 64,       # HARDCODED: 内部特征图分辨率
        "in_channels": my_channels,
        "my_channels": my_channels,
        "out_ch": my_channels,
        "ch": 64,               # HARDCODED: 基础通道数
        "ch_mult": [1, 2, 4],   # HARDCODED: 通道倍增
        "num_res_blocks": 2,    # HARDCODED: 残差块数量
        "attn_resolutions": [], # HARDCODED: 不使用Attention
        "dropout": 0.0,
        "in_dim": in_dim,
        "fdim": 4096            # HARDCODED: 全连接层维度
    }
    
    lossconfig = {
        "target": "stage1.modules.losses.CustomLosses.Myloss",
        "params": {
            "logvar_init": 0.0,
            "kl_weight": 1e-6   # overwritten by --kl_weight
        }
    }
    return ddconfig, lossconfig, my_channels, in_dim


class Step2TrainWrapper(pl.LightningModule):
    """Train wrapper around DNNWG's VAENoDiscModel with a reconstruction objective
    that correlates with our sanity metrics (cosine / relative error).
    """

    def __init__(
        self,
        vae,
        lr: float,
        mse_weight: float,
        cos_weight: float,
        rel_weight: float,
        sample_posterior: bool,
    ):
        super().__init__()
        self.vae = vae
        self.lr = lr
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight
        self.rel_weight = rel_weight
        self.sample_posterior = sample_posterior

    def forward(self, batch, sample_posterior: bool | None = None):
        if sample_posterior is None:
            sample_posterior = self.sample_posterior
        return self.vae(batch, sample_posterior=sample_posterior)

    def training_step(self, batch, batch_idx):
        inp, recon, posterior = self.forward(batch, sample_posterior=self.sample_posterior)

        mse = F.mse_loss(recon, inp)
        # Directional match (important because weights are tiny and MSE can look good even for near-zero recon)
        cos = F.cosine_similarity(recon, inp, dim=1)
        cos_loss = (1.0 - cos).mean()
        rel = (torch.norm(recon - inp, dim=1) / (torch.norm(inp, dim=1) + 1e-12)).mean()

        base_loss, log_dict = self.vae.loss(inp, recon, posterior, split="train")
        loss = base_loss + self.mse_weight * mse + self.cos_weight * cos_loss + self.rel_weight * rel

        self.log("train/base_loss", base_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/mse", mse, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/cos", cos.mean(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/rel", rel, prog_bar=False, on_step=True, on_epoch=True)
        for k, v in log_dict.items():
            if isinstance(v, torch.Tensor):
                self.log(k.replace("train/", "train/"), v, prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, recon, posterior = self.forward(batch, sample_posterior=False)
        rec_loss = torch.mean(torch.abs(inp - recon))
        mse = F.mse_loss(recon, inp)
        cos = F.cosine_similarity(recon, inp, dim=1).mean()
        rel = (torch.norm(recon - inp, dim=1) / (torch.norm(inp, dim=1) + 1e-12)).mean()

        base_loss, log_dict = self.vae.loss(inp, recon, posterior, split="val")
        loss = base_loss + self.mse_weight * mse + self.cos_weight * (1.0 - cos) + self.rel_weight * rel

        self.log("val/loss", loss, prog_bar=False)
        self.log("val/rec_loss", rec_loss, prog_bar=True)
        self.log("val/mse", mse, prog_bar=True)
        self.log("val/cos", cos, prog_bar=True)
        self.log("val/rel", rel, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters(), lr=self.lr, betas=(0.5, 0.9))

# --- 3. 主程序 ---
def main():
    parser = argparse.ArgumentParser(description="Stage 1: Train VAE Weight Encoder")
    
    # 路径参数
    parser.add_argument("--dnnwg_path", type=str, required=True, help="DNNWG 库的根目录路径")
    parser.add_argument("--weights_dir", type=str, required=True, help="ResNet18 分类头权重所在的目录")
    parser.add_argument("--output_dir", type=str, default="./vae_output", help="VAE 模型输出目录")
    
    # 训练参数
    parser.add_argument("--num_subsets", type=int, default=200, help="要加载的子集数量 (需与上一阶段一致)")
    parser.add_argument("--num_classes", type=int, default=10, help="每个子集的类别数 (用于计算维度)")
    parser.add_argument("--epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--val_split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument(
        "--weight_normalization",
        type=str,
        default="none",
        choices=["none", "l2"],
        help="Optional normalization applied to each head vector before VAE (l2 is per-vector).",
    )
    parser.add_argument("--norm_eps", type=float, default=1e-8, help="epsilon for normalization")

    # VAE loss knobs (to improve cosine/relative reconstruction)
    parser.add_argument("--kl_weight", type=float, default=0.0, help="KL weight in Myloss")
    parser.add_argument("--mse_weight", type=float, default=1.0, help="Extra MSE loss weight")
    parser.add_argument("--cos_weight", type=float, default=5.0, help="Extra cosine loss weight")
    parser.add_argument("--rel_weight", type=float, default=1.0, help="Extra relative-error loss weight")
    parser.add_argument(
        "--sample_posterior",
        action="store_true",
        help="If set, sample posterior during training (stochastic). Default trains deterministically (mode).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="训练设备: auto/cpu/mps/cuda",
    )
    parser.add_argument("--devices", type=int, default=1, help="设备数量 (mps/cuda)")

    args = parser.parse_args()

    # Resolve runtime device
    if args.device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = args.device

    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available in this environment")
    if device_str == "mps" and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        raise RuntimeError("Requested --device mps but MPS is not available in this environment")

    # 动态添加 DNNWG 路径
    sys.path.insert(0, args.dnnwg_path)
    try:
        from stage1.models.autoencoder import VAENoDiscModel
    except ImportError:
        print(f"错误: 无法在 {args.dnnwg_path} 找到 DNNWG 模块。请检查路径。")
        sys.exit(1)

    pl.seed_everything(args.seed)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 获取配置
    ddconfig, lossconfig, my_channels, in_dim = create_vae_config(args.num_classes)
    lossconfig["params"]["kl_weight"] = float(args.kl_weight)
    
    # 保存配置
    with open(output_path / 'vae_config.yaml', 'w') as f:
        yaml.dump({'ddconfig': ddconfig, 'args': vars(args)}, f)

    # 2. 数据模块
    dm = WeightDataModule(
        weights_dir=args.weights_dir,
        my_channels=my_channels,
        in_dim=in_dim,
        num_subsets=args.num_subsets,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weight_normalization=args.weight_normalization,
        eps=args.norm_eps,
    )

    # Persist normalization config next to VAE checkpoints so Step3/Step4 can apply the same transform.
    dm.setup()
    norm_cfg = {
        "method": args.weight_normalization,
        "eps": float(args.norm_eps),
        "mean_scale": float(getattr(dm, "mean_scale", 1.0)),
    }
    with open(output_path / "weight_normalization.yaml", "w") as f:
        yaml.safe_dump(norm_cfg, f)

    # 3. 模型
    model = VAENoDiscModel(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        embed_dim=4,
        learning_rate=args.lr,
        input_key="weight",
        device=device_str,
    )

    train_model = Step2TrainWrapper(
        vae=model,
        lr=args.lr,
        mse_weight=float(args.mse_weight),
        cos_weight=float(args.cos_weight),
        rel_weight=float(args.rel_weight),
        sample_posterior=bool(args.sample_posterior),
    )

    # 4. Trainer
    callbacks = [
        pl.callbacks.ModelCheckpoint(dirpath=output_path/'checkpoints', monitor='val/rec_loss', save_top_k=1, mode='min', save_last=True),
        pl.callbacks.EarlyStopping(monitor='val/rec_loss', patience=20, mode='min'),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]

    trainer = pl.Trainer(
        default_root_dir=output_path,
        max_epochs=args.epochs,
        accelerator=("cpu" if device_str == "cpu" else ("gpu" if device_str == "cuda" else "mps")),
        devices=("auto" if device_str == "cpu" else args.devices),
        callbacks=callbacks,
        log_every_n_steps=10
    )

    print(f"开始训练 VAE... 输出目录: {args.output_dir}")
    trainer.fit(train_model, dm)

    # 5. 测试重建
    print("正在进行重建测试...")
    final_ckpt = trainer.checkpoint_callback.best_model_path
    if not final_ckpt: final_ckpt = trainer.checkpoint_callback.last_model_path
    
    # 简单的重建评估逻辑
    wrapper = Step2TrainWrapper.load_from_checkpoint(
        final_ckpt,
        vae=VAENoDiscModel(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=4,
            learning_rate=args.lr,
            input_key="weight",
            device=device_str,
        ),
        lr=args.lr,
        mse_weight=float(args.mse_weight),
        cos_weight=float(args.cos_weight),
        rel_weight=float(args.rel_weight),
        sample_posterior=bool(args.sample_posterior),
    )
    model = wrapper.vae
    model.devices = device_str
    model.eval().to(torch.device(device_str))
    
    total_mse = 0
    count = 0
    dm.setup()
    with torch.no_grad():
        for batch in dm.val_dataloader():
            inp, recon, _ = model(batch, sample_posterior=False)
            total_mse += torch.nn.functional.mse_loss(inp, recon).item() * inp.size(0)
            count += inp.size(0)
    
    avg_mse = total_mse / count if count > 0 else 0.0
    print(f"最终验证集平均 MSE: {avg_mse:.6f}")
    
    result = {"avg_mse": avg_mse, "best_ckpt": final_ckpt}
    with open(output_path / 'reconstruction_result.json', 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
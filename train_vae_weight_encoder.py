#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import json

torch.set_float32_matmul_precision('medium')

# --- 1. 数据集定义 ---
class WeightDataset(Dataset):
    def __init__(self, weights_dir, my_channels, in_dim, num_subsets):
        self.weights_dir = Path(weights_dir)
        self.target_size = my_channels * in_dim
        self.weights = []
        
        print(f"正在加载 {num_subsets} 个子集的权重...")
        print(f"目标维度: {my_channels} channels × {in_dim} dim = {self.target_size} params")
        
        for i in tqdm(range(1, num_subsets + 1), desc="加载进度"):
            weight_file = self.weights_dir / f"resnet18_head_subset_{i}.pth"
            if not weight_file.exists():
                continue
            
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
                
                self.weights.append(full_vector)
            except Exception as e:
                print(f"加载错误 {weight_file}: {e}")
        
        if not self.weights:
            raise ValueError("未找到任何有效的权重文件！")
            
        self.weights = torch.stack(self.weights)
        print(f"成功加载 {len(self.weights)} 个样本，形状: {self.weights.shape}")

    def __len__(self):
        return len(self.weights)
    
    def __getitem__(self, idx):
        return {'weight': self.weights[idx]}

class WeightDataModule(pl.LightningDataModule):
    def __init__(self, weights_dir, my_channels, in_dim, num_subsets, val_split, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage=None):
        full_dataset = WeightDataset(
            self.hparams.weights_dir,
            self.hparams.my_channels,
            self.hparams.in_dim,
            self.hparams.num_subsets
        )
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
        "fdim": 2048            # HARDCODED: 全连接层维度
    }
    
    lossconfig = {
        "target": "stage1.modules.losses.CustomLosses.Myloss",
        "params": {
            "logvar_init": 0.0,
            "kl_weight": 1e-6   # HARDCODED: KL散度权重
        }
    }
    return ddconfig, lossconfig, my_channels, in_dim

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
    parser.add_argument("--lr", type=float, default=4.5e-6, help="学习率")
    parser.add_argument("--val_split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gpus", type=int, default=1, help="GPU数量 (0为CPU)")

    args = parser.parse_args()

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
        num_workers=args.num_workers
    )

    # 3. 模型
    model = VAENoDiscModel(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        embed_dim=4,
        learning_rate=args.lr,
        input_key="weight",
        device='cuda' if args.gpus > 0 and torch.cuda.is_available() else 'cpu',
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
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else "auto",
        callbacks=callbacks,
        log_every_n_steps=10
    )

    print(f"开始训练 VAE... 输出目录: {args.output_dir}")
    trainer.fit(model, dm)

    # 5. 测试重建
    print("正在进行重建测试...")
    final_ckpt = trainer.checkpoint_callback.best_model_path
    if not final_ckpt: final_ckpt = trainer.checkpoint_callback.last_model_path
    
    # 简单的重建评估逻辑
    model = VAENoDiscModel.load_from_checkpoint(final_ckpt, ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=4, input_key="weight", learning_rate=args.lr)
    model.eval()
    model.to('cuda' if args.gpus > 0 else 'cpu')
    
    total_mse = 0
    count = 0
    dm.setup()
    with torch.no_grad():
        for batch in dm.val_dataloader():
            inputs = batch['weight'].to(model.device)
            _, recon, _ = model(batch)
            total_mse += torch.nn.functional.mse_loss(inputs, recon).item() * inputs.size(0)
            count += inputs.size(0)
    
    avg_mse = total_mse / count if count > 0 else 0.0
    print(f"最终验证集平均 MSE: {avg_mse:.6f}")
    
    result = {"avg_mse": avg_mse, "best_ckpt": final_ckpt}
    with open(output_path / 'reconstruction_result.json', 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
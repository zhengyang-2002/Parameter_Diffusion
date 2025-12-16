#!/usr/bin/env python3
"""Generalization evaluation: D2NWG-generated heads vs GD-trained heads.

For each sampled 10-way task:
- Build a support set (few-shot) and query set from an unseen dataset
- D2NWG: generate a 10-way ResNet18 head (5130 params) conditioned on support images
- GD baseline: train a 10-way linear head on the same support set (frozen backbone)
- Evaluate both on query set (accuracy + cross-entropy)

Outputs CSV to --out_csv.

Example (fast, smoke):
  python experiments/eval_generalization.py \
    --diffusion_checkpoint ./components/diffusion/checkpoints/last.ckpt \
    --dataset fake --data_root ./data --tasks 3 --support 1 --query 4 \
    --ddim_steps 25 --device mps

Realistic:
  python experiments/eval_generalization.py \
    --diffusion_checkpoint ./components/diffusion/checkpoints/last.ckpt \
    --dataset cifar100 --data_root ./data --tasks 50 --support 5 --query 50 \
    --ddim_steps 50 --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models

# Ensure repo root import works
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from train_diffusion import LatentDiffusionModule
from experiments.taskbench import load_dataset, sample_task_indices, remap_subset_labels, group_support_indices_by_class, pack_support_tensor
from experiments.diffusion_sampling import ddim_sample, ddpm_sample
from experiments.utils import seed_everything, get_device, accuracy_from_logits, cross_entropy_loss


def build_frozen_resnet18(num_classes: int, device: torch.device):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = torch.nn.Linear(512, num_classes)
    return m.to(device)


@torch.no_grad()
def eval_head(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        bs = y.shape[0]
        total_loss += cross_entropy_loss(logits, y) * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        n += bs
    return total_loss / max(n, 1), total_acc / max(n, 1)


def train_gd_head(
    backbone: torch.nn.Module,
    support_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
):
    backbone.train()
    # Only head is trainable
    opt = torch.optim.SGD(backbone.fc.parameters(), lr=lr, momentum=0.9)

    it = iter(support_loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(support_loader)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = backbone(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()


def apply_generated_head(resnet: torch.nn.Module, flat_weights: torch.Tensor, num_classes: int):
    # flat_weights: (5130,) = (K*512 + K)
    w = flat_weights[: num_classes * 512].view(num_classes, 512)
    b = flat_weights[num_classes * 512 : num_classes * 512 + num_classes]
    resnet.fc.weight.data.copy_(w)
    resnet.fc.bias.data.copy_(b)


def _save_head(
    save_dir: Path,
    *,
    method: str,
    dataset: str,
    task_id: int,
    class_ids: list[int],
    head_weight: torch.Tensor,
    head_bias: torch.Tensor,
    meta: dict,
):
    task_dir = save_dir / dataset / f"task_{task_id:05d}" / method
    task_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "weight": head_weight.detach().cpu(),
        "bias": head_bias.detach().cpu(),
        "class_ids": list(map(int, class_ids)),
        **meta,
    }
    torch.save(payload, task_dir / "head.pt")
    (task_dir / "meta.json").write_text(json.dumps({k: (v if isinstance(v, (int, float, str, list, dict, bool)) else str(v)) for k, v in meta.items()}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion_checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True,
                    help="cifar10|cifar100|svhn|stl10|imagefolder|fake")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_way", type=int, default=10)
    ap.add_argument("--support", type=int, default=5, help="support shots per class")
    ap.add_argument("--query", type=int, default=50, help="query images per class")
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--pool_split", type=str, default="test", choices=["train", "test"],
                    help="Which dataset split to sample tasks from")
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--device", type=str, default="auto")

    # D2NWG sampling
    ap.add_argument("--ddim_steps", type=int, default=50,
                    help="DDIM steps for D2NWG sampling")
    ap.add_argument("--ddim_eta", type=float, default=0.0)
    ap.add_argument("--num_weight_samples", type=int, default=1,
                    help="number of generated heads per task (averaged)")

    # GD baseline
    ap.add_argument("--gd_steps", type=int, default=200)
    ap.add_argument("--gd_lr", type=float, default=0.1)

    ap.add_argument("--out_csv", type=str, default="./results/generalization.csv")
    ap.add_argument("--save_weights_dir", type=str, default=None,
                    help="If set, saves generated/GD heads to this folder for reproducibility")

    args = ap.parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dir = Path(args.save_weights_dir) if args.save_weights_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Load diffusion model
    diffusion = LatentDiffusionModule.load_from_checkpoint(args.diffusion_checkpoint, map_location="cpu")
    diffusion = diffusion.to(device)
    diffusion.eval()

    # Dataset splits
    train_ds = load_dataset(args.dataset, args.data_root, split="train", image_size=args.image_size)
    pool_ds = load_dataset(args.dataset, args.data_root, split=args.pool_split, image_size=args.image_size)

    fieldnames = [
        "dataset", "task_id", "method", "ddim_steps", "support", "query",
        "loss", "acc",
        "class_ids",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for task_id in range(args.tasks):
            task_seed = args.seed + task_id * 997

            class_ids, support_idx, query_idx = sample_task_indices(
                pool_ds,
                num_way=args.num_way,
                support_per_class=args.support,
                query_per_class=args.query,
                seed=task_seed,
            )

            # build query loader with remapped labels
            query_ds = remap_subset_labels(pool_ds, query_idx, class_ids)
            query_loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

            # build support loader for GD (remapped labels)
            support_ds = remap_subset_labels(pool_ds, support_idx, class_ids)
            support_loader = DataLoader(support_ds, batch_size=min(args.batch_size, len(support_ds)), shuffle=True, num_workers=0)

            # build conditioning tensor for D2NWG: (K, Ns, 3, H, W)
            grouped = group_support_indices_by_class(pool_ds, class_ids, support_idx)
            support_tensor = pack_support_tensor(pool_ds, grouped)
            dataset_samples_batch = [support_tensor.to(device)]
            with torch.no_grad():
                cond = diffusion.task_encoder(dataset_samples_batch)

            # --- D2NWG generated head ---
            d2_losses = []
            d2_accs = []
            for sample_id in range(args.num_weight_samples):
                weights = ddim_sample(
                    diffusion,
                    cond=cond,
                    steps=args.ddim_steps,
                    num_samples=1,
                    eta=args.ddim_eta,
                )
                flat = weights[0].detach().to(device)
                resnet = build_frozen_resnet18(args.num_way, device)
                apply_generated_head(resnet, flat, args.num_way)
                loss, acc = eval_head(resnet, query_loader, device)
                d2_losses.append(loss)
                d2_accs.append(acc)

                if save_dir is not None:
                    w = flat[: args.num_way * 512].view(args.num_way, 512)
                    b = flat[args.num_way * 512 : args.num_way * 512 + args.num_way]
                    _save_head(
                        save_dir,
                        method=f"d2nwg_ddim{args.ddim_steps}_s{sample_id}",
                        dataset=args.dataset,
                        task_id=task_id,
                        class_ids=class_ids,
                        head_weight=w,
                        head_bias=b,
                        meta={
                            "seed": args.seed,
                            "task_seed": task_seed,
                            "pool_split": args.pool_split,
                            "support": args.support,
                            "query": args.query,
                            "ddim_steps": args.ddim_steps,
                            "ddim_eta": args.ddim_eta,
                        },
                    )

            writer.writerow({
                "dataset": args.dataset,
                "task_id": task_id,
                "method": "d2nwg",
                "ddim_steps": args.ddim_steps,
                "support": args.support,
                "query": args.query,
                "loss": sum(d2_losses) / len(d2_losses),
                "acc": sum(d2_accs) / len(d2_accs),
                "class_ids": " ".join(map(str, class_ids)),
            })
            f.flush()

            # --- GD baseline ---
            resnet = build_frozen_resnet18(args.num_way, device)
            train_gd_head(resnet, support_loader, device=device, steps=args.gd_steps, lr=args.gd_lr)
            loss, acc = eval_head(resnet, query_loader, device)

            if save_dir is not None:
                _save_head(
                    save_dir,
                    method="gd",
                    dataset=args.dataset,
                    task_id=task_id,
                    class_ids=class_ids,
                    head_weight=resnet.fc.weight.data,
                    head_bias=resnet.fc.bias.data,
                    meta={
                        "seed": args.seed,
                        "task_seed": task_seed,
                        "pool_split": args.pool_split,
                        "support": args.support,
                        "query": args.query,
                        "gd_steps": args.gd_steps,
                        "gd_lr": args.gd_lr,
                    },
                )

            writer.writerow({
                "dataset": args.dataset,
                "task_id": task_id,
                "method": "gd",
                "ddim_steps": "",
                "support": args.support,
                "query": args.query,
                "loss": loss,
                "acc": acc,
                "class_ids": " ".join(map(str, class_ids)),
            })
            f.flush()

    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()

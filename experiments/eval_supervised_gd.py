#!/usr/bin/env python3
"""Full supervised baseline: train a frozen-backbone ResNet18 head on the full train split.

This is an *upper bound* baseline compared to few-shot episodic GD.
It trains only the final linear layer (fc) with gradient descent.

Outputs a one-row CSV and (optionally) saves the trained head weights.
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

import sys
from pathlib import Path as _Path

# Ensure repo root import works
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from experiments.taskbench import load_dataset
from experiments.utils import seed_everything, get_device


def build_frozen_resnet18(num_classes: int, device: torch.device):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = torch.nn.Linear(512, num_classes)
    return m.to(device)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * y.shape[0]
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.shape[0])
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def train_fc(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
):
    model.train()
    opt = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)

    for _ in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="cifar10|stl10")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_csv", type=str, default="./results/supervised_gd.csv")
    ap.add_argument(
        "--save_weights_dir",
        type=str,
        default=None,
        help="If set, saves trained head to this folder",
    )

    args = ap.parse_args()
    if args.dataset.lower() not in {"cifar10", "stl10"}:
        raise ValueError("--dataset must be cifar10 or stl10 for this baseline")

    seed_everything(args.seed)
    device = get_device(args.device)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dir = Path(args.save_weights_dir) if args.save_weights_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_dataset(args.dataset, args.data_root, split="train", image_size=args.image_size)
    test_ds = load_dataset(args.dataset, args.data_root, split="test", image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_frozen_resnet18(num_classes=10, device=device)
    train_fc(model, train_loader, device=device, epochs=args.epochs, lr=args.lr)

    loss, acc = evaluate(model, test_loader, device=device)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "image_size",
                "batch_size",
                "epochs",
                "lr",
                "loss",
                "acc",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": args.dataset,
                "method": "gd_supervised",
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "loss": loss,
                "acc": acc,
                "seed": args.seed,
            }
        )

    if save_dir is not None:
        head_dir = save_dir / args.dataset / "gd_supervised"
        head_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "weight": model.fc.weight.detach().cpu(),
                "bias": model.fc.bias.detach().cpu(),
                "dataset": args.dataset,
                "method": "gd_supervised",
                "seed": args.seed,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
            },
            head_dir / "head.pt",
        )
        (head_dir / "meta.json").write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "method": "gd_supervised",
                    "seed": args.seed,
                    "image_size": args.image_size,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "lr": args.lr,
                },
                indent=2,
            )
        )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ablation: sampling steps vs task validation loss/accuracy.

For each task (10-way):
- Use the same support set to condition the diffusion model
- For each steps value, sample a head with DDIM using that many steps
- Evaluate on the query set and record cross-entropy + accuracy

Outputs CSV to --out_csv.

Example (fast):
  python experiments/ablation_sampling_steps.py \
    --diffusion_checkpoint ./components/diffusion/checkpoints/last.ckpt \
    --dataset fake --data_root ./data --tasks 3 --support 1 --query 8 \
    --steps 5,10,25 --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.models as models

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from train_diffusion import LatentDiffusionModule
from experiments.taskbench import load_dataset, sample_task_indices, remap_subset_labels, group_support_indices_by_class, pack_support_tensor
from experiments.diffusion_sampling import ddim_sample
from experiments.utils import seed_everything, get_device, accuracy_from_logits, cross_entropy_loss


def build_frozen_resnet18(num_classes: int, device: torch.device):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = torch.nn.Linear(512, num_classes)
    return m.to(device)


def apply_generated_head(resnet: torch.nn.Module, flat_weights: torch.Tensor, num_classes: int):
    w = flat_weights[: num_classes * 512].view(num_classes, 512)
    b = flat_weights[num_classes * 512 : num_classes * 512 + num_classes]
    resnet.fc.weight.data.copy_(w)
    resnet.fc.bias.data.copy_(b)


def _save_head(
    save_dir: Path,
    *,
    dataset: str,
    task_id: int,
    steps: int,
    sample_id: int,
    class_ids: list[int],
    head_weight: torch.Tensor,
    head_bias: torch.Tensor,
    meta: dict,
):
    task_dir = save_dir / dataset / f"task_{task_id:05d}" / f"d2nwg_ddim{steps}_s{sample_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "weight": head_weight.detach().cpu(),
        "bias": head_bias.detach().cpu(),
        "class_ids": list(map(int, class_ids)),
        **meta,
    }
    torch.save(payload, task_dir / "head.pt")
    (task_dir / "meta.json").write_text(json.dumps({k: (v if isinstance(v, (int, float, str, list, dict, bool)) else str(v)) for k, v in meta.items()}, indent=2))


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


def parse_steps(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    steps = [int(p) for p in parts]
    if any(x <= 0 for x in steps):
        raise ValueError("All steps must be > 0")
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion_checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num_way", type=int, default=10)
    ap.add_argument("--support", type=int, default=5)
    ap.add_argument("--query", type=int, default=50)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--pool_split", type=str, default="test", choices=["train", "test"],
                    help="Which dataset split to sample tasks from")
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--steps", type=str, default="5,10,25,50,100",
                    help="Comma-separated DDIM steps")
    ap.add_argument("--ddim_eta", type=float, default=0.0)
    ap.add_argument("--num_weight_samples", type=int, default=1)

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_csv", type=str, default="./results/ablation_sampling_steps.csv")
    ap.add_argument("--save_weights_dir", type=str, default=None,
                    help="If set, saves sampled heads to this folder for reproducibility")

    args = ap.parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    steps_list = parse_steps(args.steps)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dir = Path(args.save_weights_dir) if args.save_weights_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    diffusion = LatentDiffusionModule.load_from_checkpoint(args.diffusion_checkpoint, map_location="cpu")
    diffusion = diffusion.to(device)
    diffusion.eval()

    pool_ds = load_dataset(args.dataset, args.data_root, split=args.pool_split, image_size=args.image_size)

    fieldnames = [
        "dataset", "task_id", "steps", "support", "query",
        "loss", "acc", "class_ids",
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

            query_ds = remap_subset_labels(pool_ds, query_idx, class_ids)
            query_loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

            grouped = group_support_indices_by_class(pool_ds, class_ids, support_idx)
            support_tensor = pack_support_tensor(pool_ds, grouped)
            dataset_samples_batch = [support_tensor.to(device)]
            with torch.no_grad():
                cond = diffusion.task_encoder(dataset_samples_batch)

            for steps in steps_list:
                losses = []
                accs = []
                for sample_id in range(args.num_weight_samples):
                    weights = ddim_sample(
                        diffusion,
                        cond=cond,
                        steps=steps,
                        num_samples=1,
                        eta=args.ddim_eta,
                    )
                    resnet = build_frozen_resnet18(args.num_way, device)
                    flat = weights[0].to(device)
                    apply_generated_head(resnet, flat, args.num_way)
                    loss, acc = eval_head(resnet, query_loader, device)
                    losses.append(loss)
                    accs.append(acc)

                    if save_dir is not None:
                        w = flat[: args.num_way * 512].view(args.num_way, 512)
                        b = flat[args.num_way * 512 : args.num_way * 512 + args.num_way]
                        _save_head(
                            save_dir,
                            dataset=args.dataset,
                            task_id=task_id,
                            steps=steps,
                            sample_id=sample_id,
                            class_ids=class_ids,
                            head_weight=w,
                            head_bias=b,
                            meta={
                                "seed": args.seed,
                                "task_seed": task_seed,
                                "pool_split": args.pool_split,
                                "support": args.support,
                                "query": args.query,
                                "ddim_steps": steps,
                                "ddim_eta": args.ddim_eta,
                            },
                        )

                writer.writerow({
                    "dataset": args.dataset,
                    "task_id": task_id,
                    "steps": steps,
                    "support": args.support,
                    "query": args.query,
                    "loss": sum(losses) / len(losses),
                    "acc": sum(accs) / len(accs),
                    "class_ids": " ".join(map(str, class_ids)),
                })
                f.flush()

    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()

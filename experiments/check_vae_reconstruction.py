#!/usr/bin/env python3
"""VAE reconstruction sanity check.

Why this matters:
Step 4 trains diffusion in *VAE latent space*. If the VAE cannot reconstruct real
classifier heads, the diffusion model cannot produce useful heads.

This script measures reconstruction quality on real heads from `--weights_dir`.

Example:
  python experiments/check_vae_reconstruction.py \
    --vae_checkpoint ./Pretrained_Components/VAE \
    --weights_dir ./Model_Zoo/Resnet18_TinyImageNet_HC \
    --num_samples 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


def _extract_vae_state_dict(state_dict: dict) -> dict:
    # Step2 saves checkpoints from a Lightning wrapper with keys like "vae.encoder.*".
    # Downstream code expects raw VAENoDiscModel keys like "encoder.*".
    if any(k.startswith("vae.") for k in state_dict.keys()):
        return {k[len("vae."):]: v for k, v in state_dict.items() if k.startswith("vae.")}
    return state_dict


def _resolve_ckpt(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        cand = p / "checkpoints" / "last.ckpt"
        if cand.exists():
            return cand
    return p


def _resolve_base_dir(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        return p
    # typical: <out>/checkpoints/last.ckpt
    if p.parent.name == "checkpoints":
        return p.parent.parent
    return p.parent


def _load_weight_norm_cfg(base_dir: Path) -> dict:
    cfg_path = base_dir / "weight_normalization.yaml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_checkpoint", type=str, required=True)
    ap.add_argument("--dnnwg_path", type=str, default="./External/DNNWG")
    ap.add_argument("--weights_dir", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # DNNWG import
    import sys
    sys.path.insert(0, args.dnnwg_path)
    from stage1.models.autoencoder import VAENoDiscModel

    # Default config (used for `Pretrained_Components/VAE` unless a `vae_config.yaml` is present)
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
        "fdim": 2048,
    }
    lossconfig = {
        "target": "stage1.modules.losses.CustomLosses.Myloss",
        "params": {"logvar_init": 0.0, "kl_weight": 1e-6},
    }

    base_dir = _resolve_base_dir(args.vae_checkpoint)
    cfg_path = base_dir / "vae_config.yaml"
    if cfg_path is not None and cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        if isinstance(cfg, dict) and "ddconfig" in cfg:
            ddconfig = cfg["ddconfig"]

    norm_cfg = _load_weight_norm_cfg(base_dir)
    norm_method = (norm_cfg.get("method") or "none").lower()
    norm_eps = float(norm_cfg.get("eps") or 1e-8)
    mean_scale = float(norm_cfg.get("mean_scale") or 1.0)

    ckpt_path = _resolve_ckpt(args.vae_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_path}")

    model = VAENoDiscModel(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        embed_dim=4,
        input_key="weight",
        device="cpu",
        learning_rate=1e-4,
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _extract_vae_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.devices = "cpu"
    model.eval().to("cpu")

    weights_dir = Path(args.weights_dir)
    head_files = sorted(weights_dir.glob("resnet18_head_subset_*.pth"))
    if not head_files:
        raise FileNotFoundError(f"No files like resnet18_head_subset_*.pth under {weights_dir}")

    # deterministic subset
    perm = torch.randperm(len(head_files))[: min(args.num_samples, len(head_files))].tolist()
    head_files = [head_files[i] for i in perm]

    mses, rels, coss = [], [], []

    for fp in head_files:
        ckpt = torch.load(fp, map_location="cpu")
        x = torch.cat([ckpt["weight"].flatten(), ckpt["bias"].flatten()]).unsqueeze(0).float()

        scale = torch.tensor(1.0)
        x_in = x
        if norm_method == "l2":
            scale = torch.norm(x).clamp_min(norm_eps)
            x_in = x / scale
        with torch.no_grad():
            inp, recon, _ = model({"weight": x_in}, sample_posterior=False)

        # Compare in original scale. For l2 normalization, use the same per-sample scale.
        if norm_method == "l2":
            inp = inp * scale
            recon = recon * scale

        mse = F.mse_loss(recon, inp).item()
        rel = (torch.norm(recon - inp) / (torch.norm(inp) + 1e-12)).item()
        cos = F.cosine_similarity(recon, inp, dim=1).item()

        mses.append(mse)
        rels.append(rel)
        coss.append(cos)

    def mean(xs):
        return float(sum(xs) / max(len(xs), 1))

    print(f"VAE ckpt: {ckpt_path}")
    if norm_method != "none":
        print(f"Weight norm: {norm_method} (mean_scale={mean_scale:.4f})")
    print(f"Checked heads: {len(head_files)}")
    print(f"MSE mean:    {mean(mses):.6f}")
    print(f"rel L2 mean: {mean(rels):.4f}  (lower is better; ~1.0 often means near-zero recon)")
    print(f"cos mean:    {mean(coss):.4f}  (higher is better; >0.9 is usually healthy)")


if __name__ == "__main__":
    main()

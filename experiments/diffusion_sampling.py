from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@torch.no_grad()
def compute_x0_from_eps(x_t: torch.Tensor, eps: torch.Tensor, alpha_cumprod_t: torch.Tensor) -> torch.Tensor:
    # alpha_cumprod_t: (B,1,1,1)
    return (x_t - torch.sqrt(1.0 - alpha_cumprod_t) * eps) / torch.sqrt(alpha_cumprod_t)


@torch.no_grad()
def ddim_sample(
    model,
    cond: torch.Tensor,
    steps: int,
    num_samples: int,
    eta: float = 0.0,
    x_T: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DDIM sampling for LatentDiffusionModule-like model.

    Assumptions:
    - model.unet(x, t, cond) predicts noise eps
    - model.alphas_cumprod is a 1D tensor of length T
    - model.decode_latent(z) converts latent to weight vector

    Returns: decoded weights (num_samples, 5130)
    """
    device = cond.device
    T = int(model.num_timesteps)
    if steps <= 0:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if steps > T:
        # Allow convenience usage with small-T checkpoints (e.g., smoketests).
        # We clip to T instead of failing hard.
        print(f"[ddim_sample] Requested steps={steps} but model has T={T}; clipping to {T}.")
        steps = T

    # Uniform timestep schedule, descending
    # timesteps are integers in [0, T-1]
    schedule = torch.linspace(T - 1, 0, steps, device=device)
    schedule = torch.round(schedule).long()
    schedule = torch.unique_consecutive(schedule)

    shape = (num_samples, int(model.latent_channels), int(model.latent_h), int(model.latent_w))
    if x_T is None:
        x = torch.randn(shape, device=device)
    else:
        x = x_T.to(device)

    # Ensure cond batch matches
    if cond.shape[0] != num_samples:
        if cond.shape[0] == 1:
            cond = cond.repeat(num_samples, 1, 1, 1)
        else:
            raise ValueError(f"cond batch {cond.shape[0]} must equal num_samples {num_samples} (or be 1)")

    alphas_cumprod = model.alphas_cumprod.to(device)

    for i, t in enumerate(schedule):
        t_batch = torch.full((num_samples,), int(t.item()), device=device, dtype=torch.long)

        eps = model.unet(x, t_batch, cond)
        a_t = alphas_cumprod[t_batch].view(-1, 1, 1, 1)
        x0 = compute_x0_from_eps(x, eps, a_t)

        if i == len(schedule) - 1:
            x = x0
            break

        t_prev = schedule[i + 1]
        t_prev_batch = torch.full((num_samples,), int(t_prev.item()), device=device, dtype=torch.long)
        a_prev = alphas_cumprod[t_prev_batch].view(-1, 1, 1, 1)

        # DDIM parameters
        sigma = (
            eta
            * torch.sqrt((1 - a_prev) / (1 - a_t))
            * torch.sqrt(1 - a_t / a_prev)
        )
        # Deterministic direction
        dir_xt = torch.sqrt(torch.clamp(1 - a_prev - sigma**2, min=0.0)) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
        x = torch.sqrt(a_prev) * x0 + dir_xt + noise

    weights = model.decode_latent(x)
    return weights


@torch.no_grad()
def ddpm_sample(model, cond: torch.Tensor, num_samples: int) -> torch.Tensor:
    # Uses the model's built-in full-step sampler
    return model.sample(cond, num_samples=num_samples)

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    device = device.lower()
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    task_id: int
    class_ids: list[int]
    support_per_class: int
    query_per_class: int
    image_size: int


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def cross_entropy_loss(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.nn.functional.cross_entropy(logits, y).item())

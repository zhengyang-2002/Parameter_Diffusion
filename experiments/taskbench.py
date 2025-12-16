from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, STL10, ImageFolder, FakeData


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int


SUPPORTED_DATASETS: dict[str, DatasetSpec] = {
    # 10-way natural baselines
    "cifar10": DatasetSpec("cifar10", 10),
    "svhn": DatasetSpec("svhn", 10),
    "stl10": DatasetSpec("stl10", 10),

    # unseen multi-class pool (we will sample 10 classes from these)
    "cifar100": DatasetSpec("cifar100", 100),

    # Tiny-ImageNet-200 (multi-class pool)
    "tinyimagenet": DatasetSpec("tinyimagenet", 200),

    # user-provided folder: root/train/<class> and root/test/<class>
    "imagefolder": DatasetSpec("imagefolder", -1),

    # smoke-test
    "fake": DatasetSpec("fake", 100),
}


def default_transform(image_size: int) -> Callable:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_dataset(
    name: str,
    root: str,
    split: str,
    image_size: int,
) -> Dataset:
    name = name.lower()
    tfm = default_transform(image_size)

    if name == "cifar10":
        train = split == "train"
        return CIFAR10(root=root, train=train, download=True, transform=tfm)

    if name == "cifar100":
        train = split == "train"
        return CIFAR100(root=root, train=train, download=True, transform=tfm)

    if name == "svhn":
        # torchvision uses split in {"train","test","extra"}
        svhn_split = "train" if split == "train" else "test"
        return SVHN(root=root, split=svhn_split, download=True, transform=tfm)

    if name == "stl10":
        stl_split = "train" if split == "train" else "test"
        return STL10(root=root, split=stl_split, download=True, transform=tfm)

    if name == "imagefolder":
        # Expect root/<split>/<class_name>/*.jpg
        return ImageFolder(root=f"{root}/{split}", transform=tfm)

    if name == "tinyimagenet":
        # Expected layout (as downloaded by train_model_zoo.py):
        #   <root>/tiny-imagenet-200/train/<wnid>/images/*.JPEG
        # Tiny-ImageNet val is not class-folder-organized by default, so for
        # split="test" we try a few common options; if none exist, we fall back
        # to using train as the pool (caller should treat tasks as within-pool).
        root_path = root
        # allow passing either the dataset root or the parent data dir
        candidate_roots = [
            f"{root_path}/tiny-imagenet-200",
            root_path,
        ]
        train_dir = None
        for base in candidate_roots:
            cand = f"{base}/train"
            if Path(cand).exists():
                train_dir = cand
                break

        if train_dir is None:
            raise ValueError(
                "tinyimagenet root not found. Provide --data_root pointing to either "
                ".../tiny-imagenet-200 or its parent folder."
            )

        if split == "train":
            return ImageFolder(root=train_dir, transform=tfm)

        # try organized val folders if user has prepared them
        for val_candidate in [
            f"{Path(train_dir).parent}/val",
            f"{Path(train_dir).parent}/val_reorg",
            f"{Path(train_dir).parent}/test",
        ]:
            if Path(val_candidate).exists() and any(Path(val_candidate).glob("*")):
                return ImageFolder(root=val_candidate, transform=tfm)

        # fallback: use train pool
        return ImageFolder(root=train_dir, transform=tfm)

    if name == "fake":
        # FakeData does not expose stable labels; create fixed targets for task sampling.
        size = 5000 if split == "train" else 2000

        class FixedFakeData(FakeData):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                rng = random.Random(123 if split == "train" else 456)
                self.targets = [rng.randrange(self.num_classes) for _ in range(size)]

            def __getitem__(self, index):
                x, _ = super().__getitem__(index)
                return x, int(self.targets[index])

        return FixedFakeData(
            size=size,
            image_size=(3, image_size, image_size),
            num_classes=100,
            transform=tfm,
        )

    raise ValueError(f"Unsupported dataset: {name}. Supported: {sorted(SUPPORTED_DATASETS.keys())}")


def _get_targets(ds: Dataset) -> list[int]:
    # CIFAR10/100: .targets
    if hasattr(ds, "targets"):
        targets = getattr(ds, "targets")
        return list(map(int, targets))

    # SVHN: .labels
    if hasattr(ds, "labels"):
        targets = getattr(ds, "labels")
        # SVHN labels can be numpy array
        return [int(x) for x in targets]

    # ImageFolder: .targets
    if hasattr(ds, "targets"):
        return [int(x) for x in getattr(ds, "targets")]

    raise ValueError("Dataset does not expose targets/labels")


def build_class_index(ds: Dataset) -> dict[int, list[int]]:
    targets = _get_targets(ds)
    class_to_indices: dict[int, list[int]] = {}
    for idx, y in enumerate(targets):
        class_to_indices.setdefault(int(y), []).append(idx)
    return class_to_indices


def sample_task_indices(
    ds: Dataset,
    num_way: int,
    support_per_class: int,
    query_per_class: int,
    seed: int,
    class_ids: Optional[list[int]] = None,
) -> tuple[list[int], list[int], list[int]]:
    """Return (class_ids, support_indices, query_indices)."""
    rng = random.Random(seed)
    class_to_indices = build_class_index(ds)

    available_classes = sorted(class_to_indices.keys())
    if class_ids is None:
        if len(available_classes) < num_way:
            raise ValueError(f"Dataset has only {len(available_classes)} classes; need {num_way}")
        class_ids = rng.sample(available_classes, num_way)
    else:
        if len(class_ids) != num_way:
            raise ValueError(f"Expected class_ids length {num_way}, got {len(class_ids)}")

    support_indices: list[int] = []
    query_indices: list[int] = []

    for cid in class_ids:
        indices = list(class_to_indices[int(cid)])
        rng.shuffle(indices)
        need = support_per_class + query_per_class
        if len(indices) < need:
            raise ValueError(f"Class {cid} has only {len(indices)} samples, need {need}")
        support_indices.extend(indices[:support_per_class])
        query_indices.extend(indices[support_per_class:need])

    return class_ids, support_indices, query_indices


def remap_subset_labels(ds: Dataset, indices: list[int], class_ids: list[int]) -> Dataset:
    """Return a Dataset wrapper that remaps labels to 0..K-1 for selected class_ids."""
    cid_to_new = {int(cid): i for i, cid in enumerate(class_ids)}
    subset = Subset(ds, indices)

    class Remapped(Dataset):
        def __len__(self):
            return len(subset)

        def __getitem__(self, i):
            x, y = subset[i]
            return x, cid_to_new[int(y)]

    return Remapped()


def pack_support_tensor(ds: Dataset, indices_by_class: list[list[int]]) -> torch.Tensor:
    """Return (K, Ns, 3, H, W) tensor for conditioning."""
    # assume transform already gives (3,H,W)
    class_tensors = []
    for class_indices in indices_by_class:
        imgs = [ds[i][0] for i in class_indices]
        class_tensors.append(torch.stack(imgs, dim=0))
    return torch.stack(class_tensors, dim=0)


def group_support_indices_by_class(
    ds: Dataset,
    class_ids: list[int],
    support_indices: list[int],
) -> list[list[int]]:
    targets = _get_targets(ds)
    cid_set = {int(c) for c in class_ids}
    grouped: dict[int, list[int]] = {int(c): [] for c in class_ids}
    for idx in support_indices:
        y = int(targets[idx])
        if y in cid_set:
            grouped[y].append(idx)
    # preserve class_ids order
    return [grouped[int(c)] for c in class_ids]

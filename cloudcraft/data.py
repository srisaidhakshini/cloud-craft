"""MNIST loading and the seeded IID / Dirichlet splits (FR-1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

NUM_CLASSES = 10


def iid_split(labels: np.ndarray, num_clients: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(labels))
    return [sorted(part.tolist()) for part in np.array_split(perm, num_clients)]


def dirichlet_split(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int, min_size: int = 50
) -> List[List[int]]:
    """Per-class Dirichlet(alpha) allocation; resample until every client has min_size."""
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        parts: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(NUM_CLASSES):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            props = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for client, chunk in enumerate(np.split(idx, cuts)):
                parts[client].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [sorted(p) for p in parts]
    raise RuntimeError("could not produce a Dirichlet split with the minimum client size")


def split_path(data_dir: str, split: str, seed: int) -> Path:
    return Path(data_dir) / "splits" / f"{split}_seed{seed}.json"


def load_mnist(data_dir: str, train: bool):
    from torchvision import datasets, transforms

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return datasets.MNIST(str(Path(data_dir) / "mnist"), train=train, download=True, transform=tfm)


def make_and_save_split(
    data_dir: str, split: str, seed: int, num_clients: int = 5, alpha: float = 0.5
) -> Path:
    """Create the split (if absent) and save indices to disk."""
    path = split_path(data_dir, split, seed)
    if path.exists():
        return path
    labels = np.asarray(load_mnist(data_dir, train=True).targets)
    if split == "iid":
        parts = iid_split(labels, num_clients, seed)
    elif split == "dirichlet":
        parts = dirichlet_split(labels, num_clients, alpha, seed)
    else:
        raise ValueError(f"unknown split '{split}' (use 'iid' or 'dirichlet')")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, List[int]] = {str(i + 1): p for i, p in enumerate(parts)}
    path.write_text(json.dumps({"split": split, "seed": seed, "alpha": alpha, "indices": payload}))
    return path


def load_split_indices(data_dir: str, split: str, seed: int, client_id: str) -> List[int]:
    path = split_path(data_dir, split, seed)
    return json.loads(path.read_text())["indices"][str(client_id)]

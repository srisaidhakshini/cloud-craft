"""Small CNN for MNIST plus train / eval helpers and parameter conversion."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


def init_model(seed: int) -> Net:
    torch.manual_seed(seed)
    return Net()


def get_weights(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    keys = model.state_dict().keys()
    model.load_state_dict({k: torch.tensor(w) for k, w in zip(keys, weights)}, strict=True)


def train(model: nn.Module, loader, epochs: int, lr: float, momentum: float) -> float:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    total, n = 0.0, 0
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader):
    model.eval()
    loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        out = model(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return loss / n, correct / n

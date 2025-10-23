# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PackageScorer(nn.Module):
    """对算子包（特征向量）进行评分（0..1）。"""

    def __init__(self, in_dim: int, hidden: Tuple[int, int] = (128, 64)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).view(-1)


def train_scorer(
        model: PackageScorer,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        *,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 3e-4,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_x.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(n)
        x = train_x[idx]
        y = train_y[idx]
        for i in range(0, n, batch_size):
            xb = x[i:i + batch_size]
            yb = y[i:i + batch_size]
            pred = model(xb)
            loss = F.binary_cross_entropy(pred, yb)
            opt.zero_grad();
            loss.backward();
            opt.step()

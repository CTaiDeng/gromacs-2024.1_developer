# Copyright (C) 2025 GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes: Tuple[int, ...], activation=nn.ReLU, output_activation=None) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 128)):
        super().__init__()
        self.net = mlp((obs_dim, *hidden, n_actions))

    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp((obs_dim, *hidden, n_actions))

    def forward(self, x: torch.Tensor):
        return self.net(x)

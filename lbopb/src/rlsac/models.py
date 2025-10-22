# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    """Actor that outputs action logits over discrete action space.

    Forward(state) -> logits (batch, n_actions) and softmax probs.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 128)):
        super().__init__()
        self.net = mlp((obs_dim, *hidden, n_actions))

    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class QNetwork(nn.Module):
    """Critic that outputs Q-values for all actions: Q(s, :)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp((obs_dim, *hidden, n_actions))

    def forward(self, x: torch.Tensor):
        return self.net(x)



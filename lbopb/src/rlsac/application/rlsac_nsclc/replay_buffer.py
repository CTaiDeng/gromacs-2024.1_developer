# Copyright (C) 2025 GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from typing import Tuple
import random
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity,), dtype=torch.long)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.float32)
        self.ptr = 0
        self.size = 0

    def push(self, s: torch.Tensor, a: int, r: float, s2: torch.Tensor, d: bool) -> None:
        i = self.ptr % self.capacity
        self.obs[i] = s
        self.actions[i] = int(a)
        self.rewards[i] = float(r)
        self.next_obs[i] = s2
        self.dones[i] = 1.0 if d else 0.0
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = min(batch_size, self.size)
        idx = random.sample(range(self.size), n)
        s = self.obs[idx]
        a = self.actions[idx]
        r = self.rewards[idx]
        s2 = self.next_obs[idx]
        d = self.dones[idx]
        return s, a, r, s2, d




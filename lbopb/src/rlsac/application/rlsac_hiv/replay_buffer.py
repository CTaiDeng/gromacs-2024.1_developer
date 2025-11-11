# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 only.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# --- 著作权独立性声明 (Copyright Independence Declaration) ---
# 本文件（“载荷”）是作者 (GaoZheng) 的原创著作物，其知识产权
# 独立于其运行平台 GROMACS（“宿主”）。
# 本文件的授权遵循上述 SPDX 标识，不受“宿主”许可证的管辖。
# 详情参见项目文档 "my_docs/project_docs/1762636780_🚩🚩gromacs-2024.1_developer项目的著作权设计策略：“宿主-载荷”与“双轨制”复合架构.md"。
# ------------------------------------------------------------------

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

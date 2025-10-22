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

from collections import deque
from typing import Deque, Tuple

import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.buffer: Deque[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=capacity)

    def push(self, s: torch.Tensor, a: int, r: float, s2: torch.Tensor, d: bool) -> None:
        self.buffer.append((s.detach().cpu(), int(a), float(r), s2.detach().cpu(), bool(d)))

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int):
        idx = torch.randint(0, len(self.buffer), (batch_size,), dtype=torch.int64)
        s = torch.stack([self.buffer[i][0] for i in idx])
        a = torch.tensor([self.buffer[i][1] for i in idx], dtype=torch.long)
        r = torch.tensor([self.buffer[i][2] for i in idx], dtype=torch.float32)
        s2 = torch.stack([self.buffer[i][3] for i in idx])
        d = torch.tensor([self.buffer[i][4] for i in idx], dtype=torch.float32)  # as 0/1 mask
        return s, a, r, s2, d



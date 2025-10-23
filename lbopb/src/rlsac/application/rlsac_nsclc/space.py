# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class SimpleBoxInt32:
    low: int
    high: int
    shape: Tuple[int, ...]

    def sample(self) -> torch.Tensor:
        return torch.randint(self.low, self.high, self.shape, dtype=torch.int32)


@dataclass
class SimpleBoxFloat32:
    low: float
    high: float
    shape: Tuple[int, ...]

    def sample(self) -> torch.Tensor:
        return (self.low + (self.high - self.low) * torch.rand(self.shape)).to(torch.float32)




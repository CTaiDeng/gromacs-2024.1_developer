# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

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


class DummyEnv:
    """Minimal RL environment using pytorch tensors.

    This is a placeholder to validate algorithm wiring. It follows the interface:
    - reset() -> state (torch.Tensor)
    - step(action: torch.Tensor) -> next_state, reward, done, info
    """

    def __init__(self) -> None:
        self.observation_space = SimpleBoxFloat32(0.0, 1.0, (4,))
        # 2 discrete actions [0, 1]
        self.action_space = SimpleBoxInt32(0, 2, (1,))
        self.state = torch.zeros(4, dtype=torch.float32)

    def reset(self) -> torch.Tensor:
        self.state = torch.zeros(4, dtype=torch.float32)
        return self.state

    def step(self, action: torch.Tensor):
        reward = torch.rand(1).item()
        done = False
        self.state = torch.rand(4)
        info: Dict[str, Any] = {}
        return self.state, reward, done, info


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

import json
from pathlib import Path
from typing import Tuple

import torch


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def select_device_from_config(cfg_path: Path) -> torch.device:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    use_gpu = bool(cfg.get("use_gpu", False))
    requested_device = cfg.get("device", "cpu")
    if use_gpu and torch.cuda.is_available():
        return torch.device(requested_device if requested_device.startswith("cuda") else "cuda:0")
    return torch.device("cpu")


def discrete_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(probs, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=-1)



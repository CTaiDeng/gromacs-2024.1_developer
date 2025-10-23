# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import json
from pathlib import Path
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


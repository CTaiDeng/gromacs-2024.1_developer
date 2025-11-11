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

import json
from pathlib import Path
import torch


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def select_device_from_config(cfg_path: Path) -> torch.device:
    """解析设备选择，兼容旧字段：
    - 旧：`use_gpu`(bool) + `device`(str: "cpu"/"cuda:0")
    - 新：`device_choose`(dict: name->id) + `device`(int id)
            或 `device`(str: name)
    返回可用的 `torch.device`；若请求 GPU 但不可用，则回退 CPU。
    """
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))

    # 1) 新风格解析：device 可为 str 名称或 int 编号
    dev_mapping = cfg.get("device_choose", {}) or {}
    dev = cfg.get("device", None)
    name: str | None = None
    if isinstance(dev, str):
        name = dev.strip().lower()
    elif isinstance(dev, int) and isinstance(dev_mapping, dict) and dev_mapping:
        for k, v in dev_mapping.items():
            try:
                if int(v) == dev:
                    name = str(k).strip().lower()
                    break
            except Exception:
                continue

    # 2) 旧风格回退
    if name is None:
        use_gpu = bool(cfg.get("use_gpu", False))
        requested_device = str(cfg.get("device", "cpu")).strip().lower()
        if use_gpu:
            name = "gpu"
        else:
            # 若旧字段直接给了 cuda:*，也按 gpu 处理
            name = "gpu" if requested_device.startswith("cuda") else "cpu"

    # 3) 映射到 torch.device
    if name == "gpu":
        if torch.cuda.is_available():
            # 允许未来扩展：如果配置提供 cuda:* 则直接使用
            requested = str(cfg.get("device", "")).strip().lower()
            return torch.device(requested if requested.startswith("cuda") else "cuda:0")
        return torch.device("cpu")
    # 默认 CPU
    return torch.device("cpu")


def discrete_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(probs, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=-1)

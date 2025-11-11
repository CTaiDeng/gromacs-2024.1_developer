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

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PackageScorer(nn.Module):
    """对算子包（特征向量）进行评分（0..1）。"""

    def __init__(self, in_dim: int, hidden: Tuple[int, int] = (128, 64)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).view(-1)


def train_scorer(
        model: PackageScorer,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        *,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 3e-4,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_x.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(n)
        x = train_x[idx]
        y = train_y[idx]
        for i in range(0, n, batch_size):
            xb = x[i:i + batch_size]
            yb = y[i:i + batch_size]
            pred = model(xb)
            loss = F.binary_cross_entropy(pred, yb)
            opt.zero_grad();
            loss.backward();
            opt.step()

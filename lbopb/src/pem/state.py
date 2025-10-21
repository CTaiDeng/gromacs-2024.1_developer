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

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict


@dataclass(frozen=True)
class PEMState:
    """PEM 状态的极简可计算表象。

    该抽象不绑定具体几何/测度实现，仅以可观测量的充分统计量近似：
    - b: 负担 B(S)=μ(S) ≥ 0
    - n_comp: 组分数 N_comp(S) ∈ N
    - perim: 边界周长 P(S)=μ(∂S) ≥ 0
    - fidelity: 功能保真 F(S) ∈ [0, 1]（或任意非负量纲化数）
    - meta: 附加元数据（可存放几何句柄、标签等），不参与等价比较
    """

    b: float
    n_comp: int
    perim: float
    fidelity: float
    meta: Dict[str, Any] | None = None

    def clamp(self) -> "PEMState":
        """裁剪到可行域（避免算子数值外逸）。"""
        b = max(0.0, float(self.b))
        n_comp = max(0, int(self.n_comp))
        perim = max(0.0, float(self.perim))
        # 允许 fidelity > 1 的扩展，但通常裁剪到 [0,1]
        fidelity = min(1.0, max(0.0, float(self.fidelity)))
        return replace(self, b=b, n_comp=n_comp, perim=perim, fidelity=fidelity)

    def with_meta(self, **meta: Any) -> "PEMState":
        d = dict(self.meta or {})
        d.update(meta)
        return replace(self, meta=d)

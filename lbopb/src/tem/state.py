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
class TEMState:
    """TEM 状态的可观测元组

    - b: 损伤负荷 B(S)=μ(S) ≥ 0
    - n_comp: 病灶/损伤灶数量 N_comp(S) ∈ N
    - perim: 损伤边界度量 P(S)=μ(∂S) ≥ 0
    - fidelity: 功能保真 F(S) ∈ [0,1]
    - meta: 元信息（上下文、标签等）
    """

    b: float
    n_comp: int
    perim: float
    fidelity: float
    meta: Dict[str, Any] | None = None

    def clamp(self) -> "TEMState":
        b = max(0.0, float(self.b))
        n_comp = max(0, int(self.n_comp))
        perim = max(0.0, float(self.perim))
        fidelity = min(1.0, max(0.0, float(self.fidelity)))
        return replace(self, b=b, n_comp=n_comp, perim=perim, fidelity=fidelity)

    def with_meta(self, **meta: Any) -> "TEMState":
        d = dict(self.meta or {})
        d.update(meta)
        return replace(self, meta=d)


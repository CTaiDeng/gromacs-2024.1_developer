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

from typing import Any, Dict, List
from pathlib import Path
import json

MODULES = ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]


def _nearest_index(values: List[float], x: float) -> int:
    if not values:
        return 0
    best_i = 0
    best_d = abs(x - float(values[0]))
    for i, v in enumerate(values):
        d = abs(x - float(v))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


class ObservationQuantizer:
    def __init__(self, map_path: str | Path) -> None:
        self.map = json.loads(Path(map_path).read_text(encoding="utf-8"))
        self.modules = self.map.get("modules", {})
        self.mode = str(self.map.get("quantize_mode", "nearest")).lower()

    def quantize_state(self, domain: str, state: Any) -> List[int]:
        dom = self.modules.get(domain, {})
        # state fields: b, perim, fidelity, n_comp
        b_seq = list(dom.get("b", []))
        p_seq = list(dom.get("perim", []))
        f_seq = list(dom.get("fidelity", []))
        n_seq = list(dom.get("n", []))
        b = float(getattr(state, "b", 0.0))
        p = float(getattr(state, "perim", 0.0))
        f = float(getattr(state, "fidelity", 0.0))
        n = float(getattr(state, "n_comp", 0.0))
        if self.mode == "nearest":
            return [
                _nearest_index(b_seq, b),
                _nearest_index(p_seq, p),
                _nearest_index(f_seq, f),
                _nearest_index(n_seq, n),
            ]
        # fallback: nearest
        return [
            _nearest_index(b_seq, b),
            _nearest_index(p_seq, p),
            _nearest_index(f_seq, f),
            _nearest_index(n_seq, n),
        ]

    def quantize_full(self, states_by_domain: Dict[str, Any], risk_by_domain: Dict[str, float]) -> List[int]:
        out: List[int] = []
        for m in MODULES:
            st = states_by_domain.get(m)
            idxs = self.quantize_state(m, st)
            # risk 映射
            r_seq = list(self.modules.get(m, {}).get("risk", []))
            r_val = float(risk_by_domain.get(m, 0.0))
            r_idx = _nearest_index(r_seq, r_val)
            out += idxs + [r_idx]
        return out

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

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

from .state import PDEMState


def B(state: PDEMState) -> float:
    return float(state.b)


def N_comp(state: PDEMState) -> int:
    return int(state.n_comp)


def P(state: PDEMState) -> float:
    return float(state.perim)


def F(state: PDEMState) -> float:
    return float(state.fidelity)


PhiFunc = Callable[[PDEMState], float]


@dataclass(frozen=True)
class Observables:
    """PDEM 可观察量集合封装"""

    funcs: Dict[str, PhiFunc]

    @staticmethod
    def default() -> "Observables":
        return Observables({"B": B, "N_comp": lambda s: float(N_comp(s)), "P": P, "F": F})

    def eval_all(self, state: PDEMState) -> Dict[str, float]:
        return {k: float(f(state)) for k, f in self.funcs.items()}

    def names(self) -> Iterable[str]:
        return self.funcs.keys()

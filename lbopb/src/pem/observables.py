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

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

from .state import PEMState


def B(state: PEMState) -> float:
    return float(state.b)


def N_comp(state: PEMState) -> int:
    return int(state.n_comp)


def P(state: PEMState) -> float:
    return float(state.perim)


def F(state: PEMState) -> float:
    return float(state.fidelity)


PhiFunc = Callable[[PEMState], float]


@dataclass(frozen=True)
class Observables:
    """PEM 可观测量族 Φ 的轻量封装。"""

    funcs: Dict[str, PhiFunc]

    @staticmethod
    def default() -> "Observables":
        return Observables({"B": B, "N_comp": lambda s: float(N_comp(s)), "P": P, "F": F})

    def eval_all(self, state: PEMState) -> Dict[str, float]:
        return {k: float(f(state)) for k, f in self.funcs.items()}

    def names(self) -> Iterable[str]:
        return self.funcs.keys()

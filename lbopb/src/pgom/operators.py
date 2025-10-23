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
from typing import Dict, Tuple

from .state import PGOMState


class PGOMOperator:
    """PGOM 调控算子基类：O: S -> S"""

    name: str = "PGOMOperator"
    params: Dict[str, float]

    def __init__(self, **params: float) -> None:
        self.params = dict(params)

    def __call__(self, s: PGOMState) -> PGOMState:
        return self.apply(s).clamp()

    def apply(self, s: PGOMState) -> PGOMState:  # pragma: no cover - abstract by convention
        return s

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"


class Identity(PGOMOperator):
    name = "Identity"

    def apply(self, s: PGOMState) -> PGOMState:
        return s


class Activate(PGOMOperator):
    """基因/通路激活：提升表达与保真，边界略增。

    - b' = b * (1 + alpha_b)
    - perim' = perim * (1 + alpha_p)
    - fidelity' = min(1, fidelity + delta_f)
    - n_comp' = n_comp + ceil(dn)
    """

    name = "Activate"

    def apply(self, s: PGOMState) -> PGOMState:
        from math import ceil

        alpha_b = float(self.params.get("alpha_b", 0.2))
        alpha_p = float(self.params.get("alpha_p", 0.05))
        delta_f = float(self.params.get("delta_f", 0.06))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 + max(0.0, alpha_b))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        f = s.fidelity + max(0.0, delta_f)
        n = s.n_comp + max(0, ceil(dn))
        return PGOMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Repress(PGOMOperator):
    """基因/通路抑制：降低表达与边界，保真下降或维持。

    - b' = b * (1 - gamma_b)
    - perim' = perim * (1 - gamma_p)
    - fidelity' = fidelity * (1 - gamma_f)
    - n_comp' 轻微变化
    """

    name = "Repress"

    def apply(self, s: PGOMState) -> PGOMState:
        from math import ceil

        gamma_b = float(self.params.get("gamma_b", 0.15))
        gamma_p = float(self.params.get("gamma_p", 0.05))
        gamma_f = float(self.params.get("gamma_f", 0.02))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 - max(0.0, min(gamma_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(gamma_p, 0.99)))
        f = s.fidelity * (1.0 - max(0.0, min(gamma_f, 0.95)))
        n = s.n_comp + max(0, ceil(dn))
        return PGOMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Mutation(PGOMOperator):
    """突变：模块与边界上升，保真下降；表达可上或下。

    - n_comp' = n_comp + ceil(alpha_n)
    - perim' = perim * (1 + alpha_p)
    - b' = b * (1 + beta_b)
    - fidelity' = fidelity * (1 - beta_f)
    """

    name = "Mutation"

    def apply(self, s: PGOMState) -> PGOMState:
        from math import ceil

        alpha_n = float(self.params.get("alpha_n", 1.0))
        alpha_p = float(self.params.get("alpha_p", 0.1))
        beta_b = float(self.params.get("beta_b", 0.0))
        beta_f = float(self.params.get("beta_f", 0.08))

        n = s.n_comp + max(1, ceil(alpha_n))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        b = s.b * (1.0 + beta_b)
        f = s.fidelity * (1.0 - max(0.0, min(beta_f, 0.95)))
        return PGOMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class RepairGenome(PGOMOperator):
    """修复：降低边界与模块数，提升保真与稳定性。

    - b' = b * (1 - rho_b)
    - perim' = perim * (1 - rho_p)
    - fidelity' = min(1, fidelity + delta_f)
    - n_comp' = max(1, round(n_comp * (1 - rho_n)))
    """

    name = "RepairGenome"

    def apply(self, s: PGOMState) -> PGOMState:
        rho_b = float(self.params.get("rho_b", 0.05))
        rho_p = float(self.params.get("rho_p", 0.15))
        rho_n = float(self.params.get("rho_n", 0.1))
        delta_f = float(self.params.get("delta_f", 0.12))

        b = s.b * (1.0 - max(0.0, min(rho_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(rho_p, 0.99)))
        f = s.fidelity + max(0.0, delta_f)
        n = max(1, round(s.n_comp * (1.0 - max(0.0, min(rho_n, 0.99)))))
        return PGOMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class EpigeneticMod(PGOMOperator):
    """表观修饰：按比例调整表达/保真/边界，可正可负。

    - b' = b * (1 + theta_b)
    - perim' = perim * (1 + theta_p)
    - fidelity' = fidelity * (1 + theta_f)
    - n_comp' = n_comp + dn
    """

    name = "EpigeneticMod"

    def apply(self, s: PGOMState) -> PGOMState:
        theta_b = float(self.params.get("theta_b", 0.0))
        theta_p = float(self.params.get("theta_p", 0.0))
        theta_f = float(self.params.get("theta_f", 0.05))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 + theta_b)
        p = s.perim * (1.0 + theta_p)
        f = s.fidelity * (1.0 + theta_f)
        n = s.n_comp + max(0, dn)
        return PGOMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class PathwayInduction(PGOMOperator):
    """通路诱导：增强表达与功能，边界适度上升。"""

    name = "PathwayInduction"

    def apply(self, s: PGOMState) -> PGOMState:
        alpha_b = float(self.params.get("alpha_b", 0.1))
        alpha_p = float(self.params.get("alpha_p", 0.05))
        delta_f = float(self.params.get("delta_f", 0.08))
        b = s.b * (1.0 + max(0.0, alpha_b))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        f = s.fidelity + max(0.0, delta_f)
        return PGOMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


class PathwayInhibition(PGOMOperator):
    """通路抑制：降低表达与功能，边界回落。"""

    name = "PathwayInhibition"

    def apply(self, s: PGOMState) -> PGOMState:
        gamma_b = float(self.params.get("gamma_b", 0.1))
        gamma_p = float(self.params.get("gamma_p", 0.05))
        gamma_f = float(self.params.get("gamma_f", 0.06))
        b = s.b * (1.0 - max(0.0, min(gamma_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(gamma_p, 0.99)))
        f = s.fidelity * (1.0 - max(0.0, min(gamma_f, 0.95)))
        return PGOMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


@dataclass(frozen=True)
class ComposedOperator(PGOMOperator):
    ops: Tuple[PGOMOperator, ...]
    name: str = "Composed"

    def __init__(self, *ops: PGOMOperator):  # type: ignore[override]
        object.__setattr__(self, "ops", tuple(ops))
        object.__setattr__(self, "params", {})

    def apply(self, s: PGOMState) -> PGOMState:
        out = s
        for op in self.ops:
            out = op(out)
        return out

    def __repr__(self) -> str:
        return "Composed(" + ", ".join(repr(o) for o in self.ops) + ")"


def compose(*ops: PGOMOperator) -> PGOMOperator:
    flat: list[PGOMOperator] = []
    for o in ops:
        if isinstance(o, Identity):
            continue
        if isinstance(o, ComposedOperator):
            flat.extend(o.ops)
        else:
            flat.append(o)
    if not flat:
        return Identity()
    if len(flat) == 1:
        return flat[0]
    return ComposedOperator(*flat)

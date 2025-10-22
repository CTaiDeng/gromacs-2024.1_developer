# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
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

from .state import TEMState


class TEMOperator:
    """TEM 调控算子基类：O: S -> S

    - 幺半群：单位元 Identity 与复合 compose
    - 非交换：一般 A∘B ≠ B∘A
    - 参数以 `params` 字典保存
    """

    name: str = "TEMOperator"
    params: Dict[str, float]

    def __init__(self, **params: float) -> None:
        self.params = dict(params)

    def __call__(self, s: TEMState) -> TEMState:
        return self.apply(s).clamp()

    def apply(self, s: TEMState) -> TEMState:  # pragma: no cover - abstract by convention
        return s

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"


class Identity(TEMOperator):
    name = "Identity"

    def apply(self, s: TEMState) -> TEMState:
        return s


class Exposure(TEMOperator):
    """暴露 O_exposure：提升损伤负荷与边界，轻度降低保真。

    - b' = b * (1 + alpha_b)
    - perim' = perim * (1 + alpha_p)
    - fidelity' = fidelity * (1 - alpha_f)
    - n_comp' = n_comp + ceil(dn)
    """

    name = "Exposure"

    def apply(self, s: TEMState) -> TEMState:
        from math import ceil

        alpha_b = float(self.params.get("alpha_b", 0.3))
        alpha_p = float(self.params.get("alpha_p", 0.15))
        alpha_f = float(self.params.get("alpha_f", 0.05))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 + max(0.0, alpha_b))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        f = s.fidelity * (1.0 - max(0.0, min(alpha_f, 0.95)))
        n = s.n_comp + max(0, ceil(dn))
        return TEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Absorption(TEMOperator):
    """吸收 O_absorb：将外源负荷转入系统，边界略增，保真下降。

    - b' = b * (1 + beta_b)
    - perim' = perim * (1 + beta_p)
    - fidelity' = fidelity * (1 - beta_f)
    - n_comp 不变或微调
    """

    name = "Absorption"

    def apply(self, s: TEMState) -> TEMState:
        beta_b = float(self.params.get("beta_b", 0.2))
        beta_p = float(self.params.get("beta_p", 0.05))
        beta_f = float(self.params.get("beta_f", 0.03))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 + max(0.0, beta_b))
        p = s.perim * (1.0 + max(0.0, beta_p))
        f = s.fidelity * (1.0 - max(0.0, min(beta_f, 0.95)))
        n = s.n_comp + max(0, dn)
        return TEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Distribution(TEMOperator):
    """分布 O_distribute：加大边界与灶数，负荷略变。

    - n_comp' = n_comp + ceil(alpha_n)
    - perim' = perim * (1 + alpha_p)
    - b' = b * (1 + alpha_b)
    - fidelity' = fidelity * (1 - alpha_f)
    """

    name = "Distribution"

    def apply(self, s: TEMState) -> TEMState:
        from math import ceil

        alpha_n = float(self.params.get("alpha_n", 1.0))
        alpha_p = float(self.params.get("alpha_p", 0.2))
        alpha_b = float(self.params.get("alpha_b", 0.05))
        alpha_f = float(self.params.get("alpha_f", 0.04))

        n = s.n_comp + max(1, ceil(alpha_n))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        b = s.b * (1.0 + max(0.0, alpha_b))
        f = s.fidelity * (1.0 - max(0.0, min(alpha_f, 0.95)))
        return TEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Lesion(TEMOperator):
    """病灶/损伤 O_lesion：显著增加边界与负荷，降低保真。

    - b' = b * (1 + k_b)
    - perim' = perim * (1 + k_p)
    - fidelity' = fidelity * (1 - k_f)
    - n_comp' = n_comp + ceil(dn)
    """

    name = "Lesion"

    def apply(self, s: TEMState) -> TEMState:
        from math import ceil

        k_b = float(self.params.get("k_b", 0.3))
        k_p = float(self.params.get("k_p", 0.35))
        k_f = float(self.params.get("k_f", 0.12))
        dn = int(self.params.get("dn", 1))

        b = s.b * (1.0 + max(0.0, k_b))
        p = s.perim * (1.0 + max(0.0, k_p))
        f = s.fidelity * (1.0 - max(0.0, min(k_f, 0.95)))
        n = s.n_comp + max(1, ceil(dn))
        return TEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Inflammation(TEMOperator):
    """炎症 O_inflam：增负荷与边界，保真下降。

    - b' = b * (1 + eta_b)
    - perim' = perim * (1 + eta_p)
    - fidelity' = fidelity * (1 - eta_f)
    - n_comp' 轻微增加
    """

    name = "Inflammation"

    def apply(self, s: TEMState) -> TEMState:
        from math import ceil

        eta_b = float(self.params.get("eta_b", 0.1))
        eta_p = float(self.params.get("eta_p", 0.25))
        eta_f = float(self.params.get("eta_f", 0.06))
        dn = int(self.params.get("dn", 1))

        b = s.b * (1.0 + max(0.0, eta_b))
        p = s.perim * (1.0 + max(0.0, eta_p))
        f = s.fidelity * (1.0 - max(0.0, min(eta_f, 0.95)))
        n = s.n_comp + max(0, ceil(dn))
        return TEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Detox(TEMOperator):
    """解毒/代谢 O_detox：降低负荷与边界，提高保真。

    - b' = b * (1 - gamma_b)
    - perim' = perim * (1 - gamma_p)
    - fidelity' = min(1, fidelity + delta_f)
    - n_comp' = max(1, round(n_comp * (1 - gamma_n)))
    """

    name = "Detox"

    def apply(self, s: TEMState) -> TEMState:
        gamma_b = float(self.params.get("gamma_b", 0.25))
        gamma_p = float(self.params.get("gamma_p", 0.2))
        gamma_n = float(self.params.get("gamma_n", 0.0))
        delta_f = float(self.params.get("delta_f", 0.08))

        b = s.b * (1.0 - max(0.0, min(gamma_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(gamma_p, 0.99)))
        f = s.fidelity + max(0.0, delta_f)
        n = max(1, round(s.n_comp * (1.0 - max(0.0, min(gamma_n, 0.99)))))
        return TEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Repair(TEMOperator):
    """修复 O_repair：强提升保真，降低边界与负荷，灶数回落。

    - b' = b * (1 - rho_b)
    - perim' = perim * (1 - rho_p)
    - fidelity' = min(1, fidelity + delta_f)
    - n_comp' = max(1, round(n_comp * (1 - rho_n)))
    """

    name = "Repair"

    def apply(self, s: TEMState) -> TEMState:
        rho_b = float(self.params.get("rho_b", 0.15))
        rho_p = float(self.params.get("rho_p", 0.25))
        rho_n = float(self.params.get("rho_n", 0.1))
        delta_f = float(self.params.get("delta_f", 0.12))

        b = s.b * (1.0 - max(0.0, min(rho_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(rho_p, 0.99)))
        f = s.fidelity + max(0.0, delta_f)
        n = max(1, round(s.n_comp * (1.0 - max(0.0, min(rho_n, 0.99)))))
        return TEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


@dataclass(frozen=True)
class ComposedOperator(TEMOperator):
    """复合算子：O_k ∘ ... ∘ O_1"""

    ops: Tuple[TEMOperator, ...]
    name: str = "Composed"

    def __init__(self, *ops: TEMOperator):  # type: ignore[override]
        object.__setattr__(self, "ops", tuple(ops))
        object.__setattr__(self, "params", {})

    def apply(self, s: TEMState) -> TEMState:
        out = s
        for op in self.ops:
            out = op(out)
        return out

    def __repr__(self) -> str:
        return "Composed(" + ", ".join(repr(o) for o in self.ops) + ")"


def compose(*ops: TEMOperator) -> TEMOperator:
    """扁平化复合，自动消去单位元"""
    flat: list[TEMOperator] = []
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


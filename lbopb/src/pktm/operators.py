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
from typing import Dict, Tuple

from .state import PKTMState


class PKTMOperator:
    """PKTM 调控算子基类：O: S -> S"""

    name: str = "PKTMOperator"
    params: Dict[str, float]

    def __init__(self, **params: float) -> None:
        self.params = dict(params)

    def __call__(self, s: PKTMState) -> PKTMState:
        return self.apply(s).clamp()

    def apply(self, s: PKTMState) -> PKTMState:  # pragma: no cover - abstract by convention
        return s

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"


class Identity(PKTMOperator):
    name = "Identity"

    def apply(self, s: PKTMState) -> PKTMState:
        return s


class Dose(PKTMOperator):
    """给药：提高系统负荷与边界（进入路径），保真轻降。

    - b' = b + delta_b (加性更贴合剂量)
    - perim' = perim * (1 + alpha_p)
    - fidelity' = fidelity * (1 - alpha_f)
    - n_comp 不变
    """

    name = "Dose"

    def apply(self, s: PKTMState) -> PKTMState:
        delta_b = float(self.params.get("delta_b", 1.0))
        alpha_p = float(self.params.get("alpha_p", 0.05))
        alpha_f = float(self.params.get("alpha_f", 0.01))
        b = s.b + max(0.0, delta_b)
        p = s.perim * (1.0 + max(0.0, alpha_p))
        f = s.fidelity * (1.0 - max(0.0, min(alpha_f, 0.95)))
        return PKTMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


class Absorb(PKTMOperator):
    """吸收：负荷与边界上升，保真小降。

    - b' = b * (1 + alpha_b)
    - perim' = perim * (1 + alpha_p)
    - fidelity' = fidelity * (1 - alpha_f)
    """

    name = "Absorb"

    def apply(self, s: PKTMState) -> PKTMState:
        alpha_b = float(self.params.get("alpha_b", 0.2))
        alpha_p = float(self.params.get("alpha_p", 0.1))
        alpha_f = float(self.params.get("alpha_f", 0.02))
        b = s.b * (1.0 + max(0.0, alpha_b))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        f = s.fidelity * (1.0 - max(0.0, min(alpha_f, 0.95)))
        return PKTMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


class Distribute(PKTMOperator):
    """分布：增加边界与隔室计数，负荷略变。

    - n_comp' = n_comp + ceil(alpha_n)
    - perim' = perim * (1 + alpha_p)
    - b' = b * (1 + alpha_b)
    """

    name = "Distribute"

    def apply(self, s: PKTMState) -> PKTMState:
        from math import ceil

        alpha_n = float(self.params.get("alpha_n", 1.0))
        alpha_p = float(self.params.get("alpha_p", 0.15))
        alpha_b = float(self.params.get("alpha_b", 0.05))
        n = s.n_comp + max(1, ceil(alpha_n))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        b = s.b * (1.0 + max(0.0, alpha_b))
        return PKTMState(b=b, n_comp=int(n), perim=p, fidelity=s.fidelity, meta=s.meta)


class Metabolize(PKTMOperator):
    """代谢：降低负荷与边界，提高保真。

    - b' = b * (1 - gamma_b)
    - perim' = perim * (1 - gamma_p)
    - fidelity' = min(1, fidelity + delta_f)
    """

    name = "Metabolize"

    def apply(self, s: PKTMState) -> PKTMState:
        gamma_b = float(self.params.get("gamma_b", 0.25))
        gamma_p = float(self.params.get("gamma_p", 0.15))
        delta_f = float(self.params.get("delta_f", 0.06))
        b = s.b * (1.0 - max(0.0, min(gamma_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(gamma_p, 0.99)))
        f = s.fidelity + max(0.0, delta_f)
        return PKTMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


class Excrete(PKTMOperator):
    """排泄：负荷减少，边界回落，保真提升。

    - b' = b * (1 - rho_b)
    - perim' = perim * (1 - rho_p)
    - fidelity' = min(1, fidelity + delta_f)
    """

    name = "Excrete"

    def apply(self, s: PKTMState) -> PKTMState:
        rho_b = float(self.params.get("rho_b", 0.2))
        rho_p = float(self.params.get("rho_p", 0.2))
        delta_f = float(self.params.get("delta_f", 0.04))
        b = s.b * (1.0 - max(0.0, min(rho_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(rho_p, 0.99)))
        f = s.fidelity + max(0.0, delta_f)
        return PKTMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


class Bind(PKTMOperator):
    """结合：功能位点被占用，负荷与边界变化，保真可增减。

    - b' = b * (1 + theta_b)
    - perim' = perim * (1 + theta_p)
    - fidelity' = fidelity * (1 + theta_f)
    """

    name = "Bind"

    def apply(self, s: PKTMState) -> PKTMState:
        theta_b = float(self.params.get("theta_b", 0.0))
        theta_p = float(self.params.get("theta_p", 0.0))
        theta_f = float(self.params.get("theta_f", 0.0))
        b = s.b * (1.0 + theta_b)
        p = s.perim * (1.0 + theta_p)
        f = s.fidelity * (1.0 + theta_f)
        return PKTMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


class Transport(PKTMOperator):
    """转运：通过转运体影响边界与负荷，保真变化。

    - b' = b * (1 + xi_b)
    - perim' = perim * (1 + xi_p)
    - fidelity' = fidelity * (1 + xi_f)
    """

    name = "Transport"

    def apply(self, s: PKTMState) -> PKTMState:
        xi_b = float(self.params.get("xi_b", 0.05))
        xi_p = float(self.params.get("xi_p", 0.1))
        xi_f = float(self.params.get("xi_f", 0.0))
        b = s.b * (1.0 + xi_b)
        p = s.perim * (1.0 + xi_p)
        f = s.fidelity * (1.0 + xi_f)
        return PKTMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


@dataclass(frozen=True)
class ComposedOperator(PKTMOperator):
    ops: Tuple[PKTMOperator, ...]
    name: str = "Composed"

    def __init__(self, *ops: PKTMOperator):  # type: ignore[override]
        object.__setattr__(self, "ops", tuple(ops))
        object.__setattr__(self, "params", {})

    def apply(self, s: PKTMState) -> PKTMState:
        out = s
        for op in self.ops:
            out = op(out)
        return out

    def __repr__(self) -> str:
        return "Composed(" + ", ".join(repr(o) for o in self.ops) + ")"


def compose(*ops: PKTMOperator) -> PKTMOperator:
    flat: list[PKTMOperator] = []
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

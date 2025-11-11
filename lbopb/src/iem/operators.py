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

from .state import IEMState


class IEMOperator:
    """IEM 调控算子基类：O: S -> S"""

    name: str = "IEMOperator"
    params: Dict[str, float]

    def __init__(self, **params: float) -> None:
        self.params = dict(params)

    def __call__(self, s: IEMState) -> IEMState:
        return self.apply(s).clamp()

    def apply(self, s: IEMState) -> IEMState:  # pragma: no cover - abstract by convention
        return s

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"


class Identity(IEMOperator):
    name = "Identity"

    def apply(self, s: IEMState) -> IEMState:
        return s


class Activate(IEMOperator):
    """免疫激活：提升负荷与保真，边界略增。

    - b' = b * (1 + alpha_b)
    - perim' = perim * (1 + alpha_p)
    - fidelity' = min(1, fidelity + delta_f)
    - n_comp' = n_comp + ceil(dn)
    """

    name = "Activate"

    def apply(self, s: IEMState) -> IEMState:
        from math import ceil

        alpha_b = float(self.params.get("alpha_b", 0.2))
        alpha_p = float(self.params.get("alpha_p", 0.08))
        delta_f = float(self.params.get("delta_f", 0.06))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 + max(0.0, alpha_b))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        f = s.fidelity + max(0.0, delta_f)
        n = s.n_comp + max(0, ceil(dn))
        return IEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Suppress(IEMOperator):
    """免疫抑制：降低负荷与保真，边界回落，克隆数可回落。

    - b' = b * (1 - gamma_b)
    - perim' = perim * (1 - gamma_p)
    - fidelity' = fidelity * (1 - gamma_f)
    - n_comp' = max(1, round(n_comp * (1 - gamma_n)))
    """

    name = "Suppress"

    def apply(self, s: IEMState) -> IEMState:
        gamma_b = float(self.params.get("gamma_b", 0.2))
        gamma_p = float(self.params.get("gamma_p", 0.15))
        gamma_n = float(self.params.get("gamma_n", 0.05))
        gamma_f = float(self.params.get("gamma_f", 0.08))

        b = s.b * (1.0 - max(0.0, min(gamma_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(gamma_p, 0.99)))
        f = s.fidelity * (1.0 - max(0.0, min(gamma_f, 0.95)))
        n = max(1, round(s.n_comp * (1.0 - max(0.0, min(gamma_n, 0.99)))))
        return IEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Proliferate(IEMOperator):
    """克隆扩增：克隆数与边界上升，负荷上升，保真略变。

    - n_comp' = n_comp + ceil(alpha_n)
    - perim' = perim * (1 + alpha_p)
    - b' = b * (1 + alpha_b)
    - fidelity' = fidelity * (1 + theta_f)
    """

    name = "Proliferate"

    def apply(self, s: IEMState) -> IEMState:
        from math import ceil

        alpha_n = float(self.params.get("alpha_n", 1.0))
        alpha_p = float(self.params.get("alpha_p", 0.2))
        alpha_b = float(self.params.get("alpha_b", 0.15))
        theta_f = float(self.params.get("theta_f", 0.0))

        n = s.n_comp + max(1, ceil(alpha_n))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        b = s.b * (1.0 + max(0.0, alpha_b))
        f = s.fidelity * (1.0 + theta_f)
        return IEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Differentiate(IEMOperator):
    """分化：功能成熟提升保真，边界微调，克隆轻变。

    - fidelity' = min(1, fidelity + delta_f)
    - perim' = perim * (1 + theta_p)
    - n_comp' = n_comp + dn
    - b' ~ b * (1 + theta_b)
    """

    name = "Differentiate"

    def apply(self, s: IEMState) -> IEMState:
        delta_f = float(self.params.get("delta_f", 0.12))
        theta_p = float(self.params.get("theta_p", 0.02))
        theta_b = float(self.params.get("theta_b", 0.0))
        dn = int(self.params.get("dn", 0))

        f = s.fidelity + max(0.0, delta_f)
        p = s.perim * (1.0 + theta_p)
        b = s.b * (1.0 + theta_b)
        n = s.n_comp + max(0, dn)
        return IEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class CytokineRelease(IEMOperator):
    """细胞因子释放/风暴：负荷与边界显著上升，保真下降，克隆略增。

    - b' = b * (1 + xi_b)
    - perim' = perim * (1 + xi_p)
    - fidelity' = fidelity * (1 - xi_f)
    - n_comp' = n_comp + ceil(dn)
    """

    name = "CytokineRelease"

    def apply(self, s: IEMState) -> IEMState:
        from math import ceil

        xi_b = float(self.params.get("xi_b", 0.3))
        xi_p = float(self.params.get("xi_p", 0.35))
        xi_f = float(self.params.get("xi_f", 0.15))
        dn = int(self.params.get("dn", 1))

        b = s.b * (1.0 + max(0.0, xi_b))
        p = s.perim * (1.0 + max(0.0, xi_p))
        f = s.fidelity * (1.0 - max(0.0, min(xi_f, 0.95)))
        n = s.n_comp + max(1, ceil(dn))
        return IEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Memory(IEMOperator):
    """免疫记忆：提升保真并降低边界/负荷，克隆稳定或微落。

    - b' = b * (1 - rho_b)
    - perim' = perim * (1 - rho_p)
    - fidelity' = min(1, fidelity + delta_f)
    - n_comp' = max(1, round(n_comp * (1 - rho_n)))
    """

    name = "Memory"

    def apply(self, s: IEMState) -> IEMState:
        rho_b = float(self.params.get("rho_b", 0.1))
        rho_p = float(self.params.get("rho_p", 0.15))
        rho_n = float(self.params.get("rho_n", 0.05))
        delta_f = float(self.params.get("delta_f", 0.15))

        b = s.b * (1.0 - max(0.0, min(rho_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(rho_p, 0.99)))
        f = s.fidelity + max(0.0, delta_f)
        n = max(1, round(s.n_comp * (1.0 - max(0.0, min(rho_n, 0.99)))))
        return IEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


@dataclass(frozen=True)
class ComposedOperator(IEMOperator):
    ops: Tuple[IEMOperator, ...]
    name: str = "Composed"

    def __init__(self, *ops: IEMOperator):  # type: ignore[override]
        object.__setattr__(self, "ops", tuple(ops))
        object.__setattr__(self, "params", {})

    def apply(self, s: IEMState) -> IEMState:
        out = s
        for op in self.ops:
            out = op(out)
        return out

    def __repr__(self) -> str:
        return "Composed(" + ", ".join(repr(o) for o in self.ops) + ")"


def compose(*ops: IEMOperator) -> IEMOperator:
    flat: list[IEMOperator] = []
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

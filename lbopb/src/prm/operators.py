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

from .state import PRMState


class PRMOperator:
    """PRM 调控算子基类：O: S -> S

    - 幺半群结构：存在单位元 Identity 与复合 compose
    - 非交换：一般 A∘B ≠ B∘A
    - 参数以 `params` 字典保存，便于日志/复现
    """

    name: str = "PRMOperator"
    params: Dict[str, float]

    def __init__(self, **params: float) -> None:
        self.params = dict(params)

    def __call__(self, s: PRMState) -> PRMState:
        return self.apply(s).clamp()

    def apply(self, s: PRMState) -> PRMState:  # pragma: no cover - abstract by convention
        return s

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"


class Identity(PRMOperator):
    name = "Identity"

    def apply(self, s: PRMState) -> PRMState:
        return s


class Ingest(PRMOperator):
    """摄入算子 O_ingest：提升能量/负荷，温和提升保真，边界略增。

    - b' = b * (1 + alpha_b)
    - perim' = perim * (1 + alpha_p)
    - fidelity' = min(1, fidelity + delta_f)
    - n_comp' = n_comp + max(0, dn)
    """

    name = "Ingest"

    def apply(self, s: PRMState) -> PRMState:
        from math import ceil

        alpha_b = float(self.params.get("alpha_b", 0.25))
        alpha_p = float(self.params.get("alpha_p", 0.05))
        delta_f = float(self.params.get("delta_f", 0.02))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 + max(0.0, alpha_b))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        f = s.fidelity + max(0.0, delta_f)
        n = s.n_comp + max(0, ceil(dn))
        return PRMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Exercise(PRMOperator):
    """运动算子 O_exercise：降低负荷、下降边界、提升保真。

    - b' = b * (1 - gamma_b)
    - perim' = perim * (1 - gamma_p)
    - fidelity' = min(1, fidelity + delta_f)
    - n_comp' = n_comp + max(0, dn)
    """

    name = "Exercise"

    def apply(self, s: PRMState) -> PRMState:
        from math import ceil

        gamma_b = float(self.params.get("gamma_b", 0.2))
        gamma_p = float(self.params.get("gamma_p", 0.1))
        delta_f = float(self.params.get("delta_f", 0.05))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 - max(0.0, min(gamma_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(gamma_p, 0.99)))
        f = s.fidelity + max(0.0, delta_f)
        n = s.n_comp + max(0, ceil(dn))
        return PRMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Hormone(PRMOperator):
    """激素调节 O_hormone：对多指标进行比例调谐（正负均可）。

    - b' = b * (1 + theta_b)
    - perim' = perim * (1 + theta_p)
    - fidelity' = fidelity * (1 + theta_f)
    - n_comp' = n_comp + dn
    """

    name = "Hormone"

    def apply(self, s: PRMState) -> PRMState:
        theta_b = float(self.params.get("theta_b", -0.02))
        theta_p = float(self.params.get("theta_p", 0.0))
        theta_f = float(self.params.get("theta_f", 0.05))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 + theta_b)
        p = s.perim * (1.0 + theta_p)
        f = s.fidelity * (1.0 + theta_f)
        n = s.n_comp + max(0, dn)
        return PRMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Proliferation(PRMOperator):
    """增殖 O_prolif：功能单元与边界上升，保真可能下调。

    - n_comp' = n_comp + ceil(alpha_n)
    - perim' = perim * (1 + alpha_p)
    - b' = b * (1 + beta_b)
    - fidelity' = fidelity * (1 - beta_f)
    """

    name = "Proliferation"

    def apply(self, s: PRMState) -> PRMState:
        from math import ceil

        alpha_n = float(self.params.get("alpha_n", 1.0))
        alpha_p = float(self.params.get("alpha_p", 0.1))
        beta_b = float(self.params.get("beta_b", 0.05))
        beta_f = float(self.params.get("beta_f", 0.05))

        n = s.n_comp + max(1, ceil(alpha_n))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        b = s.b * (1.0 + max(0.0, beta_b))
        f = s.fidelity * (1.0 - max(0.0, min(beta_f, 0.95)))
        return PRMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Adaptation(PRMOperator):
    """适应 O_adapt：趋向稳态/目标，提升保真并收敛边界/负荷。

    - b' = b * (1 - eta_b)
    - perim' = perim * (1 - eta_p)
    - fidelity' = min(1, fidelity + delta_f)
    - n_comp' = round(n_comp * (1 - eta_n))
    """

    name = "Adaptation"

    def apply(self, s: PRMState) -> PRMState:
        eta_b = float(self.params.get("eta_b", 0.1))
        eta_p = float(self.params.get("eta_p", 0.05))
        eta_n = float(self.params.get("eta_n", 0.0))
        delta_f = float(self.params.get("delta_f", 0.08))

        b = s.b * (1.0 - max(0.0, min(eta_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(eta_p, 0.99)))
        f = s.fidelity + max(0.0, delta_f)
        n = max(1, round(s.n_comp * (1.0 - max(0.0, min(eta_n, 0.99)))))
        return PRMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Stimulus(PRMOperator):
    """外界刺激 O_stim：增加负荷与边界，可能降低保真。

    - b' = b * (1 + xi_b)
    - perim' = perim * (1 + xi_p)
    - fidelity' = fidelity * (1 - xi_f)
    - n_comp' = n_comp + ceil(dn)
    """

    name = "Stimulus"

    def apply(self, s: PRMState) -> PRMState:
        from math import ceil

        xi_b = float(self.params.get("xi_b", 0.1))
        xi_p = float(self.params.get("xi_p", 0.2))
        xi_f = float(self.params.get("xi_f", 0.05))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 + max(0.0, xi_b))
        p = s.perim * (1.0 + max(0.0, xi_p))
        f = s.fidelity * (1.0 - max(0.0, min(xi_f, 0.95)))
        n = s.n_comp + max(0, ceil(dn))
        return PRMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


@dataclass(frozen=True)
class ComposedOperator(PRMOperator):
    """复合算子：O_k ∘ ... ∘ O_1

    仍保持幺半群的结合律与单位元结构
    """

    ops: Tuple[PRMOperator, ...]
    name: str = "Composed"

    def __init__(self, *ops: PRMOperator):  # type: ignore[override]
        object.__setattr__(self, "ops", tuple(ops))
        object.__setattr__(self, "params", {})

    def apply(self, s: PRMState) -> PRMState:
        out = s
        for op in self.ops:
            out = op(out)
        return out

    def __repr__(self) -> str:
        return "Composed(" + ", ".join(repr(o) for o in self.ops) + ")"


def compose(*ops: PRMOperator) -> PRMOperator:
    """扁平化复合，自动消去单位元"""
    flat: list[PRMOperator] = []
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


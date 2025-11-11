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
from typing import Callable, Dict, Iterable, Tuple

from .state import PEMState


class PEMOperator:
    """PEM 算子基类：O: S -> S。

    - 幺半群结构：存在恒等元 I，定义复合 O2∘O1(s)=O2(O1(s))，满足结合律。
    - 允许参数化（θ），用 `params` 字典持有。
    """

    name: str = "PEMOperator"
    params: Dict[str, float]

    def __init__(self, **params: float) -> None:
        self.params = dict(params)

    def __call__(self, s: PEMState) -> PEMState:
        return self.apply(s).clamp()

    # 子类应覆写该方法
    def apply(self, s: PEMState) -> PEMState:  # pragma: no cover - abstract by convention
        return s

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"


class Identity(PEMOperator):
    name = "Identity"

    def apply(self, s: PEMState) -> PEMState:
        return s


class Metastasis(PEMOperator):
    """转移算子 O_meta：倾向增加组分数、略增边界复杂度，并可能稀释负担密度。

    近似规则：
    - n_comp' = n_comp + ceil(alpha_n)
    - perim' = perim * (1 + alpha_p)
    - b' = b * (1 - beta_b)  （扩散/分裂导致密度稀释）
    - fidelity' = fidelity * (1 - beta_f)
    """

    name = "Metastasis"

    def apply(self, s: PEMState) -> PEMState:
        from math import ceil

        alpha_n = float(self.params.get("alpha_n", 1.0))
        alpha_p = float(self.params.get("alpha_p", 0.1))
        beta_b = float(self.params.get("beta_b", 0.0))
        beta_f = float(self.params.get("beta_f", 0.05))

        n = s.n_comp + max(1, ceil(alpha_n))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        b = s.b * (1.0 - max(0.0, min(beta_b, 0.95)))
        f = s.fidelity * (1.0 - max(0.0, min(beta_f, 0.95)))
        return PEMState(b=b, n_comp=n, perim=p, fidelity=f, meta=s.meta)


class Apoptosis(PEMOperator):
    """凋亡算子 O_apop：降低负担和复杂度，提升组织保真。

    近似规则：
    - b' = b * (1 - gamma_b)
    - n_comp' = max(1, round(n_comp * (1 - gamma_n)))
    - perim' = perim * (1 - gamma_p)
    - fidelity' = min(1, fidelity + delta_f)
    """

    name = "Apoptosis"

    def apply(self, s: PEMState) -> PEMState:
        gamma_b = float(self.params.get("gamma_b", 0.2))
        gamma_n = float(self.params.get("gamma_n", 0.1))
        gamma_p = float(self.params.get("gamma_p", 0.15))
        delta_f = float(self.params.get("delta_f", 0.1))

        b = s.b * (1.0 - max(0.0, min(gamma_b, 0.99)))
        n = max(1, round(s.n_comp * (1.0 - max(0.0, min(gamma_n, 0.99)))))
        p = s.perim * (1.0 - max(0.0, min(gamma_p, 0.99)))
        f = s.fidelity + max(0.0, delta_f)
        return PEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Inflammation(PEMOperator):
    """炎症算子 O_inflam：提升边界活性与负担，可能损伤保真。

    - b' = b * (1 + eta_b)
    - perim' = perim * (1 + eta_p)
    - fidelity' = fidelity * (1 - eta_f)
    - n_comp' 轻微增加
    """

    name = "Inflammation"

    def apply(self, s: PEMState) -> PEMState:
        from math import ceil

        eta_b = float(self.params.get("eta_b", 0.05))
        eta_p = float(self.params.get("eta_p", 0.25))
        eta_f = float(self.params.get("eta_f", 0.05))
        dn = int(self.params.get("dn", 1))

        b = s.b * (1.0 + max(0.0, eta_b))
        p = s.perim * (1.0 + max(0.0, eta_p))
        f = s.fidelity * (1.0 - max(0.0, min(eta_f, 0.95)))
        n = s.n_comp + max(0, ceil(dn))
        return PEMState(b=b, n_comp=n, perim=p, fidelity=f, meta=s.meta)


class Carcinogenesis(PEMOperator):
    """致癌算子 O_carcin：
    - b' = b * (1 + k_b)
    - perim' = perim * (1 + k_p)
    - fidelity' = fidelity * (1 - k_f)
    - n_comp' = n_comp （或缓增）
    """

    name = "Carcinogenesis"

    def apply(self, s: PEMState) -> PEMState:
        k_b = float(self.params.get("k_b", 0.25))
        k_p = float(self.params.get("k_p", 0.15))
        k_f = float(self.params.get("k_f", 0.1))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 + max(0.0, k_b))
        p = s.perim * (1.0 + max(0.0, k_p))
        f = s.fidelity * (1.0 - max(0.0, min(k_f, 0.95)))
        n = s.n_comp + max(0, dn)
        return PEMState(b=b, n_comp=n, perim=p, fidelity=f, meta=s.meta)


@dataclass(frozen=True)
class ComposedOperator(PEMOperator):
    """复合算子（O2∘O1∘...∘O0）。

    结合律天然成立；用于表达幺半群的封闭性与恒等元。
    """

    ops: Tuple[PEMOperator, ...]
    name: str = "Composed"

    def __init__(self, *ops: PEMOperator):  # type: ignore[override]
        object.__setattr__(self, "ops", tuple(ops))
        object.__setattr__(self, "params", {})

    def apply(self, s: PEMState) -> PEMState:
        out = s
        for op in self.ops:
            out = op(out)
        return out

    def __repr__(self) -> str:
        return "Composed(" + ", ".join(repr(o) for o in self.ops) + ")"


def compose(*ops: PEMOperator) -> PEMOperator:
    """复合多个算子；自动折叠恒等元。"""
    flat: list[PEMOperator] = []
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

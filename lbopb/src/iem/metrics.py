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

from math import exp
from typing import Iterable, Sequence

from .observables import Observables
from .operators import IEMOperator
from .state import IEMState


def delta_phi(
        A: IEMOperator,
        B: IEMOperator,
        s0: IEMState,
        phi: Observables | None = None,
) -> float:
    phi = phi or Observables.default()
    s_ab = A(B(s0))
    s_ba = B(A(s0))
    v_ab = phi.eval_all(s_ab)
    v_ba = phi.eval_all(s_ba)
    return sum(abs(v_ab[k] - v_ba[k]) for k in phi.names())


def non_commutativity_index(
        A: IEMOperator,
        B: IEMOperator,
        s0: IEMState,
        phi: Observables | None = None,
) -> float:
    phi = phi or Observables.default()
    denom = 1.0 + sum(phi.eval_all(s0).values())
    if denom <= 0:
        denom = 1.0
    return float(delta_phi(A, B, s0, phi)) / float(denom)


def imm_risk(s: IEMState, alpha1: float = 1.0, alpha2: float = 1.0) -> float:
    """Immune risk = α1·B + α2·(1 - F)"""
    return alpha1 * float(s.b) + alpha2 * float(1.0 - s.fidelity)


def topo_risk(s: IEMState, alpha1: float = 1.0, alpha2: float = 1.0) -> float:
    """兼容名：topo_risk -> imm_risk"""
    return imm_risk(s, alpha1, alpha2)


def action_cost(
        ops: Sequence[IEMOperator],
        s0: IEMState,
        w_b: float = 0.8,
        w_p: float = 0.4,
        w_n: float = 0.2,
        w_f: float = 1.0,
) -> float:
    def penalty(prev: IEMState, cur: IEMState) -> float:
        db = max(0.0, cur.b - prev.b)
        dp = max(0.0, cur.perim - prev.perim)
        dn = max(0.0, float(cur.n_comp - prev.n_comp))
        df = max(0.0, prev.fidelity - cur.fidelity)
        return w_b * db + w_p * dp + w_n * dn + w_f * df

    s = s0
    cost = 0.0
    for op in ops:
        s_next = op(s)
        cost += penalty(s, s_next)
        s = s_next
    return cost


def reach_probability(
        s0: IEMState,
        s_star: IEMState,
        candidate_sequences: Iterable[Sequence[IEMOperator]],
        temperature: float = 1.0,
) -> float:
    best = None
    for seq in candidate_sequences:
        a = action_cost(seq, s0)
        best = a if best is None else min(best, a)
    if best is None:
        return 0.0
    t = max(1e-6, float(temperature))
    return float(exp(-best / t))

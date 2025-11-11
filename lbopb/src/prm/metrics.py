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
from .operators import PRMOperator
from .state import PRMState


def delta_phi(
        A: PRMOperator,
        B: PRMOperator,
        s0: PRMState,
        phi: Observables | None = None,
) -> float:
    """Δ_Φ(A, B; S): 非交换对可观测量的影响强度

    - s_ab = A(B(s0))
    - s_ba = B(A(s0))
    - 聚合 |φ(s_ab) - φ(s_ba)| over φ ∈ Φ
    """

    phi = phi or Observables.default()
    s_ab = A(B(s0))
    s_ba = B(A(s0))
    v_ab = phi.eval_all(s_ab)
    v_ba = phi.eval_all(s_ba)
    return sum(abs(v_ab[k] - v_ba[k]) for k in phi.names())


def non_commutativity_index(
        A: PRMOperator,
        B: PRMOperator,
        s0: PRMState,
        phi: Observables | None = None,
) -> float:
    """NC(A, B; S) = Δ_Φ / (1 + Σφ(S)): 归一化非交换指数"""

    phi = phi or Observables.default()
    denom = 1.0 + sum(phi.eval_all(s0).values())
    if denom <= 0:
        denom = 1.0
    return float(delta_phi(A, B, s0, phi)) / float(denom)


def topo_risk(s: PRMState, alpha1: float = 1.0, alpha2: float = 1.0) -> float:
    """TopoRisk(S) = α1 N_comp + α2 P"""
    return alpha1 * float(s.n_comp) + alpha2 * float(s.perim)


def action_cost(
        ops: Sequence[PRMOperator],
        s0: PRMState,
        w_b: float = 1.0,
        w_p: float = 0.2,
        w_n: float = 0.1,
        w_f: float = 0.5,
) -> float:
    """路径代价：对“负向”变化加权惩罚并累加

    Σ_k [ w_b ΔB_k^+ + w_p ΔP_k^+ + w_n ΔN_k^+ + w_f (-ΔF_k)^+ ]
    """

    def penalty(prev: PRMState, cur: PRMState) -> float:
        db = max(0.0, cur.b - prev.b)
        dp = max(0.0, cur.perim - prev.perim)
        dn = max(0.0, float(cur.n_comp - prev.n_comp))
        df = max(0.0, prev.fidelity - cur.fidelity)  # F 下降为惩罚
        return w_b * db + w_p * dp + w_n * dn + w_f * df

    s = s0
    cost = 0.0
    for op in ops:
        s_next = op(s)
        cost += penalty(s, s_next)
        s = s_next
    return cost


def reach_probability(
        s0: PRMState,
        s_star: PRMState,
        candidate_sequences: Iterable[Sequence[PRMOperator]],
        temperature: float = 1.0,
) -> float:
    """可达性启发：Reach ≈ exp(- min_seq Action(seq; S→S*))

    取若干候选序列的最小代价，softmax 温度默认 1。
    """

    best = None
    for seq in candidate_sequences:
        a = action_cost(seq, s0)
        best = a if best is None else min(best, a)
    if best is None:
        return 0.0
    t = max(1e-6, float(temperature))
    return float(exp(-best / t))

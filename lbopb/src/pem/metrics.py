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

from math import exp
from typing import Iterable, Sequence

from .observables import Observables
from .operators import PEMOperator, compose
from .state import PEMState


def delta_phi(
        A: PEMOperator,
        B: PEMOperator,
        s0: PEMState,
        phi: Observables | None = None,
) -> float:
    """Δ_Φ(O_A, O_B; S) 非对易差异：先后次序对观测的影响强度。

    计算策略：
    - 执行 AB 路径：s_ab = A(B(s0))
    - 执行 BA 路径：s_ba = B(A(s0))
    - 累计 |φ(s_ab) - φ(s_ba)| over φ ∈ Φ
    """

    phi = phi or Observables.default()
    s_ab = A(B(s0))
    s_ba = B(A(s0))

    v_ab = phi.eval_all(s_ab)
    v_ba = phi.eval_all(s_ba)
    return sum(abs(v_ab[k] - v_ba[k]) for k in phi.names())


def non_commutativity_index(
        A: PEMOperator,
        B: PEMOperator,
        s0: PEMState,
        phi: Observables | None = None,
) -> float:
    """NC(O_A, O_B; S) = Δ_Φ / (1 + Σ_φ φ(S)) 的无量纲化指标。"""

    phi = phi or Observables.default()
    denom = 1.0 + sum(phi.eval_all(s0).values())
    if denom <= 0:
        denom = 1.0
    return float(delta_phi(A, B, s0, phi)) / float(denom)


def topo_risk(s: PEMState, alpha1: float = 1.0, alpha2: float = 1.0) -> float:
    """TopoRisk(S) = α1 N_comp + α2 P。"""
    return alpha1 * float(s.n_comp) + alpha2 * float(s.perim)


def action_cost(
        ops: Sequence[PEMOperator],
        s0: PEMState,
        w_b: float = 1.0,
        w_p: float = 0.2,
        w_n: float = 0.1,
        w_f: float = 0.5,
) -> float:
    """给定算子序列的“作用量”成本 𝒜(𝐎; S->S*):

    使用逐步惩罚的简化型：
    Σ_k [ w_b ΔB_k^+ + w_p ΔP_k^+ + w_n ΔN_k^+ + w_f (-ΔF_k)^+ ]
    其中 Δx_k^+ 表示不利方向的增量（例如 B/P/N 上升、F 下降）。
    """

    def penalty(prev: PEMState, cur: PEMState) -> float:
        db = max(0.0, cur.b - prev.b)
        dp = max(0.0, cur.perim - prev.perim)
        dn = max(0.0, float(cur.n_comp - prev.n_comp))
        df = max(0.0, prev.fidelity - cur.fidelity)  # F 下降惩罚
        return w_b * db + w_p * dp + w_n * dn + w_f * df

    s = s0
    cost = 0.0
    for op in ops:
        s_next = op(s)
        cost += penalty(s, s_next)
        s = s_next
    return cost


def reach_probability(
        s0: PEMState,
        s_star: PEMState,
        candidate_sequences: Iterable[Sequence[PEMOperator]],
        temperature: float = 1.0,
) -> float:
    """可达概率近似：Reach = exp(- min_Seq 𝒜(Seq; S->S*))。

    为可计算性，这里对给定候选序列集求最小作用量，再作负指数映射。
    temperature 用于柔化（默认 1）。
    """

    best = None
    for seq in candidate_sequences:
        a = action_cost(seq, s0)
        best = a if best is None else min(best, a)
    if best is None:
        return 0.0
    t = max(1e-6, float(temperature))
    return float(exp(-best / t))

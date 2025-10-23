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
from .operators import TEMOperator
from .state import TEMState


def delta_phi(
        A: TEMOperator,
        B: TEMOperator,
        s0: TEMState,
        phi: Observables | None = None,
) -> float:
    """Δ_Φ(A, B; S): 非交换对可观测量的影响强度"""

    phi = phi or Observables.default()
    s_ab = A(B(s0))
    s_ba = B(A(s0))
    v_ab = phi.eval_all(s_ab)
    v_ba = phi.eval_all(s_ba)
    return sum(abs(v_ab[k] - v_ba[k]) for k in phi.names())


def non_commutativity_index(
        A: TEMOperator,
        B: TEMOperator,
        s0: TEMState,
        phi: Observables | None = None,
) -> float:
    """NC(A, B; S) = Δ_Φ / (1 + Σφ(S))"""

    phi = phi or Observables.default()
    denom = 1.0 + sum(phi.eval_all(s0).values())
    if denom <= 0:
        denom = 1.0
    return float(delta_phi(A, B, s0, phi)) / float(denom)


def tox_risk(s: TEMState, alpha1: float = 1.0, alpha2: float = 1.0) -> float:
    """ToxRisk(S) = α1·B + α2·(1 - F)"""
    return alpha1 * float(s.b) + alpha2 * float(1.0 - s.fidelity)


def topo_risk(s: TEMState, alpha1: float = 1.0, alpha2: float = 1.0) -> float:
    """兼容名：topo_risk -> tox_risk"""
    return tox_risk(s, alpha1, alpha2)


def action_cost(
        ops: Sequence[TEMOperator],
        s0: TEMState,
        w_b: float = 1.0,
        w_p: float = 0.5,
        w_n: float = 0.2,
        w_f: float = 1.0,
) -> float:
    """路径代价：对“毒理不利”变化加权惩罚

    Σ_k [ w_b ΔB_k^+ + w_p ΔP_k^+ + w_n ΔN_k^+ + w_f (Δ(1-F)_k)^+ ]
    """

    def penalty(prev: TEMState, cur: TEMState) -> float:
        db = max(0.0, cur.b - prev.b)
        dp = max(0.0, cur.perim - prev.perim)
        dn = max(0.0, float(cur.n_comp - prev.n_comp))
        d1mF = max(0.0, (1.0 - cur.fidelity) - (1.0 - prev.fidelity))  # = max(0, -ΔF)
        return w_b * db + w_p * dp + w_n * dn + w_f * d1mF

    s = s0
    cost = 0.0
    for op in ops:
        s_next = op(s)
        cost += penalty(s, s_next)
        s = s_next
    return cost


def reach_probability(
        s0: TEMState,
        s_star: TEMState,
        candidate_sequences: Iterable[Sequence[TEMOperator]],
        temperature: float = 1.0,
) -> float:
    """可达性启发：Reach ≈ exp(- min_seq Action(seq; S→S*))"""

    best = None
    for seq in candidate_sequences:
        a = action_cost(seq, s0)
        best = a if best is None else min(best, a)
    if best is None:
        return 0.0
    t = max(1e-6, float(temperature))
    return float(exp(-best / t))

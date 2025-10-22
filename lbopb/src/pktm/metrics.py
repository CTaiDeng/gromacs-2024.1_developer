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

from math import exp
from typing import Iterable, Sequence

from .observables import Observables
from .operators import PKTMOperator
from .state import PKTMState


def delta_phi(
    A: PKTMOperator,
    B: PKTMOperator,
    s0: PKTMState,
    phi: Observables | None = None,
) -> float:
    phi = phi or Observables.default()
    s_ab = A(B(s0))
    s_ba = B(A(s0))
    v_ab = phi.eval_all(s_ab)
    v_ba = phi.eval_all(s_ba)
    return sum(abs(v_ab[k] - v_ba[k]) for k in phi.names())


def non_commutativity_index(
    A: PKTMOperator,
    B: PKTMOperator,
    s0: PKTMState,
    phi: Observables | None = None,
) -> float:
    phi = phi or Observables.default()
    denom = 1.0 + sum(phi.eval_all(s0).values())
    if denom <= 0:
        denom = 1.0
    return float(delta_phi(A, B, s0, phi)) / float(denom)


def topo_risk(s: PKTMState, alpha1: float = 1.0, alpha2: float = 1.0) -> float:
    return alpha1 * float(s.n_comp) + alpha2 * float(s.perim)


def action_cost(
    ops: Sequence[PKTMOperator],
    s0: PKTMState,
    w_b: float = 1.0,
    w_p: float = 0.4,
    w_n: float = 0.2,
    w_f: float = 0.6,
) -> float:
    def penalty(prev: PKTMState, cur: PKTMState) -> float:
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
    s0: PKTMState,
    s_star: PKTMState,
    candidate_sequences: Iterable[Sequence[PKTMOperator]],
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


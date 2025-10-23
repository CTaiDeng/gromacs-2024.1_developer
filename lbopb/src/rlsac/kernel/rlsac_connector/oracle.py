# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lbopb.src.rlsac.kernel.rlsac_pathfinder.oracle import default_init_state, apply_sequence
from lbopb.src.rlsac.kernel.common.llm_oracle import call_llm

MODULES: List[str] = ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]


def _consistency_score(changes: Dict[str, float], eps_change: float = 1e-3) -> float:
    """与 env 中逻辑一致的简化一致性评分。"""
    pairs = [("pdem", "pktm"), ("pgom", "pem"), ("tem", "pktm"), ("prm", "pem"), ("iem", "pem")]
    score = 0.0
    for a, b in pairs:
        ca = changes.get(a, 0.0)
        cb = changes.get(b, 0.0)
        if ca > eps_change and cb > eps_change:
            score += 1.0
        elif (ca > eps_change and cb <= eps_change) or (cb > eps_change and ca <= eps_change):
            score -= 1.0
    cnt = sum(1 for v in changes.values() if v > eps_change)
    if cnt >= 5:
        score += 1.0
    return score


class ConnectorAxiomOracle:
    """联络候选体（七域包）判定器（内置一致性/度量 + 可选 LLM）。"""

    def __init__(self, cost_lambda: float = 0.2, eps_change: float = 1e-3, *, use_llm: bool = False) -> None:
        self.cost_lambda = float(cost_lambda)
        self.eps_change = float(eps_change)
        self.use_llm = bool(use_llm)

    def judge(self, conn: Dict[str, List[str]], init_states: Dict[str, Any] | None = None) -> Tuple[
        int, Dict[str, float]]:
        states = dict(init_states) if init_states is not None else {m: default_init_state(m) for m in MODULES}
        deltas: Dict[str, float] = {}
        costs: Dict[str, float] = {}
        changes: Dict[str, float] = {}
        for m in MODULES:
            s0 = states[m]
            seq = conn.get(m) or []
            s1, dr, c = apply_sequence(m, s0, seq)
            states[m] = s1
            deltas[m] = float(dr)
            costs[m] = float(c)
            # 变化强度：用 |Δrisk|+cost 简化度量
            changes[m] = abs(float(dr)) + float(c)
        base = sum(deltas.values())
        cost = sum(costs.values())
        cons = _consistency_score(changes, eps_change=self.eps_change)
        score = base + cons - self.cost_lambda * cost
        ok = (score > 0.0)
        if self.use_llm:
            try:
                prompt = f"Connection candidate across domains: { {m: conn.get(m, []) for m in MODULES} }. Decide if valid (1/0) under axioms."
                txt = call_llm(prompt)
                if isinstance(txt, str):
                    ok = ok and (("1" in txt and "0" not in txt) or (txt.strip() == "1"))
            except Exception:
                pass
        return (1 if ok else 0), {"delta_risk_sum": float(base), "consistency": float(cons), "cost": float(cost)}

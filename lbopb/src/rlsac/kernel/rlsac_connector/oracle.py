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

from typing import Any, Dict, List, Tuple
import importlib

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
        fatal = False
        warns_present = False
        for m in MODULES:
            s0 = states[m]
            seq = conn.get(m) or []
            # 单域语法检查：若有显著错误，直接判 0；若有警告，记录以便启用 LLM 辅助
            try:
                mod = importlib.import_module(f"lbopb.src.{m}.syntax_checker")
                func = getattr(mod, "check_sequence", None)
                if callable(func):
                    res = func(list(seq), init_state=s0)
                    fatals = res.get("errors", []) or []
                    warns = res.get("warnings", []) or []
                    if fatals:
                        fatal = True
                    if warns:
                        warns_present = True
            except Exception:
                pass
            s1, dr, c = apply_sequence(m, s0, seq)
            states[m] = s1
            deltas[m] = float(dr)
            costs[m] = float(c)
            # 变化强度：用 |Δrisk|+cost 简化度量
            changes[m] = abs(float(dr)) + float(c)
        if fatal:
            return 0, {"delta_risk_sum": sum(deltas.values()), "consistency": 0.0, "cost": sum(costs.values())}
        base = sum(deltas.values())
        cost = sum(costs.values())
        cons = _consistency_score(changes, eps_change=self.eps_change)
        score = base + cons - self.cost_lambda * cost
        ok = (score > 0.0)
        # 仅在存在“警告”时启用 LLM 辅助
        if self.use_llm and warns_present:
            try:
                from lbopb.src.rlsac.kernel.common.llm_oracle import build_connector_prompt
                txt = call_llm(build_connector_prompt(conn))
                if isinstance(txt, str):
                    ok = ok and (("1" in txt and "0" not in txt) or (txt.strip() == "1"))
            except Exception:
                pass
        return (1 if ok else 0), {"delta_risk_sum": float(base), "consistency": float(cons), "cost": float(cost)}

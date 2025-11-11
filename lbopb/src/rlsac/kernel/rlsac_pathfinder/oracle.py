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

from typing import Any, Dict, List, Sequence, Tuple

from lbopb.src.powerset import instantiate_ops

from lbopb.src.pem import topo_risk as pem_risk, action_cost as pem_cost
from lbopb.src.pdem import eff_risk as pdem_risk, action_cost as pdem_cost
from lbopb.src.pktm import topo_risk as pktm_risk, action_cost as pktm_cost
from lbopb.src.pgom import topo_risk as pgom_risk, action_cost as pgom_cost
from lbopb.src.tem import tox_risk as tem_risk, action_cost as tem_cost
from lbopb.src.prm import topo_risk as prm_risk, action_cost as prm_cost
from lbopb.src.iem import imm_risk as iem_risk, action_cost as iem_cost

import importlib
from lbopb.src.pem import PEMState
from lbopb.src.pdem import PDEMState
from lbopb.src.pktm import PKTMState
from lbopb.src.pgom import PGOMState
from lbopb.src.tem import TEMState
from lbopb.src.prm import PRMState
from lbopb.src.iem import IEMState


def default_init_state(domain: str) -> Any:
    d = domain.lower()
    if d == "pem":
        return PEMState(b=8.0, n_comp=3, perim=2.0, fidelity=0.6)
    if d == "pdem":
        return PDEMState(b=1.5, n_comp=1, perim=0.8, fidelity=0.6)
    if d == "pktm":
        return PKTMState(b=0.5, n_comp=1, perim=0.5, fidelity=0.95)
    if d == "pgom":
        return PGOMState(b=3.0, n_comp=2, perim=1.5, fidelity=0.8)
    if d == "tem":
        return TEMState(b=5.0, n_comp=1, perim=2.0, fidelity=0.9)
    if d == "prm":
        return PRMState(b=10.0, n_comp=1, perim=5.0, fidelity=0.8)
    if d == "iem":
        return IEMState(b=2.0, n_comp=2, perim=1.0, fidelity=0.7)
    return PEMState(b=8.0, n_comp=3, perim=2.0, fidelity=0.6)


def _risk(domain: str, st: Any) -> float:
    d = domain.lower()
    if d == "pem":
        return float(pem_risk(st))
    if d == "pdem":
        return float(pdem_risk(st))
    if d == "pktm":
        return float(pktm_risk(st))
    if d == "pgom":
        return float(pgom_risk(st))
    if d == "tem":
        return float(tem_risk(st))
    if d == "prm":
        return float(prm_risk(st))
    if d == "iem":
        return float(iem_risk(st))
    return 0.0


def _cost(domain: str, ops: Sequence[Any], s0: Any) -> float:
    d = domain.lower()
    try:
        if d == "pem":
            return float(pem_cost(list(ops), s0))
        if d == "pdem":
            return float(pdem_cost(list(ops), s0))
        if d == "pktm":
            return float(pktm_cost(list(ops), s0))
        if d == "pgom":
            return float(pgom_cost(list(ops), s0))
        if d == "tem":
            return float(tem_cost(list(ops), s0))
        if d == "prm":
            return float(prm_cost(list(ops), s0))
        if d == "iem":
            return float(iem_cost(list(ops), s0))
    except Exception:
        return 0.0
    return 0.0


def apply_sequence(domain: str, s0: Any, op_names: Sequence[str]) -> Tuple[Any, float, float]:
    """应用序列并返回 (next_state, delta_risk, cost)。"""
    ops = instantiate_ops(domain, list(op_names)) if op_names else []
    s = s0
    for op in ops:
        try:
            s = op(s)
        except Exception:
            pass
    dr = _risk(domain, s0) - _risk(domain, s)
    c = _cost(domain, ops, s0) if ops else 0.0
    return s, float(dr), float(c)


class AxiomOracle:
    """公理系统判定器（双重判定：syntax_checker + 启发式；可选 LLM）。"""

    def __init__(self, cost_lambda: float = 0.2, min_improve: float = 1e-6, *, use_llm: bool = False) -> None:
        self.cost_lambda = float(cost_lambda)
        self.min_improve = float(min_improve)
        self.use_llm = bool(use_llm)

    def judge(self, domain: str, op_names: Sequence[str], s0: Any | None = None) -> int:
        s0 = s0 if s0 is not None else default_init_state(domain)
        # 1) syntax_checker 校验
        ok_syntax = True
        try:
            mod = importlib.import_module(f"lbopb.src.{domain}.syntax_checker")
            func = getattr(mod, "check_sequence", None)
            if callable(func):
                res = func(list(op_names), init_state=s0)
                fatals = res.get("errors", []) or []
                warns = res.get("warnings", []) or []
                if fatals:
                    return 1 if False else 0  # 存在显著错误，直接判定为 0
                ok_syntax = True  # 无显著错误→语法层面通过
        except Exception:
            ok_syntax = True
        # 2) 启发式度量
        _, dr, c = apply_sequence(domain, s0, op_names)
        score = dr - self.cost_lambda * c
        ok_heur = bool(dr > self.min_improve and score > 0.0)
        # 3) 可选 LLM 判定（占位，不阻断）
        ok_llm = True
        # 仅当存在“警告”时启用 LLM 辅助判定
        if self.use_llm:
            try:
                from lbopb.src.rlsac.kernel.common.llm_oracle import call_llm, build_pathfinder_prompt
                prompt = build_pathfinder_prompt(domain, list(op_names))
                txt = call_llm(prompt)
                if isinstance(txt, str):
                    ok_llm = ("1" in txt and "0" not in txt) or (txt.strip() == "1")
            except Exception:
                ok_llm = True
        # 如果存在警告，并启用了 LLM，则需要 ok_llm 与 ok_heur 同时成立；否则仅依据启发式
        try:
            warns_present = bool(res.get("warnings", []))  # type: ignore[name-defined]
        except Exception:
            warns_present = False
        if warns_present and self.use_llm:
            return 1 if (ok_syntax and ok_heur and ok_llm) else 0
        return 1 if (ok_syntax and ok_heur) else 0

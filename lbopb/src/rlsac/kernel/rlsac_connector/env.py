# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import json
import math
import torch

from lbopb.src.rlsac.application.rlsac_nsclc.space import SimpleBoxFloat32, SimpleBoxInt32
from lbopb.src.rlsac.kernel.rlsac_pathfinder.domain import get_domain_spec
from lbopb.src.powerset import instantiate_ops

# 风险指标（各域）
from lbopb.src.pem import topo_risk as pem_risk
from lbopb.src.pdem import eff_risk as pdem_risk
from lbopb.src.pktm import topo_risk as pktm_risk
from lbopb.src.pgom import topo_risk as pgom_risk
from lbopb.src.tem import tox_risk as tem_risk
from lbopb.src.prm import topo_risk as prm_risk
from lbopb.src.iem import imm_risk as iem_risk

from lbopb.src.pem import action_cost as pem_cost
from lbopb.src.pdem import action_cost as pdem_cost
from lbopb.src.pktm import action_cost as pktm_cost
from lbopb.src.pgom import action_cost as pgom_cost
from lbopb.src.tem import action_cost as tem_cost
from lbopb.src.prm import action_cost as prm_cost
from lbopb.src.iem import action_cost as iem_cost


MODULES: List[str] = ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]


def _risk_of(mod: str, st: Any) -> float:
    if mod == "pem":
        return float(pem_risk(st))
    if mod == "pdem":
        return float(pdem_risk(st))
    if mod == "pktm":
        return float(pktm_risk(st))
    if mod == "pgom":
        return float(pgom_risk(st))
    if mod == "tem":
        return float(tem_risk(st))
    if mod == "prm":
        return float(prm_risk(st))
    if mod == "iem":
        return float(iem_risk(st))
    return 0.0


def _cost_of_sequence(mod: str, ops: Sequence[Any], s0: Any) -> float:
    try:
        if mod == "pem":
            return float(pem_cost(list(ops), s0))
        if mod == "pdem":
            return float(pdem_cost(list(ops), s0))
        if mod == "pktm":
            return float(pktm_cost(list(ops), s0))
        if mod == "pgom":
            return float(pgom_cost(list(ops), s0))
        if mod == "tem":
            return float(tem_cost(list(ops), s0))
        if mod == "prm":
            return float(prm_cost(list(ops), s0))
        if mod == "iem":
            return float(iem_cost(list(ops), s0))
    except Exception:
        return 0.0
    return 0.0


def _features_of_state(st: Any) -> List[float]:
    try:
        b = float(getattr(st, "b"))
        p = float(getattr(st, "perim"))
        f = float(getattr(st, "fidelity"))
        n = float(getattr(st, "n_comp"))
    except Exception:
        b = p = f = n = 0.0
    return [b, p, f, n]


def _default_init_states() -> Dict[str, Any]:
    # 与 rlsac_hiv.SequenceEnv 中初始化取值风格一致
    from lbopb.src.pem import PEMState
    from lbopb.src.pdem import PDEMState
    from lbopb.src.pktm import PKTMState
    from lbopb.src.pgom import PGOMState
    from lbopb.src.tem import TEMState
    from lbopb.src.prm import PRMState
    from lbopb.src.iem import IEMState

    return {
        "pem": PEMState(b=8.0, n_comp=3, perim=2.0, fidelity=0.6),
        "pdem": PDEMState(b=1.5, n_comp=1, perim=0.8, fidelity=0.6),
        "pktm": PKTMState(b=0.5, n_comp=1, perim=0.5, fidelity=0.95),
        "pgom": PGOMState(b=3.0, n_comp=2, perim=1.5, fidelity=0.8),
        "tem": TEMState(b=5.0, n_comp=1, perim=2.0, fidelity=0.9),
        "prm": PRMState(b=10.0, n_comp=1, perim=5.0, fidelity=0.8),
        "iem": IEMState(b=2.0, n_comp=2, perim=1.0, fidelity=0.7),
    }


class LBOPBConnectorEnv:
    """LBOPB 联络候选体评估环境。

    - Observation: concat_{mod in 7} [B, P, F, N, risk] → 35 维
    - Action: 单一离散 id，对应 7 域各选一个“算子包”的七元组（混合基数展开）
    - Step: 一步评估（应用七域包→产生下一状态→评分）并 done=True
    - Reward: Σ域(Δrisk) + consistency_bonus − λ·Σ域(cost)
    """

    def __init__(
        self,
        *,
        packages_dir: str | Path | None = None,
        cost_lambda: float = 0.2,
        consistency_bonus: float = 1.0,
        inconsistency_penalty: float = 1.0,
        eps_change: float = 1e-3,
        init_states: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.packages_dir = Path(packages_dir) if packages_dir else (Path(__file__).resolve().parents[1] / "rlsac_pathfinder")
        self.cost_lambda = float(cost_lambda)
        self.consistency_bonus = float(consistency_bonus)
        self.inconsistency_penalty = float(inconsistency_penalty)
        self.eps_change = float(eps_change)

        # 加载每个域的“算子包辞海”（数组，元素含 id/sequence）
        self.domain_packages: Dict[str, List[Dict[str, Any]]] = {}
        self.domain_pkg_ids: Dict[str, List[str]] = {}
        for m in MODULES:
            path = self.packages_dir / f"{m}_operator_packages.json"
            arr: List[Dict[str, Any]] = []
            if path.exists():
                try:
                    arr = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    arr = []
            # 若缺失，提供空集合（动作空间将退化）
            self.domain_packages[m] = list(arr)
            self.domain_pkg_ids[m] = [str(x.get("id", f"{m}_pkg_{i}")) for i, x in enumerate(arr)]

        # 计算混合基数（各域包个数），并建立 idx<->七元组 的映射
        self.radix: List[int] = [max(1, len(self.domain_pkg_ids[m])) for m in MODULES]
        self.tot_actions: int = 1
        for r in self.radix:
            self.tot_actions *= r

        self.states: Dict[str, Any] = dict(init_states) if init_states else _default_init_states()

        # 空间定义
        self.observation_space = SimpleBoxFloat32(0.0, 1e6, (35,))
        self.action_space = SimpleBoxInt32(0, self.tot_actions, (1,))

    # 混合基数映射
    def _index_to_tuple(self, idx: int) -> Tuple[int, int, int, int, int, int, int]:
        idx = max(0, min(int(idx), self.tot_actions - 1))
        digits: List[int] = []
        x = idx
        for r in self.radix:
            digits.append(x % r)
            x //= r
        return tuple(digits)[:7]  # type: ignore[return-value]

    def _tuple_to_index(self, tup: Sequence[int]) -> int:
        x = 0
        mult = 1
        for i, r in enumerate(self.radix):
            d = int(tup[i]) if i < len(tup) else 0
            x += d * mult
            mult *= r
        return x

    def _vectorize(self) -> torch.Tensor:
        feats: List[float] = []
        for m in MODULES:
            st = self.states[m]
            feats += _features_of_state(st)
            feats.append(_risk_of(m, st))
        return torch.tensor(feats, dtype=torch.float32)

    def reset(self) -> torch.Tensor:
        self.states = _default_init_states()
        return self._vectorize()

    def _apply_package(self, mod: str, st: Any, pkg_seq: Sequence[str]) -> Tuple[Any, float, float, float]:
        """返回 (next_state, delta_risk, cost, change_norm)。"""
        r0 = _risk_of(mod, st)
        ops = instantiate_ops(mod, list(pkg_seq)) if pkg_seq else []
        s = st
        for op in ops:
            try:
                s = op(s)
            except Exception:
                pass
        r1 = _risk_of(mod, s)
        cost = _cost_of_sequence(mod, ops, st) if ops else 0.0
        # 变化范数（简单 L1 of features）
        f0 = _features_of_state(st)
        f1 = _features_of_state(s)
        change = sum(abs(f1[i] - f0[i]) for i in range(len(f0)))
        return s, (r0 - r1), cost, change

    def _consistency_score(self, changes: Dict[str, float]) -> float:
        # 若有域发生显著变化，而对偶域“近乎不变”，记为不一致
        eps = self.eps_change
        pairs = [("pdem", "pktm"), ("pgom", "pem"), ("tem", "pktm"), ("prm", "pem"), ("iem", "pem")]
        score = 0.0
        # 奖励协同变动（两者都发生变动）
        for a, b in pairs:
            ca = changes.get(a, 0.0)
            cb = changes.get(b, 0.0)
            if ca > eps and cb > eps:
                score += 1.0
            elif (ca > eps and cb <= eps) or (cb > eps and ca <= eps):
                score -= 1.0
        # 若多数域都发生非零变化，额外奖励
        cnt = sum(1 for v in changes.values() if v > eps)
        if cnt >= 5:
            score += 1.0
        return score

    def step(self, action: torch.Tensor | int):
        a = int(action if isinstance(action, int) else int(action.view(-1)[0].item()))
        digits = self._index_to_tuple(a)

        # 将 action digits 映射为 per-domain package id，并取其算子序列
        chosen_ids: Dict[str, str] = {}
        seqs: Dict[str, List[str]] = {}
        for i, m in enumerate(MODULES):
            arr = self.domain_packages[m]
            if not arr:
                chosen_ids[m] = ""
                seqs[m] = []
                continue
            j = max(0, min(digits[i], len(arr) - 1))
            chosen_ids[m] = str(arr[j].get("id", f"{m}_pkg_{j}"))
            seqs[m] = list(arr[j].get("sequence", []) or [])

        # 应用七域包并累积分数
        next_states: Dict[str, Any] = {}
        deltas: Dict[str, float] = {}
        costs: Dict[str, float] = {}
        changes: Dict[str, float] = {}
        for m in MODULES:
            s0 = self.states[m]
            s1, dr, c, ch = self._apply_package(m, s0, seqs[m])
            next_states[m] = s1
            deltas[m] = float(dr)
            costs[m] = float(c)
            changes[m] = float(ch)

        base = sum(deltas.values())
        cost = sum(costs.values())
        cons = self._consistency_score(changes)
        reward = base + self.consistency_bonus * max(0.0, cons) - self.inconsistency_penalty * max(0.0, -cons) - self.cost_lambda * cost

        self.states = next_states
        obs = self._vectorize()
        done = True  # 单步评估
        info = {
            "chosen": chosen_ids,
            "delta_risk_sum": float(base),
            "consistency": float(cons),
            "cost": float(cost),
        }
        return obs, float(reward), bool(done), info






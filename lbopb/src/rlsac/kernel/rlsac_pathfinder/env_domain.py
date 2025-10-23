# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from .domain import DomainSpec
from lbopb.src.rlsac.application.rlsac_nsclc.space import SimpleBoxFloat32, SimpleBoxInt32


@dataclass
class Goal:
    target: Any  # 目标状态（域状态类型）
    tol_b: float = 1e-3
    tol_n: float = 0.5
    tol_p: float = 1e-3
    tol_f: float = 1e-3


class DomainPathfinderEnv:
    """统一的单域路径探索环境（支持 pem/pdem/pktm/pgom/tem/prm/iem）。

    - Observation: [b, n_comp, perim, fidelity]（float32, shape=(4,)）
    - Action: 离散基本算子集的索引（int32, range [0, N)）
    - Done: 达到目标容差内，或步数上限
    - Reward: 距离改进 + 目标奖励 - 步长惩罚
    """

    def __init__(
        self,
        spec: DomainSpec,
        *,
        init_state: Any,
        goal: Goal,
        max_steps: int = 64,
        improve_weight: float = 1.0,
        step_penalty: float = 0.01,
        include_identity: bool = False,
    ) -> None:
        self._spec = spec
        self._init_state_cfg = init_state
        self._goal = goal
        self._max_steps = int(max_steps)
        self._improve_w = float(improve_weight)
        self._step_penalty = float(step_penalty)
        self._steps = 0
        self._state = init_state

        self._ops: List[Any] = []
        if include_identity:
            self._ops.append(spec.identity_cls())
        for cls in spec.op_classes:
            self._ops.append(cls())
        self._op2idx: Dict[str, int] = { self._op_name(o): i for i, o in enumerate(self._ops) }

        self.observation_space = SimpleBoxFloat32(low=0.0, high=10.0, shape=(4,))
        self.action_space = SimpleBoxInt32(low=0, high=len(self._ops), shape=(1,))

    @staticmethod
    def _op_name(o: Any) -> str:
        return getattr(o, "name", o.__class__.__name__)

    @property
    def op2idx(self) -> Dict[str, int]:
        return dict(self._op2idx)

    def reset(self) -> torch.Tensor:
        self._state = self._init_state_cfg.clamp()
        self._steps = 0
        return self._to_obs(self._state)

    def _to_obs(self, s: Any) -> torch.Tensor:
        vec = torch.tensor([float(s.b), float(s.n_comp), float(s.perim), float(s.fidelity)], dtype=torch.float32)
        return vec

    def _distance(self, s: Any) -> float:
        t = self._goal.target
        db = abs(float(s.b) - float(t.b))
        dn = abs(float(s.n_comp) - float(t.n_comp))
        dp = abs(float(s.perim) - float(t.perim))
        df = abs(float(s.fidelity) - float(t.fidelity))
        # 简单加权 L1 距离；可按域自定义
        return (1.0 * db) + (0.3 * dn) + (0.5 * dp) + (1.0 * df)

    def _is_goal(self, s: Any) -> bool:
        t = self._goal.target
        return (
            abs(float(s.b) - float(t.b)) <= self._goal.tol_b and
            abs(float(s.n_comp) - float(t.n_comp)) <= self._goal.tol_n and
            abs(float(s.perim) - float(t.perim)) <= self._goal.tol_p and
            abs(float(s.fidelity) - float(t.fidelity)) <= self._goal.tol_f
        )

    def step(self, action: torch.Tensor | int) -> Tuple[torch.Tensor, float, bool, Dict[str, float]]:
        a = int(action if isinstance(action, int) else int(action.item()))
        a = max(0, min(a, len(self._ops) - 1))
        prev = self._state
        prev_d = self._distance(prev)
        self._steps += 1

        op = self._ops[a]
        cur = op(prev).clamp()
        self._state = cur
        cur_d = self._distance(cur)

        improve = (prev_d - cur_d)
        done = self._is_goal(cur) or (self._steps >= self._max_steps)
        reward = self._improve_w * improve - self._step_penalty
        if self._is_goal(cur):
            reward += 5.0

        return self._to_obs(cur), float(reward), bool(done), {"improve": float(improve), "dist": float(cur_d)}






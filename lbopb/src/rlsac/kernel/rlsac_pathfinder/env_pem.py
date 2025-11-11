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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from lbopb.src.pem.state import PEMState
from lbopb.src.pem.operators import (
    PEMOperator,
    Identity,
    Apoptosis,
    Metastasis,
    Inflammation,
    Carcinogenesis,
)
from lbopb.src.rlsac.application.rlsac_nsclc.space import SimpleBoxFloat32, SimpleBoxInt32


@dataclass
class PEMGoal:
    target: PEMState
    tol_b: float = 1e-3
    tol_n: float = 0.5
    tol_p: float = 1e-3
    tol_f: float = 1e-3


class PEMPathfinderEnv:
    """PEM 单域路径探索环境。

    - Observation: [b, n_comp, perim, fidelity]（float32, shape=(4,)）
    - Action: 离散基本算子集的索引（int32, range [0, N)）
    - Done: 达到目标容差内，或步数上限
    - Reward: 距离改进 + 目标奖励 - 步长惩罚
    """

    def __init__(
            self,
            *,
            init_state: PEMState,
            goal: PEMGoal,
            max_steps: int = 64,
            improve_weight: float = 1.0,
            step_penalty: float = 0.01,
            include_identity: bool = False,
    ) -> None:
        self._init_state_cfg = init_state
        self._goal = goal
        self._max_steps = int(max_steps)
        self._improve_w = float(improve_weight)
        self._step_penalty = float(step_penalty)
        self._steps = 0
        self._state = init_state
        self._ops: List[PEMOperator] = self._build_opset(include_identity=include_identity)
        self._op2idx: Dict[str, int] = {self._op_name(o): i for i, o in enumerate(self._ops)}
        self.observation_space = SimpleBoxFloat32(low=0.0, high=10.0, shape=(4,))
        self.action_space = SimpleBoxInt32(low=0, high=len(self._ops), shape=(1,))

    @staticmethod
    def _op_name(o: PEMOperator) -> str:
        return getattr(o, "name", o.__class__.__name__)

    def _build_opset(self, *, include_identity: bool) -> List[PEMOperator]:
        ops: List[PEMOperator] = []
        if include_identity:
            ops.append(Identity())
        # 基本算子集（默认参数）
        ops.append(Apoptosis())
        ops.append(Metastasis())
        ops.append(Inflammation())
        ops.append(Carcinogenesis())
        return ops

    @property
    def op2idx(self) -> Dict[str, int]:
        return dict(self._op2idx)

    def reset(self) -> torch.Tensor:
        self._state = self._init_state_cfg.clamp()
        self._steps = 0
        return self._to_obs(self._state)

    def _to_obs(self, s: PEMState) -> torch.Tensor:
        vec = torch.tensor([float(s.b), float(s.n_comp), float(s.perim), float(s.fidelity)], dtype=torch.float32)
        return vec

    def _distance(self, s: PEMState) -> float:
        t = self._goal.target
        db = abs(float(s.b) - float(t.b))
        dn = abs(float(s.n_comp) - float(t.n_comp))
        dp = abs(float(s.perim) - float(t.perim))
        df = abs(float(s.fidelity) - float(t.fidelity))
        # 简单权重，可后续外置到配置
        return (1.0 * db) + (0.3 * dn) + (0.5 * dp) + (1.0 * df)

    def _is_goal(self, s: PEMState) -> bool:
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
            reward += 5.0  # 目标奖励（可调）

        return self._to_obs(cur), float(reward), bool(done), {"improve": float(improve), "dist": float(cur_d)}

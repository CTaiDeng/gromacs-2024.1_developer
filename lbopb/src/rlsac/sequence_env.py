# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

"""
基于 operator_crosswalk_train.json 的“序列跟随”环境（离散动作）。

用途：将案例包中的“模块→算子序列”转换为可用于离散 SAC 的训练样本。

设计（最小实现，便于与现有训练脚本直连）：
- observation：拼接向量 [module_one_hot(7) | next_op_one_hot(M) | pos_norm(1)]
  - 7 个模块：pem/pdem/pktm/pgom/tem/prm/iem（与案例包固定顺序一致）
  - M 为本案例包出现的“基本算子”去重后的总数（如 Inflammation/Bind/...）
  - pos_norm = 当前序列指针 / (len(seq) or 1)
- action：全局算子字典中的离散索引（0..M-1）
- 转移与奖励：
  - 正确选择 next_op 奖励 +1，并将指针推进；
  - 错误选择奖励 0（可按需设负奖励）；
  - 一个序列完成后 done=True 并重置到下一个模块（或 reset() 由上层调用）。

说明：该环境将案例 JSON 作为“模仿/预训练”式样本使用，
输出并非重建输入，而是对“给定状态应选择的下一个算子”的概率/价值预测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, List
import json
from pathlib import Path

import torch

from .env import SimpleBoxInt32, SimpleBoxFloat32  # 复用简易空间定义


MODULES: List[str] = ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]


class LBOPBSequenceEnv:
    def __init__(self, case_json: str | Path) -> None:
        p = Path(case_json)
        data = json.loads(p.read_text(encoding="utf-8"))
        cp = data.get("case_packages", {})
        # 取第一个案例包（如 HIV_Therapy_Path）
        self.case_name = next(iter(cp.keys())) if cp else ""
        case = cp.get(self.case_name, {}) if self.case_name else {}
        seqs: Dict[str, List[str]] = case.get("sequences", {})
        # 收集全局算子集合（去重）
        ops: List[str] = []
        for m in MODULES:
            for op in seqs.get(m, []) or []:
                if op not in ops:
                    ops.append(op)
        self.ops = ops
        self.op2idx = {op: i for i, op in enumerate(self.ops)}
        self.seqs = {m: list(seqs.get(m, [])) for m in MODULES}

        self.mod_idx = 0
        self.ptr = 0

        # 观测维度：7 + M + 1
        self.obs_dim = 7 + len(self.ops) + 1
        self.observation_space = SimpleBoxFloat32(0.0, 1.0, (self.obs_dim,))
        self.action_space = SimpleBoxInt32(0, len(self.ops), (1,))

    def _vectorize(self) -> torch.Tensor:
        m_onehot = [0.0] * 7
        m_onehot[self.mod_idx] = 1.0
        next_hot = [0.0] * len(self.ops)
        cur_mod = MODULES[self.mod_idx]
        seq = self.seqs.get(cur_mod, []) or []
        if self.ptr < len(seq):
            op = seq[self.ptr]
            j = self.op2idx.get(op, -1)
            if j >= 0:
                next_hot[j] = 1.0
            pos = self.ptr / float(max(1, len(seq)))
        else:
            pos = 1.0
        return torch.tensor(m_onehot + next_hot + [pos], dtype=torch.float32)

    def reset(self) -> torch.Tensor:
        self.mod_idx = 0
        self.ptr = 0
        # 若第一个模块为空序列，则前进到下一个非空模块
        while self.mod_idx < len(MODULES) and len(self.seqs.get(MODULES[self.mod_idx], []) or []) == 0:
            self.mod_idx += 1
            self.ptr = 0
        return self._vectorize()

    def step(self, action: torch.Tensor):
        a = int(action.view(-1)[0].item())
        cur_mod = MODULES[self.mod_idx]
        seq = self.seqs.get(cur_mod, []) or []
        reward = 0.0
        done = False
        # 正确选择当前需要的算子则前进，并给奖励
        if self.ptr < len(seq):
            need = self.op2idx.get(seq[self.ptr], -1)
            if a == need:
                reward = 1.0
                self.ptr += 1
        # 若当前模块已完成，切换到下一个非空模块，并标记本 episode 结束
        if self.ptr >= len(seq):
            # 切到下一个模块以便下次 reset 后继续（或由上层再 reset）
            self.mod_idx += 1
            self.ptr = 0
            while self.mod_idx < len(MODULES) and len(self.seqs.get(MODULES[self.mod_idx], []) or []) == 0:
                self.mod_idx += 1
            done = True
        return self._vectorize(), reward, done, {"module": cur_mod}


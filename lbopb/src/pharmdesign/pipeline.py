# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

"""药效基底路径积分与纤维丛联络映射工具。

包含：
- pdem_path_integral：对 PDEM 算子包进行离散“路径积分”（Lagrangian 和）
- map_pdem_sequence_to_fibers：借助 crosswalk 将 PDEM 序列映射至六切面对齐算子包
"""

from typing import Dict, Iterable, List, Tuple

from ..pdem import PDEMState, Observables as PObservables
from ..pdem import action_cost as pdem_action_cost
from ..op_crosswalk import load_crosswalk, basic_ops, crosswalk_for_tag


def pdem_path_integral(seq_ops: Iterable, s0: PDEMState, *, alpha: float = 1.0, beta: float = 1.0) -> Tuple[float, List[PDEMState]]:
    """离散路径积分（示意）：

    L_t = alpha * B(S_t) + beta * (1 - F(S_t))
    Integral ≈ sum_t L_t
    返回 (积分值, 轨迹)
    """

    phi = PObservables.default()
    s = s0
    traj = [s]
    integral = 0.0
    for op in seq_ops:
        v = phi.eval_all(s)
        L = alpha * v["B"] + beta * (1.0 - v["F"])
        integral += float(L)
        s = op(s)
        traj.append(s)
    # 终点项
    v = phi.eval_all(s)
    L = alpha * v["B"] + beta * (1.0 - v["F"])
    integral += float(L)
    return float(integral), traj


def map_pdem_sequence_to_fibers(pdem_seq: List[str]) -> Dict[str, List[str]]:
    """将 PDEM 基本算子序列映射至其他纤维丛（按语义标签的首选对齐）。

    策略：
    - 读取 JSON basic_ops('pdem') 获取 op→tags
    - 对每个 tag，用 crosswalk_by_tag 找各模块的候选基本算子，取候选列表的第一个作为默认映射
    - 对同一模块合并得到序列
    """

    cw = load_crosswalk()
    pdem_tags_map = basic_ops(cw, "pdem")  # op->tags
    # 逆向：op -> tag 列表
    out: Dict[str, List[str]] = {m: [] for m in cw.get("modules", []) if m != "pdem"}
    for op in pdem_seq:
        tags = pdem_tags_map.get(op, [])
        for tag in tags:
            xw = crosswalk_for_tag(cw, tag)
            for mod, cand in xw.items():
                if mod == "pdem" or not cand:
                    continue
                # 取首个候选作为默认映射
                if cand[0] not in out[mod]:
                    out[mod].append(cand[0])
    return out


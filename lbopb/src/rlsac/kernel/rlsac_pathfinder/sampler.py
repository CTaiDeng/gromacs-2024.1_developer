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

import random
from typing import List, Sequence

from .domain import DomainSpec


def op_name_list(spec: DomainSpec) -> List[str]:
    names: List[str] = []
    for cls in spec.op_classes:
        try:
            inst = cls()
            nm = getattr(inst, "name", inst.__class__.__name__)
        except Exception:
            nm = getattr(cls, "__name__", "UnknownOp")
        names.append(str(nm))
    return names


def sample_random_package(
        spec: DomainSpec,
        *,
        min_len: int = 1,
        max_len: int = 4,
        no_consecutive_duplicate: bool = True,
        rng: random.Random | None = None,
) -> List[str]:
    r = rng or random
    ops = op_name_list(spec)
    L = r.randint(max(1, min_len), max(1, max_len))
    seq: List[str] = []
    prev = None
    for _ in range(L):
        cand = r.choice(ops) if ops else ""
        if no_consecutive_duplicate and prev is not None and cand == prev and len(ops) > 1:
            # 重新抽一个不同的
            alt = [x for x in ops if x != prev]
            cand = r.choice(alt)
        seq.append(cand)
        prev = cand
    return seq

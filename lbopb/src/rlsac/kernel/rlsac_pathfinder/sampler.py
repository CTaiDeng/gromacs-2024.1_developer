# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

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

# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""rlsac_pathfinder：单域算子包路径探索器（支持 pem/pdem/pktm/pgom/tem/prm/iem）。

基于《O3理论的自举之路》第一阶段：
在单域上使用离散 SAC 探索从初始状态到目标状态的有效算子序列（算子包），并记录到辞海。
"""

from .env_domain import DomainPathfinderEnv, Goal
from .train import train, extract_operator_package
from .domain import get_domain_spec

__all__ = [
    "DomainPathfinderEnv",
    "Goal",
    "train",
    "extract_operator_package",
    "get_domain_spec",
]



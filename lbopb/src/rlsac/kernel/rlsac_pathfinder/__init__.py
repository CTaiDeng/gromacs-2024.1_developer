# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""rlsac_pathfinder：单域算子包路径探索器（支持 pem/pdem/pktm/pgom/tem/prm/iem）。

基于《O3理论的自举之路》第一阶段：
在单域上使用离散 SAC 探索从初始状态到目标状态的有效算子序列（算子包），并记录到辞海。
"""

from .env_domain import DomainPathfinderEnv, Goal
from .domain import get_domain_spec
from .package_store import ensure_store_dir, ingest_from_debug_dataset


def train(*args, **kwargs):
    from . import train as _train_mod
    return _train_mod.train(*args, **kwargs)


def train_all(*args, **kwargs):
    from . import train as _train_mod
    return _train_mod.train_all(*args, **kwargs)


def extract_operator_package(*args, **kwargs):
    from . import train as _train_mod
    return _train_mod.extract_operator_package(*args, **kwargs)


__all__ = [
    "DomainPathfinderEnv",
    "Goal",
    "train",
    "train_all",
    "extract_operator_package",
    "get_domain_spec",
    "ensure_store_dir",
    "ingest_from_debug_dataset",
]

# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""rlsac_connector：跨领域“法则联络”映射发现（SAC 版）。

依据《O3理论的自举之路》第二阶段：
从七本“领域辞海”中为各域各选一个“算子包”，构成联络候选七元组，
在统一的 LBOPB 全息状态上同时应用并评分其全局自洽性。
"""

from .env import LBOPBConnectorEnv

def train(*args, **kwargs):
    from . import train as _train_mod
    return _train_mod.train(*args, **kwargs)

def extract_connection(*args, **kwargs):
    from . import train as _train_mod
    return _train_mod.extract_connection(*args, **kwargs)

__all__ = [
    "LBOPBConnectorEnv",
    "train",
    "extract_connection",
]

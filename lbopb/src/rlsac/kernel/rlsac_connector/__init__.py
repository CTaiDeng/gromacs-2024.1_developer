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

"""rlsac_connector：跨领域“法则联络”映射发现（SAC 版）。

依据《O3理论的自举之路》第二阶段：
从七本“领域辞海”中为各域各选一个“算子包”，构成联络候选七元组，
在统一的 LBOPB 全息状态上同时应用并评分其全局自洽性。
"""

# 延迟导入，避免在仅运行 train/dataset 时强制依赖 torch
try:
    from .env import LBOPBConnectorEnv
except Exception:
    LBOPBConnectorEnv = None  # type: ignore


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

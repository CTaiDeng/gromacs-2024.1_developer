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

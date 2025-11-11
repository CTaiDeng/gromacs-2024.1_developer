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

"""PEM Python operator package.

病理演化幺半群（PEM）子包，包含：
 - 状态表示与可观察量
 - 幺半群（单位元/复合）与常用病理过程算子（转移/凋亡/炎症/致癌）
 - 非交换度、风险、代价与可达性等指标

参考文档：
 - `my_docs/project_docs/1761062400_病理演化幺半群 (PEM) 公理系统.md`
 - `my_docs/project_docs/1761062408_《病理演化幺半群》的核心构造及理论完备性.md`
"""

from .state import PEMState
from .observables import Observables
from .operators import (
    PEMOperator,
    Identity,
    Metastasis,
    Apoptosis,
    Inflammation,
    Carcinogenesis,
    compose,
)
from .metrics import (
    delta_phi,
    non_commutativity_index,
    topo_risk,
    action_cost,
    reach_probability,
)

__all__ = [
    "PEMState",
    "Observables",
    "PEMOperator",
    "Identity",
    "Metastasis",
    "Apoptosis",
    "Inflammation",
    "Carcinogenesis",
    "compose",
    "delta_phi",
    "non_commutativity_index",
    "topo_risk",
    "action_cost",
    "reach_probability",
]

__version__ = "0.1.0"

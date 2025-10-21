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

"""PEM Python operator package.

基于“病理演化幺半群 (PEM) 公理系统”构建：
 - 状态表示与可观测量
 - 幺半群（算子族）的组合与恒等元
 - 典型病理演化算子（转移/凋亡/炎症/致癌）
 - 非对易性指标、拓扑风险、可达性等度量

文档参阅：`my_docs/dev_docs/PEM_OPERATOR_PACKAGE.md`。
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

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

"""PGOM Python operator package.

药理基因组幺半群（PGOM）子包，包含：
 - 状态/可观察量、幺半群与常用基因调控算子（激活/抑制/突变/修复/表观/通路诱导/抑制）
 - 指标：非交换度、拓扑风险、路径代价与可达性

参考文档：
 - `my_docs/project_docs/1761062405_药理基因组幺半群 (PGOM) 公理系统.md`
 - `my_docs/project_docs/1761062412_《药理基因组算子幺半群》的核心构造及理论完备性.md`
"""

from .state import PGOMState
from .observables import Observables
from .operators import (
    PGOMOperator,
    Identity,
    Activate,
    Repress,
    Mutation,
    RepairGenome,
    EpigeneticMod,
    PathwayInduction,
    PathwayInhibition,
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
    "PGOMState",
    "Observables",
    "PGOMOperator",
    "Identity",
    "Activate",
    "Repress",
    "Mutation",
    "RepairGenome",
    "EpigeneticMod",
    "PathwayInduction",
    "PathwayInhibition",
    "compose",
    "delta_phi",
    "non_commutativity_index",
    "topo_risk",
    "action_cost",
    "reach_probability",
]

__version__ = "0.1.0"

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

"""TEM Python operator package.

毒理学效应幺半群（TEM）子包，包含：
 - 状态/可观察量、幺半群与常用毒理过程算子（暴露/吸收/分布/病灶/炎症/解毒/修复）
 - 指标：非交换度、毒理风险、路径代价与可达性

参考文档：
 - `my_docs/project_docs/1761062403_毒理学效应幺半群 (TEM) 公理系统.md`
 - `my_docs/project_docs/1761062410_《毒理学效应幺半群》的核心构造及理论完备性.md`
"""

from .state import TEMState
from .observables import Observables
from .operators import (
    TEMOperator,
    Identity,
    Exposure,
    Absorption,
    Distribution,
    Lesion,
    Inflammation,
    Detox,
    Repair,
    compose,
)
from .metrics import (
    delta_phi,
    non_commutativity_index,
    tox_risk,
    topo_risk,
    action_cost,
    reach_probability,
)

__all__ = [
    "TEMState",
    "Observables",
    "TEMOperator",
    "Identity",
    "Exposure",
    "Absorption",
    "Distribution",
    "Lesion",
    "Inflammation",
    "Detox",
    "Repair",
    "compose",
    "delta_phi",
    "non_commutativity_index",
    "tox_risk",
    "topo_risk",
    "action_cost",
    "reach_probability",
]

__version__ = "0.1.0"

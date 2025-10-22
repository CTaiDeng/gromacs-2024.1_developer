# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
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

"""PRM Python operator package.

生理调控幺半群（PRM）子包，包含：
 - 状态/可观察量、幺半群与常用生理过程算子（摄入/运动/激素/增殖/适应/刺激）
 - 指标：非交换度、风险、路径代价与可达性

参考文档：
 - `my_docs/project_docs/1761062401_生理调控幺半群 (PRM) 公理系统.md`
 - `my_docs/project_docs/1761062409_《生理调控幺半群》的核心构造及理论完备性.md`
"""

from .state import PRMState
from .observables import Observables
from .operators import (
    PRMOperator,
    Identity,
    Ingest,
    Exercise,
    Hormone,
    Proliferation,
    Adaptation,
    Stimulus,
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
    "PRMState",
    "Observables",
    "PRMOperator",
    "Identity",
    "Ingest",
    "Exercise",
    "Hormone",
    "Proliferation",
    "Adaptation",
    "Stimulus",
    "compose",
    "delta_phi",
    "non_commutativity_index",
    "topo_risk",
    "action_cost",
    "reach_probability",
]

__version__ = "0.1.0"

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

"""IEM Python operator package.

免疫效应幺半群（IEM）子包，包含：
 - 状态/可观察量、幺半群与常用免疫过程算子（激活/抑制/扩增/分化/细胞因子释放/免疫记忆）
 - 指标：非交换度、免疫风险、路径代价与可达性

参考文档：
 - `my_docs/project_docs/1761062407_免疫效应幺半群 (IEM) 公理系统.md`
 - `my_docs/project_docs/1761062414_《免疫效应幺半群》的核心构造及理论完备性.md`
"""

from .state import IEMState
from .observables import Observables
from .operators import (
    IEMOperator,
    Identity,
    Activate,
    Suppress,
    Proliferate,
    Differentiate,
    CytokineRelease,
    Memory,
    compose,
)
from .metrics import (
    delta_phi,
    non_commutativity_index,
    imm_risk,
    topo_risk,
    action_cost,
    reach_probability,
)

__all__ = [
    "IEMState",
    "Observables",
    "IEMOperator",
    "Identity",
    "Activate",
    "Suppress",
    "Proliferate",
    "Differentiate",
    "CytokineRelease",
    "Memory",
    "compose",
    "delta_phi",
    "non_commutativity_index",
    "imm_risk",
    "topo_risk",
    "action_cost",
    "reach_probability",
]

__version__ = "0.1.0"

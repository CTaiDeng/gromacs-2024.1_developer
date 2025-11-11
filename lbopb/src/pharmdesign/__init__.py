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

"""药效幺半群联动的化合物设计与分子模拟 API。

功能分层：
- requirements：药效/多维约束的需求输入结构体
- design：从需求生成小分子/大分子（抗体/肽类）设计意图（药效团/母核/取代策略）
- sim：GROMACS 退化分子对接 + 经典分子动力学 + QM/MM 接口（返回命令方案/期望产物）
- pipeline：
  * 基于 PDEM 算子包的“点集拓扑路径积分”（离散 Lagrangian 累加）
  * 借助联络（operator_crosswalk）映射至各纤维丛离散拓扑的对齐算子包

本模块为工程化接口层，不绑定具体外部安装；若未安装 GROMACS/CP2K/ORCA，仅返回命令草案和产物约定。
"""

from .requirements import PharmacodynamicRequirement, ADMETConstraint, ToxicologyConstraint, ImmunologyConstraint
from .design import (
    propose_small_molecule,
    propose_biologic,
)
from .sim import (
    DockingJob,
    MDJob,
    QMMMJob,
    docking_degenerate_gromacs,
    md_classical_gromacs,
    md_qmmm_stub,
)
from .pipeline import (
    pdem_path_integral,
    map_pdem_sequence_to_fibers,
)

__all__ = [
    # requirements
    "PharmacodynamicRequirement",
    "ADMETConstraint",
    "ToxicologyConstraint",
    "ImmunologyConstraint",
    # design
    "propose_small_molecule",
    "propose_biologic",
    # sim
    "DockingJob",
    "MDJob",
    "QMMMJob",
    "docking_degenerate_gromacs",
    "md_classical_gromacs",
    "md_qmmm_stub",
    # pipeline
    "pdem_path_integral",
    "map_pdem_sequence_to_fibers",
]

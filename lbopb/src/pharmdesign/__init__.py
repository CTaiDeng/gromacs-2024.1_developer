# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

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

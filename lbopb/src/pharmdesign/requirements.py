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

from __future__ import annotations

"""药效/多维约束需求输入结构体。

用于自顶向下表达“从需求到设计”的关键参数：
- 目标（靶点/位点/作用机理）
- 药效指标（Ki/IC50、Emax、占有度曲线等）
- ADMET/毒理/免疫侧约束
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ADMETConstraint:
    solubility_mg_per_ml: Optional[float] = None
    permeability_high: bool = True
    cyp_avoid: Optional[List[str]] = None
    bbb_penetration: Optional[bool] = None
    half_life_hours: Optional[float] = None


@dataclass
class ToxicologyConstraint:
    hERG_risk_low: bool = True
    mito_tox_low: bool = True
    liver_tox_low: bool = True


@dataclass
class ImmunologyConstraint:
    cytokine_storm_avoid: bool = True
    immunogenicity_low: bool = True


@dataclass
class PharmacodynamicRequirement:
    target_name: str
    mechanism: str  # e.g. "IN antagonist" / "RT NNRTI"
    potency_ic50_nM: Optional[float] = None
    selectivity_index: Optional[float] = None
    from dataclasses import field
    admet: ADMETConstraint = field(default_factory=ADMETConstraint)  # type: ignore
    tox: ToxicologyConstraint = field(default_factory=ToxicologyConstraint)  # type: ignore
    immuno: ImmunologyConstraint = field(default_factory=ImmunologyConstraint)  # type: ignore

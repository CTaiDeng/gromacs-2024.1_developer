# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

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
    admet: ADMETConstraint = ADMETConstraint()
    tox: ToxicologyConstraint = ToxicologyConstraint()
    immuno: ImmunologyConstraint = ImmunologyConstraint()


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

"""从需求到设计：小分子与大分子设计意图生成。

输出为可序列化字典（意向设计、非最终结构），便于后续对接分子构建/枚举工具。
"""

from dataclasses import dataclass, asdict
from typing import Dict, List

from .requirements import PharmacodynamicRequirement


def propose_small_molecule(req: PharmacodynamicRequirement) -> Dict:
    """基于药效需求生成小分子设计意图。

    - 整合酶抑制（IN antagonist）：建议“三齿螯合 + 疏水芳基 + 极性尾部”
    - 逆转录酶抑制（RT NNRTI）：建议“疏水芳环 + 翻转口袋氢键 + 柔性键”
    """

    mech = req.mechanism.lower()
    design: Dict = {
        "target": req.target_name,
        "mechanism": req.mechanism,
        "pharmacophore": [],
        "scaffold": None,
        "substituent_strategy": [],
        "admet_notes": [],
        "tox_notes": [],
    }
    if "in" in mech and "antagonist" in mech:
        design["pharmacophore"] = [
            "tridentate_metal_chelation",
            "aryl_hydrophobe",
            "tertiary_amine_sidechain",
        ]
        design["scaffold"] = "dihydroxy-aromatic + diketo-acid"
        design["substituent_strategy"] = [
            "para/meta hydrophobe fitting",
            "pKa tuned amine for solubility",
        ]
    elif "rt" in mech or "nnrti" in mech:
        design["pharmacophore"] = [
            "hydrophobic_aromatic",
            "hbond_acceptor_in_flip_pocket",
            "flexible_linker",
        ]
        design["scaffold"] = "diarylether/diaryl-aza"
        design["substituent_strategy"] = [
            "halogen tweak for lipophilicity",
            "donor/acceptor balance",
        ]
    # ADMET 备注
    if req.admet.solubility_mg_per_ml:
        design["admet_notes"].append(f"target_solubility≥{req.admet.solubility_mg_per_ml} mg/mL")
    if req.admet.bbb_penetration is False:
        design["admet_notes"].append("avoid_BBB")
    if req.admet.cyp_avoid:
        design["admet_notes"].append(f"avoid_CYP:{','.join(req.admet.cyp_avoid)}")
    # 毒理备注
    if req.tox.hERG_risk_low:
        design["tox_notes"].append("low_hERG")
    return design


def propose_biologic(req: PharmacodynamicRequirement) -> Dict:
    """大分子设计意图（抗体/肽）。"""

    design = {
        "target": req.target_name,
        "mechanism": req.mechanism,
        "format": "antibody_or_peptide",
        "epitope": "active_site_or_entry_epitope",
        "affinity_goal_nM": req.potency_ic50_nM or 10.0,
        "notes": ["optimize paratope for epitope complementarity"],
    }
    return design

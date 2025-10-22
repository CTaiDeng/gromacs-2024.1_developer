# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

"""统一 API：读取配置 JSON 并返回分子对接/动力学模拟命令方案。

配置 JSON（可选字段）示例见同目录 `example_config.json`。
未提供配置或字段缺失时，返回默认示例值（便于在无环境下演示）。
"""

import json
import os
from typing import Any, Dict, Optional

from .requirements import PharmacodynamicRequirement, ADMETConstraint, ToxicologyConstraint, ImmunologyConstraint
from .design import propose_small_molecule, propose_biologic
from .sim import (
    DockingJob,
    MDJob,
    QMMMJob,
    docking_degenerate_gromacs,
    md_classical_gromacs,
    md_qmmm_stub,
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "design": {
        "target_name": "HIV IN",
        "mechanism": "IN antagonist",
        "potency_ic50_nM": 10.0,
        "admet": {"solubility_mg_per_ml": 0.1, "bbb_penetration": False, "cyp_avoid": ["3A4"]},
        "tox": {"hERG_risk_low": True},
        "immuno": {"cytokine_storm_avoid": True},
    },
    "docking": {
        "receptor_pdb": "protein.pdb",
        "ligand_sdf": "ligand.sdf",
        "out_dir": "out/docking",
    },
    "md": {
        "system_top": "topol.top",
        "structure_gro": "system.gro",
        "mdp": "md.mdp",
        "out_dir": "out/md",
    },
    "qmmm": {
        "system_top": "topol.top",
        "structure_gro": "system.gro",
        "qmmm_config": "qmmm.inp",
        "out_dir": "out/qmmm",
    },
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置 JSON；未提供或失败时返回 DEFAULT_CONFIG。"""

    if not path:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # 简单合并默认值
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        def _merge(dst: Dict[str, Any], src: Dict[str, Any]):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _merge(dst[k], v)
                else:
                    dst[k] = v
        _merge(merged, cfg)
        return merged
    except Exception:
        return json.loads(json.dumps(DEFAULT_CONFIG))


def plan_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """从配置派生：分子设计 + 对接 + MD/QMMM 命令方案。"""

    # 设计
    d = cfg.get("design", {})
    req = PharmacodynamicRequirement(
        target_name=d.get("target_name", DEFAULT_CONFIG["design"]["target_name"]),
        mechanism=d.get("mechanism", DEFAULT_CONFIG["design"]["mechanism"]),
        potency_ic50_nM=d.get("potency_ic50_nM", DEFAULT_CONFIG["design"]["potency_ic50_nM"]),
        admet=ADMETConstraint(**d.get("admet", {})),
        tox=ToxicologyConstraint(**d.get("tox", {})),
        immuno=ImmunologyConstraint(**d.get("immuno", {})),
    )
    small = propose_small_molecule(req)
    biologic = propose_biologic(req)

    # 对接
    dk = cfg.get("docking", {})
    docking_plan = docking_degenerate_gromacs(DockingJob(
        receptor_pdb=dk.get("receptor_pdb", DEFAULT_CONFIG["docking"]["receptor_pdb"]),
        ligand_sdf=dk.get("ligand_sdf", DEFAULT_CONFIG["docking"]["ligand_sdf"]),
        out_dir=dk.get("out_dir", DEFAULT_CONFIG["docking"]["out_dir"]),
    ))

    # 经典 MD
    md_cfg = cfg.get("md", {})
    md_plan = md_classical_gromacs(MDJob(
        system_top=md_cfg.get("system_top", DEFAULT_CONFIG["md"]["system_top"]),
        structure_gro=md_cfg.get("structure_gro", DEFAULT_CONFIG["md"]["structure_gro"]),
        mdp=md_cfg.get("mdp", DEFAULT_CONFIG["md"]["mdp"]),
        out_dir=md_cfg.get("out_dir", DEFAULT_CONFIG["md"]["out_dir"]),
    ))

    # QM/MM（可选）
    qmmm_cfg = cfg.get("qmmm", {})
    qmmm_plan = md_qmmm_stub(QMMMJob(
        system_top=qmmm_cfg.get("system_top", DEFAULT_CONFIG["qmmm"]["system_top"]),
        structure_gro=qmmm_cfg.get("structure_gro", DEFAULT_CONFIG["qmmm"]["structure_gro"]),
        qmmm_config=qmmm_cfg.get("qmmm_config", DEFAULT_CONFIG["qmmm"]["qmmm_config"]),
        out_dir=qmmm_cfg.get("out_dir", DEFAULT_CONFIG["qmmm"]["out_dir"]),
    ))

    return {
        "design": {"small_molecule": small, "biologic": biologic},
        "docking": docking_plan,
        "md": md_plan,
        "qmmm": qmmm_plan,
    }


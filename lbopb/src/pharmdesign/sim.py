# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

"""GROMACS 退化分子对接 + 经典 MD + QM/MM 接口（命令方案）。

说明：
- 接口返回“命令方案”和“期望产物路径”供上层调度。
- 未强制执行外部命令；调用方可选择落盘脚本或直接子进程执行。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os


@dataclass
class DockingJob:
    receptor_pdb: str
    ligand_sdf: str
    out_dir: str


@dataclass
class MDJob:
    system_top: str
    structure_gro: str
    mdp: str
    out_dir: str


@dataclass
class QMMMJob:
    system_top: str
    structure_gro: str
    qmmm_config: str  # 外部 QM 引擎的输入片段
    out_dir: str


def docking_degenerate_gromacs(job: DockingJob) -> Dict:
    """退化分子对接（基于 rerun 能量评估的近似流程）。

    步骤建议（命令字符串，仅供参考）：
    - 将若干随机“姿势”（poses）组合为短 TRJ
    - 使用 `gmx mdrun -rerun` 在力场下评估每个姿势的能量
    - 产出：poses.scores.csv，按能量筛选 top-N
    """

    os.makedirs(job.out_dir, exist_ok=True)
    commands: List[str] = []
    exp_outputs = {
        "traj": os.path.join(job.out_dir, "poses.trr"),
        "scores": os.path.join(job.out_dir, "poses.scores.csv"),
    }
    commands.append(f"# 生成随机姿势并打包为 TRR（伪指令，需对接构建工具）")
    commands.append(
        f"python gen_poses.py --receptor {job.receptor_pdb} --ligand {job.ligand_sdf} --out {exp_outputs['traj']}")
    commands.append(f"# rerun 评估（示例命令）")
    commands.append(f"gmx mdrun -s topol.tpr -rerun {exp_outputs['traj']} -g {job.out_dir}/rerun.log")
    commands.append(f"python score_rerun.py --log {job.out_dir}/rerun.log --out {exp_outputs['scores']}")
    return {"commands": commands, "outputs": exp_outputs}


def md_classical_gromacs(job: MDJob) -> Dict:
    """经典分子动力学（GROMACS）。"""

    os.makedirs(job.out_dir, exist_ok=True)
    commands = [
        f"gmx grompp -f {job.mdp} -c {job.structure_gro} -p {job.system_top} -o {job.out_dir}/topol.tpr",
        f"gmx mdrun -deffnm {job.out_dir}/md",
    ]
    outputs = {"traj": f"{job.out_dir}/md.trr", "ener": f"{job.out_dir}/md.edr"}
    return {"commands": commands, "outputs": outputs}


def md_qmmm_stub(job: QMMMJob) -> Dict:
    """QM/MM 接口占位：返回与 CP2K/ORCA 对接的命令草案。"""

    os.makedirs(job.out_dir, exist_ok=True)
    commands = [
        f"# 准备 QM/MM 输入（片段）: {job.qmmm_config}",
        f"# 示例：调用 CP2K/ORCA 进行 QM 区域能量/力评估并回填到 MD 步进",
    ]
    outputs = {"traj": f"{job.out_dir}/qmmm.trr", "ener": f"{job.out_dir}/qmmm.edr"}
    return {"commands": commands, "outputs": outputs}

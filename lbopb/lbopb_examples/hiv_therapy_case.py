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

"""
HIV 治疗案例：以病理为基底，经联络到药效并展开六切面算子包。
数据来源：lbopb/src/operator_crosswalk.json -> case_packages["HIV_Therapy_Path"].
运行（任意工作目录均可）：  python lbopb/lbopb_examples/hiv_therapy_case.py

免责声明：本文件及其生成的报告仅用于方法学与技术演示，不构成医学建议或临床诊断/治疗方案；
亦不用于任何实际诊疗决策或药物使用指导。若需临床决策，请咨询专业医师并遵循监管要求。
"""

from __future__ import annotations

from typing import Dict, List

# 确保可以从任意工作目录运行：将仓库根目录加入 sys.path
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lbopb.src.op_crosswalk import load_crosswalk
from lbopb.src.powerset import compose_sequence, instantiate_ops
from lbopb.src.pharmdesign.api import load_config as pd_load_config, plan_from_config as pd_plan_from_config
from lbopb.src.pharmdesign.requirements import PharmacodynamicRequirement
from lbopb.src.pharmdesign.design import propose_small_molecule

# 导入各切面以便构造状态并执行
from lbopb.src.pem import PEMState
from lbopb.src.pem import topo_risk as pem_topo_risk, action_cost as pem_action_cost
from lbopb.src.pdem import PDEMState
from lbopb.src.pdem import eff_risk as pdem_eff_risk, action_cost as pdem_action_cost
from lbopb.src.pktm import PKTMState
from lbopb.src.pktm import topo_risk as pktm_topo_risk, action_cost as pktm_action_cost
from lbopb.src.pgom import PGOMState
from lbopb.src.pgom import topo_risk as pgom_topo_risk, action_cost as pgom_action_cost
from lbopb.src.tem import TEMState
from lbopb.src.tem import tox_risk as tem_tox_risk, action_cost as tem_action_cost
from lbopb.src.prm import PRMState
from lbopb.src.prm import topo_risk as prm_topo_risk, action_cost as prm_action_cost
from lbopb.src.iem import IEMState
from lbopb.src.iem import imm_risk as iem_imm_risk, action_cost as iem_action_cost


def default_states() -> Dict[str, object]:
    return dict(
        pem=PEMState(b=8.0, n_comp=3, perim=2.0, fidelity=0.6),
        pdem=PDEMState(b=1.5, n_comp=1, perim=0.8, fidelity=0.6),
        pktm=PKTMState(b=0.5, n_comp=1, perim=0.5, fidelity=0.95),
        pgom=PGOMState(b=3.0, n_comp=2, perim=1.5, fidelity=0.8),
        tem=TEMState(b=5.0, n_comp=1, perim=2.0, fidelity=0.9),
        prm=PRMState(b=10.0, n_comp=1, perim=5.0, fidelity=0.8),
        iem=IEMState(b=2.0, n_comp=2, perim=1.0, fidelity=0.7),
    )


def run_case(case_name: str = "HIV_Therapy_Path", *, pharm_cfg_path: str | None = None) -> None:
    cw = load_crosswalk()
    case = cw.get("case_packages", {}).get(case_name)
    if not case:
        raise SystemExit(f"未找到案例包：{case_name}")

    seqs: Dict[str, List[str]] = case["sequences"]
    states = default_states()

    print(f"== 案例包: {case_name}")
    if case.get("description"):
        print("描述:", case["description"])
    if case.get("notes"):
        print("说明:", case["notes"])

        # 逐模块复合并执行
    for mod, seq in seqs.items():
        if mod not in states:
            continue
        try:
            O = compose_sequence(mod, seq)
        except Exception as e:
            print(f"[WARN] 模块 {mod} 复合失败: {e}")
            continue
        s0 = states[mod]
        s1 = O(s0)  # type: ignore
        print(f"-- {mod}: seq = {seq}")
        print(f"   s0 → s1: {s0} → {s1}")

    # 生成 Markdown 报告
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{case_name}_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(gen_markdown_report(case_name, cw, states, seqs, pharm_cfg_path))
    print(f"Markdown 报告已生成: {md_path}")


def _state_tuple(s) -> tuple[float, int, float, float]:
    return float(s.b), int(s.n_comp), float(s.perim), float(s.fidelity)


def _fmt_state(s) -> str:
    b, n, p, f = _state_tuple(s)
    return f"B={b:.4g}, N={n}, P={p:.4g}, F={f:.4g}"


def _risk_and_cost(mod: str, seq: list[str], s0) -> tuple[float, float]:
    # 选择每个模块的风险函数与 action_cost
    if mod == "pem":
        risk = pem_topo_risk(s0)
        cost = pem_action_cost(instantiate_ops(mod, seq), s0)
    elif mod == "pdem":
        risk = pdem_eff_risk(s0)
        cost = pdem_action_cost(instantiate_ops(mod, seq), s0)
    elif mod == "pktm":
        risk = pktm_topo_risk(s0)
        cost = pktm_action_cost(instantiate_ops(mod, seq), s0)
    elif mod == "pgom":
        risk = pgom_topo_risk(s0)
        cost = pgom_action_cost(instantiate_ops(mod, seq), s0)
    elif mod == "tem":
        risk = tem_tox_risk(s0)
        cost = tem_action_cost(instantiate_ops(mod, seq), s0)
    elif mod == "prm":
        risk = prm_topo_risk(s0)
        cost = prm_action_cost(instantiate_ops(mod, seq), s0)
    elif mod == "iem":
        risk = iem_imm_risk(s0)
        cost = iem_action_cost(instantiate_ops(mod, seq), s0)
    else:
        risk = 0.0
        cost = 0.0
    return float(risk), float(cost)


def gen_markdown_report(case_name: str, cw: dict, states: Dict[str, object], seqs: Dict[str, List[str]],
                        pharm_cfg_path: str | None) -> str:
    lines: list[str] = []
    case = cw.get("case_packages", {}).get(case_name, {})
    # 免责声明（示例/演示用途，不构成医学建议）
    lines.append("免责声明：本文件及其生成的报告仅用于方法学与技术演示，不构成医学建议或临床诊断/治疗方案；\n")
    lines.append("亦不用于任何实际诊疗决策或药物使用指导。若需临床决策，请咨询专业医师并遵循监管要求。\n\n")
    lines.append(f"# 案例：{case_name}\n\n")
    if case.get("description"):
        lines.append("#### 说明：\n\n")
        lines.append(f"{case['description']}\n\n")
    if case.get("notes"):
        lines.append("#### 说明：\n\n")
        lines.append(f"{case['notes']}\n\n")

    # 总览表
    lines.append("## 立体序列总览\n\n")
    lines.append("| 模块 | 序列 |\n| :--- | :--- |\n")
    for mod in ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]:
        if mod in seqs:
            lines.append(f"| {mod} | {' → '.join(seqs[mod])} |\n")
    lines.append("\n")

    # 分模块详情
    for mod in ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]:
        if mod not in seqs or mod not in states:
            continue
        seq = seqs[mod]
        lines.append(f"## {mod}\n\n")
        lines.append(f"**序列**: {' → '.join(seq)}\n\n")
        # 初末态与变化
        O = compose_sequence(mod, seq)
        s0 = states[mod]
        s1 = O(s0)  # type: ignore
        b0, n0, p0, f0 = _state_tuple(s0)
        b1, n1, p1, f1 = _state_tuple(s1)
        lines.append("**状态变化**:\n\n")
        lines.append("| 指标 | s0 | s1 | Δ |\n| :--- | :--- | :--- | :--- |\n")
        lines.append(f"| B | {b0:.4g} | {b1:.4g} | {b1 - b0:+.4g} |\n")
        lines.append(f"| P | {p0:.4g} | {p1:.4g} | {p1 - p0:+.4g} |\n")
        lines.append(f"| N_comp | {n0} | {n1} | {n1 - n0:+d} |\n")
        lines.append(f"| F | {f0:.4g} | {f1:.4g} | {f1 - f0:+.4g} |\n\n")
        # 风险与代价（s0 基准）
        risk0, cost = _risk_and_cost(mod, seq, s0)
        risk1, _ = _risk_and_cost(mod, seq, s1)
        lines.append("**风险与代价（示意）**:\n\n")
        lines.append("| 项 | 数值 | 备注 |\n| :--- | :--- | :--- |\n")
        lines.append(f"| Risk(s0) | {risk0:.4g} | 模块定义的风险函数 |\n")
        lines.append(f"| Risk(s1) | {risk1:.4g} | 变化后风险 |\n")
        lines.append(f"| ActionCost(seq; s0) | {cost:.4g} | 仅示意权重 |\n\n")
        # 文本化概述
        lines.append("#### 说明：\n\n")
        if mod == "pem":
            lines.append("病理侧期望通过凋亡/清除实现负荷与边界收敛、保真提升。\n\n")
        elif mod == "pdem":
            lines.append("药效侧通过结合+拮抗抑制关键效应链，期望 F 上行。\n\n")
        elif mod == "pktm":
            lines.append("ADME 链路用于保障暴露窗口与可达性，代谢/排泄控制全身风险。\n\n")
        elif mod == "pgom":
            lines.append("通路抑制以避免不利转录级响应，辅以修复/表观调谐。\n\n")
        elif mod == "tem":
            lines.append("解毒与修复以压低损伤负荷与炎症边界，控制毒理风险。\n\n")
        elif mod == "prm":
            lines.append("刺激—适应表达稳态回归，B/P 收敛、F 提升。\n\n")
        elif mod == "iem":
            lines.append("免疫侧激活-分化-记忆，避免细胞因子过度释放。\n\n")

    # 分子设计与分子模拟计划
    lines.append("## 分子设计与分子模拟计划\n\n")
    lines.append("#### 说明：\n\n")
    lines.append("基于药效切面（PDEM）的拮抗链，给出小分子设计意图与 GROMACS 退化对接/MD/QM-MM 的命令方案。\n\n")
    cfg = pd_load_config(pharm_cfg_path)
    plan = pd_plan_from_config(cfg)
    # 设计
    sm = plan["design"]["small_molecule"]
    lines.append("### 小分子设计意图\n\n")
    lines.append(f"- 目标: {sm.get('target')}\n")
    lines.append(f"- 机制: {sm.get('mechanism')}\n")
    if sm.get("pharmacophore"):
        lines.append(f"- 药效团: {', '.join(sm['pharmacophore'])}\n")
    if sm.get("scaffold"):
        lines.append(f"- 母核: {sm['scaffold']}\n")
    if sm.get("substituent_strategy"):
        lines.append(f"- 取代策略: {', '.join(sm['substituent_strategy'])}\n")
    if sm.get("admet_notes"):
        lines.append(f"- ADMET备注: {', '.join(sm['admet_notes'])}\n")
    if sm.get("tox_notes"):
        lines.append(f"- 毒理备注: {', '.join(sm['tox_notes'])}\n")
    lines.append("\n")
    # 对接
    lines.append("### 退化分子对接（命令方案）\n\n")
    dock = plan["docking"]
    lines.append("```bash\n" + "\n".join(dock["commands"]) + "\n```\n\n")
    # 经典 MD
    lines.append("### 经典分子动力学（命令方案）\n\n")
    mdp = plan["md"]
    lines.append("```bash\n" + "\n".join(mdp["commands"]) + "\n```\n\n")
    # QM/MM
    lines.append("### QM/MM 占位（命令草案）\n\n")
    qmmm = plan["qmmm"]
    lines.append("```bash\n" + "\n".join(qmmm["commands"]) + "\n```\n\n")

    # 复现指引
    lines.append("## 复现指引\n\n")
    lines.append(
        "```\npython -c \"import sys,os; sys.path.insert(0, os.path.abspath('.')); import lbopb_examples.hiv_therapy_case as m; m.run_case(pharm_cfg_path=None)\"\n```\n\n")

    return "".join(lines)


if __name__ == "__main__":
    run_case()

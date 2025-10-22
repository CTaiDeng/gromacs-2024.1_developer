# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""HIV 娌荤枟妗堜緥锛氫互鐥呯悊涓哄熀搴曪紝缁忚仈缁滃埌鑽晥骞跺睍寮€鍏垏闈㈢畻瀛愬寘銆?
鏁版嵁鏉ユ簮锛歭bopb/src/operator_crosswalk.json -> case_packages["HIV_Therapy_Path"].
杩愯锛堜换鎰忓伐浣滅洰褰曞潎鍙級锛?  python lbopb/lbopb_examples/hiv_therapy_case.py

免责声明：本文件及其生成的报告仅用于方法学与技术演示，不构成医学建议或临床诊断/治疗方案；
亦不用于任何实际诊疗决策或药物使用指导。若需临床决策，请咨询专业医师并遵循监管要求。
"""

from __future__ import annotations

from typing import Dict, List

# 纭繚鍙互浠庝换鎰忓伐浣滅洰褰曡繍琛岋細灏嗕粨搴撴牴鐩綍鍔犲叆 sys.path
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lbopb.src.op_crosswalk import load_crosswalk
from lbopb.src.powerset import compose_sequence, instantiate_ops
from lbopb.src.pharmdesign.api import load_config as pd_load_config, plan_from_config as pd_plan_from_config
from lbopb.src.pharmdesign.requirements import PharmacodynamicRequirement
from lbopb.src.pharmdesign.design import propose_small_molecule

# 瀵煎叆鍚勫垏闈互渚挎瀯閫犵姸鎬佸苟鎵ц
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
        raise SystemExit(f"鏈壘鍒版渚嬪寘锛歿case_name}")

    seqs: Dict[str, List[str]] = case["sequences"]
    states = default_states()

    print(f"== 妗堜緥鍖? {case_name}")
    if case.get("description"):
        print("鎻忚堪:", case["description"]) 
    if case.get("notes"):
        print("璇存槑:", case["notes"]) 

    # 閫愭ā鍧楀鍚堝苟鎵ц
    for mod, seq in seqs.items():
        if mod not in states:
            continue
        try:
            O = compose_sequence(mod, seq)
        except Exception as e:
            print(f"[WARN] 妯″潡 {mod} 澶嶅悎澶辫触: {e}")
            continue
        s0 = states[mod]
        s1 = O(s0)  # type: ignore
        print(f"-- {mod}: seq = {seq}")
        print(f"   s0 鈫?s1: {s0} 鈫?{s1}")

    # 鐢熸垚 Markdown 鎶ュ憡
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{case_name}_report.md")
    with open(md_path, "w", encoding="utf-8", newline="\\r\\n") as f:
        f.write(gen_markdown_report(case_name, cw, states, seqs, pharm_cfg_path))
    print(f"Markdown 鎶ュ憡宸茬敓鎴? {md_path}")


def _state_tuple(s) -> tuple[float, int, float, float]:
    return float(s.b), int(s.n_comp), float(s.perim), float(s.fidelity)


def _fmt_state(s) -> str:
    b, n, p, f = _state_tuple(s)
    return f"B={b:.4g}, N={n}, P={p:.4g}, F={f:.4g}"


def _risk_and_cost(mod: str, seq: list[str], s0) -> tuple[float, float]:
    # 閫夋嫨姣忎釜妯″潡鐨勯闄╁嚱鏁颁笌 action_cost
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


def gen_markdown_report(case_name: str, cw: dict, states: Dict[str, object], seqs: Dict[str, List[str]], pharm_cfg_path: str | None) -> str:
    lines: list[str] = []
    case = cw.get("case_packages", {}).get(case_name, {})
    lines.append(f"# 妗堜緥锛歿case_name}\n\n")
    if case.get("description"):
        lines.append("#### 璇存槑锛歕n\n")
        lines.append(f"{case['description']}\n\n")
    if case.get("notes"):
        lines.append("#### 璇存槑锛歕n\n")
        lines.append(f"{case['notes']}\n\n")

    # 鎬昏琛?    lines.append("## 绔嬩綋搴忓垪鎬昏\n\n")
    lines.append("| 妯″潡 | 搴忓垪 |\n| :--- | :--- |\n")
    for mod in ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]:
        if mod in seqs:
            lines.append(f"| {mod} | {' 鈫?'.join(seqs[mod])} |\n")
    lines.append("\n")

    # 鍒嗘ā鍧楄鎯?    for mod in ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]:
        if mod not in seqs or mod not in states:
            continue
        seq = seqs[mod]
        lines.append(f"## {mod}\n\n")
        lines.append(f"**搴忓垪**: {' 鈫?'.join(seq)}\n\n")
        # 鍒濇湯鎬佷笌鍙樺寲
        O = compose_sequence(mod, seq)
        s0 = states[mod]
        s1 = O(s0)  # type: ignore
        b0, n0, p0, f0 = _state_tuple(s0)
        b1, n1, p1, f1 = _state_tuple(s1)
        lines.append("**鐘舵€佸彉鍖?*:\n\n")
        lines.append("| 鎸囨爣 | s0 | s1 | 螖 |\n| :--- | :--- | :--- | :--- |\n")
        lines.append(f"| B | {b0:.4g} | {b1:.4g} | {b1 - b0:+.4g} |\n")
        lines.append(f"| P | {p0:.4g} | {p1:.4g} | {p1 - p0:+.4g} |\n")
        lines.append(f"| N_comp | {n0} | {n1} | {n1 - n0:+d} |\n")
        lines.append(f"| F | {f0:.4g} | {f1:.4g} | {f1 - f0:+.4g} |\n\n")
        # 椋庨櫓涓庝唬浠凤紙s0 鍩哄噯锛?        risk0, cost = _risk_and_cost(mod, seq, s0)
        risk1, _ = _risk_and_cost(mod, seq, s1)
        lines.append("**椋庨櫓涓庝唬浠凤紙绀烘剰锛?*:\n\n")
        lines.append("| 椤?| 鏁板€?| 澶囨敞 |\n| :--- | :--- | :--- |\n")
        lines.append(f"| Risk(s0) | {risk0:.4g} | 妯″潡瀹氫箟鐨勯闄╁嚱鏁?|\n")
        lines.append(f"| Risk(s1) | {risk1:.4g} | 鍙樺寲鍚庨闄?|\n")
        lines.append(f"| ActionCost(seq; s0) | {cost:.4g} | 浠呯ず鎰忔潈閲?|\n\n")
        # 鏂囨湰鍖栨杩?        lines.append("#### 璇存槑锛歕n\n")
        if mod == "pem":
            lines.append("鐥呯悊渚ф湡鏈涢€氳繃鍑嬩骸/娓呴櫎瀹炵幇璐熻嵎涓庤竟鐣屾敹鏁涖€佷繚鐪熸彁鍗囥€俓n\n")
        elif mod == "pdem":
            lines.append("鑽晥渚ч€氳繃缁撳悎+鎷姉鎶戝埗鍏抽敭鏁堝簲閾撅紝鏈熸湜 F 涓婅銆俓n\n")
        elif mod == "pktm":
            lines.append("ADME 閾捐矾鐢ㄤ簬淇濋殰鏆撮湶绐楀彛涓庡彲杈炬€э紝浠ｈ阿/鎺掓硠鎺у埗鍏ㄨ韩椋庨櫓銆俓n\n")
        elif mod == "pgom":
            lines.append("閫氳矾鎶戝埗浠ラ伩鍏嶄笉鍒╄浆褰曠骇鍝嶅簲锛岃緟浠ヤ慨澶?琛ㄨ璋冭皭銆俓n\n")
        elif mod == "tem":
            lines.append("瑙ｆ瘨涓庝慨澶嶄互鍘嬩綆鎹熶激璐熻嵎涓庣値鐥囪竟鐣岋紝鎺у埗姣掔悊椋庨櫓銆俓n\n")
        elif mod == "prm":
            lines.append("鍒烘縺鈥旈€傚簲琛ㄨ揪绋虫€佸洖褰掞紝B/P 鏀舵暃銆丗 鎻愬崌銆俓n\n")
        elif mod == "iem":
            lines.append("鍏嶇柅渚ф縺娲?鍒嗗寲-璁板繂锛岄伩鍏嶇粏鑳炲洜瀛愯繃搴﹂噴鏀俱€俓n\n")

    # 鍒嗗瓙璁捐涓庡垎瀛愭ā鎷熻鍒?    lines.append("## 鍒嗗瓙璁捐涓庡垎瀛愭ā鎷熻鍒抃n\n")
    lines.append("#### 璇存槑锛歕n\n")
    lines.append("鍩轰簬鑽晥鍒囬潰锛圥DEM锛夌殑鎷姉閾撅紝缁欏嚭灏忓垎瀛愯璁℃剰鍥句笌 GROMACS 閫€鍖栧鎺?MD/QM-MM 鐨勫懡浠ゆ柟妗堛€俓n\n")
    cfg = pd_load_config(pharm_cfg_path)
    plan = pd_plan_from_config(cfg)
    # 璁捐
    sm = plan["design"]["small_molecule"]
    lines.append("### 灏忓垎瀛愯璁℃剰鍥綷n\n")
    lines.append(f"- 鐩爣: {sm.get('target')}\n")
    lines.append(f"- 鏈哄埗: {sm.get('mechanism')}\n")
    if sm.get("pharmacophore"):
        lines.append(f"- 鑽晥鍥? {', '.join(sm['pharmacophore'])}\n")
    if sm.get("scaffold"):
        lines.append(f"- 姣嶆牳: {sm['scaffold']}\n")
    if sm.get("substituent_strategy"):
        lines.append(f"- 鍙栦唬绛栫暐: {', '.join(sm['substituent_strategy'])}\n")
    if sm.get("admet_notes"):
        lines.append(f"- ADMET澶囨敞: {', '.join(sm['admet_notes'])}\n")
    if sm.get("tox_notes"):
        lines.append(f"- 姣掔悊澶囨敞: {', '.join(sm['tox_notes'])}\n")
    lines.append("\n")
    # 瀵规帴
    lines.append("### 閫€鍖栧垎瀛愬鎺ワ紙鍛戒护鏂规锛塡n\n")
    dock = plan["docking"]
    lines.append("```bash\n" + "\n".join(dock["commands"]) + "\n```\n\n")
    # 缁忓吀 MD
    lines.append("### 缁忓吀鍒嗗瓙鍔ㄥ姏瀛︼紙鍛戒护鏂规锛塡n\n")
    mdp = plan["md"]
    lines.append("```bash\n" + "\n".join(mdp["commands"]) + "\n```\n\n")
    # QM/MM
    lines.append("### QM/MM 鍗犱綅锛堝懡浠よ崏妗堬級\n\n")
    qmmm = plan["qmmm"]
    lines.append("```bash\n" + "\n".join(qmmm["commands"]) + "\n```\n\n")

    # 澶嶇幇鎸囧紩
    lines.append("## 澶嶇幇鎸囧紩\n\n")
    lines.append("```\npython -c \"import sys,os; sys.path.insert(0, os.path.abspath('.')); import lbopb.lbopb_examples.hiv_therapy_case as m; m.run_case(pharm_cfg_path=None)\"\n```\n\n")

    return "".join(lines)


if __name__ == "__main__":
    run_case()

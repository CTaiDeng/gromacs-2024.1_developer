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
HIV 治疗案例（HTML 报告版 v2）：
- 以病理为基底，经联络到药效并展开六切面算子包；
- 生成更美观直观的 HTML 报告（封面、标题、分节、可视化图表）。

数据来源：lbopb/src/operator_crosswalk.json -> case_packages["HIV_Therapy_Path"].
运行（任意工作目录均可）：  python lbopb/lbopb_examples/hiv_therapy_case_v2.py

免责声明：本文件及其生成的报告仅用于方法学与技术演示，不构成医学建议或临床诊断/治疗方案；
亦不用于任何实际诊疗决策或药物使用指导。若需临床决策，请咨询专业医师并遵循监管要求。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List
import urllib.request
import urllib.error
import urllib.parse

# 确保可以从任意工作目录运行：将仓库根目录加入 sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# 也将 my_scripts 目录加入（以防隐式包在某些环境下不可见）
MS = os.path.join(ROOT, "my_scripts")
if MS not in sys.path and os.path.isdir(MS):
    sys.path.insert(0, MS)

# 优先使用集中封装的 my_scripts.gemini_client
try:
    from my_scripts.gemini_client import generate_gemini_content as _gemini_generate_central
except Exception as _e:
    import sys as _sys

    _sys.stderr.write(f"[WARN] import my_scripts.gemini_client failed: {_e}\n")
    _gemini_generate_central = None  # 回退到本地 urllib 实现

# 翻译与 Markdown 组合（基于 Gemini）
try:
    from my_scripts.google_translate_client import translate_text as _gemini_translate
    from my_scripts.google_translate_client import build_markdown_with_translation as _build_md_with_cn
except Exception as _e:
    _gemini_translate = None  # type: ignore
    _build_md_with_cn = None  # type: ignore

from lbopb.src.op_crosswalk import load_crosswalk
from lbopb.src.powerset import compose_sequence, instantiate_ops
from lbopb.src.pharmdesign.api import (
    load_config as pd_load_config,
    plan_from_config as pd_plan_from_config,
)

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


def _state_tuple(s) -> tuple[float, int, float, float]:
    return float(s.b), int(s.n_comp), float(s.perim), float(s.fidelity)


def _risk_and_cost(mod: str, seq: List[str], s0) -> tuple[float, float]:
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


def gen_html_report(case_name: str, cw: dict, states: Dict[str, object], seqs: Dict[str, List[str]],
                    pharm_cfg_path: str | None) -> str:
    # 数据准备
    case = cw.get("case_packages", {}).get(case_name, {})
    modules = ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]

    # 组装模块数据（供 JS 使用）
    mod_data = {}
    for mod in modules:
        if mod not in seqs or mod not in states:
            continue
        seq = seqs[mod]
        O = compose_sequence(mod, seq)
        s0 = states[mod]
        s1 = O(s0)  # type: ignore
        b0, n0, p0, f0 = _state_tuple(s0)
        b1, n1, p1, f1 = _state_tuple(s1)
        risk0, cost0 = _risk_and_cost(mod, seq, s0)
        risk1, _ = _risk_and_cost(mod, seq, s1)
        mod_data[mod] = {
            "seq": seq,
            "s0": {"B": b0, "P": p0, "F": f0, "N": n0},
            "s1": {"B": b1, "P": p1, "F": f1, "N": n1},
            "risk": {"s0": risk0, "s1": risk1, "cost": cost0},
        }

    # 药设/对接/MD/QMMM 计划（用于报告底部）
    cfg = pd_load_config(pharm_cfg_path)
    plan = pd_plan_from_config(cfg)
    sm = plan["design"]["small_molecule"]

    # 组装 Gemini 自动评价（若配置了 API Key 则自动请求）
    def _gemini_generate(prompt: str, *, api_key: str, model: str | None = None) -> str:
        # 如已安装集中封装，直接使用（模型优先取 GEMINI_MODEL）
        if _gemini_generate_central is not None:
            return _gemini_generate_central(prompt, api_key=api_key, model=model)
        if not model:
            model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro-latest")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": api_key,
        }
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                jo = json.loads(raw)
        except urllib.error.HTTPError as e:
            try:
                err = e.read().decode("utf-8", errors="replace")
            except Exception:
                err = str(e)
            return f"[Gemini HTTPError] {e.code}: {err}"
        except Exception as e:
            return f"[Gemini Error] {e}"
        # 尽量鲁棒地抽取文本
        try:
            cands = jo.get("candidates", [])
            if not cands:
                return json.dumps(jo, ensure_ascii=False)
            parts = cands[0].get("content", {}).get("parts", [])
            texts = []
            for p in parts:
                t = p.get("text")
                if t:
                    texts.append(t)
            if texts:
                return "\n".join(texts)
            return json.dumps(jo, ensure_ascii=False)
        except Exception:
            return json.dumps(jo, ensure_ascii=False)

    gemini_result: str | None = None
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key and not os.environ.get("NO_GEMINI"):
        try:
            prompt = (
                    "请作为跨学科审评专家（病理/药效/ADME/通路/毒理/免疫），对如下 HIV 治疗路径进行结构化评价。"
                    "请返回 JSON，字段包含：summary, coherence_score, feasibility_score, risk_score, cost_score, "
                    "module_notes{pem,pdem,pktm,pgom,tem,prm,iem}, top_actions, caveats, confidence。\n"
                    "数据(UTF-8 JSON)：\n" + json.dumps(mod_data, ensure_ascii=False)
            )
            gemini_result = _gemini_generate(prompt, api_key=api_key)
        except Exception as _:
            gemini_result = None

    # HTML 模板（Chart.js 通过 CDN 引入；如需离线可替换为本地文件）
    # 统一以 LF 拼接（仓库最高规范）
    L: List[str] = []
    A = L.append

    def _j(s: str) -> None:
        # 统一 LF 行尾
        A(s)

    _j("<!DOCTYPE html>")
    _j("<html lang=\"zh-CN\">")
    _j("<head>")
    _j("  <meta charset=\"utf-8\" />")
    _j("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />")
    _j(f"  <title>HIV 治疗路径报告 · {case_name}</title>")
    _j("  <link rel=\"preconnect\" href=\"https://cdn.jsdelivr.net\" />")
    _j("  <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js\"></script>")
    _j("  <script src=\"https://cdn.jsdelivr.net/npm/marked/marked.min.js\"></script>")
    _j("  <style>")
    _j("    /* 深色主题配色与基础排版（中文注释：可按需自定义） */")
    _j("    :root{--bg:#0b1020;--bg2:#0f1830;--card:#141a2a;--txt:#e8efff;--muted:#9fb0d0;--acc:#36c;--good:#27ae60;--warn:#f39c12;--bad:#e74c3c;--chip:#223;--chip-b:#335;}")
    _j("    *{box-sizing:border-box}")
    _j("    body{margin:0;font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:linear-gradient(180deg,var(--bg),var(--bg2));color:var(--txt);}")
    _j("    .container{max-width:1120px;margin:0 auto;padding:24px}")
    _j("    .hero{padding:72px 24px 48px;background:radial-gradient(1200px 600px at 20% -10%,rgba(51,102,204,.35),transparent),radial-gradient(800px 400px at 80% -20%,rgba(231,76,60,.25),transparent);} ")
    _j("    .title{font-size:36px;font-weight:700;margin:0 0 8px}")
    _j("    .subtitle{font-size:16px;color:var(--muted);margin:0 0 16px}")
    _j("    .badge{display:inline-block;padding:4px 10px;border-radius:999px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);font-size:12px;color:#d0ddff;margin-right:8px}")
    _j("    .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}")
    _j("    .card{grid-column:span 12;background:var(--card);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:16px}")
    _j("    @media(min-width:980px){.card.half{grid-column:span 6}}");
    _j("    h2{font-size:22px;margin:10px 0 8px}")
    _j("    h3{font-size:18px;margin:10px 0 8px;color:#d7e0ff}")
    _j("    p,li{line-height:1.6;color:#cfdbff}")
    _j("    .chips{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 8px}")
    _j("    .chip{background:var(--chip);border:1px solid var(--chip-b);color:#cfe0ff;padding:4px 8px;border-radius:8px;font-size:12px}")
    _j("    .kvs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:8px 0}")
    _j("    .kv{background:#0e1322;border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:8px}")
    _j("    .kv .k{font-size:11px;color:#9fb0d0}")
    _j("    .kv .v{font-family:ui-monospace,Consolas,monospace;font-size:14px}")
    _j("    .toc a{color:#9fc8ff;text-decoration:none;margin-right:12px}")
    _j("    .muted{color:var(--muted)}")
    _j("    pre{white-space:pre-wrap;background:#0d1322;border:1px solid rgba(255,255,255,.08);padding:12px;border-radius:8px}")
    _j("    footer{color:#9fb0d0;font-size:12px;margin:32px 0 12px}")
    _j("  </style>")
    _j("</head>")
    _j("<body>")
    _j('  <!-- 封面区块（封面/标题/副标题/徽标） -->')
    _j('  <header class="hero">')
    _j('    <div class="container">')
    _j('      <div class="badge">非官方派生 / 非医学建议</div>')
    _j(f'      <h1 class="title">HIV 治疗路径报告 · {case_name}</h1>')
    _j('      <p class="subtitle">基于病理→药效与多切面算子复合的可视化示意；展示状态变更、风险-代价与分子设计计划。</p>')
    _j('      <div class="muted">本仓库为 GROMACS 的非官方派生版，仅作技术演示。</div>')
    _j('    </div>')
    _j('  </header>')
    _j('  <main class="container">')
    _j('    <!-- 目录区块（页面快速跳转） -->')
    _j('    <section class="card">')
    _j('      <h2 id="toc">目录</h2>')
    _j('      <div class="toc">')
    _j('        <a href="#overview">立体序列总览</a>')
    _j('        <a href="#modules">分模块详情</a>')
    _j('        <a href="#pharm">分子设计与模拟</a>')
    _j('        <a href="#llm_result">Gemini 评价结果</a>')
    _j('      </div>')
    _j('      <p class="muted">免责声明：本报告仅用于方法学与技术演示，不构成医学建议或临床诊断/治疗方案；亦不用于任何实际诊疗决策或药物使用指导。</p>')
    _j('    </section>')

    # Gemini 自动评价结果（如已获取）
    if gemini_result:
        _j('    <section id="llm_result" class="card">')
        _j('      <h2>Gemini 评价结果</h2>')
        _j('      <p class="muted">以下内容由 Google Gemini API 自动生成，供方法学参考，不构成医学建议。</p>')
        # 若为结构化 JSON，优先生成中文 Markdown 并渲染；否则按原文展示
        _rendered = False
        try:
            jo = json.loads(gemini_result)
            if _gemini_translate and _build_md_with_cn:
                original_json_text = json.dumps(jo, ensure_ascii=False, indent=2)
                zh = _gemini_translate(original_json_text, target="zh-CN")
                md_cn = _build_md_with_cn(original_json_text, zh, title="LLM 评价（英文/中文）")
                _j('      <div id="gemini_md_cn"></div>')
                _j('      <script>')
                _j('      (function(){')
                _j('        const MD = ' + json.dumps(md_cn, ensure_ascii=False) + ';')
                _j('        const el = document.getElementById("gemini_md_cn");')
                _j('        if (window.marked && el) { el.innerHTML = marked.parse(MD); } else { el.textContent = MD; }')
                _j('      })();')
                _j('      </script>')
                _rendered = True
        except Exception:
            _rendered = False
        if not _rendered:
            esc = gemini_result.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            _j('      <pre>' + esc + '</pre>')
        _j('    </section>')
    elif api_key is None:
        _j('    <section id="llm_result" class="card">')
        _j('      <h2>Gemini 评价结果</h2>')
        _j('      <p class="muted">未检测到环境变量 GEMINI_API_KEY/GOOGLE_API_KEY，已跳过自动评价。</p>')
        _j('    </section>')

    # 总览
    _j('    <!-- 概览区块：展示各模块序列（chips）与案例说明 -->')
    _j('    <section id="overview" class="card">')
    _j("      <h2>立体序列总览</h2>")
    if case.get("description"):
        desc = str(case.get("description", ""))
        _j("      <p class=\"muted\">" + desc + "</p>")
    if case.get("notes"):
        notes = str(case.get("notes", ""))
        _j("      <p class=\"muted\">" + notes + "</p>")
    _j('      <div class="grid">')
    for mod in modules:
        if mod not in seqs:
            continue
        chips = ''.join([f"<span class='chip'>{step}</span>" for step in seqs[mod]])
        _j('        <div class="card half">')
        _j(f"          <h3>{mod}</h3>")
        _j(f'          <div class="chips">{chips}</div>')
        _j('        </div>')
    _j('      </div>')
    _j('      <h3>名词与符号注释</h3>')
    _j('      <ul>')
    _j('        <li>模块缩写：PEM（病理演化）、PDEM（药效效应）、PKTM（药代转运/ADME）、PGOM（药理基因组/通路）、TEM（毒理效应）、PRM（生理调控/稳态）、IEM（免疫效应）。</li>')
    _j('        <li>指标：B=Benefit（效益/负荷）、P=Perimeter（边界/周长，示意）、F=Fidelity（保真度）、N=Components（组件数）。</li>')
    _j('        <li>符号：s0→s1 表示初始到变化后状态；Δ 表示增量；Risk=风险函数；ActionCost=操作序列代价（示意）。</li>')
    _j('      </ul>')
    _j('    </section>')

    # 分模块详情 + 图表
    _j('    <!-- 分模块详情：指标卡片 + 图表（Chart.js） -->')
    _j('    <section id="modules" class="card">')
    _j("      <h2>分模块详情</h2>")
    _j('      <div class="grid">')
    for mod in modules:
        if mod not in mod_data:
            continue
        d = mod_data[mod]
        # 指标卡片
        _j('        <div class="card half">')
        _j(f"          <h3>{mod} · 指标</h3>")
        _j('          <div class="kvs">')
        for k in ["B", "P", "F", "N"]:
            _j('            <div class="kv">')
            _j(f'              <div class="k">{k} (s0 → s1)</div>')
            _j(f"              <div class=\"v\">{d['s0'][k]:.4g} → {d['s1'][k]:.4g}</div>")
            _j('            </div>')
        _j('          </div>')
        _j('          <p class="muted">注释：B=Benefit（效益/负荷）；P=Perimeter（边界/周长，示意）；F=Fidelity（保真度）；N=Components（组件数）；s0→s1=初末状态；Δ=增量；Risk=风险函数；ActionCost=操作序列代价（示意）。</p>')
        _j(f'          <canvas id="chart_metrics_{mod}" height="150"></canvas>')
        _j('        </div>')
        # 风险与代价
        _j('        <div class="card half">')
        _j(f"          <h3>{mod} · 风险与代价</h3>")
        _j('          <div class="kvs">')
        _j('            <div class="kv"><div class="k">Risk(s0)</div><div class="v">' + f"{d['risk']['s0']:.4g}" + '</div></div>')
        _j('            <div class="kv"><div class="k">Risk(s1)</div><div class="v">' + f"{d['risk']['s1']:.4g}" + '</div></div>')
        _j('            <div class="kv"><div class="k">ActionCost</div><div class="v">' + f"{d['risk']['cost']:.4g}" + '</div></div>')
        _j('          </div>')
        _j(f'          <canvas id="chart_risk_{mod}" height="150"></canvas>')
        _j("        </div>")
    _j("      </div>")
    _j("    </section>")

    # —— LLM 模板与示例已移除，直接在“Gemini 评价结果”小节展示在线结果 ——

    # 分子设计与分子模拟计划
    _j("    <!-- 药设与模拟计划：小分子设计要点与命令示例 -->")
    _j("    <section id=\"pharm\" class=\"card\">")
    _j("      <h2>分子设计与分子模拟计划</h2>")
    _j("      <h3>小分子设计意图</h3>")
    _j("      <ul>")
    _j(f"        <li>目标: {sm.get('target')}</li>")
    _j(f"        <li>机制: {sm.get('mechanism')}</li>")
    if sm.get("pharmacophore"):
        _j(f"        <li>药效团: {', '.join(sm['pharmacophore'])}</li>")
    if sm.get("scaffold"):
        _j(f"        <li>母核: {sm['scaffold']}</li>")
    if sm.get("substituent_strategy"):
        _j(f"        <li>取代策略: {', '.join(sm['substituent_strategy'])}</li>")
    if sm.get("admet_notes"):
        _j(f"        <li>ADMET备注: {', '.join(sm['admet_notes'])}</li>")
    if sm.get("tox_notes"):
        _j(f"        <li>毒理备注: {', '.join(sm['tox_notes'])}</li>")
    _j("      </ul>")
    _j("      <h3>设计要点评估（人工摘要）</h3>")
    _j("      <ul>")
    _j("        <li><strong>优势：</strong>核心药效团及母核经过验证，具高结合亲和力与特定机制（三齿金属螯合），辅以可调 pKa 叔胺优化溶解度。</li>")
    _j("        <li><strong>优势：</strong>ADMET/毒理目标明确且具前瞻性，尤其避开 BBB、CYP3A4 及 hERG，显著降低早期开发风险，提升成药性。</li>")
    _j("        <li><strong>风险：</strong>作为 IN 拮抗剂，其针对常见耐药株的效力及耐药屏障需重点评估，现有取代策略或影响耐药谱。</li>")
    _j("        <li><strong>风险：</strong>三齿金属螯合虽利于靶点结合，但需警惕潜在的非特异性金属螯合毒性或体内其他金属酶的干扰。</li>")
    _j("        <li><strong>改进：</strong>叔胺结构需更精细设计，在保证溶解度的同时，彻底规避 hERG、其他 CYP 代谢及磷脂沉积等潜在脱靶风险。</li>")
    _j("        <li><strong>挑战：</strong>实现高溶解度、低 BBB 渗透、低 CYP3A4 代谢及低 hERG 活性的多参数平衡优化，可能存在结构—活性/ADMET 间的权衡。</li>")
    _j("      </ul>")

    # 基于小分子设计要点的 Gemini 简评（若可用）
    if api_key and not os.environ.get("NO_GEMINI"):
        try:
            design_lines = []
            design_lines.append(f"目标: {sm.get('target')}")
            design_lines.append(f"机制: {sm.get('mechanism')}")
            if sm.get("pharmacophore"): design_lines.append(f"药效团: {', '.join(sm['pharmacophore'])}")
            if sm.get("scaffold"): design_lines.append(f"母核: {sm.get('scaffold')}")
            if sm.get("substituent_strategy"): design_lines.append(f"取代策略: {', '.join(sm['substituent_strategy'])}")
            if sm.get("admet_notes"): design_lines.append(f"ADMET备注: {', '.join(sm['admet_notes'])}")
            if sm.get("tox_notes"): design_lines.append(f"毒理备注: {', '.join(sm['tox_notes'])}")
            design_text = "\n".join(design_lines)
            prompt_design = (
                    "请以药物化学/ADMET/毒理视角，对下述小分子设计要点做中文要点式简评：\n"
                    "- 给出3-6条结论，涵盖优势、潜在风险与改进建议；\n"
                    "- 语言精炼，避免套话；仅输出条目列表；\n\n"
                    + design_text
            )
            gemini_design_eval = _gemini_generate(prompt_design, api_key=api_key)
            _j("      <h3>Gemini 评价（小分子设计）</h3>")
            _j('      <div id="design_eval_md"></div>')
            _j('      <script>')
            _j('      (function(){')
            _j('        const MD = ' + json.dumps(str(gemini_design_eval), ensure_ascii=False) + ';')
            _j('        const el = document.getElementById("design_eval_md");')
            _j('        if (window.marked && el) { el.innerHTML = marked.parse(MD); } else { el.textContent = MD; }')
            _j('      })();')
            _j('      </script>')
        except Exception:
            _j("      <h3>Gemini 评价（小分子设计）</h3>")
            _j("      <p class=\"muted\">Gemini 请求失败，已跳过。</p>")
    elif not api_key:
        _j("      <h3>Gemini 评价（小分子设计）</h3>")
        _j("      <p class=\"muted\">未检测到 GEMINI_API_KEY/GOOGLE_API_KEY，未执行在线简评。</p>")

    def _block(title: str, cmds: List[str]) -> None:
        _j(f"      <h3>{title}</h3>")
        _j("      <pre>")
        for c in cmds:
            _j("" + c.replace("<", "&lt;").replace(">", "&gt;") + "")
        _j("      </pre>")

    _block("退化分子对接（命令方案）", plan["docking"]["commands"])  # type: ignore
    _block("经典分子动力学（命令方案）", plan["md"]["commands"])  # type: ignore
    _block("QM/MM 占位（命令草案）", plan["qmmm"]["commands"])  # type: ignore

    _j("    </section>")

    _j("    <footer>© 2010– GROMACS Authors · © 2025 GaoZheng · GPL-3.0-only · 本仓库为非官方派生，报告仅用于演示。</footer>")
    _j("  </main>")

    # 图表脚本：逐模块渲染
    _j("  <script>")
    _j("    const MOD_DATA = ")
    _j(json.dumps(mod_data, ensure_ascii=False))
    _j(";")
    _j("    const makeBar = (id, labels, s0, s1) => {\n      const el = document.getElementById(id); if(!el) return;\n      new Chart(el.getContext('2d'), {type:'bar', data:{labels, datasets:[{label:'s0', data:s0, backgroundColor:'rgba(54,162,235,.6)'},{label:'s1', data:s1, backgroundColor:'rgba(39,174,96,.7)'}]}, options:{plugins:{legend:{labels:{color:'#cfe0ff'}}}, scales:{x:{ticks:{color:'#9fb0d0'}}, y:{ticks:{color:'#9fb0d0'}, grid:{color:'rgba(255,255,255,.08)'}}}}});\n    };")
    _j("    const makeRisk = (id, r0, r1, cost) => {\n      const el = document.getElementById(id); if(!el) return;\n      new Chart(el.getContext('2d'), {type:'bar', data:{labels:['Risk(s0)','Risk(s1)','Cost'], datasets:[{label:'值', data:[r0,r1,cost], backgroundColor:['#e67e22','#27ae60','#8e44ad']}]}, options:{plugins:{legend:{display:false}}, scales:{x:{ticks:{color:'#9fb0d0'}}, y:{ticks:{color:'#9fb0d0'}, grid:{color:'rgba(255,255,255,.08)'}}}}});\n    };")
    _j("    for (const [mod, d] of Object.entries(MOD_DATA)) {\n      const labels=['B','P','F'];\n      makeBar(`chart_metrics_${mod}`, labels, [d.s0.B, d.s0.P, d.s0.F], [d.s1.B, d.s1.P, d.s1.F]);\n      makeRisk(`chart_risk_${mod}`, d.risk.s0, d.risk.s1, d.risk.cost);\n    }")
    _j("  </script>")

    _j("</body>")
    _j("</html>")

    # 返回统一 LF 的完整 HTML 文本
    return ("\n").join(L) + "\n"


def run_case(case_name: str = "HIV_Therapy_Path", *, pharm_cfg_path: str | None = None) -> None:
    cw = load_crosswalk()
    case = cw.get("case_packages", {}).get(case_name)
    if not case:
        raise SystemExit(f"未找到案例包：{case_name}")

    seqs: Dict[str, List[str]] = case["sequences"]
    states = default_states()

    # 控制台输出简要检查信息
    print(f"== 案例包: {case_name}")
    if case.get("description"):
        print("描述:", case["description"])  # type: ignore
    if case.get("notes"):
        print("说明:", case["notes"])  # type: ignore

    # 生成 HTML 报告
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"{case_name}_report.html")
    html = gen_html_report(case_name, cw, states, seqs, pharm_cfg_path)
    # 明确以 UTF-8 + LF 写入（仓库最高规范）
    with open(html_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"HTML 报告已生成: {html_path}")


if __name__ == "__main__":
    run_case()

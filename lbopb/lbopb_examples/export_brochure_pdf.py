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
基于现有 HTML 的结构化信息，生成“商业宣传风格”的 PDF：
- simple 模式：不依赖原页面样式/JS，纯 Python 重新排版；图表由 matplotlib 生成；
- browser 模式：可选，使用 Playwright 保留原 HTML 的渲染（包含 JS 图表）。

依赖：
- simple：pip install reportlab matplotlib
- browser：pip install playwright && python -m playwright install chromium

用法：
  python lbopb/lbopb_examples/export_brochure_pdf.py \
    --html lbopb/lbopb_examples/out/HIV_Therapy_Path_report.html \
    --out  lbopb/lbopb_examples/out/HIV_Therapy_Path_brochure.pdf \
    --mode simple

说明：simple 模式为“简约风格提取重构”，不做页面灰度化、不依赖 JS 渲染；browser 模式才会注入样式。
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
import re

# ReportLab (纯 Python 导出 PDF，简约重排，无 JS)
_HAVE_REPORTLAB = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
        Image, Preformatted, ListFlowable, ListItem
    )
    from reportlab.lib.utils import ImageReader

    _HAVE_REPORTLAB = True
except Exception:
    # 允许缺失；缺失时 simple 模式将自动降级为 browser 模式
    _HAVE_REPORTLAB = False


def _resolve_repo_root() -> Path:
    p = Path(__file__).resolve()
    # 尝试上溯至仓库根（包含 my_scripts/、lbopb/ 等）
    for up in [3, 4, 5]:
        root = p.parents[up] if len(p.parents) > up else p.parent
        if (root / "lbopb").is_dir():
            return root
    return p.parents[2]


def _register_cjk_font() -> str:
    """注册并返回一个可显示中文的字体名。
    优先顺序：仿宋(仿宋_GB2312) → 黑体(SimHei) → 楷体(KaiTi) → 内置 STSong-Light → Helvetica
    注：ReportLab 对 .ttc 支持有限，优先使用 .ttf。
    """
    candidates = [
        ("FangSong", [
            r"C:\\Windows\\Fonts\\simfang.ttf",
            r"C:/Windows/Fonts/simfang.ttf",
        ]),
        ("SimHei", [
            r"C:\\Windows\\Fonts\\simhei.ttf",
            r"C:/Windows/Fonts/simhei.ttf",
        ]),
        ("KaiTi", [
            r"C:\\Windows\\Fonts\\simkai.ttf",
            r"C:/Windows/Fonts/simkai.ttf",
        ]),
    ]
    for name, paths in candidates:
        for p in paths:
            if os.path.exists(p):
                try:
                    if name not in pdfmetrics.getRegisteredFontNames():
                        pdfmetrics.registerFont(TTFont(name, p))
                    return name
                except Exception:
                    continue
    # 内置 CID 中文字体（无需本地 TTF）
    try:
        if "STSong-Light" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"
    except Exception:
        pass
    return "Helvetica"


def _extract_overview_sequences(html_text: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    # 查找模块 + 芯片序列
    pattern = re.compile(r"<h3>([a-z]+)</h3>\s*<div class=\"chips\">(.*?)</div>", re.S)
    for m in pattern.finditer(html_text):
        mod = m.group(1)
        chips_html = m.group(2)
        chips = re.findall(r"<span class='chip'>(.*?)</span>", chips_html)
        out.append((mod, chips))
    return out


def _extract_small_molecule_points(html_text: str) -> list[str]:
    pts: list[str] = []
    sec = re.search(r"<h3>小分子设计意图</h3>\s*<ul>(.*?)</ul>", html_text, re.S)
    if not sec:
        return pts
    ul = sec.group(1)
    for li in re.findall(r"<li>(.*?)</li>", ul):
        # 去除简单标签
        txt = re.sub(r"<.*?>", "", li)
        pts.append(txt.strip())
    return pts


def _parse_modules(html: str) -> dict:
    modules = {}
    mod_list = ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]
    for mod in mod_list:
        # 指标块：从标题起向后截取一段，提取后续 4 个 <div class="v">...</div>
        B = P = F = N = None
        m = re.search(rf"<h3>\s*{mod}\s*·\s*指标\s*</h3>", html, re.S)
        if m:
            seg = html[m.end(): m.end() + 3000]
            vals = re.findall(r"<div class=\"v\">([^<]+)</div>", seg)

            def _pair(v):
                parts = re.split(r"→|->", v)
                if len(parts) == 2:
                    try:
                        return float(parts[0].strip()), float(parts[1].strip())
                    except Exception:
                        return None
                return None

            if len(vals) >= 4:
                B = _pair(vals[0])
                P = _pair(vals[1])
                F = _pair(vals[2])
                N = _pair(vals[3])
        # 风险块：同理提取后续 3 个值
        R0 = R1 = AC = None
        r = re.search(rf"<h3>\s*{mod}\s*·\s*风险与代价\s*</h3>", html, re.S)
        if r:
            seg = html[r.end(): r.end() + 2000]
            vals = re.findall(r"<div class=\"v\">([^<]+)</div>", seg)
            try:
                if len(vals) >= 3:
                    R0 = float(vals[0].strip())
                    R1 = float(vals[1].strip())
                    AC = float(vals[2].strip())
            except Exception:
                pass
        if any(x is not None for x in [B, P, F, N, R0, R1, AC]):
            modules[mod] = {"B": B, "P": P, "F": F, "N": N, "Risk0": R0, "Risk1": R1, "ActionCost": AC}
    return modules


def _matplotlib_image(fig_maker):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 确保中文字体与负号显示
        try:
            matplotlib.rcParams['font.sans-serif'] = [
                'FangSong', 'SimSun', 'NSimSun', 'SimHei', 'Microsoft YaHei',
                'Noto Sans CJK SC', 'Source Han Sans SC', 'Noto Sans CJK',
                'WenQuanYi Zen Hei', 'Arial Unicode MS', 'DejaVu Sans'
            ]
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
    except Exception:
        return None
    from io import BytesIO
    buf = BytesIO()
    try:
        fig = fig_maker()
        fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
        return None


def _export_simple_pdf(in_html: Path, out_pdf: Path) -> None:
    # 读取并抽取结构化数据
    html = in_html.read_text(encoding="utf-8", errors="ignore")
    title = "报告"
    t = re.search(r"<title>(.*?)</title>", html, re.S)
    if t:
        title = re.sub(r"\s+", " ", t.group(1)).strip()
    overview = _extract_overview_sequences(html)
    sm_points = _extract_small_molecule_points(html)
    modules = _parse_modules(html)

    # 辅助：提取 Gemini JSON 与“小分子设计”MD 文本
    def _extract_gemini_json(html_text: str):
        # 查找『Gemini 评价结果』附近的 JSON（可能在 ```json 代码块中）
        m = re.search(r"<h2>\s*Gemini\s*评价结果\s*</h2>.*?<pre>(.*?)</pre>", html_text, re.S)
        if not m:
            return None
        block = m.group(1)
        block = re.sub(r"^```json\s*|\s*```$", "", block.strip())
        try:
            return json.loads(block)
        except Exception:
            # 去除 HTML 实体与多余标签再试
            cleaned = re.sub(r"<.*?>", "", block)
            cleaned = cleaned.replace("&quot;", '"').replace("&amp;", "&")
            try:
                return json.loads(cleaned)
            except Exception:
                return None

    def _extract_design_eval_md(html_text: str) -> str | None:
        # 在脚本中查找 const MD = "..." 的多行字符串
        m = re.search(r"const\s+MD\s*=\s*\"([\s\S]*?)\";", html_text)
        if not m:
            return None
        s = m.group(1)
        s = s.replace("\\n", "\n")
        return s

    gem_json = _extract_gemini_json(html)
    design_md = _extract_design_eval_md(html)

    # 文本清洗：去除所有 Markdown 加粗标记 **（代码块不处理）
    def _strip_bold_markers(s: str) -> str:
        try:
            return s.replace("**", "")
        except Exception:
            return s

    # 字体与样式
    font_name = _register_cjk_font()
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = font_name
    styles["Normal"].fontSize = 10
    styles["Heading1"].fontName = font_name
    styles["Heading2"].fontName = font_name
    styles["Heading3"].fontName = font_name
    for k in ("Heading1", "Heading2", "Heading3"):
        styles[k].textColor = colors.black
    # 免责声明样式
    disclaimer = ParagraphStyle(
        name="Disclaimer",
        parent=styles["Normal"],
        textColor=colors.black,
        leading=14,
        spaceBefore=4,
        spaceAfter=8,
    )
    h1, h2, h3, normal = styles["Heading1"], styles["Heading2"], styles["Heading3"], styles["Normal"]

    story = []
    # 封面
    story.append(Paragraph(title, h1))
    story.append(Spacer(1, 6))
    story.append(Paragraph("（简约重排 · 由 Python 生成图表 · 不依赖原页面样式）", normal))
    # 强化免责声明（前置强调）
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "免责声明：本报告仅用于方法学与技术演示，不构成医学建议或临床诊断/治疗方案；亦不用于任何实际诊疗决策或药物使用指导。",
        disclaimer,
    ))
    story.append(Spacer(1, 14))

    # 目录（简化为章节顺序）
    story.append(Paragraph("目录", h2))
    story.append(Paragraph("1. 原文主体重构", normal))
    story.append(Paragraph("2. 序列总览", normal))
    story.append(Paragraph("3. 方法学命令方案", normal))
    story.append(Paragraph("4. 附：汇总图表", normal))
    story.append(Paragraph("5. 模块指标与风险摘要", normal))
    if sm_points or design_md:
        story.append(Paragraph("6. 小分子设计要点/评价", normal))
    story.append(Spacer(1, 12))

    # 1) 原文主体重构（前置）
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main") or soup.body or soup

        def _text_of(el):
            txt = (el.get_text(" ", strip=True) or "").replace("\u00A0", " ")
            return _strip_bold_markers(txt)

        def _handle_list(ul, ordered=False):
            items = []
            for li in ul.find_all("li", recursive=False):
                items.append(ListItem(Paragraph(_text_of(li), normal)))
            if items:
                story.append(ListFlowable(items, bulletType='1' if ordered else 'bullet'))
                story.append(Spacer(1, 6))

        def _handle_table(tbl):
            rows = []
            for tr in tbl.find_all("tr", recursive=False):
                cells = []
                for td in tr.find_all(["th", "td"], recursive=False):
                    cells.append(_text_of(td))
                if cells:
                    rows.append(cells)
            if rows:
                t = Table(rows)
                t.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("LINEABOVE", (0, 0), (-1, -1), 0.25, colors.black),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.black),
                ]))
                story.append(t)
                story.append(Spacer(1, 6))

        def _handle_img(img):
            src = img.get("src")
            if not src:
                return
            try:
                from io import BytesIO
                data = None
                if src.startswith("data:image/"):
                    import base64
                    head, b64 = src.split(",", 1)
                    data = BytesIO(base64.b64decode(b64))
                else:
                    from urllib.parse import urlparse
                    u = urlparse(src)
                    if u.scheme in ("http", "https"):
                        import urllib.request
                        with urllib.request.urlopen(src, timeout=10) as r:
                            data = BytesIO(r.read())
                    else:
                        p = (in_html.parent / src).resolve()
                        if p.exists():
                            data = BytesIO(p.read_bytes())
                if data:
                    story.append(Image(data, width=420, height=None))
                    story.append(Spacer(1, 6))
            except Exception:
                pass

        story.append(Paragraph("1. 原文主体重构", h2))
        # 尝试将 Gemini JSON 转为文档块
        if gem_json and isinstance(gem_json, dict):
            story.append(Paragraph("Gemini 评价（结构化）", h3))
            if 'summary' in gem_json:
                story.append(Paragraph(_strip_bold_markers(str(gem_json.get('summary', ''))), normal))
            # 指标
            kvs = []
            for k in ['coherence_score', 'feasibility_score', 'risk_score', 'cost_score', 'confidence']:
                if k in gem_json:
                    kvs.append([k, str(gem_json.get(k))])
            if kvs:
                t = Table([['指标', '值']] + kvs)
                t.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("LINEABOVE", (0, 0), (-1, -1), 0.25, colors.black),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.black),
                ]))
                story.append(t)
                story.append(Spacer(1, 6))
            # 列表型
            for name in ['risk_flags', 'signals', 'top_actions', 'caveats']:
                arr = gem_json.get(name)
                if isinstance(arr, list) and arr:
                    story.append(Paragraph(name, h3))
                    items = [ListItem(Paragraph(_strip_bold_markers(str(x)), normal)) for x in arr]
                    story.append(ListFlowable(items, bulletType='bullet'))
                    story.append(Spacer(1, 6))

        # 若“Gemini 评价（小分子设计）”通过 JS 注入，补全为文档
        if design_md:
            story.append(Paragraph("Gemini 评价（小分子设计）", h3))
            for line in design_md.splitlines():
                s = line.strip()
                if not s:
                    continue
                if s.startswith(('*', '-')):
                    story.append(
                        ListFlowable([ListItem(Paragraph(_strip_bold_markers(s.lstrip('*- ').strip()), normal))],
                                     bulletType='bullet'))
                else:
                    story.append(Paragraph(_strip_bold_markers(s), normal))
            story.append(Spacer(1, 8))

        # 遍历 main 的直接子元素
        for el in main.children:
            if getattr(el, 'name', None) is None:
                continue
            name = el.name.lower()
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                lvl = int(name[1])
                style = h1 if lvl == 1 else h2 if lvl == 2 else h3 if lvl == 3 else normal
                story.append(Paragraph(_strip_bold_markers(_text_of(el)), style))
                story.append(Spacer(1, 6))
            elif name == "p":
                story.append(Paragraph(_strip_bold_markers(_text_of(el)), normal))
                story.append(Spacer(1, 4))
            elif name == "pre":
                txt = el.get_text("\n")
                # 如为 JSON，则已在前文结构化渲染；这里仍保留原始代码块供核对
                story.append(Preformatted(txt, normal))
                story.append(Spacer(1, 6))
            elif name == "ul":
                items = [ListItem(Paragraph(_strip_bold_markers(_text_of(li)), normal)) for li in
                         el.find_all('li', recursive=False)]
                if items:
                    story.append(ListFlowable(items, bulletType='bullet'))
                    story.append(Spacer(1, 6))
            elif name == "ol":
                items = [ListItem(Paragraph(_strip_bold_markers(_text_of(li)), normal)) for li in
                         el.find_all('li', recursive=False)]
                if items:
                    story.append(ListFlowable(items, bulletType='1'))
                    story.append(Spacer(1, 6))
            elif name == "table":
                _handle_table(el)
            elif name == "img":
                _handle_img(el)
        story.append(Spacer(1, 12))
    except Exception:
        pass

    # 2) 序列总览（表格）
    story.append(Paragraph("2. 序列总览", h2))
    data = [["模块", "序列（→ 分隔）"]]
    for mod, chips in overview:
        data.append([mod, " → ".join(chips)])
    tbl = Table(data, repeatRows=1, colWidths=[80, 400])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("LINEBEFORE", (0, 0), (-1, -1), 0.5, colors.black),
        ("LINEAFTER", (0, 0), (-1, -1), 0.5, colors.black),
        ("LINEABOVE", (0, 0), (-1, -1), 0.5, colors.black),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    # 3) 方法学命令方案（确保代码排版）
    story.append(Paragraph("3. 方法学命令方案", h2))
    cmd_docking = (
        "退化分子对接（命令方案）\n"
        "# 生成随机姿势并打包为 TRR（伪指令，需对接构建工具）\n"
        "python gen_poses.py --receptor protein.pdb --ligand ligand.sdf --out out/docking\\poses.trr\n"
        "# rerun 评估（示例命令）\n"
        "gmx mdrun -s topol.tpr -rerun out/docking\\poses.trr -g out/docking/rerun.log\n"
        "python score_rerun.py --log out/docking/rerun.log --out out/docking\\poses.scores.csv\n"
    )
    story.append(Preformatted(cmd_docking, normal))
    story.append(Spacer(1, 6))
    cmd_md = (
        "经典分子动力学（命令方案）\n"
        "gmx grompp -f md.mdp -c system.gro -p topol.top -o out/md/topol.tpr\n"
        "gmx mdrun -deffnm out/md/md\n"
    )
    story.append(Preformatted(cmd_md, normal))
    story.append(Spacer(1, 6))
    cmd_qmmm = (
        "QM/MM 占位（命令草案）\n"
        "# 准备 QM/MM 输入（片段）: qmmm.inp\n"
        "# 示例：调用 CP2K/ORCA 进行 QM 区域能量/力评估并回填到 MD 步进\n"
    )
    story.append(Preformatted(cmd_qmmm, normal))
    story.append(Spacer(1, 10))

    # 4) 附：汇总图表（由 matplotlib 生成）
    order = ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]
    mods = [m for m in order if m in modules]

    # 图A：ΔB（s1 - s0）
    def _fig_deltaB():
        import matplotlib.pyplot as plt
        vals = []
        for m in mods:
            b = modules[m].get("B")
            if b and isinstance(b, tuple):
                vals.append(b[1] - b[0])
            else:
                vals.append(0.0)
        colors_bar = ['#2ecc71' if v >= 0 else '#e74c3c' for v in vals]
        fig, ax = plt.subplots(figsize=(7.2, 3))
        ax.bar(range(len(mods)), vals, color=colors_bar)
        ax.set_xticks(range(len(mods)))
        ax.set_xticklabels(mods)
        ax.set_ylabel('ΔB (s1 - s0)')
        ax.set_title('模块 ΔB 概览')
        ax.grid(True, axis='y', alpha=0.3)
        return fig

    imgA = _matplotlib_image(_fig_deltaB)
    if imgA:
        story.append(Paragraph("4.1 模块 ΔB 概览", h3))
        story.append(Image(imgA, width=480, height=200))
        story.append(Spacer(1, 10))

    # 图B：Risk(s0) 与 Risk(s1)
    def _fig_risk():
        import matplotlib.pyplot as plt
        r0 = [];
        r1 = []
        for m in mods:
            r0.append(modules[m].get("Risk0") or 0.0)
            r1.append(modules[m].get("Risk1") or 0.0)
        x = list(range(len(mods)))
        w = 0.38
        fig, ax = plt.subplots(figsize=(7.2, 3))
        ax.bar([i - w / 2 for i in x], r0, width=w, label='Risk(s0)', color='#95a5a6')
        ax.bar([i + w / 2 for i in x], r1, width=w, label='Risk(s1)', color='#34495e')
        ax.set_xticks(x)
        ax.set_xticklabels(mods)
        ax.set_ylabel('Risk')
        ax.set_title('模块风险对比')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        return fig

    imgB = _matplotlib_image(_fig_risk)
    if imgB:
        story.append(Paragraph("4.2 模块风险对比", h3))
        story.append(Image(imgB, width=480, height=200))
        story.append(Spacer(1, 10))

    # 图C：ActionCost
    def _fig_cost():
        import matplotlib.pyplot as plt
        ac = [(modules[m].get("ActionCost") or 0.0) for m in mods]
        fig, ax = plt.subplots(figsize=(7.2, 3))
        ax.bar(range(len(mods)), ac, color='#7f8c8d')
        ax.set_xticks(range(len(mods)))
        ax.set_xticklabels(mods)
        ax.set_ylabel('ActionCost')
        ax.set_title('操作代价概览')
        ax.grid(True, axis='y', alpha=0.3)
        return fig

    imgC = _matplotlib_image(_fig_cost)
    if imgC:
        story.append(Paragraph("4.3 操作代价概览", h3))
        story.append(Image(imgC, width=480, height=200))
        story.append(Spacer(1, 12))

    # 5) 模块指标与风险摘要（表格）
    story.append(Paragraph("5. 模块指标与风险摘要", h2))
    data = [["模块", "B s0", "B s1", "P s0", "P s1", "F s0", "F s1", "N s0", "N s1", "Risk0", "Risk1", "ActionCost"]]
    for m in mods:
        md = modules[m]

        def vpair(x):
            return (x[0], x[1]) if (isinstance(x, tuple) and len(x) == 2) else (None, None)

        b0, b1 = vpair(md.get("B"));
        p0, p1 = vpair(md.get("P"));
        f0, f1 = vpair(md.get("F"));
        n0, n1 = vpair(md.get("N"))
        row = [m, b0, b1, p0, p1, f0, f1, n0, n1, md.get("Risk0"), md.get("Risk1"), md.get("ActionCost")]
        data.append(row)
    tbl2 = Table(data, repeatRows=1)
    tbl2.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("LINEABOVE", (0, 0), (-1, -1), 0.25, colors.black),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.black),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    story.append(tbl2)
    story.append(Spacer(1, 12))

    # 6) 小分子设计要点/评价（如有）
    if sm_points or design_md:
        story.append(Paragraph("6. 小分子设计要点/评价", h2))
        for p in sm_points:
            story.append(Paragraph("• " + p, normal))
        story.append(Spacer(1, 10))

    # 5) 原文主体重构（尽量覆盖文本/列表/表格/图片）
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main") or soup.body or soup

        def _text_of(el):
            return (el.get_text(" ", strip=True) or "").replace("\u00A0", " ")

        def _handle_list(ul, ordered=False):
            items = []
            for li in ul.find_all("li", recursive=False):
                items.append(ListItem(Paragraph(_text_of(li), normal)))
            if items:
                story.append(ListFlowable(items, bulletType='1' if ordered else 'bullet'))
                story.append(Spacer(1, 6))

        def _handle_table(tbl):
            rows = []
            for tr in tbl.find_all("tr", recursive=False):
                cells = []
                for td in tr.find_all(["th", "td"], recursive=False):
                    cells.append(_text_of(td))
                if cells:
                    rows.append(cells)
            if rows:
                t = Table(rows)
                t.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("LINEABOVE", (0, 0), (-1, -1), 0.25, colors.black),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.black),
                ]))
                story.append(t)
                story.append(Spacer(1, 6))

        def _handle_img(img):
            src = img.get("src")
            if not src:
                return
            try:
                from io import BytesIO
                data = None
                if src.startswith("data:image/"):
                    import base64
                    head, b64 = src.split(",", 1)
                    data = BytesIO(base64.b64decode(b64))
                else:
                    from urllib.parse import urlparse
                    u = urlparse(src)
                    if u.scheme in ("http", "https"):
                        import urllib.request
                        with urllib.request.urlopen(src, timeout=10) as r:
                            data = BytesIO(r.read())
                    else:
                        p = (in_html.parent / src).resolve()
                        if p.exists():
                            data = BytesIO(p.read_bytes())
                if data:
                    story.append(Image(data, width=420, height=None))
                    story.append(Spacer(1, 6))
            except Exception:
                pass

        story.append(PageBreak())
        story.append(Paragraph("原文主体重构", h2))
        # 遍历 main 的直接子元素，尽量保持顺序
        for el in main.children:
            if getattr(el, 'name', None) is None:
                continue
            name = el.name.lower()
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                lvl = int(name[1])
                style = h1 if lvl == 1 else h2 if lvl == 2 else h3 if lvl == 3 else normal
                story.append(Paragraph(_text_of(el), style))
                story.append(Spacer(1, 6))
            elif name == "p":
                story.append(Paragraph(_text_of(el), normal))
                story.append(Spacer(1, 4))
            elif name == "pre":
                story.append(Preformatted(el.get_text("\n"), normal))
                story.append(Spacer(1, 6))
            elif name == "ul":
                _handle_list(el, ordered=False)
            elif name == "ol":
                _handle_list(el, ordered=True)
            elif name == "table":
                _handle_table(el)
            elif name == "img":
                _handle_img(el)
            elif name == "section" or name == "div":
                # 递归处理常见子块（浅递归，避免过度复杂）
                for sub in el.find_all(recursive=False):
                    if sub.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                        lvl = int(sub.name[1])
                        style = h1 if lvl == 1 else h2 if lvl == 2 else h3 if lvl == 3 else normal
                        story.append(Paragraph(_text_of(sub), style))
                        story.append(Spacer(1, 6))
                    elif sub.name == "p":
                        story.append(Paragraph(_text_of(sub), normal))
                        story.append(Spacer(1, 4))
                    elif sub.name == "pre":
                        story.append(Preformatted(sub.get_text("\n"), normal))
                        story.append(Spacer(1, 6))
                    elif sub.name == "ul":
                        _handle_list(sub, ordered=False)
                    elif sub.name == "ol":
                        _handle_list(sub, ordered=True)
                    elif sub.name == "table":
                        _handle_table(sub)
                    elif sub.name == "img":
                        _handle_img(sub)
        story.append(Spacer(1, 12))
    except Exception:
        # 忽略重构失败，不影响主体导出
        pass

    # 页眉页脚
    def _header_footer(canvas, doc):
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.black)
        w, h = A4
        canvas.drawString(28, h - 20, title)
        canvas.drawRightString(w - 28, 20, f"第 {doc.page} 页")

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=18 * 1.5, rightMargin=18 * 1.5, topMargin=18 * 1.5,
                            bottomMargin=18 * 1.5)
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


def main() -> None:
    ap = argparse.ArgumentParser(description="将 HIV HTML 报告导出为商业宣传风格 PDF")
    # 默认路径：基于仓库根
    repo = _resolve_repo_root()
    default_html = repo / "lbopb/lbopb_examples/out/HIV_Therapy_Path_report.html"
    default_out = repo / "lbopb/lbopb_examples/out/HIV_Therapy_Path_report.pdf"
    ap.add_argument("--html", default=str(default_html), help=f"输入 HTML 路径 (默认: {default_html})")
    ap.add_argument("--out", default=str(default_out), help=f"输出 PDF 路径 (默认: {default_out})")
    ap.add_argument("--wait", type=float, default=2.0, help="渲染等待秒数（仅 browser 模式用于 Chart.js 绘图）")
    ap.add_argument("--mode", choices=["simple", "browser"], default="simple",
                    help="导出引擎：simple=纯 Python 黑白简约；browser=Playwright 渲染")
    args = ap.parse_args()

    in_html = Path(args.html).resolve()
    out_pdf = Path(args.out).resolve()
    if not in_html.exists():
        print(f"[ERR] 输入 HTML 不存在: {in_html}")
        sys.exit(2)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "simple":
        # 简约重排：强制不使用 JS 渲染，不再自动降级为 browser。缺依赖时直接给出指引。
        if not _HAVE_REPORTLAB:
            print("[ERR] simple 模式需要 reportlab，请先执行：pip install reportlab")
            sys.exit(3)
        try:
            # 检查 matplotlib
            import matplotlib  # noqa: F401
            import matplotlib.pyplot  # noqa: F401
        except Exception as e:
            print("[ERR] simple 模式需要 matplotlib，请先执行：pip install matplotlib")
            print(f"详细错误: {e}")
            sys.exit(3)
        try:
            _export_simple_pdf(in_html, out_pdf)
            print(f"PDF 已生成: {out_pdf}")
            return
        except Exception as e:
            print(f"[ERR] simple 模式导出失败：{e}")
            sys.exit(3)

    # browser 模式：保留原渲染（含封面/目录/页眉页脚）
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:
        print("[ERR] 未安装 Playwright。请先执行：pip install playwright")
        print(f"详细错误: {e}")
        sys.exit(3)

    # 准备封面/目录注入脚本（在浏览器中执行）
    inject_js = """
    (function() {
      const d = document;
      // 注入全局印刷样式（楷体）
      const style = d.createElement('style');
      style.type = 'text/css';
      style.textContent = `
        @page { size: A4; margin: 16mm; }
        body {
          font-family: KaiTi, "KaiTi_GB2312", "STKaiti", "Noto Serif SC", serif !important;
          -webkit-print-color-adjust: exact; print-color-adjust: exact;
          line-height: 1.35;
          background: #ffffff !important;
          color: #111111 !important;
        }
        h1, h2, h3, h4, h5, h6 { color: #111111 !important; }
        .page-break { page-break-after: always; }
        .cover {
          height: calc(100vh - 32mm);
          display: flex; flex-direction: column; justify-content: center; align-items: center;
          background: #ffffff !important;
          border: 1px solid #222222; border-radius: 8px; padding: 24px;
        }
        .cover h1 { font-size: 28pt; margin: 0 0 8pt; }
        .cover h2 { font-size: 14pt; font-weight: 400; color: #333333; margin: 4pt 0 18pt; }
        .cover .brand { color: #444444; margin-top: 12pt; }
        .toc h2 { font-size: 16pt; margin: 0 0 6pt; }
        .toc ol { margin: 6pt 0 0 16pt; }
        .toc li { margin: 3pt 0; color: #111111; }
        /* 卡片、弱文案、页眉区域均改为黑白简约 */
        .card { background:#ffffff !important; border:1px solid #222222 !important; color:#111111 !important; }
        .muted { color:#333333 !important; }
        .hero { background:#ffffff !important; color:#111111 !important; }
        /* 图表转灰度，线条清晰 */
        canvas { filter: grayscale(100%); }
      `;
      d.head.appendChild(style);

      // 标题信息取自 <title> 与页面 h1（避免可选链，兼容旧解析）
      const titleEl = d.querySelector('title');
      const title = ((titleEl && titleEl.textContent) || '').trim() || 'HIV 治疗路径报告';
      const h1El = d.querySelector('main h2, h1');
      const h1 = ((h1El && h1El.textContent) || '').trim();

      // 封面
      const cover = d.createElement('section');
      cover.className = 'cover page-break';
      cover.innerHTML = `
        <h1>${title}</h1>
        <h2>${h1 ? h1 : '基于病理—药效切面复合的商业宣传稿'}</h2>
        <div class="brand">© 2010– GROMACS Authors · © 2025 GaoZheng · 非官方派生 · 方法学演示</div>
      `;

      // 目录：提取主文档中的 h2/h3
      const toc = d.createElement('section');
      toc.className = 'toc page-break';
      const heads = Array.from(d.querySelectorAll('main h2, main h3'));
      const items = heads.map((h, idx) => {
        if (!h.id) h.id = 'sec_' + (idx+1);
        const tag = h.tagName.toLowerCase();
        const text = (h.textContent || '').trim();
        return { level: (tag === 'h2'? 2 : 3), id: h.id, text };
      });
      const ol = d.createElement('ol');
      let html = '';
      for (const it of items) {
        const indent = it.level === 3 ? ' style="margin-left:12pt"' : '';
        html += `<li${indent}><a href="#${it.id}">${it.text}</a></li>`;
      }
      ol.innerHTML = html;
      toc.innerHTML = '<h2>目录</h2>';
      toc.appendChild(ol);

      // 将封面与目录插入到 main 前面
      const container = d.querySelector('main') || d.body;
      container.parentElement.insertBefore(toc, container);
      container.parentElement.insertBefore(cover, toc);
    }})();
    """

    with sync_playwright() as pw:
        # 确保已安装 chromium 浏览器；若缺失则自动安装后重试
        try:
            browser = pw.chromium.launch()
        except Exception as e:
            msg = str(e)
            if "Executable doesn't exist" in msg or "playwright install" in msg:
                print("[INFO] 未找到 Chromium 浏览器，正在自动安装（python -m playwright install chromium）...")
                try:
                    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
                    browser = pw.chromium.launch()
                except Exception as ee:
                    print("[ERR] 自动安装 Chromium 失败，请手动执行：python -m playwright install chromium")
                    print(f"详细错误: {ee}")
                    sys.exit(4)
            else:
                raise
        try:
            ctx = browser.new_context()
            page = ctx.new_page()
            page.goto(in_html.as_uri(), wait_until="domcontentloaded")
            # 等待 Chart.js 等异步渲染（简单延时，或可检测特定 canvas 数量）
            if args.wait and args.wait > 0:
                page.wait_for_timeout(int(args.wait * 1000))
            # 注入封面/目录/字体与打印样式
            try:
                page.evaluate(inject_js)
            except Exception as _inj_err:
                # 兼容性降级：使用不含模板字符串/箭头函数的精简版脚本再次注入
                _inject_js_fallback = (
                    "(function(){\n"
                    "  var d=document;\n"
                    "  try{\n"
                    "    var style=d.createElement('style'); style.type='text/css';\n"
                    "    style.textContent='@page { size: A4; margin: 16mm; }\\n' +\n"
                    "      'body {\\n  font-family: KaiTi, \"KaiTi_GB2312\", \"STKaiti\", \"Noto Serif SC\", serif !important;\\n  -webkit-print-color-adjust: exact; print-color-adjust: exact;\\n  line-height: 1.35;\\n  background: #ffffff !important;\\n  color: #111111 !important;\\n}\\n' +\n"
                    "      'h1, h2, h3, h4, h5, h6 { color: #111111 !important; }\\n' +\n"
                    "      '.page-break { page-break-after: always; }\\n' +\n"
                    "      '.cover { height: calc(100vh - 32mm); display: flex; flex-direction: column; justify-content: center; align-items: center; background: #ffffff !important; border: 1px solid #222222; border-radius: 8px; padding: 24px; }\\n' +\n"
                    "      '.cover h1 { font-size: 28pt; margin: 0 0 8pt; }\\n' +\n"
                    "      '.cover h2 { font-size: 14pt; font-weight: 400; color: #333333; margin: 4pt 0 18pt; }\\n' +\n"
                    "      '.cover .brand { color: #444444; margin-top: 12pt; }\\n' +\n"
                    "      '.toc h2 { font-size: 16pt; margin: 0 0 6pt; }\\n.toc ol { margin: 6pt 0 0 16pt; }\\n.toc li { margin: 3pt 0; color: #111111; }\\n' +\n"
                    "      '.card { background:#ffffff !important; border:1px solid #222222 !important; color:#111111 !important; }\\n.muted { color:#333333 !important; }\\n.hero { background:#ffffff !important; color:#111111 !important; }\\n' +\n"
                    "      'canvas { filter: grayscale(100%); }\\n';\n"
                    "    if(d.head){ d.head.appendChild(style); }\n"
                    "    var titleEl=d.querySelector('title'); var title=(titleEl&&titleEl.textContent||'').replace(/\\s+/g,' ').trim(); if(!title)title='HIV 治疗路径报告';\n"
                    "    var h1El=d.querySelector('main h2, h1'); var h1=(h1El&&h1El.textContent||'').replace(/\\s+/g,' ').trim();\n"
                    "    var cover=d.createElement('section'); cover.className='cover page-break';\n"
                    "    var h1html='<h1>'+title+'</h1>'; var h2txt=h1?h1:'基于病理—药效切面复合的商业宣传稿'; var h2html='<h2>'+h2txt+'</h2>'; var brand='<div class=\"brand\">© 2010– GROMACS Authors · © 2025 GaoZheng · 非官方派生 · 方法学演示</div>';\n"
                    "    cover.innerHTML=h1html+'\\n'+h2html+'\\n'+brand;\n"
                    "    var toc=d.createElement('section'); toc.className='toc page-break'; var heads=(function(n){return Array.prototype.slice.call(n);})(d.querySelectorAll('main h2, main h3'));\n"
                    "    var ol=d.createElement('ol'); var html='';\n"
                    "    for(var i=0;i<heads.length;i++){ var el=heads[i]; if(!el.id) el.id='sec_'+(i+1); var lvl=(el.tagName&&el.tagName.toLowerCase()==='h2'?2:3); var indent=(lvl===3)?' style=\"margin-left:12pt\"':''; var text=(el.textContent||'').replace(/\\s+/g,' ').trim(); html += '<li'+indent+'><a href=\\\'#'+el.id+'\\\'>'+text+'</a></li>'; }\n"
                    "    ol.innerHTML=html; toc.innerHTML='<h2>目录</h2>'; toc.appendChild(ol);\n"
                    "    var container=d.querySelector('main')||d.body; if(container&&container.parentElement){ container.parentElement.insertBefore(toc, container); container.parentElement.insertBefore(cover, toc);}\n"
                    "  }catch(e){}\n"
                    "})();"
                )
                try:
                    page.evaluate(_inject_js_fallback)
                    print("[INFO] 已使用精简注入脚本成功生成封面与目录。")
                except Exception as _inj_err2:
                    print(f"[WARN] 页面注入失败：{_inj_err}; 精简脚本失败：{_inj_err2}")
            # 组装页眉页脚模板（白底、楷体、小号灰字）
            try:
                title_txt = page.title() or "LBOPB 报告"
            except Exception:
                title_txt = "LBOPB 报告"
            header_tpl = (
                '<div style="width:100%; font-family:KaiTi,\'KaiTi_GB2312\',\'STKaiti\',\'Noto Serif SC\',serif;'
                ' font-size:9px; color:#666; padding:0 10mm;">'
                f'{title_txt}'
                '</div>'
            )
            footer_tpl = (
                '<div style="width:100%; font-family:KaiTi,\'KaiTi_GB2312\',\'STKaiti\',\'Noto Serif SC\',serif;'
                ' font-size:9px; color:#666; padding:0 10mm; display:flex; justify-content:space-between;">'
                '<span></span>'
                '<span>第 <span class="pageNumber"></span> / <span class="totalPages"></span> 页</span>'
                '</div>'
            )
            # 再次等待版式稳定
            page.wait_for_timeout(500)
            page.pdf(
                path=str(out_pdf),
                format="A4",
                print_background=True,
                display_header_footer=True,
                header_template=header_tpl,
                footer_template=footer_tpl,
                margin={"top": "16mm", "bottom": "16mm", "left": "12mm", "right": "12mm"},
            )
        finally:
            browser.close()

    print(f"PDF 已生成: {out_pdf}")


if __name__ == "__main__":
    main()

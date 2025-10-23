# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

"""
基于现有 HTML 报告（含 Chart.js 图表）生成“商业宣传风格”的 PDF：
- 添加“封面 + 目录（自动提取 H1/H2）”；
- 全文使用楷体优先（KaiTi/KaiTi_GB2312/STKaiti/Noto Serif SC 兜底）；
- 保留页面内的交互绘图（Chart.js）渲染结果；
- 适配 A4 纵向排版，页边距及分页优化；
- 输出至 lbopb/lbopb_examples/out。

依赖：Playwright（Chromium）
  1) pip install playwright
  2) python -m playwright install chromium

用法：
  python lbopb/lbopb_examples/export_brochure_pdf.py \
    --html lbopb/lbopb_examples/out/HIV_Therapy_Path_report.html \
    --out  lbopb/lbopb_examples/out/HIV_Therapy_Path_brochure.pdf

注意：
- 本脚本会在浏览器上下文中动态注入“封面/目录/字体/CSS”，不修改原 HTML 文件；
- 若内网/CSP 拦截 CDN，请将 HTML 中的 Chart.js 改为本地引用后再生成。
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path


def _resolve_repo_root() -> Path:
    p = Path(__file__).resolve()
    # 尝试上溯至仓库根（包含 my_scripts/、lbopb/ 等）
    for up in [3, 4, 5]:
        root = p.parents[up] if len(p.parents) > up else p.parent
        if (root / "lbopb").is_dir():
            return root
    return p.parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="将 HIV HTML 报告导出为商业宣传风格 PDF")
    # 默认路径：基于仓库根
    repo = _resolve_repo_root()
    default_html = repo / "lbopb/lbopb_examples/out/HIV_Therapy_Path_report.html"
    default_out = repo / "lbopb/lbopb_examples/out/HIV_Therapy_Path_report.pdf"
    ap.add_argument("--html", default=str(default_html), help=f"输入 HTML 路径 (默认: {default_html})")
    ap.add_argument("--out", default=str(default_out), help=f"输出 PDF 路径 (默认: {default_out})")
    ap.add_argument("--wait", type=float, default=2.0, help="渲染等待秒数（等待 Chart.js 绘图）")
    args = ap.parse_args()

    in_html = Path(args.html).resolve()
    out_pdf = Path(args.out).resolve()
    if not in_html.exists():
        print(f"[ERR] 输入 HTML 不存在: {in_html}")
        sys.exit(2)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:
        print("[ERR] 未安装 Playwright。请先执行：pip install playwright")
        print(f"详细错误: {e}")
        sys.exit(3)

    # 准备封面/目录注入脚本（在浏览器中执行）
    inject_js = """
    (function() {{
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
        }
        .page-break { page-break-after: always; }
        .cover {
          height: calc(100vh - 32mm);
          display: flex; flex-direction: column; justify-content: center; align-items: center;
          background: radial-gradient(800px 400px at 20% 10%, rgba(51,102,204,.25), transparent),
                      radial-gradient(600px 300px at 80% 0%, rgba(231,76,60,.2), transparent);
          border: 1px solid rgba(0,0,0,.06); border-radius: 8px; padding: 24px;
        }
        .cover h1 { font-size: 28pt; margin: 0 0 8pt; }
        .cover h2 { font-size: 14pt; font-weight: 400; color: #555; margin: 4pt 0 18pt; }
        .cover .brand { color: #666; margin-top: 12pt; }
        .toc h2 { font-size: 16pt; margin: 0 0 6pt; }
        .toc ol { margin: 6pt 0 0 16pt; }
        .toc li { margin: 3pt 0; }
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
            page.evaluate(inject_js)
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

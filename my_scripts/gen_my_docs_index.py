#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
import time
from pathlib import Path


ROOT = Path("my_docs")
REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "my_scripts" / "docs_whitelist.json"


def _load_excludes() -> list[str]:
    try:
        if CFG_PATH.exists():
            data = json.loads(CFG_PATH.read_text(encoding="utf-8"))
            return [str(x).replace("\\", "/").rstrip("/") for x in data.get("doc_write_exclude", [])]
    except Exception:
        pass
    return []


EX = _load_excludes()


def _is_excluded(p: Path) -> bool:
    try:
        rp = p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        rp = p.as_posix().replace("\\", "/")
    for e in EX:
        if rp == e or rp.startswith(e + "/"):
            return True
    return False


def fmt_date(ts: int | None = None) -> str:
    if ts is None:
        ts = int(time.time())
    return time.strftime("%Y年%m月%d日", time.localtime(ts))


def summarize_markdown(p: Path, max_len: int | None = None) -> str:
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return "(无法提取摘要)"
    lines = text.splitlines()

    # Skip front-matter: 标题、作者/日期、O3 注释
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        if ln.startswith("# "):
            i += 1
            continue
        if ln.startswith("- 作者：") or ln.startswith("- 日期：") or ln.startswith("日期："):
            i += 1
            continue
        if ln.startswith("#### ***注："):
            i += 1
            continue
        break

    # Prefer 摘要 section if present
    for j in range(i, len(lines)):
        if re.search(r"^\s*##\s*摘要\s*$", lines[j]):
            # collect next non-empty paragraph block
            k = j + 1
            buf: list[str] = []
            while k < len(lines) and lines[k].strip() == "":
                k += 1
            while k < len(lines) and lines[k].strip() and not lines[k].lstrip().startswith("#"):
                buf.append(lines[k].strip())
                k += 1
            if buf:
                s = re.sub(r"\s+", " ", " ".join(buf))
                if max_len is None:
                    return s
                return (s[: max_len - 1] + "…") if len(s) > max_len else s
            break

    # Fallback: 第一段正文
    buf: list[str] = []
    k = i
    while k < len(lines) and (not lines[k].strip() or lines[k].lstrip().startswith("#")):
        k += 1
    while k < len(lines) and lines[k].strip() and not lines[k].lstrip().startswith("#"):
        buf.append(lines[k].strip())
        k += 1
    s = re.sub(r"\s+", " ", " ".join(buf)) if buf else "(缺少摘要内容)"
    if max_len is None:
        return s
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


def build_index() -> str:
    header_block = (
        "# README\n\n"
        "- 作者：GaoZheng\n"
        "- 日期：2025-10-13\n\n"
        "#### ***注：O3 理论/O3 元理论/多维宇宙生成论（PFB-GNLA）相关参考请见 [作者 GaoZheng 的资料库](https://drive.google.com/drive/folders/1lrgVtvhEq8cNal0Aa0AjeCNQaRA8WERu?usp=sharing) 与 [作者 GaoZheng 主页](https://mymetamathematics.blogspot.com)***\n\n"
    )

    out: list[str] = []
    out.append("# my_docs 文档索引")
    out.append(f"日期：{fmt_date()}")
    out.append("")
    out.append("说明：本页罗列 `my_docs/dev_docs` 与 `my_docs/project_docs` 的文档路径与要点摘要。")
    out.append("")

    # Excludes note for external references
    ex_proj = [e for e in EX if e.startswith("my_docs/project_docs/")]
    if ex_proj:
        out.append("注意：以下路径为外部知识参考（只读）")
        for e in ex_proj:
            out.append(f"- `{e}`")

    # dev_docs
    dev_dir = ROOT / "dev_docs"
    out.append("## dev_docs")
    if dev_dir.exists():
        files = sorted([p for p in dev_dir.glob("*.md") if p.is_file() and not _is_excluded(p)])
    else:
        files = []
    if not files:
        out.append("- 暂无文档")
    else:
        for p in files:
            summary = summarize_markdown(p)
            out.append(f"- `{p.as_posix()}`：")
            if summary:
                out.append(f"  {summary}")
    out.append("")

    # project_docs：按时间戳前缀 + 标题排序
    proj_dir = ROOT / "project_docs"
    out.append("## project_docs")
    if proj_dir.exists():
        files = [p for p in proj_dir.glob("*.md") if p.is_file() and not _is_excluded(p)]

        def _ts_key(p: Path) -> tuple[int, str]:
            name = p.name
            ts = 10**18
            rest = name
            if "_" in name:
                pref, rest = name.split("_", 1)
                if pref.isdigit() and len(pref) == 10:
                    try:
                        ts = int(pref)
                    except Exception:
                        ts = 10**18
            return (ts, rest)

        files = sorted(files, key=_ts_key)
    else:
        files = []
    if not files:
        out.append("- 暂无文档")
    else:
        for p in files:
            summary = summarize_markdown(p)
            bullet = f"- `{p.as_posix()}`："
            out.append(bullet)
            if summary:
                out.append(f"  {summary}")
    out.append("")

    body = "\n".join(out) + "\n"
    return header_block + body


def main() -> int:
    if not ROOT.exists():
        print("my_docs not found")
        return 0
    content = build_index()
    # 写为 UTF-8 with BOM，兼容 Windows 控制台
    with open(ROOT / "README.md", "w", encoding="utf-8-sig") as f:
        f.write(content)
    print("Wrote my_docs/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

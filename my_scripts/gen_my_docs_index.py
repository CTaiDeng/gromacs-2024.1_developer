#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2025 GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.

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


def build_index_section() -> str:
    """构建 README.md 中的 `# my_docs 文档索引` 小节内容。

    仅生成索引小节文本，不包含 README 顶部说明/捐赠等固定内容，确保这些固定内容不被改动。
    """
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
    return body


def replace_or_append_index_section(readme_text: str, index_section: str) -> str:
    """在现有 README 文本中替换或追加 `# my_docs 文档索引` 小节。

    - 若存在以一级标题 `# my_docs 文档索引` 开头的小节，则替换该小节至下一个一级标题（`# `）或文件末尾。
    - 若不存在该小节，则在文末以 1 个空行间隔后追加该小节。
    """
    lines = readme_text.splitlines()
    start = -1
    for i, ln in enumerate(lines):
        if ln.strip() == "# my_docs 文档索引":
            start = i
            break
    if start == -1:
        # 直接追加
        buf: list[str] = lines[:]
        if buf and buf[-1].strip() != "":
            buf.append("")
        buf.extend(index_section.rstrip("\n").splitlines())
        buf.append("")
        return "\n".join(buf) + "\n"

    # 查找下一个一级标题作为小节结束
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].lstrip().startswith("# "):
            end = j
            break
    # 拼接新内容：保留 start 之前所有行，插入索引小节，保留 end 之后所有行
    before = lines[:start]
    after = lines[end:]
    new_section_lines = index_section.rstrip("\n").splitlines()
    out_lines = before + new_section_lines + after
    return "\n".join(out_lines) + "\n"


def main() -> int:
    if not ROOT.exists():
        print("my_docs not found")
        return 0
    index_sec = build_index_section()
    readme_path = REPO_ROOT / "README.md"
    existing = ""
    if readme_path.exists():
        existing = readme_path.read_text(encoding="utf-8")
    updated = replace_or_append_index_section(existing, index_sec)
    # 严格遵循仓库规范：UTF-8（无 BOM）+ LF
    with open(readme_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(updated)
    print("Updated README.md (# my_docs 文档索引)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


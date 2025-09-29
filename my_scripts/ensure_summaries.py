#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path("my_docs")


def summarize_text(text: str, max_len: int = 220) -> str:
    lines = text.splitlines()
    # Skip title/date/O3 note lines
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        if ln.startswith("# "):
            i += 1
            continue
        if ln.startswith("日期："):
            i += 1
            continue
        if ln.startswith("#### ***注："):
            i += 1
            continue
        break

    # Prefer section near 摘要/简介
    for j in range(i, len(lines)):
        if re.search(r"摘要|简介", lines[j]):
            # take the next non-empty paragraph block
            k = j + 1
            buf: list[str] = []
            while k < len(lines) and lines[k].strip() == "":
                k += 1
            while k < len(lines) and lines[k].strip() and not lines[k].lstrip().startswith("#"):
                buf.append(lines[k].strip())
                k += 1
            if buf:
                s = re.sub(r"\s+", " ", " ".join(buf))
                return (s[: max_len - 1] + "…") if len(s) > max_len else s
            break

    # Fallback: first non-empty paragraph from i
    k = i
    while k < len(lines) and (not lines[k].strip() or lines[k].lstrip().startswith("#")):
        k += 1
    buf: list[str] = []
    while k < len(lines) and lines[k].strip() and not lines[k].lstrip().startswith("#"):
        buf.append(lines[k].strip())
        k += 1
    s = re.sub(r"\s+", " ", " ".join(buf)) if buf else "(暂缺摘要，可后续补充)"
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


def has_summary(lines: list[str]) -> bool:
    patt_h = re.compile(r"^\s*#{2,4}\s*(摘要|简介)\s*$")
    patt_inline = re.compile(r"^(摘要|简介)[:：]", re.I)
    for ln in lines:
        if patt_h.match(ln.strip()) or patt_inline.match(ln.strip()):
            return True
    return False


def insert_summary(md: Path) -> bool:
    try:
        text = md.read_text(encoding="utf-8")
    except Exception:
        return False
    lines = text.splitlines(True)
    if has_summary(lines):
        return False

    # Find date line
    title_idx = None
    date_idx = None
    date_pat = re.compile(r"^\s*日期[:：]\s*\d{4}年\d{2}月\d{2}日\s*$")
    for i, ln in enumerate(lines):
        if title_idx is None and ln.lstrip().startswith("# "):
            title_idx = i
        if date_idx is None and date_pat.match(ln.strip()):
            date_idx = i
        if title_idx is not None and date_idx is not None:
            break

    # Generate summary text from full content (raw text)
    summary = summarize_text(text)
    block = ["## 摘要\n", summary + "\n", "\n"]

    # Determine insertion position: after date; if O3 note follows date, insert after it
    insert_pos = None
    if date_idx is not None:
        insert_pos = date_idx + 1
        # skip over O3 note if present immediately
        sig = "#### ***注："
        if insert_pos < len(lines) and lines[insert_pos].startswith(sig):
            insert_pos += 1
            # skip one empty line after note
            if insert_pos < len(lines) and lines[insert_pos].strip() == "":
                insert_pos += 1
    elif title_idx is not None:
        insert_pos = title_idx + 1
    else:
        insert_pos = 0

    lines[insert_pos:insert_pos] = block
    md.write_text("".join(lines), encoding="utf-8")
    return True


def main() -> int:
    changed = []
    for sub in ("dev_docs", "project_docs"):
        d = ROOT / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.md")):
            if insert_summary(p):
                changed.append(str(p))
    print(f"Inserted summary in {len(changed)} file(s)")
    for f in changed:
        print(" -", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


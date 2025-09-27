#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Align my_docs knowledge-base files:
1) Rename files whose name starts with `<digits>_` so that the `<digits>` equals
   the file's first Git add timestamp (Unix seconds).
2) For Markdown files, insert a date line `日期：YYYY年MM月DD日` right after the
   first H1 title, derived from the timestamp prefix.

Safe to run repeatedly (idempotent). Uses `git mv` to preserve history.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("my_docs")


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True)


def first_add_timestamp(path: Path) -> int | None:
    # First, try first add commit
    try:
        out = run_git([
            "log",
            "--follow",
            "--diff-filter=A",
            "--format=%at",
            "--",
            str(path),
        ])
    except subprocess.CalledProcessError:
        out = ""
    ts = out.strip().splitlines()[-1] if out.strip() else ""
    if not ts:
        try:
            out = run_git(["log", "--follow", "--format=%at", "--", str(path)])
        except subprocess.CalledProcessError:
            out = ""
        ts = out.strip().splitlines()[-1] if out.strip() else ""
    return int(ts) if ts.isdigit() else None


def fmt_date(ts: int) -> str:
    return time.strftime("%Y年%m月%d日", time.localtime(ts))


def ensure_date_in_markdown(md_path: Path, ts: int) -> bool:
    """Insert or normalize the date line under the first H1. Returns True if changed."""
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return False
    lines = text.splitlines(True)
    title_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("# "):
            title_idx = i
            break
    date_line = f"日期：{fmt_date(ts)}\n"
    changed = False
    if title_idx is None:
        # prepend a synthetic title from filename stem
        title = md_path.stem
        new_text = f"# {title}\n{date_line}\n" + text
        if new_text != text:
            md_path.write_text(new_text, encoding="utf-8")
            return True
        return False
    # search a small window for an existing date line and replace
    date_pat = re.compile(r"^\s*日期[:：]\s*\d{4}年\d{2}月\d{2}日\s*$")
    window_end = min(len(lines), title_idx + 5)
    for k in range(title_idx + 1, window_end):
        if date_pat.match(lines[k]):
            if lines[k] != date_line:
                lines[k] = date_line
                changed = True
            break
    else:
        # no date found; insert after title (keep an empty line spacing)
        insert_pos = title_idx + 1
        if insert_pos < len(lines) and lines[insert_pos].strip() != "":
            lines.insert(insert_pos, date_line)
            lines.insert(insert_pos + 1, "\n")
        else:
            lines.insert(insert_pos + 1, date_line)
        changed = True
    if changed:
        md_path.write_text("".join(lines), encoding="utf-8")
    return changed


def main() -> int:
    if not ROOT.exists():
        print("my_docs not found; nothing to do.")
        return 0
    renamed: list[str] = []
    dated: list[str] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if "_" not in name:
            continue
        prefix, rest = name.split("_", 1)
        if not prefix.isdigit():
            continue
        ts = first_add_timestamp(p)
        if ts is None:
            continue
        # 1) rename if needed
        new_name = f"{ts}_{rest}"
        if new_name != name:
            new_path = p.with_name(new_name)
            subprocess.check_call(["git", "mv", "-f", "--", str(p), str(new_path)])
            p = new_path
            renamed.append(str(new_path))
        # 2) ensure date line in markdown
        if p.suffix.lower() == ".md":
            if ensure_date_in_markdown(p, ts):
                dated.append(str(p))
    print(f"Renamed {len(renamed)} file(s)")
    for f in renamed:
        print(" -", f)
    print(f"Updated date in {len(dated)} markdown file(s)")
    for f in dated:
        print(" -", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

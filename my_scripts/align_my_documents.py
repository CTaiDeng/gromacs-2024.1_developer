#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Align my_docs knowledge-base files and keep docs consistent:
1) For files named as `<digits>_...`, rename so `<digits>` equals first Git add
   timestamp (Unix seconds), preserving history via `git mv`.
2) For Markdown files, insert/normalize the date line `日期：YYYY年MM月DD日`
   right below the first H1 title, derived from the timestamp used in step 1.
3) When the content contains any O3-related keywords, insert a canonical O3 note
   line right below the date line (idempotent and de-duplicated).

Special-case exemption:
- For `my_docs/project_docs` with filename timestamp prefixes in
  {1752417159..1752417168}, do NOT rename; use their filename timestamp directly
  for the date line.

Safe to run repeatedly (idempotent).
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
import json
import os


ROOTS = [Path("my_docs"), Path("my_project")]

# Config: doc write whitelist/exclude
REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "my_scripts" / "docs_whitelist.json"


def _load_whitelist_config() -> tuple[list[str], list[str]]:
    wl: list[str] = []
    ex: list[str] = []
    try:
        if CFG_PATH.exists():
            data = json.loads(CFG_PATH.read_text(encoding="utf-8"))
            wl = [str(x).replace("\\", "/").rstrip("/") for x in data.get("doc_write_whitelist", [])]
            ex = [str(x).replace("\\", "/").rstrip("/") for x in data.get("doc_write_exclude", [])]
    except Exception:
        pass
    return wl, ex


WL, EX = _load_whitelist_config()


def _rel_posix(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix().replace("\\", "/")


def _is_allowed(p: Path) -> bool:
    rp = _rel_posix(p)
    # Exclude takes precedence
    for e in EX:
        if rp == e or rp.startswith(e + "/"):
            return False
    # If whitelist provided, require match; else allow
    if WL:
        for w in WL:
            if rp == w or rp.startswith(w + "/"):
                return True
        return False
    return True

# Trigger keywords for adding O3-related reference note
O3_KEYWORDS = [
    "O3理论",
    "O3元数学理论",
    "主纤维丛版广义非交换李代数",
    "PFB-GNLA",
]

O3_NOTE = (
    "#### ***注：“O3理论/O3元数学理论/主纤维丛版广义非交换李代数(PFB-GNLA)”相关理论参见： "
    "[作者（GaoZheng）网盘分享](https://drive.google.com/drive/folders/1lrgVtvhEq8cNal0Aa0AjeCNQaRA8WERu?usp=sharing) "
    "或 [作者（GaoZheng）主页](https://mymetamathematics.blogspot.com)***\n"
)

# Exempt these filename timestamp prefixes under my_docs/project_docs
EXEMPT_PREFIXES = {
    1759156359, 1759156360, 1759156361, 1759156362, 1759156363,
    1759156364, 1759156365, 1759156366, 1759156367, 1759156368,
}


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True)


def first_add_timestamp(path: Path) -> int | None:
    """Return first add (A) commit timestamp for path, falling back to first commit."""
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
    lines = list(text.splitlines(True))
    title_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("# "):
            title_idx = i
            break
    date_line = f"日期：{fmt_date(ts)}\n"
    changed = False
    if title_idx is None:
        # Prepend a synthetic title derived from filename (drop numeric prefix)
        stem = md_path.stem
        if "_" in stem and stem.split("_", 1)[0].isdigit():
            title = stem.split("_", 1)[1]
        else:
            title = stem
        new_text = f"# {title}\n{date_line}\n" + text
        if new_text != text:
            md_path.write_text(new_text, encoding="utf-8")
            return True
        return False
    # Search a small window for an existing date line and replace
    date_pat = re.compile(r"^\s*日期[:：]\s*\d{4}年\d{2}月\d{2}日\s*$")
    window_end = min(len(lines), title_idx + 5)
    for k in range(title_idx + 1, window_end):
        if date_pat.match(lines[k].strip()):
            if lines[k] != date_line:
                lines[k] = date_line
                changed = True
            break
    else:
        # No date found; insert after title (keep an empty line spacing)
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


def contains_o3_keyword(text: str) -> bool:
    low = text.lower()
    if "pfb-gnla" in low:
        return True
    return any(k in text for k in O3_KEYWORDS if k != "PFB-GNLA")


def ensure_o3_note(md_path: Path) -> bool:
    """Ensure the O3 reference note is placed directly below the date line when
    any of the configured keywords appear. Idempotent.
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return False
    if not contains_o3_keyword(text):
        return False

    lines = text.splitlines(True)
    # Locate first H1 title
    title_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("# "):
            title_idx = i
            break
    # Find date line near title
    date_pat = re.compile(r"^\s*日期[:：]\s*\d{4}年\d{2}月\d{2}日\s*$")
    date_idx = None
    if title_idx is not None:
        window_end = min(len(lines), title_idx + 6)
        for k in range(title_idx + 1, window_end):
            if date_pat.match(lines[k].strip()):
                date_idx = k
                break

    changed = False
    if title_idx is None or date_idx is None:
        # Fallback: synthesize heading/date at top
        name = md_path.name
        ts_val = None
        if "_" in name and name.split("_", 1)[0].isdigit():
            try:
                ts_val = int(name.split("_", 1)[0])
            except Exception:
                ts_val = None
        date_line = f"日期：{fmt_date(ts_val or int(time.time()))}\n"
        title = md_path.stem
        new_lines = [f"# {title}\n", date_line, O3_NOTE, "\n"]
        new_lines.extend(lines)
        md_path.write_text("".join(new_lines), encoding="utf-8")
        return True

    # Remove existing O3 note instances (to keep single canonical copy)
    sig_substr = "O3理论/O3元数学理论/主纤维丛版广义非交换李代数(PFB-GNLA)"
    if any(sig_substr in ln for ln in lines):
        lines = [ln for ln in lines if sig_substr not in ln]
        changed = True
        # Recompute indices after removal
        title_idx = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith("# ")), None)
        date_idx = None
        if title_idx is not None:
            window_end = min(len(lines), title_idx + 6)
            for k in range(title_idx + 1, window_end):
                if date_pat.match(lines[k].strip()):
                    date_idx = k
                    break

    insert_pos = (date_idx or 0) + 1
    if insert_pos < len(lines) and lines[insert_pos].strip() == "":
        lines.insert(insert_pos, O3_NOTE)
    else:
        lines.insert(insert_pos, O3_NOTE)
        if insert_pos + 1 < len(lines) and lines[insert_pos + 1].strip() != "":
            lines.insert(insert_pos + 1, "\n")
    changed = True
    if changed:
        md_path.write_text("".join(lines), encoding="utf-8")
    return changed


def normalize_h1_prefix(md_path: Path) -> bool:
    """If the first H1 line looks like "# <digits>_<title>", drop the numeric prefix.
    Returns True if the file was modified.
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return False
    lines = text.splitlines(True)
    for i, ln in enumerate(lines):
        if ln.startswith("# "):
            m = re.match(r"^#\s+(\d{10})_(.+)$", ln.rstrip("\n"))
            if m:
                title_rest = m.group(2)
                lines[i] = f"# {title_rest}\n"
                md_path.write_text("".join(lines), encoding="utf-8")
                return True
            break
    return False

def iter_target_files() -> list[Path]:
    files: list[Path] = []
    # my_docs/**
    docs_root = ROOTS[0]
    if docs_root.exists():
        files.extend([p for p in docs_root.rglob("*") if p.is_file() and _is_allowed(p)])
    # my_project/**/docs/**
    proj_root = ROOTS[1]
    if proj_root.exists():
        for p in proj_root.rglob("*"):
            if not p.is_file():
                continue
            if "docs" in p.parts:
                if _is_allowed(p):
                    files.append(p)
    return files


def main() -> int:
    renamed: list[str] = []
    dated: list[str] = []
    noted: list[str] = []
    targets = iter_target_files()
    if not targets:
        print("No target files found under my_docs/ or my_project/**/docs")
        return 0
    for p in targets:
        name = p.name
        ts_use: int | None = None
        if "_" in name:
            prefix, rest = name.split("_", 1)
            if prefix.isdigit():
                ts_filename = int(prefix)
                exempt = (
                    p.as_posix().startswith("my_docs/project_docs/")
                    and ts_filename in EXEMPT_PREFIXES
                )
                if exempt:
                    ts_use = ts_filename
                else:
                    ts_git = first_add_timestamp(p)
                    if ts_git is not None:
                        # 1) rename if needed
                        new_name = f"{ts_git}_{rest}"
                        if new_name != name:
                            new_path = p.with_name(new_name)
                            subprocess.check_call(["git", "mv", "-f", "--", str(p), str(new_path)])
                            p = new_path
                            name = p.name
                            renamed.append(str(new_path))
                        ts_use = ts_git
        # 2) ensure date line in markdown
        if p.suffix.lower() == ".md" and ts_use is not None:
            if ensure_date_in_markdown(p, ts_use):
                dated.append(str(p))
            if normalize_h1_prefix(p):
                # treat as date-updated category for reporting simplicity
                if str(p) not in dated:
                    dated.append(str(p))
        # 3) ensure O3 note when keywords present (markdown only)
        if p.suffix.lower() == ".md":
            if ensure_o3_note(p):
                noted.append(str(p))
    print(f"Renamed {len(renamed)} file(s)")
    for f in renamed:
        print(" -", f)
    print(f"Updated date in {len(dated)} markdown file(s)")
    for f in dated:
        print(" -", f)
    print(f"Inserted O3 note in {len(noted)} markdown file(s)")
    for f in noted:
        print(" -", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Append a canonical CC BY-NC-ND 4.0 footer to Markdown docs named as
  `^\d{10}_.+\.md$`
under specified roots:
  - my_docs/project_docs (excluding kernel_reference)
  - my_project/gmx_split_20250924_011827/docs
Also copy my_docs/LICENSE.md to my_project/gmx_split_20250924_011827/docs.

Idempotent: will not append if a footer marker already exists.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERN = re.compile(r"^\d{10}_.+\.md$")
FOOTER = (
    "\n---\n"
    "**许可声明 (License)**\n\n"
    "Copyright (C) 2025 GaoZheng\n\n"
    "本文档采用[知识共享-署名-非商业性使用-禁止演绎 4.0 国际许可协议 (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.zh-Hans)进行许可。\n"
)
MARKER = "许可声明 (License)"
LINK_SNIPPET = "creativecommons.org/licenses/by-nc-nd/4.0"


def iter_target_files() -> list[Path]:
    targets: list[Path] = []
    # 1) my_docs/project_docs (exclude kernel_reference)
    proj = ROOT / "my_docs" / "project_docs"
    if proj.exists():
        for p in proj.rglob("*.md"):
            if not p.is_file():
                continue
            try:
                rel = p.resolve().relative_to(ROOT.resolve()).as_posix()
            except Exception:
                rel = p.as_posix().replace("\\", "/")
            if rel.startswith("my_docs/project_docs/kernel_reference/"):
                continue
            if PATTERN.match(p.name):
                targets.append(p)
    # 2) my_project/gmx_split_20250924_011827/docs
    proj_docs = ROOT / "my_project" / "gmx_split_20250924_011827" / "docs"
    if proj_docs.exists():
        for p in proj_docs.rglob("*.md"):
            if not p.is_file():
                continue
            if PATTERN.match(p.name):
                targets.append(p)
    return targets


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(errors="ignore")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def ensure_footer(path: Path) -> bool:
    """Append footer if not present. Returns True if file modified."""
    text = read_text(path)
    hay = text[-2000:] if len(text) > 2000 else text
    if (MARKER in hay) and (LINK_SNIPPET in hay):
        return False
    # Normalize line endings and trailing whitespace
    body = text.rstrip("\r\n") + "\n" + FOOTER
    write_text(path, body)
    return True


def copy_license_into_project() -> bool:
    src = ROOT / "my_docs" / "project_docs" / "LICENSE.md"
    dst_dir = ROOT / "my_project" / "gmx_split_20250924_011827" / "docs"
    dst = dst_dir / "LICENSE.md"
    if not src.exists() or not dst_dir.exists():
        return False
    try:
        s = read_text(src)
        d = read_text(dst) if dst.exists() else None
        if d == s:
            return False
    except Exception:
        pass
    shutil.copyfile(src, dst)
    return True


def main() -> int:
    modified = 0
    for p in iter_target_files():
        if ensure_footer(p):
            print(f"[footer] appended: {p}")
            modified += 1
    if copy_license_into_project():
        print("[footer] copied LICENSE.md into project docs")
    print(f"[footer] done. modified={modified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

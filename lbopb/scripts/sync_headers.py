# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""同步 lbopb/ 与 lbopb/lbopb_examples/ 头注：

- 仅保留本地版权行（GaoZheng），移除上游版权行；
- 确保存在 SPDX 许可标识；

用法：
  python lbopb/scripts/sync_headers.py
"""

from __future__ import annotations

import pathlib
import re
from typing import Iterable

ROOTS = [
    pathlib.Path(__file__).resolve().parents[1],
    pathlib.Path(__file__).resolve().parents[1] / 'lbopb_examples',
]

SPDX = "# SPDX-License-Identifier: GPL-3.0-only"
COPY = "# Copyright (C) 2025 GaoZheng"
UPSTREAM_RE = re.compile(r"^#\s*Copyright\s*\(C\)\s*2010-\s*The GROMACS Authors\s*$", re.I)


def iter_files() -> Iterable[pathlib.Path]:
    for root in ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            yield p


def sync_file(p: pathlib.Path) -> bool:
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = False
    # 移除上游版权行
    new_lines = [ln for ln in lines if not UPSTREAM_RE.match(ln)]
    if len(new_lines) != len(lines):
        lines = new_lines
        changed = True

    # 确保 SPDX 与本地版权行在文件顶部
    def ensure_at_top(marker: str) -> None:
        nonlocal lines, changed
        if not any(ln.strip() == marker for ln in lines[:5]):
            lines.insert(0, marker)
            changed = True

    ensure_at_top(COPY)
    ensure_at_top(SPDX)
    if changed:
        # 强制使用 UTF-8（无 BOM）+ CRLF
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
    return changed


def main() -> None:
    total = 0
    for f in iter_files():
        if sync_file(f):
            total += 1
    print(f"Synced headers in {total} files.")


if __name__ == "__main__":
    main()

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

"""同步 lbopb/ 与 lbopb/lbopb_examples/ 源码头注至统一规范。

规范模板（严格顺序与内容，保留 shebang 与编码行）：

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

用法：
  python lbopb/scripts/sync_headers.py
"""

from __future__ import annotations

import pathlib
import re
from typing import Iterable, List, Tuple

ROOTS = [
    pathlib.Path(__file__).resolve().parents[1],
    pathlib.Path(__file__).resolve().parents[1] / 'lbopb_examples',
]

SPDX = "# SPDX-License-Identifier: GPL-3.0-only"
COPY = "# Copyright (C) 2025 GaoZheng"
UPSTREAM_RE = re.compile(r"^#\s*Copyright\s*\(C\)\s*2010-\s*The GROMACS Authors\s*$", re.I)

HEADER_TEMPLATE: List[str] = [
    "# SPDX-License-Identifier: GPL-3.0-only",
    "# Copyright (C) 2025 GaoZheng",
    "#",
    "# This program is free software: you can redistribute it and/or modify",
    "# it under the terms of the GNU General Public License as published by",
    "# the Free Software Foundation, version 3 only.",
    "#",
    "# This program is distributed in the hope that it will be useful,",
    "# but WITHOUT ANY WARRANTY; without even the implied warranty of",
    "# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the",
    "# GNU General Public License for more details.",
    "#",
    "# You should have received a copy of the GNU General Public License",
    "# along with this program.  If not, see <https://www.gnu.org/licenses/>.",
    "#",
    "# --- 著作权独立性声明 (Copyright Independence Declaration) ---",
    "# 本文件（“载荷”）是作者 (GaoZheng) 的原创著作物，其知识产权",
    "# 独立于其运行平台 GROMACS（“宿主”）。",
    "# 本文件的授权遵循上述 SPDX 标识，不受“宿主”许可证的管辖。",
    "# 详情参见项目文档 \"my_docs/project_docs/1762636780_🚩🚩gromacs-2024.1_developer项目的著作权设计策略：“宿主-载荷”与“双轨制”复合架构.md\"。",
    "# ------------------------------------------------------------------",
]

CODING_RE = re.compile(r"^#.*coding[:=]\s*[-_.a-zA-Z0-9]+", re.I)


def iter_files() -> Iterable[pathlib.Path]:
    for root in ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            yield p


def sync_file(p: pathlib.Path) -> bool:
    text = p.read_text(encoding="utf-8")
    # 归一化换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    prelude: List[str] = []
    i = 0
    # shebang
    if i < len(lines) and lines[i].startswith("#!"):
        prelude.append(lines[i])
        i += 1
    # coding
    if i < len(lines) and CODING_RE.search(lines[i] or ""):
        prelude.append(lines[i])
        i += 1

    # 跳过紧随其后的空行（保持一个空行由我们控制）
    while i < len(lines) and (lines[i].strip() == ""):
        i += 1

    # 识别并移除现有头注（连续以 # 开头的注释块），仅当包含许可证/版权关键字时替换
    j = i
    while j < len(lines) and (lines[j].startswith('#') or lines[j].strip() == ''):
        j += 1
    header_block = lines[i:j]
    def looks_like_license(block: List[str]) -> bool:
        text = "\n".join(block)
        return (
            "SPDX-License-Identifier" in text
            or "GNU General Public License" in text
            or "This program is free software" in text
            or "GROMACS Authors" in text
            or "著作权独立性声明" in text
        )

    body = lines[j:] if looks_like_license(header_block) else lines[i:]

    # 清理 body 顶部可能残留的上游版权行
    body = [ln for ln in body if not UPSTREAM_RE.match(ln)]

    # 组装新内容：prelude + 规范头注 + 空行 + body（保持末尾换行）
    new_lines: List[str] = []
    new_lines.extend(prelude)
    if new_lines and new_lines[-1] != "":
        new_lines.append("")
    new_lines.extend(HEADER_TEMPLATE)
    if body and (body[0] != ""):
        new_lines.append("")
    new_lines.extend(body)

    new_text = "\n".join(new_lines).rstrip("\n") + "\n"
    changed = new_text != text + ("\n" if not text.endswith("\n") else "")
    if changed:
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(new_text)
    return changed


def main() -> None:
    total = 0
    for f in iter_files():
        if sync_file(f):
            total += 1
    print(f"Synced headers in {total} files.")


if __name__ == "__main__":
    main()

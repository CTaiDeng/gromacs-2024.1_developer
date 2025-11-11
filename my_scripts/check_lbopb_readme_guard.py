#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
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

"""
只读审查脚本（lbopb/README.md 受保护区域）

- 目标：
  1) 校验 lbopb/README.md 顶部是否存在受保护块（guard begin/end 标记）。
  2) 校验受保护块内是否包含关键法务与开发协议要点（关键词检测）。
  3) 检索 my_scripts/** 与 lbopb/scripts/** 是否存在对 lbopb/README.md 的写入型自动化风险。

- 特点：
  - 绝不对文件进行任何“写入/修复”；仅返回非零退出码提示人工介入。
  - 严格 UTF-8（无 BOM）+ LF 读取；在 Windows/WSL 场景下做容错换行归一。

用法：
  python3 my_scripts/check_lbopb_readme_guard.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
READ_ME = ROOT / "lbopb" / "README.md"
GUARD_BEGIN = "<!-- @guard-begin: lbopb-legal-guard"
GUARD_END = "<!-- @guard-end: lbopb-legal-guard"


def read_text_utf8(path: Path) -> str:
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    # 归一化换行到 LF
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_guard_block(text: str) -> Tuple[int, int, str]:
    b = text.find(GUARD_BEGIN)
    e = text.find(GUARD_END)
    if b == -1 or e == -1:
        return -1, -1, ""
    # guard-end 注释本身不计入内容，取到其前一行
    # 但此处保留更宽松：截到 end 注释之前
    return b, e, text[b:e]


def check_keywords(block: str) -> List[str]:
    problems: List[str] = []
    # 必要关键词（法律与开发协议）
    required = [
        "LBOPB 子项目：著作权与独立性声明",
        "重要法律声明",
        "宿主（GROMACS）",
        "载荷（本项目 `lbopb`）",
        "CC-BY-NC-ND 4.0",
        "GPL-3.0-only",
        "开发协议（受保护区域）",
        "审查脚本",
        "DO NOT EDIT BY SCRIPTS",
    ]
    for k in required:
        if k not in block:
            problems.append(f"受保护块缺少关键要点：{k}")
    return problems


def scan_automation_writes() -> List[str]:
    problems: List[str] = []
    # 扫描 my_scripts/** 与 lbopb/scripts/**
    targets = [ROOT / "my_scripts", ROOT / "lbopb" / "scripts"]
    for base in targets:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            try:
                txt = read_text_utf8(p)
            except Exception:
                continue
            if "lbopb/README.md" not in txt:
                continue
            # 粗略检测写入风险：出现 open(..., 'w'/'a'/'r+') 或常见就地修改命令
            risky = False
            if re.search(r"open\s*\(.*lbopb/README\.md.*,[^)]*['\"]([wa]|r\+)['\"]", txt):
                risky = True
            # 同行就地修改：仅当同一行同时出现命令与目标路径时才视为风险
            if re.search(r"^.*sed\s+-i[^\n]*lbopb/README\.md.*$", txt, re.M):
                risky = True
            if re.search(r"^.*perl\s+-pi[^\n]*lbopb/README\.md.*$", txt, re.M):
                risky = True
            if risky:
                problems.append(f"检测到可能写入/就地修改 lbopb/README.md 的自动化脚本：{p.as_posix()}")
            # 仅提及但未命中写入/就地修改模式的情况作为信息提示，不计为失败。
    return problems


def main() -> int:
    if not READ_ME.exists():
        print("[lbopb-guard] 未找到 lbopb/README.md", file=sys.stderr)
        return 2
    text = read_text_utf8(READ_ME)
    b, e, block = extract_guard_block(text)
    if b == -1 or e == -1:
        print("[lbopb-guard] 未检测到受保护块（缺少 guard-begin/end 标记）", file=sys.stderr)
        return 3
    problems: List[str] = []
    problems += check_keywords(block)
    problems += scan_automation_writes()
    if problems:
        print("[lbopb-guard] 检查发现以下问题：", file=sys.stderr)
        for m in problems:
            print(f" - {m}", file=sys.stderr)
        print("\n该脚本不进行任何自动修复；请人工审阅与修正。", file=sys.stderr)
        return 4
    print("[lbopb-guard] OK — 受保护区存在且关键要点完整；未发现写入型自动化风险。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

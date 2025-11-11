#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

"""
检查 lbopb 子项目是否：
1) 依赖并导入了 GPL 许可的 Python 包（非 LGPL）。
2) 与 GROMACS 的交互是否保持“臂长通信”（子进程/命令行），避免直接导入 gromacs/gmx/gmxapi 等库。

使用方法：
  python lbopb/scripts/check_gpl_usage_and_boundary.py [--list]

退出码：
  0 通过；
  1 发现 GPL 依赖；
  2 发现与宿主的直接库级导入；
  3 同时命中 1 和 2；
  4 运行时错误。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:  # py3.10+
    import importlib.metadata as md  # type: ignore
except Exception:  # pragma: no cover
    md = None  # type: ignore

REPO = Path(__file__).resolve().parents[2]
LBOPB_ROOT = REPO / "lbopb"
SELF_PATH = Path(__file__).resolve()

# 关键：判定 GPL 的关键词（排除 Lesser）
GPL_TOKENS = ("GPL", "GNU General Public License")
LGPL_TOKENS = ("LGPL", "Lesser General Public License")

# 受限制的“宿主库”导入名（大小写敏感按实际模块名处理）
RESTRICTED_HOST_IMPORTS: Set[str] = {
    "gmx",           # hypothetical python module
    "gromacs",       # e.g., gromacs wrappers
    "gmxapi",        # GROMACS Python API
    "MDAnalysis",    # GPL-2+ project, 用于提醒（与宿主不直接相关）
}

# 常见三方包到发行包名映射（import 名 -> distribution 名）
MODULE_TO_DIST = {
    "bs4": "beautifulsoup4",
    "playwright": "playwright",
    "reportlab": "reportlab",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "networkx": "networkx",
    "pymbar": "pymbar",
}


def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # 跳过缓存
        if "__pycache__" in p.parts:
            continue
        yield p


def collect_imports(paths: Iterable[Path]) -> Tuple[Set[str], Dict[Path, List[str]]]:
    """收集顶级 import 模块名（不含相对导入），并记录每个文件的导入清单。"""
    topmods: Set[str] = set()
    per_file: Dict[Path, List[str]] = {}
    for p in paths:
        try:
            src = p.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(src, filename=str(p))
        except Exception:
            continue
        mods: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    mods.append(name)
                    if name and name not in {"lbopb", "my_scripts"}:
                        topmods.add(name)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    # 相对导入跳过
                    continue
                if node.module:
                    name = node.module.split(".")[0]
                    mods.append(name)
                    if name and name not in {"lbopb", "my_scripts"}:
                        topmods.add(name)
        per_file[p] = mods
    return topmods, per_file


def is_gpl_license(lic: str) -> bool:
    L = lic or ""
    if any(tok.lower() in L.lower() for tok in LGPL_TOKENS):
        return False
    return any(tok.lower() in L.lower() for tok in GPL_TOKENS)


def dist_for_module(mod: str) -> List[str]:
    names: List[str] = []
    if md is not None and hasattr(md, "packages_distributions"):
        try:
            mapping = md.packages_distributions()  # type: ignore[attr-defined]
            if mod in mapping:
                names.extend(mapping[mod] or [])
        except Exception:
            pass
    # 备用映射
    if not names and mod in MODULE_TO_DIST:
        names.append(MODULE_TO_DIST[mod])
    # import 名即发行包名的常见情况
    if not names:
        names.append(mod)
    return list(dict.fromkeys(names))


def license_of_distribution(dist: str) -> Tuple[str, List[str]]:
    lic = ""
    classifiers: List[str] = []
    if md is None:
        return lic, classifiers
    try:
        meta = md.metadata(dist)
        lic = meta.get("License", "") or ""
        classifiers = [c for c in meta.get_all("Classifier") or []]
    except Exception:
        pass
    return lic, classifiers


def check_gpl_usage(mods: Set[str]) -> Tuple[bool, List[str]]:
    flagged: List[str] = []
    for m in sorted(mods):
        # 跳过常见标准库前缀
        if m in {
            "sys", "os", "re", "json", "typing", "pathlib", "subprocess", "time", "math", "random", "itertools", "dataclasses", "argparse", "hashlib", "io", "urllib", "base64",
        }:
            continue
        # 解析分发
        for dist in dist_for_module(m):
            lic, classifiers = license_of_distribution(dist)
            lic_tokens = " | ".join([lic] + classifiers)
            if lic or classifiers:
                if is_gpl_license(lic) or any(("GNU General Public License" in c and "Lesser" not in c) for c in classifiers):
                    flagged.append(f"{m} -> {dist} :: {lic_tokens}")
            # 未获取到许可信息：保守不判定
    return (len(flagged) > 0), flagged


def check_host_boundary(per_file_imports: Dict[Path, List[str]]) -> Tuple[bool, List[str], List[str]]:
    direct_import_hits: List[str] = []
    evidence_cli: List[str] = []
    for p, mods in per_file_imports.items():
        for m in mods:
            if m in RESTRICTED_HOST_IMPORTS:
                direct_import_hits.append(f"{p.as_posix()} : import {m}")
        # 粗略文本扫描：查找 gmx 命令行作为“臂长通信”证据
        if p.resolve() != SELF_PATH:
            try:
                txt = p.read_text(encoding="utf-8")
            except Exception:
                txt = ""
            if "gmx " in txt or " gromacs" in txt:
                # 截取一行示例
                for line in txt.splitlines():
                    if "gmx " in line or " gromacs" in line:
                        evidence_cli.append(f"{p.name}: {line.strip()}")
                        break
    return (len(direct_import_hits) > 0), direct_import_hits, evidence_cli


def main(argv: List[str]) -> int:
    list_only = "--list" in argv
    no_color = "--no-color" in argv
    force_color = "--force-color" in argv

    # 轻量彩色输出（无依赖），可通过 --no-color 关闭
    use_color = (sys.stdout.isatty() and not no_color) or force_color

    def color(s: str, code: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if use_color else s

    C = {
        "RED": "31", "GREEN": "32", "YELLOW": "33", "BLUE": "34", "MAGENTA": "35", "CYAN": "36",
        "BOLD": "1", "DIM": "2",
    }
    files = list(iter_py_files(LBOPB_ROOT))
    mods, per_file = collect_imports(files)

    gpl_found, gpl_items = check_gpl_usage(mods)
    host_link_found, host_hits, cli_evidence = check_host_boundary(per_file)

    print(color("[lbopb GPL usage check]", C["BOLD"]))
    print(color(f" - scanned files: {len(files)}", C["DIM"]))
    print(color(f" - unique imports: {len(mods)}", C["DIM"]))
    if list_only:
        print(color(" - imports:", C["CYAN"]))
        for m in sorted(mods):
            print(f"   * {m}")

    if gpl_found:
        print(color(" - GPL packages detected:", f"{C['BOLD']};{C['RED']}"))
        for item in gpl_items:
            print(color(f"   ! {item}", C["RED"]))
    else:
        print(color(" - GPL packages: none detected (based on local metadata)", C["GREEN"]))

    print(color("[host boundary check]", C["BOLD"]))
    if host_link_found:
        print(color(" - Direct host-library import detected:", f"{C['BOLD']};{C['RED']}"))
        for h in host_hits:
            print(color(f"   ! {h}", C["RED"]))
    else:
        print(color(" - No direct imports of gromacs/gmx/gmxapi/MDAnalysis detected", C["GREEN"]))

    if cli_evidence:
        print(color(" - Evidence of CLI usage (arm's length):", C["CYAN"]))
        for ev in cli_evidence[:5]:
            print(color(f"   + {ev}", C["BLUE"]))

    rc = 0
    if gpl_found:
        rc |= 1
    if host_link_found:
        rc |= 2
    if rc == 0:
        print(color("[OK] lbopb 未发现 GPL 依赖且与宿主保持臂长通信（基于当前环境可用的元数据与静态扫描）", f"{C['BOLD']};{C['GREEN']}"))
    else:
        print(color("[FAIL] 请根据上方 ‘!’ 项完成整改或给出许可依据", f"{C['BOLD']};{C['RED']}"))
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(4)

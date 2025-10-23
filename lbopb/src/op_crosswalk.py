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

"""LBOPB 幺半群算子联络映射加载器。

本模块加载 `lbopb/src/operator_crosswalk.json`，提供对以下结构的便捷访问：
 - 基本算子（各模块）→ 语义标签（B/P/N/F 的升降、激活/抑制/炎症/修复/ADMET 等）
 - 标签 → 跨模块可类比的基本算子（联络 crosswalk）
 - 仅由基本算子构成的“规范化算子包”（normal-form packages）

知识库引用：
 - my_docs/project_docs/1761062400_病理演化幺半群 (PEM) 公理系统.md
 - my_docs/project_docs/1761062401_生理调控幺半群 (PRM) 公理系统.md
 - my_docs/project_docs/1761062403_毒理学效应幺半群 (TEM) 公理系统.md
 - my_docs/project_docs/1761062404_药代转运幺半群 (PKTM) 公理系统.md
 - my_docs/project_docs/1761062405_药理基因组幺半群 (PGOM) 公理系统.md
 - my_docs/project_docs/1761062406_药效效应幺半群 (PDEM) 公理系统.md
 - my_docs/project_docs/1761062407_免疫效应幺半群 (IEM) 公理系统.md
 - 以及配套的“核心构造及理论完备性”系列文档（1761062408~1761062414）

注意：映射为“类比联络”，用于工程对齐与快速原型，并非严格等价替换；应结合具体公理与情境校准。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

_DEFAULT_JSON = os.path.join(os.path.dirname(__file__), "operator_crosswalk.json")


def load_crosswalk(path: Optional[str] = None) -> Dict[str, Any]:
    """加载联络映射 JSON 数据。

    - path: 可选，自定义 JSON 路径；默认使用包内 `operator_crosswalk.json`。
    返回：解析后的字典。
    """

    p = path or _DEFAULT_JSON
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def list_modules(cw: Mapping[str, Any]) -> List[str]:
    """返回纳入联络的模块列表（pem/prm/tem/pktm/pgom/pdem/iem）。"""

    return list(cw.get("modules", []))


def basic_ops(cw: Mapping[str, Any], module: str) -> Mapping[str, List[str]]:
    """获取某模块的基本算子 → 标签列表映射。"""

    return cw.get("basic_ops", {}).get(module, {})  # type: ignore[return-value]


def crosswalk_for_tag(cw: Mapping[str, Any], tag: str) -> Mapping[str, List[str]]:
    """按语义标签获取跨模块基本算子对齐（联络）。"""

    return cw.get("crosswalk_by_tag", {}).get(tag, {})  # type: ignore[return-value]


def canonical_package(cw: Mapping[str, Any], name: str) -> Mapping[str, List[str]]:
    """获取规范化算子包（仅基本算子构成）的跨模块对齐。"""

    return cw.get("canonical_packages", {}).get(name, {})  # type: ignore[return-value]


__all__ = [
    "load_crosswalk",
    "list_modules",
    "basic_ops",
    "crosswalk_for_tag",
    "canonical_package",
]

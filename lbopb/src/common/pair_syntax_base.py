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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.

from __future__ import annotations

from typing import Any, Dict, List, Tuple


RuleMsg = Tuple[str, str]  # (level: 'error'|'warning', message)


def _list_of_str(x: Any) -> bool:
    try:
        return isinstance(x, list) and all(isinstance(t, str) and t.strip() for t in x)
    except Exception:
        return False


def _adjacent_duplicates(seq: List[str]) -> bool:
    try:
        for a, b in zip(seq, seq[1:]):
            if str(a) == str(b):
                return True
    except Exception:
        return False
    return False


def check_pair(dom_a: str, dom_b: str, seq_a: List[str] | None, seq_b: List[str] | None) -> Dict[str, Any]:
    """最小可用的跨域联络语法检查。

    - error 条件：任一域序列缺失/空；元素非字符串；
    - warning 条件：总长>6；两域长度差>5；存在相邻重复；
    - result：无 error 且无 warning → "通过"；无 error 但有 warning → "警告"；有 error → "错误"。
    """
    errors: List[str] = []
    warnings: List[str] = []

    a = [str(x).strip() for x in (seq_a or []) if str(x).strip()]
    b = [str(x).strip() for x in (seq_b or []) if str(x).strip()]

    if not _list_of_str(a):
        errors.append(f"{dom_a}: 非法序列或含非字符串元素")
    if not _list_of_str(b):
        errors.append(f"{dom_b}: 非法序列或含非字符串元素")

    if len(a) == 0:
        errors.append(f"{dom_a}: 序列为空")
    if len(b) == 0:
        errors.append(f"{dom_b}: 序列为空")

    # soft warnings
    total_len = len(a) + len(b)
    if total_len > 6:
        warnings.append(f"联络总长度偏大: {total_len} > 6")
    if abs(len(a) - len(b)) > 5:
        warnings.append(f"两域长度差过大: |{len(a)}-{len(b)}| > 5")
    if _adjacent_duplicates(a):
        warnings.append(f"{dom_a}: 存在相邻重复操作")
    if _adjacent_duplicates(b):
        warnings.append(f"{dom_b}: 存在相邻重复操作")

    result = "错误" if errors else ("警告" if warnings else "通过")
    return {
        "result": result,
        "errors": len(errors),
        "warnings": len(warnings),
        "errors_text": errors,
        "warnings_text": warnings,
    }


def check_conn(dom_a: str, dom_b: str, conn: Dict[str, List[str]]) -> Dict[str, Any]:
    return check_pair(dom_a, dom_b, conn.get(dom_a) or [], conn.get(dom_b) or [])


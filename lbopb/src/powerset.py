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

from __future__ import annotations

"""幂集算法（自由幺半群序列）与常用幂集生成。

依据“算子幂集算法”系列文档：
 - my_docs/project_docs/1761062415_病理演化幺半群 (PEM) 的算子幂集算法.md
 - my_docs/project_docs/1761062416_生理调控幺半群 (PRM) 的算子幂集算法.md
 - my_docs/project_docs/1761062417_毒理学效应幺半群 (TEM) 的算子幂集算法.md
 - my_docs/project_docs/1761062418_药代转运幺半群 (PKTM) 的算子幂集算法.md
 - my_docs/project_docs/1761062419_药理基因组幺半群 (PGOM) 的算子幂集算法.md
 - my_docs/project_docs/1761062420_药效效应幺半群 (PDEM) 的算子幂集算法.md

实现要点：
 - 从 `operator_crosswalk.json` 中读取各模块幂集配置（基本算子集/约束/常用家族）。
 - 生成自由幺半群（仅基本算子构成）的算子序列（支持最大长度、跳过 Identity、禁止相邻重复等约束）。
 - 支持将算子名序列实例化为对应模块的算子实例序列，或直接复合为 compose(…) 的复合算子。
"""

import importlib
from itertools import product
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .op_crosswalk import load_crosswalk


def _get_module(module: str):
    return importlib.import_module(f"lbopb.src.{module}")


def _get_op_class(module: str, op_name: str):
    mod = _get_module(module)
    try:
        return getattr(mod, op_name)
    except AttributeError as e:
        raise RuntimeError(f"模块 {module} 不包含算子类 {op_name}") from e


def instantiate_ops(module: str, op_names: Sequence[str]) -> List[Any]:
    """将算子名序列实例化为算子对象列表（使用默认参数）。"""

    return [(_get_op_class(module, name))() for name in op_names]


def compose_sequence(module: str, op_names: Sequence[str]) -> Any:
    """将算子名序列复合为一个复合算子（使用模块的 compose）。"""

    mod = _get_module(module)
    ops = instantiate_ops(module, op_names)
    return mod.compose(*ops)


def enumerate_sequences(
        base: Sequence[str],
        max_len: int,
        *,
        include_empty: bool = False,
        no_consecutive_duplicate: bool = True,
        skip_identity: bool = True,
) -> Iterator[List[str]]:
    """生成算子名序列（仅基本算子构成）。

    - base: 基本算子名集合
    - max_len: 最大长度（>=1）
    - include_empty: 是否包含空序列
    - no_consecutive_duplicate: 禁止相邻重复（O_i != O_{i+1}）
    - skip_identity: 跳过 Identity（若 base 中包含）
    """

    bset = [x for x in base if not (skip_identity and x == "Identity")]
    if include_empty:
        yield []
    for L in range(1, max_len + 1):
        for tup in product(bset, repeat=L):
            if no_consecutive_duplicate and any(tup[i] == tup[i + 1] for i in range(L - 1)):
                continue
            yield list(tup)


def get_powerset_config(cw: Mapping[str, Any], module: str) -> Mapping[str, Any]:
    """读取某模块的幂集配置（来自 JSON powersets[module]）。"""

    ps = cw.get("powersets", {}).get(module)
    if not ps:
        raise KeyError(f"联络 JSON 未定义模块 {module} 的幂集配置 powersets[{module}]")
    return ps  # type: ignore[return-value]


def generate_powerset(
        module: str,
        *,
        json_path: Optional[str] = None,
        include_empty: bool = False,
) -> Iterator[List[str]]:
    """按 JSON 配置生成某模块的“仅基本算子构成”的序列（幂集枚举）。"""

    cw = load_crosswalk(json_path)
    cfg = get_powerset_config(cw, module)
    base = cfg.get("base", [])
    max_len = int(cfg.get("max_len", 3))
    cons = cfg.get("constraints", {})
    no_dup = bool(cons.get("no_consecutive_duplicate", True))
    skip_id = bool(cons.get("skip_identity", True))
    yield from enumerate_sequences(base, max_len, include_empty=include_empty, no_consecutive_duplicate=no_dup,
                                   skip_identity=skip_id)


def list_families(module: str, *, json_path: Optional[str] = None) -> Dict[str, List[List[str]]]:
    """返回某模块的“常用幂集家族”（仅基本算子序列集合）。"""

    cw = load_crosswalk(json_path)
    cfg = get_powerset_config(cw, module)
    fam = cfg.get("families", {})
    return {k: list(v) for k, v in fam.items()}  # type: ignore[return-value]


# -------- 常用序列生成器（基于 JSON 模式） --------

def list_generators(module: str, *, json_path: Optional[str] = None) -> List[Mapping[str, Any]]:
    """列出模块定义的常用序列生成器（JSON powersets[module].generators）。"""

    cw = load_crosswalk(json_path)
    cfg = get_powerset_config(cw, module)
    gens = cfg.get("generators", [])
    return list(gens)  # type: ignore[return-value]


def _expand_chain_step(step: Any) -> List[List[str]]:
    """将链式模式中的一步展开为若干可选的算子名列表。

    支持：
    - 字符串：单个算子名 → [[name]]
    - {"choice": [name1, name2, ...]} → [[name1], [name2], ...]
    - {"repeat": {"op": name, "min": a, "max": b}} → [[name]*k for k in [a..b]]
    """

    if isinstance(step, str):
        return [[step]]
    if isinstance(step, dict):
        if "choice" in step:
            return [[x] for x in list(step["choice"])]
        if "repeat" in step:
            spec = dict(step["repeat"])
            op = str(spec.get("op"))
            mi = int(spec.get("min", 1))
            ma = int(spec.get("max", mi))
            if mi < 0:
                mi = 0
            if ma < mi:
                ma = mi
            return [[op] * k for k in range(mi, ma + 1)]
    # 未知结构，忽略
    return [[]]


def expand_chain_pattern(chain: Sequence[Any]) -> Iterator[List[str]]:
    """展开链式生成器的模式为若干算子名序列。"""

    # 将每一步转为若干备选序列，再做笛卡尔积
    steps_opts: List[List[List[str]]] = [_expand_chain_step(st) for st in chain]

    # 笛卡尔积
    def _prod(acc: List[List[str]], rest: List[List[List[str]]]) -> List[List[str]]:
        if not rest:
            return acc
        head = rest[0]
        if not acc:
            new_acc = [seq for seq in head]
        else:
            new_acc = [a + b for a in acc for b in head]
        return _prod(new_acc, rest[1:])

    for seq in _prod([], steps_opts):
        yield seq


def generate_by_generator(module: str, name: str, *, json_path: Optional[str] = None) -> Iterator[List[str]]:
    """按生成器名生成常用序列（可包含非基本算子如 Identity，依模式定义）。"""

    gens = list_generators(module, json_path=json_path)
    target = None
    for g in gens:
        if str(g.get("name")) == name:
            target = g
            break
    if not target:
        raise KeyError(f"模块 {module} 未定义生成器 {name}")
    chain = target.get("chain", [])
    for seq in expand_chain_pattern(chain):
        yield seq


def generate_common_sequences(module: str, *, json_path: Optional[str] = None) -> List[List[str]]:
    """合并‘常用幂集家族’与‘常用序列生成器’的全部序列（去重）。"""

    out: List[List[str]] = []
    seen: set[Tuple[str, ...]] = set()

    # families
    fam = list_families(module, json_path=json_path)
    for _, seqs in fam.items():
        for s in seqs:
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                out.append(list(s))

    # generators
    gens = list_generators(module, json_path=json_path)
    for g in gens:
        for seq in generate_by_generator(module, str(g.get("name")), json_path=json_path):
            t = tuple(seq)
            if t not in seen:
                seen.add(t)
                out.append(list(seq))

    return out


__all__ = [
    "instantiate_ops",
    "compose_sequence",
    "enumerate_sequences",
    "get_powerset_config",
    "generate_powerset",
    "list_families",
    "list_generators",
    "expand_chain_pattern",
    "generate_by_generator",
    "generate_common_sequences",
]

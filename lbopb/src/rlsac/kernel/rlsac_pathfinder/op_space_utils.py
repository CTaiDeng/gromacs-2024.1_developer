# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json


def load_op_space(ref: str | Path) -> Dict[str, Any]:
    """加载算子空间定义 JSON。

    ref 可为相对路径（相对仓库根或当前工作目录）或绝对路径。
    """
    p = Path(ref)
    if not p.exists():
        # 允许相对于本模块所在目录的相对路径
        p2 = Path(__file__).resolve().parent / p
        if p2.exists():
            p = p2
    if not p.exists():
        raise FileNotFoundError(f"op-space not found: {ref}")
    return json.loads(p.read_text(encoding="utf-8"))


def param_grid_of(space: Dict[str, Any], op_name: str) -> Tuple[List[str], List[List[Any]]]:
    """获取某算子的参数名列表与对应离散网格（按 JSON 中的声明顺序）。"""
    ops = space.get("operators", {})
    if op_name not in ops:
        raise KeyError(f"operator not in space: {op_name}")
    params = ops[op_name].get("params", {})
    # 保持 JSON 字段顺序（Python 3.7+ 默认保持插入顺序）
    names = list(params.keys())
    grids = [list(params[n]) for n in names]
    return names, grids


def params_from_grid(space: Dict[str, Any], op_name: str, grid_index: List[int]) -> Dict[str, Any]:
    """根据 grid_index 反查参数字典。"""
    names, grids = param_grid_of(space, op_name)
    if len(grid_index) != len(names):
        raise ValueError(f"grid_index length mismatch for {op_name}: expect {len(names)}, got {len(grid_index)}")
    out: Dict[str, Any] = {}
    for i, name in enumerate(names):
        idx = int(grid_index[i])
        vals = grids[i]
        if idx < 0 or idx >= len(vals):
            raise IndexError(f"grid_index out of range for {op_name}.{name}: {idx} not in [0, {len(vals)-1}]")
        out[name] = vals[idx]
    return out


def normalize_ops_detailed(ops_detailed: List[Dict[str, Any]], space: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """验证并补全 ops_detailed：
    - 校验每步的 name 与 grid_index；
    - 若缺少 params，则由 grid_index 反查填充；若存在 params，则校验一致性；
    - 返回 (规范化后的 steps, warnings, errors)。
    """
    norm: List[Dict[str, Any]] = []
    warns: List[str] = []
    errs: List[str] = []
    for i, step in enumerate(ops_detailed or []):
        try:
            name = str(step.get("name"))
            if not name:
                raise ValueError("missing name")
            if "grid_index" not in step:
                raise ValueError("missing grid_index")
            gi = [int(x) for x in list(step.get("grid_index") or [])]
            params = params_from_grid(space, name, gi)
            user_params = step.get("params")
            if user_params is None:
                step = dict(step)
                step["params"] = params
            else:
                # 比较一致性
                for k, v in params.items():
                    if k not in user_params or user_params[k] != v:
                        warns.append(f"step {i}: params mismatch for {name}.{k}; expected {v}, got {user_params.get(k)}; normalized to expected")
                step = dict(step)
                step["params"] = params
            norm.append(step)
        except Exception as e:
            errs.append(f"step {i}: {e}")
    return norm, warns, errs


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

from typing import Any, Dict, List, Tuple, Callable

from .state import IEMState
from .operators import (
    IEMOperator,
    Identity,
    Activate,
    Suppress,
    Proliferate,
    Differentiate,
    CytokineRelease,
    Memory,
)


def default_init_state() -> IEMState:
    return IEMState(b=2.0, n_comp=2, perim=1.0, fidelity=0.7)


def _sign(x: float, eps: float = 1e-9) -> str:
    return "up" if x > eps else ("down" if x < -eps else "same")


RULES: Dict[str, Dict[str, str]] = {
    "Identity": {},
    "Activate": {"b": "up", "perim": "up", "f": "up", "n": "up"},
    "Suppress": {"b": "down", "perim": "down", "f": "down", "n": "down"},
    "Proliferate": {"n": "up", "perim": "up", "b": "up"},
    "Differentiate": {"f": "up"},
    "CytokineRelease": {"b": "up", "perim": "up", "f": "down", "n": "up"},
    "Memory": {"b": "down", "perim": "down", "f": "up", "n": "down"},
}


def _instantiate(name: str, params: Dict[str, Any] | None = None) -> IEMOperator:
    mapping = {
        "Identity": Identity,
        "Activate": Activate,
        "Suppress": Suppress,
        "Proliferate": Proliferate,
        "Differentiate": Differentiate,
        "CytokineRelease": CytokineRelease,
        "Memory": Memory,
    }
    if name not in mapping:
        raise KeyError(f"Unknown operator: {name}")
    cls = mapping[name]
    return cls(**(params or {}))


class RuleEngine:
    """更强的规则引擎：支持上下文依赖、阈值/停机条件、不可交换、序次模式等。

    - threshold_bounds: 运行区间限制（如 fidelity ∈ [0,1]）
    - stop_conditions: 触发后终止（后续操作视为非法）
    - forbidden_pairs: 不可交换/禁止相邻序对
    - require_followups: 要求某算子后续窗口内必须出现某些算子之一
    """

    def __init__(self) -> None:
        self.threshold_bounds: Dict[str, Tuple[float, float]] = {
            "fidelity": (0.0, 1.0),
            "b": (0.0, float("inf")),
            "perim": (0.0, float("inf")),
            "n": (0.0, float("inf")),
        }
        self.stop_conditions: List[Callable[[IEMState], bool]] = []
        self.forbidden_pairs: List[Tuple[str, str]] = [
            ("CytokineRelease", "Proliferate"),  # 风暴后立即扩增通常不合法
        ]
        # 某些操作需要在 K 步内跟随“收敛/缓解类”操作
        self.require_followups: Dict[str, Tuple[int, List[str]]] = {
            "CytokineRelease": (3, ["Suppress", "Memory", "Repair"]),
        }

    def add_stop_if(self, fn: Callable[[IEMState], bool]) -> None:
        self.stop_conditions.append(fn)

    def within_bounds(self, s: IEMState) -> bool:
        lo, hi = self.threshold_bounds["fidelity"];
        if not (lo - 1e-9 <= s.fidelity <= hi + 1e-9):
            return False
        if s.b < -1e-9 or s.perim < -1e-9 or s.n_comp < 0:
            return False
        return True

    def check_forbidden_pair(self, prev_op: str | None, cur_op: str) -> bool:
        if prev_op is None:
            return True
        return (prev_op, cur_op) not in self.forbidden_pairs

    def check_followups(self, seq: List[str], idx: int) -> Tuple[bool, str | None]:
        name = seq[idx]
        if name in self.require_followups:
            k, allowed = self.require_followups[name]
            window = seq[idx + 1: idx + 1 + k]
            if not any(x in allowed for x in window):
                return False, f"{name} must be followed within {k} steps by one of {allowed}"
        return True, None


def check_sequence(seq: List[str], *, init_state: IEMState | None = None) -> Dict[str, Any]:
    engine = RuleEngine()
    state = init_state or default_init_state()
    ok = True
    errors: List[str] = []
    warnings: List[str] = []
    steps: List[Dict[str, Any]] = []
    halted = False
    prev_name: str | None = None
    for i, name in enumerate(seq):
        try:
            op = _instantiate(name)
        except KeyError as e:
            ok = False
            errors.append(str(e))
            break
        # 不可交换/禁止邻接对检查
        if not engine.check_forbidden_pair(prev_name, name):
            ok = False
            errors.append(f"Step {i}: forbidden pair ({prev_name} -> {name})")
        # follow-up 模式检查（上下文依赖）
        ok_follow, why = engine.check_followups(seq, i)
        if not ok_follow and why:
            warnings.append(f"Step {i}: {why}")
        prev = state
        cur = op(prev)
        db = float(cur.b - prev.b)
        dn = float(cur.n_comp - prev.n_comp)
        dp = float(cur.perim - prev.perim)
        df = float(cur.fidelity - prev.fidelity)
        sg = {"b": _sign(db), "n": _sign(dn), "perim": _sign(dp), "f": _sign(df)}
        rule = RULES.get(name, {})
        violated = []
        for k, expect in rule.items():
            if sg[k] != expect:
                violated.append(f"{k}: expect {expect}, got {sg[k]}")
        if violated:
            warnings.append(f"Step {i}: {name} violates: " + "; ".join(violated))
        steps.append({"op": name, "delta": {"b": db, "n": dn, "perim": dp, "f": df}, "sign": sg})
        state = cur
        prev_name = name
        # 阈值/停机条件
        if not engine.within_bounds(state):
            ok = False
            errors.append(f"Step {i}: state out of bounds")
            halted = True
            break
        for cond in engine.stop_conditions:
            try:
                if cond(state):
                    halted = True
                    errors.append(f"Step {i}: stop condition triggered")
                    break
            except Exception:
                continue
        if halted:
            break
    return {
        "valid": ok,
        "errors": [
            {"index": i, "op": steps[i]["op"] if i < len(steps) else None, "message": msg,
             "doc": "my_docs/project_docs/1761062407_免疫效应幺半群 (IEM) 公理系统.md"}
            for i, msg in enumerate(errors)
        ] if errors else [],
        "warnings": [
            {"index": i, "op": steps[i]["op"] if i < len(steps) else None, "message": msg,
             "doc": "my_docs/project_docs/1761062407_免疫效应幺半群 (IEM) 公理系统.md"}
            for i, msg in enumerate(warnings)
        ] if warnings else [],
        "steps": steps,
    }


def check_package(pkg: Dict[str, Any]) -> Dict[str, Any]:
    seq: List[str] = list(pkg.get("sequence", []) or [])
    warnings: List[str] = []
    errors: List[str] = []
    params_list: List[Dict[str, Any]] = [dict() for _ in seq]

    if "op_param_seq" in pkg:
        warnings.append("op_param_seq is discouraged; use ops_detailed.grid_index with op_space_ref")

    ops_detailed = pkg.get("ops_detailed")
    op_space_ref = pkg.get("op_space_ref")
    if isinstance(ops_detailed, list) and op_space_ref:
        try:
            from lbopb.src.rlsac.kernel.rlsac_pathfinder.op_space_utils import load_op_space, normalize_ops_detailed
            space = load_op_space(str(op_space_ref))
            norm_steps, warns, errs = normalize_ops_detailed(ops_detailed, space)
            warnings.extend([f"ops_detailed: {m}" for m in warns])
            errors.extend([f"ops_detailed: {m}" for m in errs])
            for i, st in enumerate(norm_steps):
                if i < len(params_list):
                    params_list[i] = dict(st.get("params", {}))
                if i < len(seq) and st.get("name") and seq[i] != st.get("name"):
                    warnings.append(f"step {i}: sequence name '{seq[i]}' != ops_detailed name '{st.get('name')}'")
        except Exception as e:
            warnings.append(f"ops_detailed check failed: {e}")

    engine = RuleEngine()
    state = default_init_state()
    ok = True
    steps: List[Dict[str, Any]] = []
    halted = False
    prev_name: str | None = None
    for i, name in enumerate(seq):
        try:
            op = _instantiate(name, params=params_list[i] if i < len(params_list) else None)
        except KeyError as e:
            ok = False
            errors.append(str(e))
            break
        if not engine.check_forbidden_pair(prev_name, name):
            ok = False
            errors.append(f"Step {i}: forbidden pair ({prev_name} -> {name})")
        ok_follow, why = engine.check_followups(seq, i)
        if not ok_follow and why:
            warnings.append(f"Step {i}: {why}")
        prev = state
        cur = op(prev)
        db = float(cur.b - prev.b)
        dn = float(cur.n_comp - prev.n_comp)
        dp = float(cur.perim - prev.perim)
        df = float(cur.fidelity - prev.fidelity)
        sg = {"b": _sign(db), "n": _sign(dn), "perim": _sign(dp), "f": _sign(df)}
        rule = RULES.get(name, {})
        violated = []
        for k, expect in rule.items():
            if sg[k] != expect:
                violated.append(f"{k}: expect {expect}, got {sg[k]}")
        if violated:
            warnings.append(f"Step {i}: {name} violates: " + "; ".join(violated))
        steps.append({"op": name, "delta": {"b": db, "n": dn, "perim": dp, "f": df}, "sign": sg})
        state = cur
        prev_name = name
        if not engine.within_bounds(state):
            ok = False
            errors.append(f"Step {i}: state out of bounds")
            halted = True
            break
        for cond in engine.stop_conditions:
            try:
                if cond(state):
                    halted = True
                    errors.append(f"Step {i}: stop condition triggered")
                    break
            except Exception:
                continue
        if halted:
            break

    return {"valid": ok and not errors, "errors": errors, "warnings": warnings, "steps": steps}


if __name__ == "__main__":
    import json, sys

    seq = sys.argv[1:] if len(sys.argv) > 1 else ["Activate", "Proliferate"]
    res = check_sequence(seq)
    print(json.dumps(res, ensure_ascii=False, indent=2))

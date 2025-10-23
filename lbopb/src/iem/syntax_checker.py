# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

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


def _instantiate(name: str) -> IEMOperator:
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
    return mapping[name]()


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
            ok = False
            errors.append(f"Step {i}: {why}")
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
            ok = False
            errors.append(f"Step {i}: {name} violates: " + "; ".join(violated))
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
    return {"valid": ok, "errors": errors, "steps": steps}


if __name__ == "__main__":
    import json, sys

    seq = sys.argv[1:] if len(sys.argv) > 1 else ["Activate", "Proliferate"]
    res = check_sequence(seq)
    print(json.dumps(res, ensure_ascii=False, indent=2))

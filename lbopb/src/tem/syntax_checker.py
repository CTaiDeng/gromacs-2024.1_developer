# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable

from .state import TEMState
from .operators import (
    TEMOperator,
    Identity,
    Exposure,
    Absorption,
    Distribution,
    Lesion,
    Inflammation,
    Detox,
    Repair,
)


def default_init_state() -> TEMState:
    return TEMState(b=5.0, n_comp=1, perim=2.0, fidelity=0.9)


def _sign(x: float, eps: float = 1e-9) -> str:
    return "up" if x > eps else ("down" if x < -eps else "same")


RULES: Dict[str, Dict[str, str]] = {
    "Identity": {},
    "Exposure": {"b": "up", "perim": "up", "f": "down", "n": "up"},
    "Absorption": {"b": "up", "perim": "up", "f": "down"},
    "Distribution": {"n": "up", "perim": "up", "b": "up", "f": "down"},
    "Lesion": {"b": "up", "perim": "up", "f": "down", "n": "up"},
    "Inflammation": {"b": "up", "perim": "up", "f": "down"},
    "Detox": {"b": "down", "perim": "down", "f": "up"},
    "Repair": {"b": "down", "perim": "down", "f": "up", "n": "down"},
}


def _instantiate(name: str) -> TEMOperator:
    mapping = {
        "Identity": Identity,
        "Exposure": Exposure,
        "Absorption": Absorption,
        "Distribution": Distribution,
        "Lesion": Lesion,
        "Inflammation": Inflammation,
        "Detox": Detox,
        "Repair": Repair,
    }
    if name not in mapping:
        raise KeyError(f"Unknown operator: {name}")
    return mapping[name]()


class RuleEngine:
    def __init__(self) -> None:
        self.threshold_bounds: Dict[str, Tuple[float, float]] = {
            "fidelity": (0.0, 1.0),
            "b": (0.0, float("inf")),
            "perim": (0.0, float("inf")),
            "n": (0.0, float("inf")),
        }
        self.stop_conditions: List[Callable[[TEMState], bool]] = []
        self.forbidden_pairs: List[Tuple[str, str]] = [
            ("Lesion", "Inflammation"),
        ]
        self.require_followups: Dict[str, Tuple[int, List[str]]] = {
            "Exposure": (3, ["Absorption", "Detox", "Repair"]),
            "Lesion": (5, ["Detox", "Repair"]),
        }

    def within_bounds(self, s: TEMState) -> bool:
        lo, hi = self.threshold_bounds["fidelity"];
        if not (lo - 1e-9 <= s.fidelity <= hi + 1e-9):
            return False
        if s.b < -1e-9 or s.perim < -1e-9 or s.n_comp < 0:
            return False
        return True

    def check_forbidden_pair(self, prev_op: str | None, cur_op: str) -> bool:
        if prev_op is None: return True
        return (prev_op, cur_op) not in self.forbidden_pairs

    def check_followups(self, seq: List[str], idx: int) -> Tuple[bool, str | None]:
        name = seq[idx]
        if name in self.require_followups:
            k, allowed = self.require_followups[name]
            window = seq[idx + 1: idx + 1 + k]
            if not any(x in allowed for x in window):
                return False, f"{name} must be followed within {k} steps by one of {allowed}"
        return True, None


def check_sequence(seq: List[str], *, init_state: TEMState | None = None) -> Dict[str, Any]:
    engine = RuleEngine()
    state = init_state or default_init_state()
    ok = True\r\n    errors: List[str] = []\r\n    warnings: List[str] = []
    steps: List[Dict[str, Any]] = []
    prev_name: str | None = None
    for i, name in enumerate(seq):
        try:
            op = _instantiate(name)
        except KeyError as e:
            ok = False
            errors.append(str(e))
            break
        if not engine.check_forbidden_pair(prev_name, name):
            ok = False
            errors.append(f"Step {i}: forbidden pair ({prev_name} -> {name})")
        ok_follow, why = engine.check_followups(seq, i)
        if not ok_follow and why:\r\n            warnings.append(f"Step {i}: {why}")
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
        if violated:\r\n            warnings.append(f"Step {i}: {name} violates: " + "; ".join(violated))
        steps.append({"op": name, "delta": {"b": db, "n": dn, "perim": dp, "f": df}, "sign": sg})
        state = cur
        prev_name = name
        if not engine.within_bounds(state):
            ok = False
            errors.append(f"Step {i}: state out of bounds")
            break
    return {
        "valid": ok,
        "errors": [
            {"index": i, "op": steps[i]["op"] if i < len(steps) else None, "message": msg, "doc": "my_docs/project_docs/1761062403_毒理学效应幺半群 (TEM) 公理系统.md"}
            for i, msg in enumerate(errors)
        ] if errors else [],
        "steps": steps,
    }


if __name__ == "__main__":
    import json, sys

    seq = sys.argv[1:] if len(sys.argv) > 1 else ["Exposure", "Detox"]
    res = check_sequence(seq)
    print(json.dumps(res, ensure_ascii=False, indent=2))


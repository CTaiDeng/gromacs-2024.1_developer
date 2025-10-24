# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path
import json

MODULES = ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]


def _nearest_index(values: List[float], x: float) -> int:
    if not values:
        return 0
    best_i = 0
    best_d = abs(x - float(values[0]))
    for i, v in enumerate(values):
        d = abs(x - float(v))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


class ObservationQuantizer:
    def __init__(self, map_path: str | Path) -> None:
        self.map = json.loads(Path(map_path).read_text(encoding="utf-8"))
        self.modules = self.map.get("modules", {})
        self.mode = str(self.map.get("quantize_mode", "nearest")).lower()

    def quantize_state(self, domain: str, state: Any) -> List[int]:
        dom = self.modules.get(domain, {})
        # state fields: b, perim, fidelity, n_comp
        b_seq = list(dom.get("b", []))
        p_seq = list(dom.get("perim", []))
        f_seq = list(dom.get("fidelity", []))
        n_seq = list(dom.get("n", []))
        b = float(getattr(state, "b", 0.0))
        p = float(getattr(state, "perim", 0.0))
        f = float(getattr(state, "fidelity", 0.0))
        n = float(getattr(state, "n_comp", 0.0))
        if self.mode == "nearest":
            return [
                _nearest_index(b_seq, b),
                _nearest_index(p_seq, p),
                _nearest_index(f_seq, f),
                _nearest_index(n_seq, n),
            ]
        # fallback: nearest
        return [
            _nearest_index(b_seq, b),
            _nearest_index(p_seq, p),
            _nearest_index(f_seq, f),
            _nearest_index(n_seq, n),
        ]

    def quantize_full(self, states_by_domain: Dict[str, Any], risk_by_domain: Dict[str, float]) -> List[int]:
        out: List[int] = []
        for m in MODULES:
            st = states_by_domain.get(m)
            idxs = self.quantize_state(m, st)
            # risk 映射
            r_seq = list(self.modules.get(m, {}).get("risk", []))
            r_val = float(risk_by_domain.get(m, 0.0))
            r_idx = _nearest_index(r_seq, r_val)
            out += idxs + [r_idx]
        return out

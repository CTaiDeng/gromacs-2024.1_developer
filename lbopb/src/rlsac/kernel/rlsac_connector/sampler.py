# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

MODULES: List[str] = ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]


def load_domain_packages(packages_dir: str | Path) -> Dict[str, List[Dict]]:
    """读取各域的算子包辞海（若不存在则为空列表）。"""
    p = Path(packages_dir)
    out: Dict[str, List[Dict]] = {}
    for m in MODULES:
        fp = p / f"{m}_operator_packages.json"
        arr: List[Dict] = []
        if fp.exists():
            try:
                arr = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                arr = []
        out[m] = arr
    return out


def sample_random_connection(pkg_map: Dict[str, List[Dict]], rng: random.Random | None = None) -> Dict[str, Dict]:
    """随机从每个域的辞海中各取一个算子包，返回七元映射（若某域为空则取空包）。"""
    r = rng or random
    chosen: Dict[str, Dict] = {}
    for m in MODULES:
        arr = pkg_map.get(m) or []
        if arr:
            chosen[m] = r.choice(arr)
        else:
            chosen[m] = {"id": f"{m}_empty", "sequence": []}
    return chosen

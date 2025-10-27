# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import hashlib
import json
import random
import time as _t
from pathlib import Path
from typing import Any, Dict, List, Tuple


DOMAINS = ["pem", "prm", "tem", "pktm", "pgom", "pdem", "iem"]


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for anc in [p.parent] + list(p.parents):
        try:
            if (anc / ".git").exists():
                return anc
        except Exception:
            continue
    try:
        return p.parents[6]
    except Exception:
        return p.parents[-1]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_ops_from_monoid() -> Dict[str, List[str]]:
    root = Path(__file__).resolve().parents[2] / "rlsac_pathfinder" / "monoid_packages"
    ops_by_dom: Dict[str, List[str]] = {}
    for d in DOMAINS:
        names: List[str] = []
        f = root / f"{d}_operator_packages.json"
        arr = _read_json(f) or []
        for it in (arr or []):
            try:
                seq = list(it.get("sequence", []) or [])
            except Exception:
                seq = []
            for s in seq:
                s = str(s).strip()
                if s and s not in names:
                    names.append(s)
        ops_by_dom[d] = names
    return ops_by_dom


def _stable_id(pair: Tuple[str, str], seqs: Dict[str, List[str]]) -> str:
    payload = {
        "pair": list(pair),
        "sequences": {pair[0]: list(seqs.get(pair[0], []) or []), pair[1]: list(seqs.get(pair[1], []) or [])},
    }
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_pair_packages(n_per_pair: int = 32, seed: int | None = None) -> None:
    if seed is not None:
        random.seed(int(seed))

    base = Path(__file__).resolve().parent
    base.mkdir(parents=True, exist_ok=True)
    ops = _collect_ops_from_monoid()
    ts = int(_t.time())

    pairs: List[Tuple[str, str]] = []
    for i, a in enumerate(DOMAINS):
        for b in DOMAINS[i + 1 :]:
            pairs.append((a, b))

    for a, b in pairs:
        ops_a = ops.get(a, [])
        ops_b = ops.get(b, [])
        items: List[Dict[str, Any]] = []
        if not ops_a or not ops_b:
            # 无可用原子操作，写空文件以便后续替换
            (base / f"{a}_{b}_operator_packages.json").write_text("[]", encoding="utf-8")
            continue
        for _ in range(n_per_pair):
            la = random.choice([1, 2])
            lb = random.choice([1, 2])
            seq_a = random.sample(ops_a, k=min(la, len(ops_a)))
            seq_b = random.sample(ops_b, k=min(lb, len(ops_b)))
            seqs = {a: seq_a, b: seq_b}
            pid = _stable_id((a, b), seqs)
            items.append(
                {
                    "id": f"pkg_{a}_{b}_{pid}",
                    "pair": {"a": a, "b": b},
                    "sequences": seqs,
                    "length_a": len(seq_a),
                    "length_b": len(seq_b),
                    "created_at": ts,
                    "updated_at": ts,
                    "source": "pair_generated",
                    "validation": {
                        "mode": "dual",
                        "syntax": {"result": "待校验", "errors": 0, "warnings": 0},
                        "gemini": {"used": False, "result": "未知"},
                    },
                }
            )

        out = base / f"{a}_{b}_operator_packages.json"
        out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[build] written: {out} items={len(items)}")


def main() -> None:
    import sys

    n = 32
    seed = None
    for a in sys.argv[1:]:
        if a.startswith("--seed="):
            try:
                seed = int(a.split("=", 1)[1])
            except Exception:
                seed = None
        else:
            try:
                n = int(a)
            except Exception:
                continue
    build_pair_packages(n_per_pair=n, seed=seed)


if __name__ == "__main__":
    main()


# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import random
import time as _t
from pathlib import Path
from typing import Any, Dict, List


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def main() -> None:
    # 输出到 out/out_connector/dataset_<ts>/debug_dataset.json
    root = _repo_root()
    out_root = root / "out" / "out_connector"
    ts = int(_t.time())
    run = out_root / f"dataset_{ts}"
    run.mkdir(parents=True, exist_ok=True)

    # 读取成对辞海，合并样本并随机抽样
    base = Path(__file__).resolve().parent / "monoid_packages"
    items: List[Dict[str, Any]] = []
    for f in sorted(base.glob("*_operator_packages.json")):
        arr = _read_json(f) or []
        for it in (arr or []):
            try:
                pair = it.get("pair") or {}
                a = str(pair.get("a")).lower()
                b = str(pair.get("b")).lower()
                seqs = it.get("sequences") or {}
                seq_a = list(seqs.get(a, []) or [])
                seq_b = list(seqs.get(b, []) or [])
                items.append(
                    {
                        "id": it.get("id"),
                        "pair": f"{a}_{b}",
                        "sequences": {a: seq_a, b: seq_b},
                        "length": int(len(seq_a) + len(seq_b)),
                        "created_at": int(it.get("created_at", ts)),
                        "updated_at": int(it.get("updated_at", ts)),
                        "source": "pair_monoid",
                    }
                )
            except Exception:
                continue
    random.shuffle(items)
    (run / "debug_dataset.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dataset] written: {run / 'debug_dataset.json'} items={len(items)}")


if __name__ == "__main__":
    main()


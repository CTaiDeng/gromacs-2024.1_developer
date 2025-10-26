# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def discretize(p: float) -> float:
    # round to nearest 0.5
    try:
        return round(float(p) * 2.0) / 2.0
    except Exception:
        return 0.0


def label_of(s3: float) -> str:
    if s3 >= 0.75:
        return "正确"
    if s3 >= 0.25:
        return "警告"
    return "错误"


def reward_of(lbl: str) -> float:
    return 1.0 if lbl == "正确" else (0.5 if lbl == "警告" else 0.0)


def main() -> None:
    # usage: python apply_triscore.py <run_dir> [infile] [outfile]
    import sys
    args = sys.argv[1:]
    if not args:
        print("usage: python apply_triscore.py <run_dir> [infile] [outfile]")
        return
    run_dir = Path(args[0]).resolve()
    in_path = run_dir / (args[1] if len(args) > 1 else "samples.output.json")
    out_path = run_dir / (args[2] if len(args) > 2 else "samples.output.tri.json")
    try:
        data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    except Exception:
        print(f"[apply_triscore] read failed: {in_path}")
        return
    if not isinstance(data, list):
        print("[apply_triscore] input not a list")
        return
    out: List[Dict[str, Any]] = []
    for it in data:
        try:
            s = float(it.get("score", 0.0))
            s3 = discretize(s)
            lbl = label_of(s3)
            rwd = reward_of(lbl)
            o = dict(it)
            o.update({"score_tri": s3, "label3": lbl, "reward": rwd})
            out.append(o)
        except Exception:
            out.append(dict(it))
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[apply_triscore] written: {out_path}")


if __name__ == "__main__":
    main()


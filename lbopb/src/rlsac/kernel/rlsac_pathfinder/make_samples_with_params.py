# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for anc in [p.parent] + list(p.parents):
        if (anc / ".git").exists():
            return anc
    return p.parents[-1]


def synth_ops(domain: str, seq: List[str], base: Path) -> List[Dict[str, Any]]:
    try:
        from lbopb.src.rlsac.kernel.rlsac_pathfinder.op_space_utils import load_op_space, param_grid_of, params_from_grid  # type: ignore
    except Exception:
        return [{"name": nm} for nm in seq]
    space_ref = base / "operator_spaces" / f"{domain}_op_space.v1.json"
    if not space_ref.exists():
        return [{"name": nm} for nm in seq]
    try:
        space = load_op_space(str(space_ref))
        steps: List[Dict[str, Any]] = []
        for nm in seq:
            try:
                _, grids = param_grid_of(space, nm)
            except Exception:
                steps.append({"name": nm})
                continue
            gi = [max(0, (len(g) - 1) // 2) for g in grids]
            prs = params_from_grid(space, nm, gi)
            steps.append({"name": nm, "grid_index": gi, "params": prs})
        return steps
    except Exception:
        return [{"name": nm} for nm in seq]


def main() -> None:
    # usage: python make_samples_with_params.py <run_dir> <domain> <seq1,seq2,...;seq1,seq2,...>
    import sys
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: python make_samples_with_params.py <run_dir> <domain> [sequences]")
        print("  sequences: optional; semicolon-separated lists, each list comma-separated ops")
        return
    run_dir = Path(args[0]).resolve()
    domain = str(args[1]).strip().lower()
    seqs: List[List[str]]
    if len(args) >= 3:
        raw = args[2]
        seqs = [list(filter(None, s.split(','))) for s in raw.split(';')]
    else:
        # fallback: read op_names from meta and craft 2 demo sequences
        meta = json.loads((run_dir / "train_meta.json").read_text(encoding="utf-8"))
        ops = meta.get("op_names", []) or []
        a = [ops[0]] if len(ops) > 0 else []
        b = [ops[1], ops[2]] if len(ops) > 2 else a
        seqs = [a, b]

    base = Path(__file__).resolve().parent
    items: List[Dict[str, Any]] = []
    for i, seq in enumerate(seqs, start=1):
        steps = synth_ops(domain, seq, base)
        items.append({
            "id": f"demo_{i}",
            "domain": domain,
            "sequence": seq,
            "ops_detailed": steps,
        })
    (run_dir / "samples.input.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[samples] written: {run_dir / 'samples.input.json'} items={len(items)}")


if __name__ == "__main__":
    main()


# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _t
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hashlib


def _repo_root() -> Path:
    """Return the project root (directory that contains .git),
    falling back to a reasonable ancestor if not found.
    This ensures out/out_pathfinder resolves to the repo-root/out/out_pathfinder.
    """
    p = Path(__file__).resolve()
    # Walk up to locate a .git directory which marks repo root
    for anc in [p.parent] + list(p.parents):
        try:
            if (anc / ".git").exists():
                return anc
        except Exception:
            continue
    # Fallback: go up a fixed number of levels to approximate repo root
    try:
        return p.parents[6]
    except Exception:
        return p.parents[-1]


def _load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _space_steps(domain: str, seq: List[str]) -> Tuple[List[Dict[str, Any]] | None, Dict[str, Any] | None]:
    try:
        from lbopb.src.rlsac.kernel.rlsac_pathfinder.op_space_utils import load_op_space, param_grid_of, params_from_grid  # type: ignore
        base = Path(__file__).resolve().parents[1]
        space_ref = base / "operator_spaces" / f"{domain}_op_space.v1.json"
        if not space_ref.exists():
            return None, None
        space = load_op_space(str(space_ref))
        steps: List[Dict[str, Any]] = []
        for nm in seq:
            try:
                names, grids = param_grid_of(space, nm)
            except Exception:
                steps.append({"name": nm})
                continue
            gi: List[int] = []
            for g in grids:
                L = len(g)
                gi.append(max(0, (L - 1) // 2))
            prs = params_from_grid(space, nm, gi)
            steps.append({"name": nm, "grid_index": gi, "params": prs})
        meta = {
            "op_space_id": space.get("space_id", f"{domain}.v1"),
            "op_space_ref": str(space_ref.relative_to(_repo_root())).replace("\\", "/"),
        }
        return steps, meta
    except Exception:
        return None, None


def main() -> None:
    repo = _repo_root()
    # Always resolve out/out_pathfinder under the repository root
    out_root = (repo / "out" / "out_pathfinder").resolve()
    print(f"[collect] repo_root={repo}")
    print(f"[collect] scan dir={out_root}")
    if not out_root.exists():
        print(f"[collect] out root not found: {out_root}")
        return

    # 读取 config 以获取 cost_lambda（若需要）
    cfg_path = Path(__file__).resolve().parents[1] / "config.json"
    cfg = _load_json(cfg_path)
    cost_lambda = float(cfg.get("cost_lambda", 0.2))

    # 聚合：按 (domain + 序列 + 参数化取值) 去重，保留 score 更高
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    now = int(_t.time())
    for p in out_root.glob("train_*/debug_dataset.json"):
        data = _load_json(p)
        domain = str(data.get("domain", "")).lower() or "pem"
        samples = data.get("samples", []) or []
        for s in samples:
            try:
                seq = list(s.get("sequence", []) or [])
                feats = s.get("features", {}) or {}
                dr = float(feats.get("delta_risk", 0.0))
                c = float(feats.get("cost", 0.0))
                length = int(feats.get("length", len(seq)))
                score = dr - cost_lambda * c
                label = int(s.get("label", 0))
                # 优先从样本中提取参数化步骤（如存在），否则按 v1 空间补全中位参数
                steps = None
                meta = None
                try:
                    cand = s.get("ops_detailed") or (s.get("judge", {}) or {}).get("ops_detailed")
                    if isinstance(cand, list) and len(cand) > 0:
                        steps = cand
                        meta = {k: s.get(k) for k in ("op_space_id", "op_space_ref") if k in s}
                except Exception:
                    steps = None
                    meta = None
                if not steps:
                    steps, meta = _space_steps(domain, seq)
                # 生成稳定 ID：基于序列+参数化取值的哈希
                payload = {
                    "sequence": list(seq),
                    "ops_detailed": steps or [],
                }
                sblob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                h = hashlib.sha1(sblob).hexdigest()
                num = int(h, 16) % (10**10)
                sid = f"pkg_{domain}_{num}"
                key = (domain, sid)
                item = {
                    "id": sid,
                    "domain": domain,
                    "sequence": seq,
                    "length": int(length),
                    "delta_risk": float(dr),
                    "cost": float(c),
                    "score": float(score),
                    "created_at": now,
                    "updated_at": now,
                    "source": "debug_dataset",
                    "label": int(label),
                }
                # 始终写入 ops_detailed（若无参数网格则用仅 name 的占位）
                item["ops_detailed"] = steps or [{"name": nm} for nm in seq]
                if meta:
                    item.update(meta)
                prev = by_key.get(key)
                if (prev is None) or (float(score) > float(prev.get("score", -1e9))):
                    by_key[key] = item
            except Exception:
                continue

    # 排序并写入 train_datas/debug_dataset.json
    items = list(by_key.values())
    items.sort(key=lambda d: (
        -float(d.get("score", 0.0)),
        int(d.get("length", 0)),
        tuple(str(x) for x in d.get("sequence", []))
    ))
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "debug_dataset.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[collect] written: {out_path} items={len(items)}")


if __name__ == "__main__":
    main()

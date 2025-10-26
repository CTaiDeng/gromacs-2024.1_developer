# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def _truth_from_item(it: Dict[str, Any]) -> str | None:
    val = it.get("validation") or {}
    if isinstance(val, dict):
        gem = val.get("gemini") or {}
        if isinstance(gem, dict) and gem.get("used") and isinstance(gem.get("result"), str):
            r = str(gem.get("result")).strip()
            if r in ("正确", "错误"):
                return r
        syn = val.get("syntax") or {}
        if isinstance(syn, dict) and isinstance(syn.get("result"), str):
            r = str(syn.get("result")).strip()
            if r in ("正确", "警告", "错误"):
                return r
    if "label" in it:
        try:
            return "正确" if int(it.get("label", 0)) == 1 else "错误"
        except Exception:
            return None
    return None


def main() -> None:
    # usage:
    #   python apply_model_cli.py <run_dir> [infile] [outfile]
    #   python apply_model_cli.py <infile.json> [outfile]
    import sys
    args = sys.argv[1:]
    if not args:
        print("usage: python apply_model_cli.py <run_dir>|<infile.json> [infile|outfile] [outfile]")
        return
    first = Path(args[0]).resolve()
    # 兼容：若第一个参数是文件，则将其父目录视为 run_dir
    if first.is_file() and first.suffix.lower() == ".json":
        run_dir = first.parent
        in_path = first
        out_path = run_dir / (args[1] if len(args) > 1 else "samples.output.json")
    else:
        run_dir = first
        if not run_dir.exists():
            print(f"[apply_cli] run_dir not found: {run_dir}")
            return
        in_path = run_dir / (args[1] if len(args) > 1 else "samples.input.json")
        out_path = run_dir / (args[2] if len(args) > 2 else "samples.output.json")

    meta_path = run_dir / "train_meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[apply_cli] meta load failed: {meta_path} err={e}")
        return

    # --- import scorer with robust fallback ---
    PackageScorer = None  # type: ignore
    try:
        from lbopb.src.rlsac.kernel.rlsac_pathfinder.scorer import PackageScorer as _PS  # type: ignore
        PackageScorer = _PS  # type: ignore
    except Exception:
        # try add repo root to sys.path
        try:
            import sys as _sys
            root = run_dir
            for anc in [run_dir] + list(run_dir.parents):
                if (anc / ".git").exists():
                    root = anc
                    break
            if str(root) not in _sys.path:
                _sys.path.insert(0, str(root))
            from lbopb.src.rlsac.kernel.rlsac_pathfinder.scorer import PackageScorer as _PS2  # type: ignore
            PackageScorer = _PS2  # type: ignore
        except Exception:
            PackageScorer = None  # type: ignore
    if PackageScorer is None:
        # final fallback: define a minimal scorer consistent with training
        import torch.nn as nn  # type: ignore
        class PackageScorer(nn.Module):  # type: ignore
            def __init__(self, in_dim: int, hidden=(128, 64)) -> None:
                super().__init__()
                h1, h2 = hidden
                self.net = nn.Sequential(
                    nn.Linear(in_dim, h1), nn.ReLU(),
                    nn.Linear(h1, h2), nn.ReLU(),
                    nn.Linear(h2, 1)
                )
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(self.net(x)).view(-1)

    feat_dim = int(meta.get("feat_dim", 0))
    doms: List[str] = list(meta.get("domains", []))
    ops: List[str] = list(meta.get("op_names", []))

    model = PackageScorer(feat_dim)
    model.load_state_dict(torch.load(run_dir / "scorer.pt", map_location="cpu"))
    model.eval()

    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[apply_cli] read input failed: {in_path} err={e}")
        data = []
    if not isinstance(data, list):
        print(f"[apply_cli] input not a list: {in_path}")
        return

    def _feat_of(item: Dict[str, Any]) -> torch.Tensor:
        dom = str(item.get("domain", "")).lower()
        seq = list(item.get("sequence", []) or [])
        cnts = [float(seq.count(nm)) for nm in ops]
        length = float(len(seq))
        onehot = [0.0] * len(doms)
        try:
            idx = doms.index(dom)
            onehot[idx] = 1.0
        except Exception:
            pass
        return torch.tensor([*cnts, length, *onehot], dtype=torch.float32).unsqueeze(0)

    out: List[Dict[str, Any]] = []
    for it in data:
        try:
            x = _feat_of(it)
            with torch.no_grad():
                s = float(model(x).item())
            # tri discretization + reward
            s3 = round(s * 2.0) / 2.0
            label3 = ("正确" if s3 >= 0.75 else ("警告" if s3 >= 0.25 else "错误"))
            reward = 1.0 if label3 == "正确" else (0.5 if label3 == "警告" else 0.0)
            pred = "正确" if s >= 0.5 else "错误"
            truth = _truth_from_item(it)
            corr = (pred == truth) if truth in ("正确", "错误") else None
            out.append({
                "id": it.get("id"),
                "domain": it.get("domain"),
                "sequence": it.get("sequence"),
                "ops_detailed": it.get("ops_detailed"),
                "score": s,
                "score_tri": s3,
                "label3": label3,
                "pred": pred,
                "reward": reward,
                **({"truth": truth} if truth is not None else {}),
                **({"correct": bool(corr)} if corr is not None else {}),
                "threshold": 0.5,
            })
        except Exception:
            continue

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[apply_cli] written: {out_path} items={len(out)}")


if __name__ == "__main__":
    main()

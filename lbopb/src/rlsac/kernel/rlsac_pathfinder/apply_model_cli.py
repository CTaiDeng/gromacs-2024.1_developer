# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 only.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# --- 著作权独立性声明 (Copyright Independence Declaration) ---
# 本文件（“载荷”）是作者 (GaoZheng) 的原创著作物，其知识产权
# 独立于其运行平台 GROMACS（“宿主”）。
# 本文件的授权遵循上述 SPDX 标识，不受“宿主”许可证的管辖。
# 详情参见项目文档 "my_docs/project_docs/1762636780_🚩🚩gromacs-2024.1_developer项目的著作权设计策略：“宿主-载荷”与“双轨制”复合架构.md"。
# ------------------------------------------------------------------

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
    #   python apply_model_cli.py <run_dir> [infile] [outfile] [--model <model.pt>]
    #   python apply_model_cli.py <infile.json> [outfile] [--model <model.pt>]
    import sys
    args = sys.argv[1:]
    if not args:
        print(
            "usage: python apply_model_cli.py <run_dir>|<infile.json> [infile|outfile] [outfile] [--model <model.pt>] [--out <outfile>]")
        return

    # parse flags
    model_override: str | None = None
    out_override: str | None = None
    pos: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--model"):
            val = None
            if a == "--model" and i + 1 < len(args):
                val = args[i + 1];
                i += 2
            elif "=" in a:
                val = a.split("=", 1)[1];
                i += 1
            else:
                val = a[len("--model"):];
                i += 1
            if val:
                model_override = str(Path(val))
            continue
        if a.startswith("--out"):
            val = None
            if a == "--out" and i + 1 < len(args):
                val = args[i + 1];
                i += 2
            elif "=" in a:
                val = a.split("=", 1)[1];
                i += 1
            else:
                val = a[len("--out"):];
                i += 1
            if val:
                out_override = str(Path(val))
            continue
        pos.append(a);
        i += 1

    if not pos:
        print("[apply_cli] missing run_dir or infile.json")
        return

    first = Path(pos[0]).resolve()
    # run_dir vs infile mode
    if first.is_file() and first.suffix.lower() == ".json":
        run_dir = first.parent
        in_path = first
        out_path = Path(out_override) if out_override else (
                    run_dir / (pos[1] if len(pos) > 1 else "samples.output.json"))
    else:
        run_dir = first
        if not run_dir.exists():
            print(f"[apply_cli] run_dir not found: {run_dir}")
            return
        in_path = run_dir / (pos[1] if len(pos) > 1 else "samples.input.json")
        out_path = Path(out_override) if out_override else (
                    run_dir / (pos[2] if len(pos) > 2 else "samples.output.json"))

    # choose model dir (override or run_dir)
    if model_override:
        model_path = Path(model_override).resolve()
        model_dir = model_path.parent
    else:
        model_dir = run_dir
        model_path = model_dir / "scorer.pt"
    meta_path = model_dir / "train_meta.json"
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
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
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
            pred = "正确" if s >= 0.5 else "错误"
            reward = 1.0 if pred == "正确" else 0.0
            truth = _truth_from_item(it)
            corr = (pred == truth) if truth in ("正确", "错误") else None
            out.append({
                "id": it.get("id"),
                "domain": it.get("domain"),
                "sequence": it.get("sequence"),
                "ops_detailed": it.get("ops_detailed"),
                "score": s,
                "model_output": round(s, 4),
                "pred": pred,
                "reward": reward,
                **({"truth": truth} if truth is not None else {}),
                **({"correct": bool(corr)} if corr is not None else {}),
                "threshold": 0.5,
            })
        except Exception:
            continue

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[apply_cli] written: {out_path} items={len(out)}")


if __name__ == "__main__":
    main()

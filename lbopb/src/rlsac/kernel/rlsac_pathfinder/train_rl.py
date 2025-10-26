# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _t
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    from .domain import get_domain_spec
    from .scorer import PackageScorer, train_scorer
except Exception:
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.domain import get_domain_spec  # type: ignore
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.scorer import PackageScorer, train_scorer  # type: ignore


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_domain(cfg: Dict[str, Any]) -> str:
    d = cfg.get("domain")
    if isinstance(d, str) and d:
        return d.strip().lower()
    try:
        dom_num = int(d)
        mapping = cfg.get("domain_choose", {}) or {}
        for k, v in mapping.items():
            try:
                if int(v) == dom_num:
                    return str(k).strip().lower()
            except Exception:
                continue
    except Exception:
        pass
    return "pem"

def _all_domains(cfg: Dict[str, Any]) -> List[str]:
    mapping = cfg.get("domain_choose", {}) or {}
    if isinstance(mapping, dict) and mapping:
        try:
            return [k for k, _ in sorted(mapping.items(), key=lambda kv: int(kv[1]))]
        except Exception:
            return list(mapping.keys())
    return ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]


def _reward_from_record(rec: Dict[str, Any]) -> float:
    # 优先使用 labeled 的 validation
    val = rec.get("validation")
    if isinstance(val, dict):
        # Gemini 结果优先（若存在）
        gem = val.get("gemini") or {}
        if isinstance(gem, dict) and gem.get("used") and isinstance(gem.get("result"), str):
            r = str(gem.get("result")).strip()
            if r == "正确":
                return 1.0
            if r == "错误":
                return 0.0
        # 其次看 syntax
        syn = val.get("syntax") or {}
        if isinstance(syn, dict) and isinstance(syn.get("result"), str):
            r = str(syn.get("result")).strip()
            if r == "正确":
                return 1.0
            if r == "警告":
                return 0.5
            if r == "错误":
                return 0.0
    # 回退：使用 judge 结构
    j = rec.get("judge", {}) or {}
    syn = j.get("syntax", {}) or {}
    try:
        if (syn.get("errors") or []) or False:
            return 0.0
        if (syn.get("warnings") or []):
            return 0.5
        return 1.0
    except Exception:
        pass
    # 最后回退 label
    try:
        return 1.0 if int(rec.get("label", 0)) == 1 else 0.0
    except Exception:
        return 0.0


def train_from_debug_dataset(config_path: str | Path | None = None, domain_override: str | None = None) -> Path:
    mod_dir = Path(__file__).resolve().parent
    cfg_path = Path(config_path) if config_path else (mod_dir / "config.json")
    cfg = _load_json(cfg_path) or {}
    domains: List[str] = [str(domain_override).lower()] if domain_override else _all_domains(cfg)

    # 输出目录
    repo_root = Path(__file__).resolve().parents[5]
    out_root = repo_root / "out" / (cfg.get("output_dir", "out_pathfinder"))
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / ("train_" + str(int(_t.time())))
    run_dir.mkdir(parents=True, exist_ok=True)

    # 读取聚合数据
    ds_path = mod_dir / "train_datas" / "debug_dataset.json"
    data = _load_json(ds_path)
    if not isinstance(data, list):
        data = []
    # 特征构造：bag-of-ops + length（基于域的 op 集）
    op_set = set()
    for dom in domains:
        try:
            spec = get_domain_spec(dom)
            for cls in spec.op_classes:
                try:
                    nm = cls().name
                except Exception:
                    nm = cls.__name__
                op_set.add(str(nm))
        except Exception:
            continue
    op_names: List[str] = sorted(op_set)
    dom_index = {d: i for i, d in enumerate(domains)}

    X: List[List[float]] = []
    y: List[float] = []
    for rec in data:
        try:
            dom = str(rec.get("domain", "")).lower()
            if domain_override and dom != domains[0]:
                continue
            seq = list(rec.get("sequence", []) or [])
            length = float(len(seq))
            cnts = [float(seq.count(nm)) for nm in op_names]
            onehot = [0.0] * len(domains)
            if dom in dom_index:
                onehot[dom_index[dom]] = 1.0
            X.append([*cnts, length, *onehot])
            y.append(float(_reward_from_record(rec)))
        except Exception:
            continue

    # 训练
    feat_dim = len(op_names) + 1 + len(domains)
    model = PackageScorer(feat_dim)
    if len(X) > 0:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        train_scorer(model, X_t, y_t, epochs=int(cfg.get("epochs", 20)), batch_size=int(cfg.get("batch_size", 64)),
                     lr=float(cfg.get("learning_rate_actor", 3e-4)))

    torch.save(model.state_dict(), run_dir / "scorer.pt")
    # 写入简单元数据
    meta = {
        "domains": domains,
        "samples": len(X),
        "op_names": op_names,
        "feat_dim": feat_dim,
        "source": str(ds_path),
    }
    (run_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train_rl] saved scorer to: {run_dir / 'scorer.pt'}")
    return run_dir


if __name__ == "__main__":
    train_from_debug_dataset()



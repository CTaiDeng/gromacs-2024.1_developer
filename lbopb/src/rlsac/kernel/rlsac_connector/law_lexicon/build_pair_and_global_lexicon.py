#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

MODULES: List[str] = ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for anc in [p.parent] + list(p.parents):
        try:
            if (anc / ".git").exists():
                return anc
        except Exception:
            continue
    return p.parents[-1]


def _ensure_repo_in_sys_path() -> None:
    try:
        import lbopb  # type: ignore
        return
    except Exception:
        pass
    try:
        import sys as _sys
        root = _repo_root()
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
    except Exception:
        pass


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(p: Path, data: Any) -> None:
    txt = json.dumps(data, ensure_ascii=False, indent=2).replace("\r\n", "\n")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)


def collect_pairwise_entries(mono_dir: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    _ensure_repo_in_sys_path()
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for i, a in enumerate(MODULES):
        for b in MODULES[i + 1:]:
            fp = mono_dir / f"{a}_{b}_operator_packages.json"
            if not fp.exists():
                fp2 = mono_dir / f"{b}_{a}_operator_packages.json"
                if fp2.exists():
                    fp = fp2
                else:
                    continue
            arr = _read_json(fp)
            if not isinstance(arr, list):
                continue
            norm: List[Dict[str, Any]] = []
            # 延迟导入参数空间工具（存在则生成带参数的 ops_detailed）
            try:
                from lbopb.src.rlsac.kernel.rlsac_pathfinder.make_samples_with_params import synth_ops  # type: ignore
                base_root = Path(__file__).resolve().parents[2] / 'rlsac_pathfinder'
            except Exception:
                synth_ops = None  # type: ignore
                base_root = None  # type: ignore

            def build_steps(domain: str, seq: List[str]) -> List[Dict[str, Any]]:
                if synth_ops is None or base_root is None:
                    return [{"name": nm, "grid_index": [], "params": {}} for nm in seq]
                try:
                    steps = list(synth_ops(domain, seq, base_root))
                    # 兜底补齐结构
                    for st in steps:
                        if "params" not in st:
                            st["params"] = {}
                        if "grid_index" not in st:
                            st["grid_index"] = []
                    return steps
                except Exception:
                    return [{"name": nm, "grid_index": [], "params": {}} for nm in seq]
            for it in arr:
                seqs = it.get("sequences") or {}
                pa = list(seqs.get(a) or [])
                pb = list(seqs.get(b) or [])
                entry: Dict[str, Any] = {
                    "id": str(it.get("id", "")),
                    "pair": {"a": a, "b": b},
                    "sequences": {a: pa, b: pb},
                    "created_at": int(it.get("created_at", time.time())),
                    "updated_at": int(it.get("updated_at", time.time())),
                    "source": str(it.get("source", "pair_generated")),
                    "validation": it.get("validation", {}),
                }
                # 生成 ops_detailed（包含参数取值）
                entry["ops_detailed"] = {a: build_steps(a, pa), b: build_steps(b, pb)}
                norm.append(entry)
            out[(a, b)] = norm
    return out


def build_global_lexicon_from_pairs(pairs: Dict[Tuple[str, str], List[Dict[str, Any]]], *, max_items: int = 100) -> List[Dict[str, Any]]:
    _ensure_repo_in_sys_path()
    # 将每个域在全部 pair 列表中出现过的序列汇总，形成“七维可选池”
    pool: Dict[str, List[List[str]]] = {m: [] for m in MODULES}
    for (a, b), arr in pairs.items():
        for it in arr:
            seqs = it.get("sequences") or {}
            sa = list(seqs.get(a) or [])
            sb = list(seqs.get(b) or [])
            if sa:
                pool[a].append(sa)
            if sb:
                pool[b].append(sb)

    # 条件：七维都至少存在一个候选序列，才生成全局词条
    if not all(pool[m] for m in MODULES):
        return []

    # 策略：为每个域选“最长序列”作为代表，拼成一个全局词条；
    # 可随机采样若干条（此处生成 1 条 + 最多 max_items-1 条随机）
    def longest(lst: List[List[str]]) -> List[str]:
        return sorted(lst, key=lambda s: len(s), reverse=True)[0]

    import random
    items: List[Dict[str, Any]] = []

    # 1) 基准条目（各域最长）
    baseline = {m: longest(pool[m]) for m in MODULES}
    base_item: Dict[str, Any] = {
        "id": f"global_{int(time.time())}",
        "chosen": {m: baseline[m] for m in MODULES},
        "meta": {"strategy": "longest_per_domain"},
        "score": 0.0,
    }
    # 生成 per-domain ops_detailed（含参数取值），失败则回退为仅 name
    try:
        _ensure_repo_in_sys_path()
        from lbopb.src.rlsac.kernel.rlsac_pathfinder.make_samples_with_params import synth_ops  # type: ignore
        base_root = Path(__file__).resolve().parents[2] / 'rlsac_pathfinder'
        def build_steps(domain: str, seq: List[str]) -> List[Dict[str, Any]]:
            try:
                steps = list(synth_ops(domain, seq, base_root))
                for st in steps:
                    if "params" not in st:
                        st["params"] = {}
                    if "grid_index" not in st:
                        st["grid_index"] = []
                return steps
            except Exception:
                return [{"name": nm, "grid_index": [], "params": {}} for nm in seq]
        base_item["ops_detailed"] = {m: build_steps(m, baseline[m]) for m in MODULES}
    except Exception:
        base_item["ops_detailed"] = {m: [{"name": nm, "grid_index": [], "params": {}} for nm in baseline[m]] for m in MODULES}
    items.append(base_item)

    # 2) 随机条目（可选）
    for _ in range(max(0, max_items - 1)):
        choice = {m: list(random.choice(pool[m])) for m in MODULES}
        gi: Dict[str, Any] = {
            "id": f"global_{int(time.time())}_{random.randint(0, 10**6)}",
            "chosen": choice,
            "meta": {"strategy": "random_per_domain"},
            "score": 0.0,
        }
        try:
            _ensure_repo_in_sys_path()
            from lbopb.src.rlsac.kernel.rlsac_pathfinder.make_samples_with_params import synth_ops  # type: ignore
            base_root = Path(__file__).resolve().parents[2] / 'rlsac_pathfinder'
            def build_steps(domain: str, seq: List[str]) -> List[Dict[str, Any]]:
                try:
                    steps = list(synth_ops(domain, seq, base_root))
                    for st in steps:
                        if "params" not in st:
                            st["params"] = {}
                        if "grid_index" not in st:
                            st["grid_index"] = []
                    return steps
                except Exception:
                    return [{"name": nm, "grid_index": [], "params": {}} for nm in seq]
            gi["ops_detailed"] = {m: build_steps(m, choice[m]) for m in MODULES}
        except Exception:
            gi["ops_detailed"] = {m: [{"name": nm, "grid_index": [], "params": {}} for nm in choice[m]] for m in MODULES}
        items.append(gi)
    return items


def main() -> None:
    import sys
    mod_dir = Path(__file__).resolve().parents[1]
    mono_dir = mod_dir / "monoid_packages"
    out_dir = mod_dir / "law_lexicon"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 生成 pairs 级辞海
    pairs = collect_pairwise_entries(mono_dir)
    for (a, b), arr in pairs.items():
        _write_json(out_dir / f"lexicon_{a}_{b}.json", arr)

    # 2) 生成全局七维辞海（条件：每个域至少存在一个候选）
    global_items = build_global_lexicon_from_pairs(pairs, max_items=50)
    # 角色重复：仅保留 global_law_connections.json
    _write_json(out_dir / "global_law_connections.json", global_items)
    # 若历史存在 global_lexicon.json，则清理
    old = out_dir / "global_lexicon.json"
    try:
        if old.exists():
            old.unlink()
    except Exception:
        pass

    # 可选：清理由本次生成的 pairwise lexicon（临时文件）
    if any(arg in ("--cleanup", "--cleanup-pairwise") for arg in sys.argv[1:]):
        for (a, b) in pairs.keys():
            try:
                (out_dir / f"lexicon_{a}_{b}.json").unlink(missing_ok=True)
            except Exception:
                pass
    print(f"[lexicon] pairwise={len(pairs)} files, global={len(global_items)} entries written under {out_dir}")


if __name__ == "__main__":
    main()

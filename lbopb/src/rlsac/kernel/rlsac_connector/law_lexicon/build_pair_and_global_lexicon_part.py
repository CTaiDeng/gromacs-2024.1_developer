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
            for it in arr:
                seqs = it.get("sequences") or {}
                pa = list(seqs.get(a) or [])
                pb = list(seqs.get(b) or [])
                norm.append({
                    "id": str(it.get("id", "")),
                    "pair": {"a": a, "b": b},
                    "sequences": {a: pa, b: pb},
                })
            out[(a, b)] = norm
    return out


def build_partial_global_from_pairs(pairs: Dict[Tuple[str, str], List[Dict[str, Any]]], *, max_items: int = 100) -> List[Dict[str, Any]]:
    """寻找在两两约束集合下的“最大覆盖”部分七域组合（不完备），使用缺省参数输出。

    - 返回若干条覆盖域数尽可能多的组合；每条仅包含已覆盖域的 chosen/ops_detailed。
    - 每步 ops_detailed 采用缺省参数（grid_index: [], params: {}）。
    """
    _ensure_repo_in_sys_path()

    def key_of(seq: List[str]) -> Tuple[str, ...]:
        return tuple(str(x) for x in (seq or []))

    domain_index: Dict[str, int] = {m: i for i, m in enumerate(MODULES)}
    allowed: Dict[Tuple[str, str], set[Tuple[Tuple[str, ...], Tuple[str, ...]]]] = {}
    domain_pool: Dict[str, set[Tuple[str, ...]]] = {m: set() for m in MODULES}

    for (a, b), arr in pairs.items():
        a2, b2 = (a, b) if domain_index[a] < domain_index[b] else (b, a)
        st: set = allowed.setdefault((a2, b2), set())
        for it in arr:
            seqs = it.get("sequences") or {}
            sa = key_of(seqs.get(a2) or [])
            sb = key_of(seqs.get(b2) or [])
            st.add((sa, sb))
            if sa:
                domain_pool[a2].add(sa)
            if sb:
                domain_pool[b2].add(sb)

    # 允许域候选为空的情况（会导致最大覆盖 < 7）
    order = sorted(MODULES, key=lambda m: len(domain_pool[m]) if domain_pool[m] else 10**9)

    best_size = 0
    solutions: List[Dict[str, Tuple[str, ...]]] = []

    def consistent(assign: Dict[str, Tuple[str, ...]], m: str, val: Tuple[str, ...]) -> bool:
        mi = domain_index[m]
        for n, v in assign.items():
            a, b = (m, n) if domain_index[m] < domain_index[n] else (n, m)
            va, vb = (val, v) if a == m else (v, val)
            st = allowed.get((a, b))
            if st is None:
                # 没有该 pair 的约束，视为不允许（保守策略）
                return False
            if (va, vb) not in st:
                return False
        return True

    def backtrack(i: int, assign: Dict[str, Tuple[str, ...]]):
        nonlocal best_size
        if i == len(order):
            size = len(assign)
            if size >= best_size:
                if size > best_size:
                    best_size = size
                    solutions.clear()
                solutions.append(dict(assign))
            return
        m = order[i]
        # 分支1：选择一个候选（若存在）
        pool_m = domain_pool.get(m) or set()
        if pool_m:
            for val in pool_m:
                if consistent(assign, m, val):
                    assign[m] = val
                    backtrack(i + 1, assign)
                    assign.pop(m, None)
        # 分支2：跳过该域，继续尝试其它域的覆盖
        backtrack(i + 1, assign)

    backtrack(0, {})

    # 仅保留最大覆盖的若干解
    sols = [s for s in solutions if len(s) == best_size]
    # 限制条数
    sols = sols[:max_items]

    def default_steps(seq: List[str]) -> List[Dict[str, Any]]:
        return [{"name": nm, "grid_index": [], "params": {}} for nm in seq]

    items: List[Dict[str, Any]] = []
    for sol in sols:
        chosen = {m: list(map(str, sol[m])) for m in sol.keys()}
        missing = [m for m in MODULES if m not in sol]
        item: Dict[str, Any] = {
            "id": f"global_part_{int(time.time())}",
            "chosen": chosen,
            "missing_domains": missing,
            "meta": {"strategy": "partial_pairwise_max_cover", "size": len(chosen), "missing_count": len(missing)},
            "score": 0.0,
        }
        item["ops_detailed"] = {m: default_steps(chosen[m]) for m in chosen.keys()}
        items.append(item)
    return items


def main() -> None:
    mod_dir = Path(__file__).resolve().parents[1]
    mono_dir = mod_dir / "monoid_packages"
    out_dir = mod_dir / "law_lexicon"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairwise_entries(mono_dir)
    part_items = build_partial_global_from_pairs(pairs, max_items=100)
    _write_json(out_dir / "global_law_connections_part.json", part_items)
    print(f"[lexicon-part] written: {out_dir / 'global_law_connections_part.json'} items={len(part_items)} (best_cover={max((len(it.get('chosen',{})) for it in part_items), default=0)})")


if __name__ == "__main__":
    main()


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


def _color(txt: str, code: int) -> str:
    # ANSI color helper: 31=red, 32=green, 33=yellow, 36=cyan
    try:
        return f"\033[{code}m{txt}\033[0m"
    except Exception:
        return txt


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
    """在两两联络约束下，搜索“完备七域联络”的一致七域算子包（每域一条序列）。

    - 仅当找到满足所有 21 对约束的七域组合时才产出（每步含参数/取值）。
    - 若无解，返回空列表。
    """
    _ensure_repo_in_sys_path()

    # 1) 构建每个 pair 的“允许组合”集合，以及每个域的候选序列池
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

    # 覆盖检查：每个域必须至少有一个候选序列
    if not all(domain_pool[m] for m in MODULES):
        return []

    # 2) 回溯搜索一致赋值（满足所有 pair 约束）
    solutions: List[Dict[str, List[str]]] = []
    order = sorted(MODULES, key=lambda m: len(domain_pool[m]))  # 先搜候选少的域

    def consistent(assign: Dict[str, Tuple[str, ...]], m: str, val: Tuple[str, ...]) -> bool:
        mi = domain_index[m]
        for n, v in assign.items():
            a, b = (m, n) if domain_index[m] < domain_index[n] else (n, m)
            va, vb = (val, v) if a == m else (v, val)
            if (a, b) not in allowed:
                return False
            if (va, vb) not in allowed[(a, b)]:
                return False
        return True

    def backtrack(i: int, assign: Dict[str, Tuple[str, ...]]):
        if len(solutions) >= max_items:
            return
        if i == len(order):
            solutions.append({k: list(map(str, v)) for k, v in assign.items()})
            return
        m = order[i]
        for val in domain_pool[m]:
            if consistent(assign, m, val):
                assign[m] = val
                backtrack(i + 1, assign)
                assign.pop(m, None)

    backtrack(0, {})

    if not solutions:
        return []

    # 3) 构造含参数与取值的输出条目
    items: List[Dict[str, Any]] = []
    try:
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
    except Exception:
        def build_steps(domain: str, seq: List[str]) -> List[Dict[str, Any]]:  # type: ignore
            return [{"name": nm, "grid_index": [], "params": {}} for nm in seq]

    for sol in solutions[:max_items]:
        item: Dict[str, Any] = {
            "id": f"global_{int(time.time())}",
            "chosen": {m: list(sol[m]) for m in MODULES},
            "meta": {"strategy": "pairwise_complete_clique"},
            "score": 0.0,
        }
        item["ops_detailed"] = {m: build_steps(m, list(sol[m])) for m in MODULES}
        items.append(item)
    return items


def main() -> None:
    import sys
    mod_dir = Path(__file__).resolve().parents[1]
    mono_dir = mod_dir / "monoid_packages"
    out_dir = mod_dir / "law_lexicon"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载 pairs 级数据（不落盘 pairwise 文件）
    pairs = collect_pairwise_entries(mono_dir)

    # 覆盖性检查：必须具备“完备七域联络”，否则放弃并彩色打印结果
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
    missing = [m for m in MODULES if not pool[m]]
    if missing:
        print(_color("[global-lexicon] 覆盖性不足：缺少完备七域联络；放弃写入 global_law_connections.json。", 31))
        print(_color("域覆盖统计：", 36))
        for m in MODULES:
            cnt = len(pool[m])
            col = 32 if cnt > 0 else 31
            print(_color(f" - {m}: {cnt}", col))
        # 清理残留的 pairwise lexicon 文件，确保目录整洁
        for p in out_dir.glob("lexicon_*.json"):
            try:
                p.unlink()
            except Exception:
                pass
        return

    # 2) 搜索完备七域联络并写入（每次重写）
    global_items = build_global_lexicon_from_pairs(pairs, max_items=50)
    # 始终重写，若无解写入空数组
    _write_json(out_dir / "global_law_connections.json", global_items)
    # 若历史存在 global_lexicon.json，则清理
    old = out_dir / "global_lexicon.json"
    try:
        if old.exists():
            old.unlink()
    except Exception:
        pass

    # 始终清理残留的 pairwise lexicon 文件，确保只保留全局文件
    for p in out_dir.glob("lexicon_*.json"):
        try:
            p.unlink()
        except Exception:
            pass
    print(f"[lexicon] pairwise=0 files (skipped), global={len(global_items)} entries written under {out_dir}")


if __name__ == "__main__":
    main()

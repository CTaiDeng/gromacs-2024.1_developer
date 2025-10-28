# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _t
from pathlib import Path
from typing import Any, Dict, List
import importlib
import sys


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


def _ensure_repo_in_sys_path() -> None:
    try:
        import lbopb  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    try:
        root = _repo_root()
        sys.path.insert(0, str(root))
    except Exception:
        pass


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))


def _fmt(x: float | int) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def _stat(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": float(min(vals)),
        "max": float(max(vals)),
        "avg": float(sum(vals) / max(1, len(vals))),
    }


def _build_steps(domain: str, seq: List[str]) -> List[Dict[str, Any]]:
    """为给定 domain/seq 生成带参数与取值的 ops_detailed。

    - 优先使用 pathfinder 的参数空间工具合成（存在则返回带 params/grid_index 的 steps）
    - 若缺失参数空间，则回退为仅 name 的占位结构，并补齐空的 params 与 grid_index
    """
    try:
        _ensure_repo_in_sys_path()
        from lbopb.src.rlsac.kernel.rlsac_pathfinder.make_samples_with_params import synth_ops  # type: ignore
        base_root = _repo_root() / 'lbopb' / 'src' / 'rlsac' / 'kernel' / 'rlsac_pathfinder'
        steps = list(synth_ops(domain, seq, base_root))
        for st in steps:
            if 'params' not in st:
                st['params'] = {}
            if 'grid_index' not in st:
                st['grid_index'] = []
        return steps
    except Exception:
        return [{"name": nm, "grid_index": [], "params": {}} for nm in seq]


def _latest_dataset_debug(repo_root: Path) -> Path | None:
    ds_root = repo_root / 'out' / 'out_connector'
    ds_dirs = [p for p in ds_root.glob('dataset_*') if p.is_dir()]
    if not ds_dirs:
        return None
    def _ts_of(p: Path) -> int:
        try:
            return int(str(p.name).split('_', 1)[1])
        except Exception:
            return 0
    ds_dirs.sort(key=_ts_of, reverse=True)
    return ds_dirs[0] / 'debug_dataset.json'


def main() -> None:
    # 读取 out/out_connector/dataset_<ts>/debug_dataset.json（可传入路径或自动取最新），写入 train_datas
    mod_dir = Path(__file__).resolve().parents[1]
    out_dir = mod_dir / 'train_datas'
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_repo_in_sys_path()

    repo_root = _repo_root()
    # 可选参数：直接传入 debug_dataset.json 路径或 dataset 目录
    in_arg = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else None
    in_path: Path | None = None
    if in_arg is not None:
        if in_arg.is_dir():
            in_path = in_arg / 'debug_dataset.json'
        else:
            in_path = in_arg
    if in_path is None or not in_path.exists():
        in_path = _latest_dataset_debug(repo_root)
    if in_path is None or not in_path.exists():
        print("[collect] input debug_dataset.json not found")
        return

    data = _read_json(in_path) or {}
    samples = (data if isinstance(data, list) else (data.get('samples', []) or []))

    items: List[Dict[str, Any]] = []
    for rec in samples:
        try:
            pair_val = rec.get("pair")
            a = b = ""
            if isinstance(pair_val, str) and "_" in pair_val:
                parts = pair_val.split("_", 1)
                a, b = parts[0].strip().lower(), parts[1].strip().lower()
            elif isinstance(pair_val, dict):
                a = str((pair_val.get("a") or '')).lower()
                b = str((pair_val.get("b") or '')).lower()
            elif isinstance(pair_val, list) and len(pair_val) == 2:
                a = str(pair_val[0]).lower(); b = str(pair_val[1]).lower()

            # 兼容不同数据结构的序列来源
            seq_a = []
            seq_b = []
            if isinstance(rec.get('sequences'), dict):
                seqs = rec.get('sequences') or {}
                seq_a = list(seqs.get(a, []) or [])
                seq_b = list(seqs.get(b, []) or [])
            else:
                pa = rec.get('package_a') or rec.get('src') or {}
                pb = rec.get('package_b') or rec.get('dst') or {}
                seq_a = list((pa.get('sequence') or []))
                seq_b = list((pb.get('sequence') or []))
            # syntax check
            syn = None
            try:
                mod = importlib.import_module(f"lbopb.src.common.{a}_{b}_syntax_checker")
                syn = getattr(mod, "check")(seq_a, seq_b)
            except Exception:
                syn = None
            val = {
                "mode": "dual",
                "syntax": syn if isinstance(syn, dict) else {"result": "δ֪", "errors": 0, "warnings": 0},
                "gemini": {"used": False, "result": "δ֪"},
            }
            label = 1 if (isinstance(syn, dict) and int(syn.get("errors", 0)) == 0) else 0
            items.append(
                {
                    "id": rec.get("id"),
                    "pair": f"{a}_{b}",
                    "sequences": {a: seq_a, b: seq_b},
                    "ops_detailed": {a: _build_steps(a, seq_a), b: _build_steps(b, seq_b)},
                    "length": int(len(seq_a) + len(seq_b)),
                    "created_at": int(rec.get("created_at", _t.time())),
                    "updated_at": int(rec.get("updated_at", _t.time())),
                    "source": str(rec.get("source", "pair_from_packages")),
                    "label": label,
                    "validation": val,
                }
            )
        except Exception:
            continue

    # 排序与写出
    items.sort(key=lambda d: (d.get("pair", ""), int(d.get("length", 0))))
    _write_json(out_dir / "debug_dataset.json", items)
    print(f"[collect] written: {out_dir / 'debug_dataset.json'} items={len(items)}")

    # 统计
    lengths: List[float] = []
    by_pair: Dict[str, int] = {}
    by_label: Dict[str, int] = {"1": 0, "0": 0, "unknown": 0}
    per_pair: Dict[str, Dict[str, Any]] = {}
    for d in items:
        pr = str(d.get("pair", "")).lower()
        by_pair[pr] = int(by_pair.get(pr, 0)) + 1
        lv = d.get("label", None)
        try:
            by_label[str(int(lv))] = int(by_label.get(str(int(lv)), 0)) + 1
        except Exception:
            by_label["unknown"] += 1
        try:
            lengths.append(float(d.get("length", 0)))
        except Exception:
            pass
        st = per_pair.setdefault(pr, {"count": 0, "len_sum": 0.0, "labels": {"1": 0, "0": 0, "unknown": 0}})
        st["count"] = int(st.get("count", 0)) + 1
        st["len_sum"] = float(st.get("len_sum", 0.0)) + float(d.get("length", 0))
        try:
            st["labels"][str(int(lv))] = int(st["labels"].get(str(int(lv)), 0)) + 1
        except Exception:
            st["labels"]["unknown"] = int(st["labels"].get("unknown", 0)) + 1

    stats = {
        "updated_at": int(_t.time()),
        "total": len(items),
        "pairs": by_pair,
        "labels": by_label,
        "length": _stat(lengths),
        "per_pair": {k: {"count": v["count"], "avg_length": float(v["len_sum"]) / max(1, int(v["count"])) , "labels": v["labels"]} for k, v in per_pair.items()},
    }
    _write_json(out_dir / "debug_dataset.stats.json", stats)

    # markdown
    ts_local = _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(stats.get("updated_at", int(_t.time()))))
    md: List[str] = []
    md.append("# connector debug_dataset 统计（自动生成）")
    md.append("")
    md.append(f"- 生成时间：{ts_local}")
    md.append(f"- 样本总数：{int(stats.get('total', 0))}")
    md.append("")
    md.append("## 按 pair 计数（pairs）")
    for k, v in (stats.get("pairs", {}) or {}).items():
        md.append(f"- {k}: {int(v)}")
    md.append("")
    md.append("## 标签统计（labels）")
    labs = stats.get("labels", {}) or {}
    md.append(f"- 正确(1)：{int(labs.get('1', 0))}")
    md.append(f"- 错误(0)：{int(labs.get('0', 0))}")
    md.append(f"- 未知(unknown)：{int(labs.get('unknown', 0))}")
    md.append("")
    md.append("## 数值指标（min / max / avg）")
    sec = stats.get("length", {}) or {}
    md.append(f"- length: min={_fmt(sec.get('min', 0))} max={_fmt(sec.get('max', 0))} avg={_fmt(sec.get('avg', 0))}")
    md.append("")
    md.append("## 分 pair 统计（per_pair）")
    per = stats.get("per_pair", {}) or {}
    if isinstance(per, dict) and per:
        for k, st in per.items():
            md.append(f"### - {k}")
            md.append(f"- 样本数：{int(st.get('count', 0))}")
            md.append(f"- 平均 length：{_fmt(st.get('avg_length', 0.0))}")
            ls = st.get("labels", {}) or {}
            md.append(f"- 标签：1={int(ls.get('1', 0))} 0={int(ls.get('0', 0))} unknown={int(ls.get('unknown', 0))}")
    else:
        md.append("- <空>")
    with (out_dir / "debug_dataset.stats.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print(f"[collect] written: {out_dir / 'debug_dataset.stats.json'} and .md")


if __name__ == "__main__":
    main()

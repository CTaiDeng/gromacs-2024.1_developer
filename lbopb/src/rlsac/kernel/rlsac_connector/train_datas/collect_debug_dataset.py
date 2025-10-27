# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _t
from pathlib import Path
from typing import Any, Dict, List
import importlib


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


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _ensure_repo_in_sys_path() -> None:
    try:
        import lbopb  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    try:
        import sys as _sys
        root = _repo_root()
        _sys.path.insert(0, str(root))
    except Exception:
        pass


def main() -> None:
    # 读取 pair 辭海，生成 debug_dataset.json + 统计
    mod_dir = Path(__file__).resolve().parents[1]
    mono = mod_dir / "monoid_packages"
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_repo_in_sys_path()

    items: List[Dict[str, Any]] = []
    for f in sorted(mono.glob("*_operator_packages.json")):
        arr = _read_json(f) or []
        for it in (arr or []):
            try:
                pair = it.get("pair") or {}
                a = str(pair.get("a")).lower()
                b = str(pair.get("b")).lower()
                seqs = it.get("sequences") or {}
                seq_a = list(seqs.get(a, []) or [])
                seq_b = list(seqs.get(b, []) or [])
                # syntax check to fill validation
                syn = None
                try:
                    mod = importlib.import_module(f"lbopb.src.common.{a}_{b}_syntax_checker")
                    syn = getattr(mod, "check")(seq_a, seq_b)
                except Exception:
                    syn = None
                val = {
                    "mode": "dual",
                    "syntax": syn if isinstance(syn, dict) else {"result": "未知", "errors": 0, "warnings": 0},
                    "gemini": {"used": False, "result": "未知"},
                }
                label = 1 if (isinstance(syn, dict) and int(syn.get("errors", 0)) == 0) else 0
                items.append(
                    {
                        "id": it.get("id"),
                        "pair": f"{a}_{b}",
                        "sequences": {a: seq_a, b: seq_b},
                        "length": int(len(seq_a) + len(seq_b)),
                        "created_at": int(it.get("created_at", _t.time())),
                        "updated_at": int(it.get("updated_at", _t.time())),
                        "source": "connector_monoid",
                        "label": label,
                        "validation": val,
                    }
                )
            except Exception:
                continue

    # sort
    items.sort(key=lambda d: (d.get("pair", ""), int(d.get("length", 0))))
    _write_json(out_dir / "debug_dataset.json", items)
    print(f"[collect] written: {out_dir / 'debug_dataset.json'} items={len(items)}")

    # stats
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

    # md
    ts_local = _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(stats.get("updated_at", int(_t.time()))))
    md: List[str] = []
    md.append("# connector debug_dataset 统计（自动生成）")
    md.append("")
    md.append(f"- 更新时间：{ts_local}")
    md.append(f"- 样本总数：{int(stats.get('total', 0))}")
    md.append("")
    md.append("## 分布（pairs）")
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
            md.append(f"### · {k}")
            md.append(f"- 样本数：{int(st.get('count', 0))}")
            md.append(f"- 平均 length：{_fmt(st.get('avg_length', 0.0))}")
            ls = st.get("labels", {}) or {}
            md.append(f"- 标签：1={int(ls.get('1', 0))} 0={int(ls.get('0', 0))} unknown={int(ls.get('unknown', 0))}")
    else:
        md.append("- <空>")
    (out_dir / "debug_dataset.stats.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[collect] written: {out_dir / 'debug_dataset.stats.json'} and .md")


if __name__ == "__main__":
    main()

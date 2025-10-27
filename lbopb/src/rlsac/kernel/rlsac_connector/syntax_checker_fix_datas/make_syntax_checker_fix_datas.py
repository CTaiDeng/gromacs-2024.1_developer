# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _t
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def _write_json(p: Path, obj: Any) -> None:
    _write_text(p, json.dumps(obj, ensure_ascii=False, indent=2))


def _stat(vals: Iterable[float]) -> Dict[str, float]:
    arr = [float(x) for x in vals]
    if not arr:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {"min": min(arr), "max": max(arr), "avg": sum(arr) / max(1, len(arr))}


def _fmt(x: float | int) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    in_path = base / "train_datas" / "debug_dataset.json"
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    src = _read_json(in_path)
    if not isinstance(src, list):
        print(f"[split] 输入不可用：{in_path}")
        return

    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for it in src:
        try:
            pr = str(it.get("pair", "")).lower()
        except Exception:
            pr = "unknown"
        by_pair.setdefault(pr, []).append(it)

    for pr, arr in sorted(by_pair.items()):
        data_path = out_dir / f"{pr}_debug_dataset.json"
        stats_json_path = out_dir / f"{pr}_debug_dataset.stats.json"
        stats_md_path = out_dir / f"{pr}_debug_dataset.stats.md"

        _write_json(data_path, arr)
        lengths = [float(x.get("length", 0)) for x in arr]
        stats = {
            "updated_at": int(_t.time()),
            "total": len(arr),
            "pairs": {pr: len(arr)},
            "labels": {
                "1": sum(1 for x in arr if int(x.get("label", 0)) == 1),
                "0": sum(1 for x in arr if int(x.get("label", 0)) == 0),
                "unknown": 0,
            },
            "length": _stat(lengths),
            "per_pair": {
                pr: {
                    "count": len(arr),
                    "labels": {
                        "1": sum(1 for x in arr if int(x.get("label", 0)) == 1),
                        "0": sum(1 for x in arr if int(x.get("label", 0)) == 0),
                        "unknown": 0,
                    },
                    "avg_length": (sum(lengths) / max(1, len(lengths))) if lengths else 0.0,
                }
            },
        }
        _write_json(stats_json_path, stats)

        # md
        ts_local = _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(stats.get("updated_at", int(_t.time()))))
        md: List[str] = []
        md.append(f"# {pr}_debug_dataset 统计（自动生成）")
        md.append("")
        md.append(f"- 更新时间：{ts_local}")
        md.append(f"- 样本总数：{int(stats.get('total', 0))}")
        md.append("")
        md.append("## 分布（pairs）")
        md.append(f"- {pr}: {len(arr)}")
        md.append("")
        md.append("## 标签统计（labels）")
        labs = stats.get("labels", {}) or {}
        md.append(f"- 正确(1)：{int(labs.get('1', 0))}")
        md.append(f"- 错误(0)：{int(labs.get('0', 0))}")
        md.append(f"- 未知(unknown)：{int(labs.get('unknown', 0))}")
        md.append("")
        sec = stats.get("length", {}) or {}
        md.append("## 数值指标（min / max / avg）")
        md.append(f"- length: min={_fmt(sec.get('min', 0))} max={_fmt(sec.get('max', 0))} avg={_fmt(sec.get('avg', 0))}")
        md.append("")
        md.append("> 本文件由 make_syntax_checker_fix_datas.py 自动生成，基于 connector/train_datas/debug_dataset.json 的可读摘要。")
        _write_text(stats_md_path, "\n".join(md))
        print(f"[split] pair={pr} items={len(arr)} -> {data_path.name}")


if __name__ == "__main__":
    main()


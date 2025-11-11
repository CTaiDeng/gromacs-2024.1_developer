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
import time as _t
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def _write_json(path: Path, obj: Any) -> None:
    _write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _load_cost_lambda(base_dir: Path) -> float:
    try:
        cfg = _read_json(base_dir / "config.json") or {}
        return float(cfg.get("cost_lambda", 0.2))
    except Exception:
        return 0.2


def _stat(vals: Iterable[float]) -> Dict[str, float]:
    arr = [float(v) for v in vals]
    if not arr:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": float(min(arr)),
        "max": float(max(arr)),
        "avg": float(sum(arr) / max(1, len(arr))),
    }


def _fmt_num(x: float | int) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def _make_stats(subset: List[Dict[str, Any]], cost_lambda: float) -> Dict[str, Any]:
    total = len(subset)
    by_domain: Dict[str, int] = {}
    by_label: Dict[str, int] = {"1": 0, "0": 0, "unknown": 0}
    lengths: List[float] = []
    scores: List[float] = []
    risks: List[float] = []
    costs: List[float] = []
    per_domain: Dict[str, Dict[str, Any]] = {}

    for d in subset:
        dom = str(d.get("domain", "")).lower() or "unknown"
        by_domain[dom] = int(by_domain.get(dom, 0)) + 1
        lv = d.get("label", None)
        if lv is None:
            by_label["unknown"] += 1
        else:
            try:
                by_label[str(int(lv))] = int(by_label.get(str(int(lv)), 0)) + 1
            except Exception:
                by_label["unknown"] += 1
        try:
            lengths.append(float(d.get("length", 0)))
        except Exception:
            pass
        try:
            scores.append(float(d.get("score", 0.0)))
        except Exception:
            pass
        try:
            risks.append(float(d.get("delta_risk", 0.0)))
        except Exception:
            pass
        try:
            costs.append(float(d.get("cost", 0.0)))
        except Exception:
            pass

        st = per_domain.setdefault(
            dom,
            {
                "count": 0,
                "labels": {"1": 0, "0": 0, "unknown": 0},
                "score_sum": 0.0,
                "risk_sum": 0.0,
                "cost_sum": 0.0,
                "len_sum": 0.0,
            },
        )
        st["count"] = int(st.get("count", 0)) + 1
        try:
            st["score_sum"] = float(st.get("score_sum", 0.0)) + float(d.get("score", 0.0))
        except Exception:
            pass
        try:
            st["risk_sum"] = float(st.get("risk_sum", 0.0)) + float(d.get("delta_risk", 0.0))
        except Exception:
            pass
        try:
            st["cost_sum"] = float(st.get("cost_sum", 0.0)) + float(d.get("cost", 0.0))
        except Exception:
            pass
        try:
            st["len_sum"] = float(st.get("len_sum", 0.0)) + float(d.get("length", 0))
        except Exception:
            pass

    stats: Dict[str, Any] = {
        "updated_at": int(_t.time()),
        "total": total,
        "domains": by_domain,
        "labels": by_label,
        "length": _stat(lengths),
        "score": _stat(scores),
        "delta_risk": _stat(risks),
        "cost": _stat(costs),
        "per_domain": {},
    }
    for k, st in per_domain.items():
        cnt = int(st.get("count", 0))
        stats["per_domain"][k] = {
            "count": cnt,
            "labels": st.get("labels", {}),
            "avg_score": float(st.get("score_sum", 0.0)) / max(1, cnt),
            "avg_delta_risk": float(st.get("risk_sum", 0.0)) / max(1, cnt),
            "avg_cost": float(st.get("cost_sum", 0.0)) / max(1, cnt),
            "avg_length": float(st.get("len_sum", 0.0)) / max(1, cnt),
        }

    # attach a human-friendly summary text block for md generation
    stats["_meta"] = {"cost_lambda": float(cost_lambda)}
    return stats


def _write_stats_md(md_path: Path, title: str, stats: Dict[str, Any]) -> None:
    def _gz(x: Any, k: str, d: Any = 0) -> Any:
        try:
            v = (x or {}).get(k, d)
        except Exception:
            v = d
        return v

    ts_local = _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(int(stats.get("updated_at", int(_t.time())))))
    cost_lambda = float(_gz(stats, "_meta", {}).get("cost_lambda", 0.2))

    lines: List[str] = []
    lines.append(f"# {title} 说明（自动生成）")
    lines.append("")
    lines.append(f"- 更新时间：{ts_local}")
    lines.append(f"- 样本总数：{int(stats.get('total', 0))}")
    lines.append("")
    lines.append("## 分布（domains）")
    doms = stats.get("domains", {}) or {}
    if isinstance(doms, dict) and doms:
        for k, v in doms.items():
            lines.append(f"- {k}: {int(v)}")
    else:
        lines.append("- <空>")
    lines.append("")
    lines.append("## 标签统计（labels）")
    labs = stats.get("labels", {}) or {}
    lines.append(f"- 正确(1)：{int(labs.get('1', 0))}")
    lines.append(f"- 可疑(0)：{int(labs.get('0', 0))}")
    lines.append(f"- 未知(unknown)：{int(labs.get('unknown', 0))}")
    lines.append("")
    lines.append("## 数值指标（min / max / avg）")
    for key in ("length", "score", "delta_risk", "cost"):
        sec = stats.get(key, {}) or {}
        lines.append(
            f"- {key}: min={_fmt_num(sec.get('min', 0))} max={_fmt_num(sec.get('max', 0))} avg={_fmt_num(sec.get('avg', 0))}"
        )
    lines.append("")
    lines.append("## 指标解释")
    lines.append("- length：操作序列长度。一般越长越难，且综合评分可能更低。")
    lines.append(
        f"- score：综合评分；约定义为 score = delta_risk − cost_lambda × cost，cost_lambda={_fmt_num(cost_lambda)}。"
    )
    lines.append("- delta_risk：风险净减少，越大越好；体现为治疗有效度或负效应降低等。")
    lines.append("- cost：代价，越小越好；可综合时长、药物用量/副作用、工程复杂度或二次验证成本等。")
    try:
        dr_avg = float((stats.get("delta_risk", {}) or {}).get("avg", 0.0))
        c_avg = float((stats.get("cost", {}) or {}).get("avg", 0.0))
        sc_avg_est = dr_avg - float(cost_lambda) * c_avg
        sc_avg = float((stats.get("score", {}) or {}).get("avg", 0.0))
        lines.append(
            f"- 校验：约有 avg_score ≈ avg_delta_risk − cost_lambda × avg_cost = {_fmt_num(dr_avg)} − {_fmt_num(cost_lambda)} × {_fmt_num(c_avg)} = {_fmt_num(sc_avg_est)}；当前统计 avg_score={_fmt_num(sc_avg)}。"
        )
    except Exception:
        pass
    lines.append("")
    lines.append(
        "> 本文件由 make_syntax_checker_fix_datas.py 自动生成，基于 train_datas/debug_dataset.json 的可读摘要。"
    )

    _write_text(md_path, "\n".join(lines))


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]  # .../rlsac_pathfinder
    in_path = base_dir / "train_datas" / "debug_dataset.json"
    out_dir = base_dir / "syntax_checker_fix_datas"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = _read_json(in_path)
    if not isinstance(src, list):
        print(f"[split] 输入不是列表或无法解析：{in_path}")
        return

    # group by domain
    by_dom: Dict[str, List[Dict[str, Any]]] = {}
    for it in src:
        try:
            dom = str(it.get("domain", "")).lower() or "unknown"
        except Exception:
            dom = "unknown"
        by_dom.setdefault(dom, []).append(it)

    cost_lambda = _load_cost_lambda(base_dir)

    # per-domain emit json + stats
    for dom, arr in sorted(by_dom.items()):
        safe_dom = dom.replace("/", "_").replace("\\", "_")
        data_path = out_dir / f"{safe_dom}_debug_dataset.json"
        stats_json_path = out_dir / f"{safe_dom}_debug_dataset.stats.json"
        stats_md_path = out_dir / f"{safe_dom}_debug_dataset.stats.md"

        _write_json(data_path, arr)
        stats = _make_stats(arr, cost_lambda)
        _write_json(stats_json_path, stats)
        _write_stats_md(stats_md_path, f"{safe_dom}_debug_dataset.stats.json", stats)
        print(f"[split] domain={dom} items={len(arr)} -> {data_path.name}")

    # quick summary
    print(f"[split] 输入：{in_path}")
    print(f"[split] 输出目录：{out_dir}")


if __name__ == "__main__":
    main()

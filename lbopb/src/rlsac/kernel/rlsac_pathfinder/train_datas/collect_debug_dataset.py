# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _t
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hashlib


def _repo_root() -> Path:
    """Return the project root (directory that contains .git),
    falling back to a reasonable ancestor if not found.
    This ensures out/out_pathfinder resolves to the repo-root/out/out_pathfinder.
    """
    p = Path(__file__).resolve()
    # Walk up to locate a .git directory which marks repo root
    for anc in [p.parent] + list(p.parents):
        try:
            if (anc / ".git").exists():
                return anc
        except Exception:
            continue
    # Fallback: go up a fixed number of levels to approximate repo root
    try:
        return p.parents[6]
    except Exception:
        return p.parents[-1]


def _load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _space_steps(domain: str, seq: List[str]) -> Tuple[List[Dict[str, Any]] | None, Dict[str, Any] | None]:
    try:
        from lbopb.src.rlsac.kernel.rlsac_pathfinder.op_space_utils import load_op_space, param_grid_of, params_from_grid  # type: ignore
        base = Path(__file__).resolve().parents[1]
        space_ref = base / "operator_spaces" / f"{domain}_op_space.v1.json"
        if not space_ref.exists():
            return None, None
        space = load_op_space(str(space_ref))
        steps: List[Dict[str, Any]] = []
        for nm in seq:
            try:
                names, grids = param_grid_of(space, nm)
            except Exception:
                steps.append({"name": nm})
                continue
            gi: List[int] = []
            for g in grids:
                L = len(g)
                gi.append(max(0, (L - 1) // 2))
            prs = params_from_grid(space, nm, gi)
            steps.append({"name": nm, "grid_index": gi, "params": prs})
        meta = {
            "op_space_id": space.get("space_id", f"{domain}.v1"),
            "op_space_ref": str(space_ref.relative_to(_repo_root())).replace("\\", "/"),
        }
        return steps, meta
    except Exception:
        return None, None


def main() -> None:
    repo = _repo_root()
    # Always resolve out/out_pathfinder under the repository root
    out_root = (repo / "out" / "out_pathfinder").resolve()
    print(f"[collect] repo_root={repo}")
    print(f"[collect] scan dir={out_root}")
    if not out_root.exists():
        print(f"[collect] out root not found: {out_root}")
        return

    # 读取 config 以获取 cost_lambda（若需要）
    cfg_path = Path(__file__).resolve().parents[1] / "config.json"
    cfg = _load_json(cfg_path)
    cost_lambda = float(cfg.get("cost_lambda", 0.2))

    # 聚合：按 (domain + 序列 + 参数化取值) 去重，保留 score 更高
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    now = int(_t.time())
    # 1) 汇总 train_*/debug_dataset.json
    for p in out_root.glob("train_*/debug_dataset.json"):
        data = _load_json(p)
        domain = str(data.get("domain", "")).lower() or "pem"
        samples = data.get("samples", []) or []
        for s in samples:
            try:
                seq = list(s.get("sequence", []) or [])
                feats = s.get("features", {}) or {}
                dr = float(feats.get("delta_risk", 0.0))
                c = float(feats.get("cost", 0.0))
                length = int(feats.get("length", len(seq)))
                score = dr - cost_lambda * c
                label = int(s.get("label", 0))
                # 优先从样本中提取参数化步骤（如存在），否则按 v1 空间补全中位参数
                steps = None
                meta = None
                try:
                    cand = s.get("ops_detailed") or (s.get("judge", {}) or {}).get("ops_detailed")
                    if isinstance(cand, list) and len(cand) > 0:
                        steps = cand
                        meta = {k: s.get(k) for k in ("op_space_id", "op_space_ref") if k in s}
                except Exception:
                    steps = None
                    meta = None
                if not steps:
                    steps, meta = _space_steps(domain, seq)
                # 生成稳定 ID：基于序列+参数化取值的哈希
                payload = {
                    "sequence": list(seq),
                    "ops_detailed": steps or [],
                }
                sblob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                h = hashlib.sha1(sblob).hexdigest()
                num = int(h, 16) % (10**10)
                sid = f"pkg_{domain}_{num}"
                key = (domain, sid)
                item = {
                    "id": sid,
                    "domain": domain,
                    "sequence": seq,
                    "length": int(length),
                    "delta_risk": float(dr),
                    "cost": float(c),
                    "score": float(score),
                    "created_at": now,
                    "updated_at": now,
                    "source": "debug_dataset",
                    "label": int(label),
                }
                # 始终写入 ops_detailed（若无参数网格则用仅 name 的占位）
                item["ops_detailed"] = steps or [{"name": nm} for nm in seq]
                if meta:
                    item.update(meta)
                prev = by_key.get(key)
                if (prev is None) or (float(score) > float(prev.get("score", -1e9))):
                    by_key[key] = item
            except Exception:
                continue

    # 2) 汇总各次训练产出的 *_operator_packages_labeled.json（项目根 out/out_pathfinder 下）
    for p in out_root.rglob("*_operator_packages_labeled.json"):
        try:
            arr = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for it in (arr or []):
            try:
                domain = str(it.get("domain", "")).lower() or p.name.split("_")[0].lower()
                seq = list(it.get("sequence", []) or [])
                steps = it.get("ops_detailed") if isinstance(it.get("ops_detailed"), list) else []
                payload = {"sequence": list(seq), "ops_detailed": steps or []}
                sblob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                h = hashlib.sha1(sblob).hexdigest()
                num = int(h, 16) % (10**10)
                sid = f"pkg_{domain}_{num}"
                key = (domain, sid)
                dr = float(it.get("delta_risk", 0.0))
                c = float(it.get("cost", 0.0))
                length = int(it.get("length", len(seq)))
                score = float(it.get("score", dr - cost_lambda * c))
                now2 = int(it.get("updated_at", now))
                label = int(it.get("label", 0))
                meta = {k: it.get(k) for k in ("op_space_id", "op_space_ref") if k in it}
                item = {
                    "id": sid,
                    "domain": domain,
                    "sequence": seq,
                    "length": length,
                    "delta_risk": dr,
                    "cost": c,
                    "score": score,
                    "created_at": int(it.get("created_at", now2)),
                    "updated_at": now2,
                    "source": "labeled_operator_packages",
                    "label": label,
                    "ops_detailed": steps or [{"name": nm} for nm in seq],
                }
                if meta:
                    item.update(meta)
                prev = by_key.get(key)
                if (prev is None) or (float(score) > float(prev.get("score", -1e9))):
                    by_key[key] = item
            except Exception:
                continue

    # 排序并写入 train_datas/debug_dataset.json
    items = list(by_key.values())
    items.sort(key=lambda d: (
        -float(d.get("score", 0.0)),
        int(d.get("length", 0)),
        tuple(str(x) for x in d.get("sequence", []))
    ))
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "debug_dataset.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[collect] written: {out_path} items={len(items)}")

    # 同步生成配套统计：debug_dataset.stats.json（针对上述 debug_dataset.json）
    try:
        total = len(items)
        by_domain: Dict[str, int] = {}
        by_label: Dict[str, int] = {"1": 0, "0": 0, "unknown": 0}
        lengths: List[int] = []
        scores: List[float] = []
        risks: List[float] = []
        costs: List[float] = []
        per_domain: Dict[str, Dict[str, Any]] = {}

        for d in items:
            dom = str(d.get("domain", "")).lower() or "unknown"
            by_domain[dom] = int(by_domain.get(dom, 0)) + 1
            # label 统计
            lv = d.get("label", None)
            if lv is None:
                by_label["unknown"] += 1
            else:
                try:
                    by_label[str(int(lv))] = int(by_label.get(str(int(lv)), 0)) + 1
                except Exception:
                    by_label["unknown"] += 1
            # 累计标量
            try:
                lengths.append(int(d.get("length", 0)))
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

            # 域内统计
            st = per_domain.setdefault(dom, {
                "count": 0,
                "labels": {"1": 0, "0": 0, "unknown": 0},
                "score_sum": 0.0,
                "risk_sum": 0.0,
                "cost_sum": 0.0,
                "len_sum": 0,
            })
            st["count"] += 1
            try:
                st["labels"][str(int(lv))] += 1  # type: ignore[index]
            except Exception:
                st["labels"]["unknown"] += 1
            try:
                st["score_sum"] += float(d.get("score", 0.0))
            except Exception:
                pass
            try:
                st["risk_sum"] += float(d.get("delta_risk", 0.0))
            except Exception:
                pass
            try:
                st["cost_sum"] += float(d.get("cost", 0.0))
            except Exception:
                pass
            try:
                st["len_sum"] += int(d.get("length", 0))
            except Exception:
                pass

        def _summary(nums: List[float | int]) -> Dict[str, float]:
            if not nums:
                return {"min": 0.0, "max": 0.0, "avg": 0.0}
            vals = [float(x) for x in nums]
            return {"min": min(vals), "max": max(vals), "avg": (sum(vals) / len(vals))}

        # 整体统计
        stats: Dict[str, Any] = {
            "total": total,
            "domains": by_domain,
            "labels": by_label,
            "length": _summary(lengths),
            "score": _summary(scores),
            "delta_risk": _summary(risks),
            "cost": _summary(costs),
            "updated_at": int(_t.time()),
        }
        # 分域平均
        stats_domain: Dict[str, Any] = {}
        for dom, st in per_domain.items():
            c = max(int(st.get("count", 0)), 1)
            stats_domain[dom] = {
                "count": int(st.get("count", 0)),
                "labels": st.get("labels", {}),
                "avg_score": float(st.get("score_sum", 0.0)) / c,
                "avg_delta_risk": float(st.get("risk_sum", 0.0)) / c,
                "avg_cost": float(st.get("cost_sum", 0.0)) / c,
                "avg_length": float(st.get("len_sum", 0)) / c,
            }
        stats["per_domain"] = stats_domain

        stats_path = out_dir / "debug_dataset.stats.json"
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[collect] written: {stats_path}")

        # 生成 Markdown 配套说明（动态）并在每次运行时覆盖：debug_dataset.stats.md
        def _fmt_num(x: float | int) -> str:
            try:
                return f"{float(x):.3f}"
            except Exception:
                return str(x)

        ts_local = _t.strftime('%Y-%m-%d %H:%M:%S', _t.localtime(stats.get('updated_at', int(_t.time()))))
        md_lines: List[str] = []
        md_lines.append("# debug_dataset.stats.json 配套说明（自动生成）")
        md_lines.append("")
        md_lines.append(f"- 生成时间：{ts_local}")
        md_lines.append(f"- 样本总数：{int(stats.get('total', 0))}")
        md_lines.append("")
        md_lines.append("## 域分布（domains）")
        doms = stats.get('domains', {}) or {}
        if isinstance(doms, dict) and doms:
            for k, v in doms.items():
                md_lines.append(f"- {k}: {int(v)}")
        else:
            md_lines.append("- <空>")
        md_lines.append("")
        md_lines.append("## 标签统计（labels）")
        labs = stats.get('labels', {}) or {}
        md_lines.append(f"- 正样本(1)：{int(labs.get('1', 0))}")
        md_lines.append(f"- 负样本(0)：{int(labs.get('0', 0))}")
        md_lines.append(f"- 未知(unknown)：{int(labs.get('unknown', 0))}")
        md_lines.append("")
        md_lines.append("## 数值汇总（min / max / avg）")
        for key in ("length", "score", "delta_risk", "cost"):
            sec = stats.get(key, {}) or {}
            md_lines.append(f"- {key}: min={_fmt_num(sec.get('min', 0))} max={_fmt_num(sec.get('max', 0))} avg={_fmt_num(sec.get('avg', 0))}")
        md_lines.append("")
        # 指标解读与举例
        md_lines.append("## 指标解读（含举例）")
        md_lines.append("- length：算子包序列的长度（操作步数）。一般越短越精简，但需结合得分与收益综合评估。")
        md_lines.append(
            f"- score：综合得分，用于训练与筛选。定义为 score = delta_risk − cost_lambda × cost（当前 cost_lambda={_fmt_num(cost_lambda)}）。"
        )
        md_lines.append("- delta_risk：收益项（越大越好），可理解为风险下降/效用提升的度量；为负表示变差。")
        md_lines.append("- cost：代价/资源消耗项（越小越好），是多因素的抽象，例如时间、药物毒性/不良反应、价格、操作难度或临床风险等。")
        # 一行校核（用均值做近似校核）
        try:
            dr_avg = float((stats.get('delta_risk', {}) or {}).get('avg', 0.0))
            c_avg = float((stats.get('cost', {}) or {}).get('avg', 0.0))
            sc_avg_est = dr_avg - float(cost_lambda) * c_avg
            sc_avg = float((stats.get('score', {}) or {}).get('avg', 0.0))
            md_lines.append(
                f"- 校核：约有 avg_score ≈ avg_delta_risk − cost_lambda × avg_cost ≈ {_fmt_num(dr_avg)} − {_fmt_num(cost_lambda)} × {_fmt_num(c_avg)} ≈ {_fmt_num(sc_avg_est)}（当前统计 avg_score={_fmt_num(sc_avg)}）"
            )
        except Exception:
            pass
        md_lines.append("")
        md_lines.append("## 按域统计（per_domain）")
        per = stats.get('per_domain', {}) or {}
        if isinstance(per, dict) and per:
            for k, st in per.items():
                try:
                    md_lines.append(f"### 域：{k}")
                    md_lines.append(f"- 样本数：{int(st.get('count', 0))}")
                    ls = st.get('labels', {}) or {}
                    md_lines.append(f"- 标签：1={int(ls.get('1', 0))} 0={int(ls.get('0', 0))} unknown={int(ls.get('unknown', 0))}")
                    md_lines.append(
                        "- 均值：score="
                        + _fmt_num(st.get('avg_score', 0))
                        + ", delta_risk="
                        + _fmt_num(st.get('avg_delta_risk', 0))
                        + ", cost="
                        + _fmt_num(st.get('avg_cost', 0))
                        + ", length="
                        + _fmt_num(st.get('avg_length', 0))
                    )
                except Exception:
                    continue
        else:
            md_lines.append("- <空>")

        md_lines.append("")
        md_lines.append("> 本文件由 collect_debug_dataset.py 每次运行自动覆盖生成，用于配合 debug_dataset.stats.json 的可读化展示。")
        md_path = out_dir / "debug_dataset.stats.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"[collect] written: {md_path}")
    except Exception as e:
        print(f"[collect] stats failed: {e}")


if __name__ == "__main__":
    main()

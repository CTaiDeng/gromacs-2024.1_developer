# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _t
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hashlib


def _repo_root() -> Path:
    """Locate repository root by walking up to find .git."""
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


def _load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _space_steps(domain: str, seq: List[str]) -> Tuple[List[Dict[str, Any]] | None, Dict[str, Any] | None]:
    """Fill ops_detailed using v1 operator space with median grid, repo-relative ref."""
    try:
        from lbopb.src.rlsac.kernel.rlsac_pathfinder.op_space_utils import (  # type: ignore
            load_op_space, param_grid_of, params_from_grid,
        )
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
    out_root = (repo / "out" / "out_pathfinder").resolve()
    print(f"[collect] repo_root={repo}")
    print(f"[collect] scan dir={out_root}")
    if not out_root.exists():
        print(f"[collect] out root not found: {out_root}")
        return

    # load cost_lambda
    cfg_path = Path(__file__).resolve().parents[1] / "config.json"
    cfg = _load_json(cfg_path)
    cost_lambda = float(cfg.get("cost_lambda", 0.2))

    # merge key: (domain, sid) where sid uses sha1(domain+sequence+ops_detailed)
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    now = int(_t.time())

    # 1) collect from dataset_*_<domain>/debug_dataset.json
    for p in out_root.glob("dataset_*_*/debug_dataset.json"):
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

                # prefer provided ops_detailed; else synthesize from v1
                steps: List[Dict[str, Any]] | None = None
                meta: Dict[str, Any] | None = None
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

                payload = {"domain": domain, "sequence": list(seq), "ops_detailed": steps or []}
                sblob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                # 使用十六进制哈希（SHA-256），不再转为十进制
                h = hashlib.sha256(sblob).hexdigest()
                sid = f"pkg_{domain}_{h}"
                key = (domain, sid)

                # 可用 judge 结构构造 validation（若存在）
                validation = None
                try:
                    j = s.get("judge", {}) or {}
                    syn = j.get("syntax", {}) or {}
                    syn_errs = list(syn.get("errors", []) or [])
                    syn_warns = list(syn.get("warnings", []) or [])
                    if syn_errs:
                        _syn_res_cn = "错误"
                    elif syn_warns:
                        _syn_res_cn = "警告"
                    else:
                        _syn_res_cn = "正确"
                    _llm_attempted = bool(j.get("llm_attempted", False))
                    _gem_res_cn = None
                    try:
                        _used = bool(j.get("llm_used", False)) and str(j.get("llm_status", "")) == "used"
                        _raw = j.get("llm_raw", None)
                        if _llm_attempted and _used and isinstance(_raw, str):
                            _s = _raw.strip()
                            if (_s == "1") or ("1" in _s and "0" not in _s):
                                _gem_res_cn = "正确"
                            elif (_s == "0") or ("0" in _s and "1" not in _s):
                                _gem_res_cn = "错误"
                    except Exception:
                        _gem_res_cn = None
                    validation = {
                        "mode": "dual" if _llm_attempted else "syntax_only",
                        "syntax": {
                            "result": _syn_res_cn,
                            "errors": len(syn_errs),
                            "warnings": len(syn_warns),
                        },
                        "gemini": ({
                            "used": True,
                            **({"result": _gem_res_cn} if _gem_res_cn is not None else {}),
                        } if _llm_attempted else {"used": False}),
                    }
                except Exception:
                    validation = None

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
                    "ops_detailed": steps or [{"name": nm} for nm in seq],
                }
                if validation:
                    item["validation"] = validation
                if meta:
                    item.update(meta)
                prev = by_key.get(key)
                if (prev is None) or (float(score) > float(prev.get("score", -1e9))):
                    by_key[key] = item
            except Exception:
                continue

    # 2) merge labeled operator packages within dataset_*_<domain> subtrees
    for ds in out_root.glob("dataset_*_*"):
        for p in ds.rglob("*_operator_packages_labeled.json"):
            try:
                arr = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            for it in (arr or []):
                try:
                    domain = str(it.get("domain", "")).lower() or ds.name.split("_")[-1].lower()
                    seq = list(it.get("sequence", []) or [])
                    steps = it.get("ops_detailed") if isinstance(it.get("ops_detailed"), list) else []
                    payload = {"domain": domain, "sequence": list(seq), "ops_detailed": steps or []}
                    sblob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    h = hashlib.sha256(sblob).hexdigest()
                    sid = f"pkg_{domain}_{h}"
                    key = (domain, sid)
                    dr = float(it.get("delta_risk", 0.0))
                    c = float(it.get("cost", 0.0))
                    length = int(it.get("length", len(seq)))
                    score = float(it.get("score", dr - cost_lambda * c))
                    now2 = int(it.get("updated_at", now))
                    label = int(it.get("label", 0))
                    meta = {k: it.get(k) for k in ("op_space_id", "op_space_ref") if k in it}
                    validation = it.get("validation", None)
                    if not isinstance(validation, dict):
                        validation = None
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
                    if validation:
                        item["validation"] = validation
                    if meta:
                        item.update(meta)
                    prev = by_key.get(key)
                    if (prev is None) or (float(score) > float(prev.get("score", -1e9))):
                        by_key[key] = item
                except Exception:
                    continue

    # write merged dataset
    items = list(by_key.values())
    items.sort(key=lambda d: (
        -float(d.get("score", 0.0)),
        int(d.get("length", 0)),
        tuple(str(x) for x in d.get("sequence", []))
    ))
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "debug_dataset.json"
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(items, ensure_ascii=False, indent=2))
    print(f"[collect] written: {out_path} items={len(items)}")

    # write stats (json + md)
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
            lv = d.get("label", None)
            if lv is None:
                by_label["unknown"] += 1
            else:
                try:
                    by_label[str(int(lv))] = int(by_label.get(str(int(lv)), 0)) + 1
                except Exception:
                    by_label["unknown"] += 1
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

            st = per_domain.setdefault(dom, {
                "count": 0,
                "labels": {"1": 0, "0": 0, "unknown": 0},
                "score_sum": 0.0,
                "risk_sum": 0.0,
                "cost_sum": 0.0,
                "len_sum": 0,
            })
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
                st["len_sum"] = int(st.get("len_sum", 0)) + int(d.get("length", 0))
            except Exception:
                pass

        def _stat(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"min": 0.0, "max": 0.0, "avg": 0.0}
            return {
                "min": float(min(vals)),
                "max": float(max(vals)),
                "avg": float(sum(vals) / max(1, len(vals)))
            }

        stats = {
            "updated_at": int(_t.time()),
            "total": total,
            "domains": by_domain,
            "labels": by_label,
            "length": _stat([float(x) for x in lengths]),
            "score": _stat([float(x) for x in scores]),
            "delta_risk": _stat([float(x) for x in risks]),
            "cost": _stat([float(x) for x in costs]),
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
                "avg_length": float(st.get("len_sum", 0)) / max(1, cnt),
            }

        out_stats = out_dir / "debug_dataset.stats.json"
        with out_stats.open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"[collect] written: {out_stats}")

        # also write human-readable markdown
        def _fmt_num(x: float | int) -> str:
            try:
                return f"{float(x):.3f}"
            except Exception:
                return str(x)

        ts_local = _t.strftime('%Y-%m-%d %H:%M:%S', _t.localtime(stats.get('updated_at', int(_t.time()))))
        md_lines: List[str] = []
        md_lines.append("# debug_dataset.stats.json 说明（自动生成）")
        md_lines.append("")
        md_lines.append(f"- 生成时间：{ts_local}")
        md_lines.append(f"- 样本总数：{int(stats.get('total', 0))}")
        md_lines.append("")
        md_lines.append("## 分布（domains）")
        doms = stats.get('domains', {}) or {}
        if isinstance(doms, dict) and doms:
            for k, v in doms.items():
                md_lines.append(f"- {k}: {int(v)}")
        else:
            md_lines.append("- <空>")
        md_lines.append("")
        md_lines.append("## 标签统计（labels）")
        labs = stats.get('labels', {}) or {}
        md_lines.append(f"- 正确(1)：{int(labs.get('1', 0))}")
        md_lines.append(f"- 错误(0)：{int(labs.get('0', 0))}")
        md_lines.append(f"- 未知(unknown)：{int(labs.get('unknown', 0))}")
        md_lines.append("")
        md_lines.append("## 数值概览（min / max / avg）")
        for key in ("length", "score", "delta_risk", "cost"):
            sec = stats.get(key, {}) or {}
            md_lines.append(f"- {key}: min={_fmt_num(sec.get('min', 0))} max={_fmt_num(sec.get('max', 0))} avg={_fmt_num(sec.get('avg', 0))}")
        md_lines.append("")
        md_lines.append("## 指标解读与示例")
        md_lines.append("- length：算子包序列长度。一般越短越优，但需综合评分考量。")
        md_lines.append(f"- score：综合评分，当前筛选规则为 score = delta_risk − cost_lambda × cost（cost_lambda={_fmt_num(cost_lambda)}）。")
        md_lines.append("- delta_risk：收益指标，越大越好。可理解为病灶负担下降/疗效提升等抽象。")
        md_lines.append("- cost：成本（越小越好），综合抽象了时间、药物毒性/不良反应、价格、操作难度或临床风险等。")
        try:
            dr_avg = float((stats.get('delta_risk', {}) or {}).get('avg', 0.0))
            c_avg = float((stats.get('cost', {}) or {}).get('avg', 0.0))
            sc_avg_est = dr_avg - float(cost_lambda) * c_avg
            sc_avg = float((stats.get('score', {}) or {}).get('avg', 0.0))
            md_lines.append(
                f"- 校验：约有 avg_score ≈ avg_delta_risk − cost_lambda × avg_cost = {_fmt_num(dr_avg)} − {_fmt_num(cost_lambda)} × {_fmt_num(c_avg)} = {_fmt_num(sc_avg_est)}；当前统计 avg_score={_fmt_num(sc_avg)}。"
            )
        except Exception:
            pass
        md_lines.append("")
        md_lines.append("## 分域统计（per_domain）")
        per = stats.get('per_domain', {}) or {}
        if isinstance(per, dict) and per:
            for k, st in per.items():
                try:
                    md_lines.append(f"### 域 {k}")
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
        md_lines.append("> 本文件由 collect_debug_dataset.py 每次运行自动覆盖生成，配合 debug_dataset.json 的可读呈现。")
        md_path = out_dir / "debug_dataset.stats.md"
        with md_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(md_lines))
        print(f"[collect] written: {md_path}")
    except Exception as e:
        print(f"[collect] stats failed: {e}")


if __name__ == "__main__":
    main()

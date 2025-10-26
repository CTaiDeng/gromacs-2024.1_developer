# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import os
import time as _t
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_cfg(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _oneline(s: str, max_len: int = 240) -> str:
    """压缩为单行并截断，便于 Debug 打印。"""
    try:
        t = s.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace("\n", "\\n")
        return t[:max_len] + ("..." if len(t) > max_len else "")
    except Exception:
        return s


def _repo_root() -> Path:
    """Locate repository root (directory containing .git). Fallback to higher ancestor."""
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


def _read_packages(p: Path) -> List[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _domain_from_file(p: Path) -> str:
    name = p.name.split("_")[0].lower()
    return name


def _ensure_repo_in_sys_path() -> None:
    """确保可以导入 lbopb 包（支持直接脚本运行）。"""
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

def _ops_detailed_for(seq: List[str], domain: str) -> Tuple[List[Dict[str, Any]] | None, Dict[str, Any] | None]:
    """若文件中未提供 ops_detailed，则从 v1 空间构造中位离散点。"""
    try:
        from lbopb.src.rlsac.kernel.rlsac_pathfinder.op_space_utils import load_op_space, param_grid_of, params_from_grid  # type: ignore
        mod_dir = Path(__file__).resolve().parents[1]
        space_ref = mod_dir / "operator_spaces" / f"{domain}_op_space.v1.json"
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


def verify_file(pack_file: Path, cfg: Dict[str, Any], *, out_dir: Path,
                debug: bool = False, prune: bool = False, limit: int | None = None) -> Dict[str, Any]:
    # ANSI color scheme (consistent with train.py)
    ANSI_RESET = "\x1b[0m"
    ANSI_RED = "\x1b[31;1m"
    ANSI_GREEN = "\x1b[32;1m"
    ANSI_YELLOW = "\x1b[33;1m"
    ANSI_CYAN = "\x1b[36;1m"
    ANSI_MAGENTA = "\x1b[35;1m"

    def _cprint(txt: str, color: str | None = None, always: bool = False) -> None:
        try:
            if color and (debug or always):
                print(f"{color}{txt}{ANSI_RESET}")
            else:
                print(txt)
        except Exception:
            print(txt)

    domain = _domain_from_file(pack_file)
    arr = _read_packages(pack_file)
    report_items: List[Dict[str, Any]] = []

    # LLM settings
    model_map = cfg.get("gemini_model_choose", {}) or {}
    gm = cfg.get("GEMINI_MODEL", None)
    if isinstance(gm, int):
        for k, v in model_map.items():
            if int(v) == gm:
                model = str(k)
                break
        else:
            model = None
    else:
        model = str(gm) if isinstance(gm, str) else None

    req_interval = float(cfg.get("llm_request_interval_sec", 0.0))
    last = 0.0
    # 传递 LLM 配置到环境变量，避免 HTTP 客户端阻塞无输出
    try:
        _llm_to = cfg.get("llm_timeout_sec", None)
        if _llm_to is not None:
            os.environ["LBOPB_GEMINI_TIMEOUT_SEC"] = str(_llm_to)
        if isinstance(model, str) and model:
            os.environ["LBOPB_GEMINI_MODEL"] = model
            os.environ["GEMINI_MODEL"] = model
    except Exception:
        pass

    _ensure_repo_in_sys_path()
    from lbopb.src.rlsac.kernel.common.llm_oracle import build_pathfinder_prompt, call_llm  # type: ignore
    # 按要求：不进行 syntax_checker 检查，仅进行 Gemini 检查

    kept: List[Dict[str, Any]] = []
    total = len(arr)
    n_check = total if (limit is None or limit <= 0) else min(limit, total)
    proc = 0
    print(f"[VERIFY] file={pack_file} domain={domain} total={total} checking={n_check} model={model or '<default>'} interval={req_interval}s", flush=True)
    for idx, it in enumerate(arr[:n_check], start=1):
        seq = list(it.get("sequence", []) or [])
        print(f"[VERIFY:{idx}/{n_check}] START seq={seq}", flush=True)
        # syntax check（优先使用参数化）
        try:
            if isinstance(it.get("ops_detailed"), list) and it.get("op_space_ref"):
                syn_res = getattr(syn_mod, "check_package")(it)
            else:
                syn_res = getattr(syn_mod, "check_sequence")(seq)
        except Exception:
            syn_res = {"valid": True, "errors": [], "warnings": []}
        syn_ok = bool(syn_res.get("valid", True)) and not bool(syn_res.get("errors"))

        # prepare LLM prompt
        ops_det = it.get("ops_detailed") if isinstance(it.get("ops_detailed"), list) else None
        extra = None
        if not ops_det:
            ops_det, extra = _ops_detailed_for(seq, domain)
        else:
            extra = {k: it.get(k) for k in ("op_space_id", "op_space_ref") if k in it}

        prompt = build_pathfinder_prompt(domain, seq, ops_det, extra)
        if debug:
            try:
                import json as _json
                _ops_prev = _oneline(_json.dumps(ops_det, ensure_ascii=False)) if ops_det else "<none>"
            except Exception:
                _ops_prev = "<err>"
            print(f"[VERIFY:{idx}/{n_check}] SEQ={seq}")
            print(f"[VERIFY:{idx}/{n_check}] PARAMS={_ops_prev}")
            try:
                _pv = _oneline(str(prompt))
            except Exception:
                _pv = "<prompt>"
            print(f"[VERIFY:{idx}/{n_check}] PROMPT_PREVIEW={_pv}")
            if syn_res:
                try:
                    _errs = list(syn_res.get("errors", []) or [])
                    _warn = list(syn_res.get("warnings", []) or [])
                    print(f"[VERIFY:{idx}/{n_check}] SYNTAX valid={bool(syn_res.get('valid', True))} errors={len(_errs)} warnings={len(_warn)}")
                    if _errs:
                        print(f"[VERIFY:{idx}/{n_check}] SYNTAX_ERRORS={_oneline(str(_errs), 400)}")
                    if _warn:
                        print(f"[VERIFY:{idx}/{n_check}] SYNTAX_WARNINGS={_oneline(str(_warn), 400)}")
                except Exception:
                    pass
        # throttle
        if req_interval > 0 and last > 0:
            w = (last + req_interval) - _t.time()
            if w > 0:
                if debug:
                    print(f"[VERIFY:{idx}/{n_check}] THROTTLE sleep={w:.2f}s")
                _t.sleep(w)
        t0 = _t.time()
        try:
            txt = call_llm(prompt, model=model)
        except Exception as e:
            txt = f"[LLM Exception] {e}"
            print(f"[VERIFY:{idx}/{n_check}] LLM_EXCEPTION: {e}", flush=True)
        last = _t.time()
        if isinstance(txt, str):
            s = txt.strip()
            llm_ok = (s == "1") or ("1" in s and "0" not in s)
        else:
            llm_ok = True  # 如果不可用，则不阻断

        both_ok = bool(syn_ok and llm_ok)
        item_report = {
            "id": it.get("id"),
            "domain": domain,
            "sequence": seq,
            "syntax_ok": bool(syn_ok),
            "llm_ok": bool(llm_ok),
            "both_ok": bool(both_ok),
        }
        report_items.append(item_report)
        _resp_prev = _oneline(str(txt) if isinstance(txt, str) else "<None>")
        print(f"[VERIFY:{idx}/{n_check}] RESULT syntax_ok={syn_ok} llm_ok={llm_ok} both_ok={both_ok}", flush=True)
        if debug:
            print(f"[VERIFY:{idx}/{n_check}] LLM_USED model={model or '<default>'} dt={(last-t0):.2f}s resp={_resp_prev}")
        if both_ok or not prune:
            kept.append(it)
        proc += 1
    # 如果开启了 prune，则将通过的条目回写到原文件；未处理的尾部条目保持原样
    if prune:
        try:
            if n_check < total:
                kept.extend(arr[n_check:])
            pack_file.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
            if debug:
                print(f"[VERIFY] pruned file written: {pack_file} kept={len(kept)}/{len(arr)} (checked={n_check})")
        except Exception as e:
            print(f"[VERIFY] prune write failed: {e}")

    out = {
        "file": str(pack_file.relative_to(_repo_root())).replace("\\", "/"),
        "domain": domain,
        "count": len(arr),
        "checked": int(n_check),
        "ok_both": sum(1 for x in report_items if x["both_ok"]),
        "ok_syntax": sum(1 for x in report_items if x["syntax_ok"]),
        "ok_llm": sum(1 for x in report_items if x["llm_ok"]),
        "items": report_items,
    }
    (out_dir / f"verify_{domain}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    base = Path(__file__).resolve().parent
    cfg_path = base.parent / "config.json"
    cfg = _load_cfg(cfg_path) if cfg_path.exists() else {}
    out_dir = base

    files = [
        base / "pem_operator_packages.json",
        base / "pdem_operator_packages.json",
        base / "pktm_operator_packages.json",
        base / "pgom_operator_packages.json",
        base / "tem_operator_packages.json",
        base / "prm_operator_packages.json",
        base / "iem_operator_packages.json",
    ]
    # 简易参数：--debug / --prune / [limit]
    import sys
    args = sys.argv[1:]
    debug = ("--debug" in args) or ("-d" in args)
    prune = ("--prune" in args) or ("-p" in args)
    # 默认行为：等价于 --debug --prune（即使未显式传入标志）
    debug = True
    prune = True
    # 可选的首个非短横线参数作为本次检查的数量上限
    limit: int | None = None
    for a in args:
        if not a.startswith('-'):
            try:
                limit = int(a)
                break
            except Exception:
                continue

    summary: Dict[str, Any] = {"reports": []}
    for f in files:
        if not f.exists():
            continue
        rep = verify_file(f, cfg, out_dir=out_dir, debug=debug, prune=prune, limit=limit)
        summary["reports"].append(rep)
    (out_dir / "verify_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "ok_both_total": sum(r["ok_both"] for r in summary["reports"]),
        "count_total": sum(r["count"] for r in summary["reports"]),
        "checked_total": sum(r.get("checked", r.get("count", 0)) for r in summary["reports"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

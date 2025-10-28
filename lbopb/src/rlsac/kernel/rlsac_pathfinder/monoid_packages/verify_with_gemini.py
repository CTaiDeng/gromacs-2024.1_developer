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
    try:
        t = s.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace("\n", "\\n")
        return t[:max_len] + ("..." if len(t) > max_len else "")
    except Exception:
        return s


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


def _read_packages(p: Path) -> List[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _domain_from_file(p: Path) -> str:
    return p.name.split("_")[0].lower()


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


def _ops_detailed_for(seq: List[str], domain: str) -> Tuple[List[Dict[str, Any]] | None, Dict[str, Any] | None]:
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
                debug: bool = True, prune: bool = True, limit: int | None = None) -> Dict[str, Any]:
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
        model = None
        for k, v in model_map.items():
            if int(v) == gm:
                model = str(k)
                break
    else:
        model = str(gm) if isinstance(gm, str) else None

    req_interval = float(cfg.get("llm_request_interval_sec", 0.0))
    last = 0.0
    # propagate LLM timeout and model to env (for HTTP client)
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

    kept: List[Dict[str, Any]] = []
    total = len(arr)
    n_check = total if (limit is None or limit <= 0) else min(limit, total)
    _cprint(f"[校验] 文件={pack_file} 域={domain} 总数={total} 检查={n_check} 模型={model or '<default>'} 间隔={req_interval}s", ANSI_CYAN, always=True)

    consecutive_unparsed = 0
    for idx, it in enumerate(arr[:n_check], start=1):
        seq = list(it.get("sequence", []) or [])
        _cprint(f"[校验:{idx}/{n_check}] 开始 序列={seq}", ANSI_YELLOW, always=True)

        # 不进行 syntax_checker，固定视为 True（仅 LLM 判定）
        syn_ok = True

        # Prepare ops_detailed/meta
        ops_det = it.get("ops_detailed") if isinstance(it.get("ops_detailed"), list) else None
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
            _cprint(f"[校验:{idx}/{n_check}] 参数={_ops_prev}", ANSI_YELLOW)
            _cprint(f"[校验:{idx}/{n_check}] 提示词预览={_oneline(str(prompt))}", ANSI_CYAN)

        # throttle
        if req_interval > 0 and last > 0:
            w = (last + req_interval) - _t.time()
            if w > 0:
                _cprint(f"[校验:{idx}/{n_check}] 限流 等待={w:.2f}s", ANSI_CYAN)
                _t.sleep(w)

        # LLM 调用：失败/不可解析重试 3 次；连续失败 3 条则退出
        max_attempts = 3
        attempt = 0
        llm_ok = False
        llm_raw: str | None = None
        t0 = _t.time()
        parsed = False
        while attempt < max_attempts:
            attempt += 1
            try:
                txt = call_llm(prompt, model=model)
            except Exception as e:
                txt = f"[Gemini Exception] {e}"
            llm_raw = str(txt) if txt is not None else None
            s = (llm_raw or "").strip()
            if s:
                if s == "1" or ("1" in s and "0" not in s):
                    llm_ok = True
                    parsed = True
                elif s == "0" or ("0" in s and "1" not in s):
                    llm_ok = False
                    parsed = True
            if parsed:
                break
            # 未解析或 HTTP 错误：读取 429 的 retry 提示或使用配置间隔
            wait_sec = 0.0
            try:
                import re as _re
                m1 = _re.search(r"Please retry in\s*([0-9.]+)s", s)
                m2 = _re.search(r"\"retryDelay\"\s*:\s*\"([0-9.]+)s\"", s)
                if m1:
                    wait_sec = float(m1.group(1))
                elif m2:
                    wait_sec = float(m2.group(1))
            except Exception:
                wait_sec = 0.0
            if wait_sec <= 0.0:
                wait_sec = max(1.0, float(req_interval or 0.0))
            _cprint(f"[校验:{idx}/{n_check}] 重试 次数={attempt}/{max_attempts} 等待={wait_sec:.2f}s", ANSI_CYAN)
            _t.sleep(wait_sec)
        last = _t.time()

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
        _resp_prev = _oneline(llm_raw or "<None>")
        if both_ok:
            _cprint(f"[校验:{idx}/{n_check}] 结果 通过 (LLM=1)", ANSI_GREEN, always=True)
        else:
            _cprint(f"[校验:{idx}/{n_check}] 结果 不通过 (LLM=0 或未解析)", ANSI_RED, always=True)
        if debug:
            _cprint(f"[校验:{idx}/{n_check}] LLM调用 模型={model or '<default>'} 用时={(last-t0):.2f}s 响应预览={_resp_prev}", ANSI_MAGENTA)

        # 记录“请求失败/未解析”的连续次数；若本条已解析（无论 1/0）则清零
        if not parsed:
            consecutive_unparsed += 1
        else:
            consecutive_unparsed = 0
        if consecutive_unparsed >= 3:
            raise SystemExit(f"[校验] 终止：连续失败条目达到 {consecutive_unparsed}")

        if both_ok or not prune:
            kept.append(it)

    # 写回（剪除失败项）
    if prune:
        try:
            if n_check < total:
                kept.extend(arr[n_check:])
            with pack_file.open("w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(kept, ensure_ascii=False, indent=2))
            if debug:
                _cprint(f"[校验] 已写回：{pack_file} 保留={len(kept)}/{len(arr)} (已检={n_check})", ANSI_CYAN)
        except Exception as e:
            print(f"[校验] 剪除写回失败: {e}")

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
    with (out_dir / f"verify_{domain}.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
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
    # 默认等价于 --debug --prune；可附加整数 N 只检查前 N 条
    import sys
    args = sys.argv[1:]
    debug = True
    prune = True
    limit: int | None = None
    for a in args:
        if a in ("--debug", "-d"):
            debug = True
        elif a in ("--prune", "-p"):
            prune = True
        elif not a.startswith('-'):
            try:
                limit = int(a)
            except Exception:
                continue

    # 每次运行先清空 PEM 专用输出文件，确保只存本次
    pem_out = base / "verify_pem.json"
    try:
        with pem_out.open("w", encoding="utf-8", newline="\n") as f:
            f.write("{}")
    except Exception:
        pass

    summary: Dict[str, Any] = {"reports": []}
    for f in files:
        if not f.exists():
            continue
        rep = verify_file(f, cfg, out_dir=out_dir, debug=debug, prune=prune, limit=limit)
        summary["reports"].append(rep)
        # 若为 PEM 域，单独写入 verify_pem.json（仅保存本次）
        try:
            if str(rep.get("domain","")) == "pem":
                with pem_out.open("w", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(rep, ensure_ascii=False, indent=2))
        except Exception:
            pass
    with (out_dir / "verify_summary.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({
        "ok_both_total": sum(r["ok_both"] for r in summary["reports"]),
        "count_total": sum(r["count"] for r in summary["reports"]),
        "checked_total": sum(r.get("checked", r.get("count", 0)) for r in summary["reports"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

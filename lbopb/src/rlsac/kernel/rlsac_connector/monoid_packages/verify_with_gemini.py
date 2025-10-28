# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import importlib
import json
import os
import time as _t
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _oneline(s: str, max_len: int = 240) -> str:
    try:
        t = s.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace("\n", "\\n")
        return t[:max_len] + ("..." if len(t) > max_len else "")
    except Exception:
        return s


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


def _pair_from_filename(p: Path) -> Tuple[str, str]:
    name = p.stem  # <a>_<b>_operator_packages
    if name.endswith("_operator_packages"):
        name = name[: -len("_operator_packages")]
    parts = name.split("_")
    return parts[0].lower(), parts[1].lower()


def verify_file(pack_file: Path, cfg: Dict[str, Any], *, out_dir: Path,
                debug: bool = True, prune: bool = True, limit: int | None = None) -> Dict[str, Any]:
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

    pair_a, pair_b = _pair_from_filename(pack_file)
    arr = _read_json(pack_file) or []
    report_items: List[Dict[str, Any]] = []

    # LLM settings
    model_map = cfg.get("gemini_model_choose", {}) or {}
    gm = cfg.get("GEMINI_MODEL", None)
    if isinstance(gm, int):
        model = None
        for k, v in model_map.items():
            try:
                if int(v) == gm:
                    model = str(k)
                    break
            except Exception:
                continue
    else:
        model = str(gm) if isinstance(gm, str) else None

    req_interval = float(cfg.get("llm_request_interval_sec", 0.0))
    last = 0.0
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
    from lbopb.src.rlsac.kernel.common.llm_oracle import build_connector_prompt, call_llm  # type: ignore

    kept: List[Dict[str, Any]] = []
    total = len(arr)
    n_check = total if (limit is None or limit <= 0) else min(limit, total)
    _cprint(
        f"[校验] 文件={pack_file} 对={pair_a}_{pair_b} 总数={total} 检查={n_check} 模型={model or '<default>'} 间隔={req_interval}s",
        ANSI_CYAN,
        always=True,
    )

    consecutive_unparsed = 0
    for idx, it in enumerate(arr[:n_check], start=1):
        seqs = it.get("sequences") or {}
        seq_a = list(seqs.get(pair_a, []) or [])
        seq_b = list(seqs.get(pair_b, []) or [])
        _cprint(f"[校验:{idx}/{n_check}] 开始 seq_a={seq_a} seq_b={seq_b}", ANSI_YELLOW, always=True)

        # syntax check via pair module
        syn_ok = False
        warns_present = False
        try:
            mod_name = f"lbopb.src.common.{pair_a}_{pair_b}_syntax_checker"
            mod = importlib.import_module(mod_name)
            syn = getattr(mod, "check")
            syn_res: Dict[str, Any] = syn(seq_a, seq_b)
            syn_ok = int(syn_res.get("errors", 0)) == 0
            warns_present = int(syn_res.get("warnings", 0)) > 0
        except Exception as e:
            _cprint(f"[校验:{idx}/{n_check}] 语法检查异常: {e}", ANSI_RED, always=True)
            syn_ok = False

        llm_ok = True
        llm_raw: str | None = None
        parsed = False
        if warns_present and bool(cfg.get("use_llm_oracle", False)):
            conn = {pair_a: seq_a, pair_b: seq_b}
            prompt = build_connector_prompt(conn)
            if debug:
                _cprint(f"[校验:{idx}/{n_check}] 提示词预览={_oneline(str(prompt))}", ANSI_CYAN)

            # throttle
            if req_interval > 0 and last > 0:
                w = (last + req_interval) - _t.time()
                if w > 0:
                    _cprint(f"[校验:{idx}/{n_check}] 限速 等待={w:.2f}s", ANSI_CYAN)
                    _t.sleep(w)

            max_attempts = 3
            attempt = 0
            t0 = _t.time()
            llm_ok = False
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
                # basic 429/retry-delay parsing
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
                _cprint(f"[校验:{idx}/{n_check}] 重试 {attempt}/{max_attempts} 等待={wait_sec:.2f}s", ANSI_CYAN)
                _t.sleep(wait_sec)
            last = _t.time()
            if debug:
                _cprint(
                    f"[校验:{idx}/{n_check}] LLM={model or '<default>'} 用时={(last - t0):.2f}s 响应预览={_oneline(llm_raw or '<None>')}",
                    ANSI_MAGENTA,
                )

        both_ok = bool(syn_ok and (llm_ok if warns_present and bool(cfg.get("use_llm_oracle", False)) else True))
        item_report = {
            "id": it.get("id"),
            "pair": f"{pair_a}_{pair_b}",
            "sequences": {pair_a: seq_a, pair_b: seq_b},
            "syntax_ok": bool(syn_ok),
            "llm_ok": bool(llm_ok),
            "both_ok": bool(both_ok),
        }
        report_items.append(item_report)
        if both_ok:
            _cprint(f"[校验:{idx}/{n_check}] 通过 (LLM={'1' if warns_present else '-'})", ANSI_GREEN, always=True)
        else:
            _cprint(f"[校验:{idx}/{n_check}] 未通过 (LLM={'0' if warns_present else '-'})", ANSI_RED, always=True)

        if both_ok or not prune:
            # keep item; optionally update validation
            kept_item = dict(it)
            try:
                val = kept_item.get("validation") or {}
                syn = val.get("syntax") or {}
                syn["result"] = "通过" if syn_ok else "错误"
                kept_item.setdefault("validation", {})["syntax"] = syn
                if warns_present and bool(cfg.get("use_llm_oracle", False)):
                    g = kept_item["validation"].get("gemini", {})
                    g.update({"used": True, "result": "正确" if llm_ok else "错误"})
                    kept_item["validation"]["gemini"] = g
            except Exception:
                pass
            kept.append(kept_item)

        # stop when too many unparsed LLM outputs
        if warns_present and bool(cfg.get("use_llm_oracle", False)) and not parsed:
            consecutive_unparsed += 1
        else:
            consecutive_unparsed = 0
        if consecutive_unparsed >= 3:
            raise SystemExit(f"[校验] 停止：连续无法解析的 LLM 输出达到 {consecutive_unparsed}")

    if prune:
        try:
            if n_check < total:
                kept.extend(arr[n_check:])
            with pack_file.open("w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(kept, ensure_ascii=False, indent=2))
            if debug:
                _cprint(f"[校验] 回写：{pack_file} 保留={len(kept)}/{len(arr)} (已检={n_check})", ANSI_CYAN)
        except Exception as e:
            print(f"[校验] 回写失败: {e}")

    out = {
        "file": str(pack_file.relative_to(_repo_root())).replace("\\", "/"),
        "pair": f"{pair_a}_{pair_b}",
        "count": len(arr),
        "checked": int(n_check),
        "ok_both": sum(1 for x in report_items if x["both_ok"]),
        "ok_syntax": sum(1 for x in report_items if x["syntax_ok"]),
        "ok_llm": sum(1 for x in report_items if x["llm_ok"]),
        "items": report_items,
    }
    with (out_dir / f"verify_{pair_a}_{pair_b}.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main() -> None:
    base = Path(__file__).resolve().parent
    cfg_path = base.parent / "config.json"
    cfg = _read_json(cfg_path) if cfg_path.exists() else {}
    out_dir = base

    files = sorted(list(base.glob("*_operator_packages.json")))
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
        elif not a.startswith("-"):
            try:
                limit = int(a)
            except Exception:
                continue

    summary: Dict[str, Any] = {"reports": []}
    for f in files:
        if not f.exists():
            continue
        rep = verify_file(f, cfg, out_dir=out_dir, debug=debug, prune=prune, limit=limit)
        summary["reports"].append(rep)
    with (out_dir / "verify_summary.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({
        "ok_both_total": sum(r["ok_both"] for r in summary["reports"]),
        "count_total": sum(r["count"] for r in summary["reports"]),
        "checked_total": sum(r.get("checked", r.get("count", 0)) for r in summary["reports"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

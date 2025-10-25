# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _pytime
from pathlib import Path
import os
import re
import re
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

try:
    from .env_domain import DomainPathfinderEnv, Goal
    from .domain import get_domain_spec
except ImportError:
    # 兼容直接脚本执行：将仓库根加入 sys.path 后用绝对导入
    from pathlib import Path as _Path
    import sys as _sys

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.env_domain import DomainPathfinderEnv, Goal
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.domain import get_domain_spec
try:
    from .env_domain import DomainPathfinderEnv, Goal
    from .domain import get_domain_spec
    from .sampler import sample_random_package
    from .oracle import AxiomOracle, default_init_state, apply_sequence
    from .scorer import PackageScorer, train_scorer
except ImportError:
    # 兼容直接脚本执行：将仓库根加入 sys.path 后用绝对导入
    from pathlib import Path as _Path
    import sys as _sys

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.env_domain import DomainPathfinderEnv, Goal
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.domain import get_domain_spec
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.sampler import sample_random_package
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.oracle import AxiomOracle, default_init_state, apply_sequence
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.scorer import PackageScorer, train_scorer
from lbopb.src.rlsac.application.rlsac_nsclc.utils import select_device_from_config

try:
    from lbopb.src.rlsac.application.rlsac_nsclc.models import DiscretePolicy  # type: ignore
except Exception:
    DiscretePolicy = None  # type: ignore


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _resolve_domain(cfg: Dict[str, Any], override: str | None = None) -> str:
    """解析域标识：
    - 若提供 override（字符串），优先返回其小写；
    - 若 cfg["domain"] 为字符串，直接返回其小写；
    - 若 cfg["domain"] 为数字，则在 cfg["domain_choose"] 中按值反查对应 key；
    - 兜底返回 "pem"。
    """
    if override is not None:
        try:
            return str(override).strip().lower()
        except Exception:
            pass
    dom = cfg.get("domain", None)
    if isinstance(dom, str):
        return dom.strip().lower()
    try:
        dom_num = int(dom)  # type: ignore[arg-type]
        mapping = cfg.get("domain_choose", {}) or {}
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                try:
                    if int(v) == dom_num:
                        return str(k).strip().lower()
                except Exception:
                    continue
    except Exception:
        pass
    return "pem"


def build_env_from_config(cfg_path: Path, domain_override: str | None = None) -> DomainPathfinderEnv:
    cfg = _load_config(cfg_path)
    domain = _resolve_domain(cfg, domain_override)
    # 读取域差异配置（config.dict.json）
    mod_dir = Path(__file__).resolve().parent
    dict_path = mod_dir / "config.dict.json"
    dict_cfg: Dict[str, Any] = {}
    try:
        if dict_path.exists():
            dict_cfg = json.loads(dict_path.read_text(encoding="utf-8"))
    except Exception:
        dict_cfg = {}
    dom_defaults = (dict_cfg.get("domains", {}) or {}).get(domain, {}) if isinstance(dict_cfg, dict) else {}
    spec = get_domain_spec(domain)
    # 优先使用 config.json 中的覆盖项，否则采用 config.dict.json 的域默认
    s0_cfg = cfg.get("initial_state",
                     dom_defaults.get("initial_state", {"b": 3.0, "n_comp": 3, "perim": 5.0, "fidelity": 0.4}))
    st_cfg = cfg.get("target_state",
                     dom_defaults.get("target_state", {"b": 0.5, "n_comp": 1, "perim": 2.0, "fidelity": 0.9}))
    tol_cfg = cfg.get("tolerance", dom_defaults.get("tolerance", {"b": 0.1, "n": 0.5, "p": 0.2, "f": 0.05}))
    init_state = spec.state_cls(
        b=float(s0_cfg.get("b", 3.0)),
        n_comp=int(s0_cfg.get("n_comp", 3)),
        perim=float(s0_cfg.get("perim", 5.0)),
        fidelity=float(s0_cfg.get("fidelity", 0.4)),
    )
    goal = Goal(
        target=spec.state_cls(
            b=float(st_cfg.get("b", 0.5)),
            n_comp=int(st_cfg.get("n_comp", 1)),
            perim=float(st_cfg.get("perim", 2.0)),
            fidelity=float(st_cfg.get("fidelity", 0.9)),
        ),
        tol_b=float(tol_cfg.get("b", 0.1)),
        tol_n=float(tol_cfg.get("n", 0.5)),
        tol_p=float(tol_cfg.get("p", 0.2)),
        tol_f=float(tol_cfg.get("f", 0.05)),
    )
    max_steps = int(cfg.get("episode_max_steps", 64))
    improve_w = float(cfg.get("improve_weight", 1.0))
    step_penalty = float(cfg.get("step_penalty", 0.01))
    include_identity = bool(cfg.get("include_identity", False))
    env = DomainPathfinderEnv(
        spec,
        init_state=init_state,
        goal=goal,
        max_steps=max_steps,
        improve_weight=improve_w,
        step_penalty=step_penalty,
        include_identity=include_identity,
    )
    return env


def train(config_path: str | Path | None = None, domain_override: str | None = None) -> Path:
    mod_dir = Path(__file__).resolve().parent
    cfg_path = Path(config_path) if config_path else (mod_dir / "config.json")
    cfg = _load_config(cfg_path)
    device = select_device_from_config(cfg_path)

    debug = bool(cfg.get("debug", False))
    log_every_step = bool(cfg.get("log_every_step", True))
    log_to_file = bool(cfg.get("log_to_file", True))

    # Debug 模式下，打开 Gemini 全程调试（通过环境变量传递给 HTTP 客户端）
    if debug:
        try:
            os.environ.setdefault("LBOPB_GEMINI_DEBUG", "1")
        except Exception:
            pass
    # 从配置传递 LLM 超时时间（秒）给 HTTP 客户端
    try:
        _llm_to = cfg.get("llm_timeout_sec", None)
        if _llm_to is not None:
            os.environ["LBOPB_GEMINI_TIMEOUT_SEC"] = str(_llm_to)
    except Exception:
        pass
    # 从配置解析 Gemini 模型选择（名称或编号），并下发给环境变量
    gemini_model: str | None = None
    try:
        gm = cfg.get("GEMINI_MODEL", None)
        name = None
        if isinstance(gm, str):
            name = gm.strip()
        elif isinstance(gm, int):
            mp = cfg.get("gemini_model_choose", {}) or cfg.get("GEMINI_MODEL_CHOOSE", {}) or {}
            if isinstance(mp, dict):
                for k, v in mp.items():
                    try:
                        if int(v) == gm:
                            name = str(k)
                            break
                    except Exception:
                        continue
        if name:
            os.environ["LBOPB_GEMINI_MODEL"] = name
            # 强制覆盖以避免外部默认值干扰
            os.environ["GEMINI_MODEL"] = name
            gemini_model = name
    except Exception:
        pass

    # LLM 请求节流（秒）
    llm_req_interval = float(cfg.get("llm_request_interval_sec", 0.0))
    last_llm_time: float = 0.0

    class RunLogger:
        def __init__(self, path: Path, append: bool = False):
            self.path = path
            self.f = open(path, 'a' if append else 'w', encoding='utf-8', newline='')

        def write_line(self, text: str) -> None:
            try:
                self.f.write(text + "\n")
                self.f.flush()
            except Exception:
                pass

        def close(self) -> None:
            try:
                self.f.flush()
                self.f.close()
            except Exception:
                pass

    def dbg(msg: str) -> None:
        if debug:
            print(f"[DEBUG] {msg}")
            if logger:
                logger.write_line(f"[DEBUG] {msg}")

    def step_log(msg: str) -> None:
        if log_every_step or debug:
            print(msg)
            if logger:
                logger.write_line(msg)

    # 控制台颜色与日志辅助（控制台彩色，日志去色）
    ANSI_RESET = "\x1b[0m"
    ANSI_RED = "\x1b[31;1m"
    ANSI_GREEN = "\x1b[32;1m"
    ANSI_YELLOW = "\x1b[33;1m"
    ANSI_CYAN = "\x1b[36;1m"
    ANSI_MAGENTA = "\x1b[35;1m"

    _ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def _strip_ansi(s: str) -> str:
        try:
            return _ansi_re.sub("", s)
        except Exception:
            return s

    def logc(text: str, color: str | None = None) -> None:
        s = f"{color}{text}{ANSI_RESET}" if (color and debug) else text
        print(s)
        if logger:
            logger.write_line(_strip_ansi(text))

    # 环境
    # 确定本次训练的域
    this_domain = _resolve_domain(cfg, domain_override)

    # 采样-监督：随机生成算子包 → 公理系统打分(0/1) → 训练打分网络
    min_len = int(cfg.get("min_len", 1))
    max_len = int(cfg.get("max_len", 4))
    no_dup = bool(cfg.get("sampler_no_consecutive_dup", True))
    n_samples = int(cfg.get("samples", 2000))
    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch_size", 64))
    topk = int(cfg.get("topk", 50))
    cost_lambda = float(cfg.get("cost_lambda", 0.2))
    oracle = AxiomOracle(cost_lambda=cost_lambda, use_llm=bool(cfg.get("use_llm_oracle", False)))

    spec = get_domain_spec(this_domain)
    op_names = []
    # 确定特征维度：每个基本算子一个计数 + 序列长度 + Δrisk + cost
    for cls in spec.op_classes:
        try:
            nm = cls().name
        except Exception:
            nm = cls.__name__
        op_names.append(str(nm))
    feat_dim = len(op_names) + 3

    # 准备运行目录（便于调试产物写入）
    repo_root = Path(__file__).resolve().parents[5]
    out_root = repo_root / "out"
    base_out = out_root / Path(cfg.get("output_dir", "out_pathfinder"))
    base_out.mkdir(parents=True, exist_ok=True)
    _stamp = str(int(_pytime.time()))
    run_name = "train_" + _stamp + (f"_{this_domain}" if this_domain else "")
    run_dir = base_out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    if log_to_file:
        logger = RunLogger(log_path, append=False)
        logger.write_line(f"# TRAIN START {_pytime.strftime('%Y-%m-%d %H:%M:%S', _pytime.localtime())}")
        logger.write_line(f"device={device}")
        logger.write_line(f"config={json.dumps(cfg, ensure_ascii=False)}")

    # 训练样本将按纳入规则动态收集，不再预先分配张量
    debug_dump = bool(cfg.get("debug_dump", True))
    dataset_dump: list[dict[str, Any]] = [] if debug_dump else []
    init_state = default_init_state(this_domain)

    def evaluate_with_details(domain: str, seq: List[str], s0: Any | None = None) -> dict[str, Any]:
        import importlib
        s0 = s0 if s0 is not None else default_init_state(domain)
        # syntax check
        syntax = {"errors": [], "warnings": [], "valid": True}
        try:
            mod = importlib.import_module(f"lbopb.src.{domain}.syntax_checker")
            func = getattr(mod, "check_sequence", None)
            if callable(func):
                res = func(list(seq), init_state=s0)
                syntax = {
                    "errors": list(res.get("errors", []) or []),
                    "warnings": list(res.get("warnings", []) or []),
                    "valid": bool(res.get("valid", True)),
                }
        except Exception:
            pass
        # heuristics（仅用于调试观测，不作为训练标签决策依据）
        _, dr, c = apply_sequence(domain, s0, seq)
        heur_ok = bool((dr - cost_lambda * c) > 0.0)
        # LLM only when warnings present and enabled
        llm_used = False
        llm_attempted = False
        llm_raw: str | None = None
        llm_prompt: str | None = None
        llm_status: str = "skipped_no_warn"  # used | skipped_no_warn | skipped_use_llm_false | skipped_errors | skipped_exception
        # 训练样本纳入规则：
        # - syntax 有 errors: 明确错误，纳入训练（负样本）
        # - syntax 无错误但有 warnings:
        #     - 若启用 LLM 且调用成功返回 '1'/'0'：按返回纳入训练；
        #     - 否则（未启用或调用失败）：不纳入训练；
        # - 无错误且无警告：明确正确，纳入训练（正样本）。
        train_include = False
        train_reason = ""
        # 确保任何路径下都有 label 初值，避免未绑定错误
        label = 0
        if syntax["errors"]:
            label = 0
            train_include = True
            train_reason = "syntax_error"
            llm_status = "skipped_errors"
        else:
            warns_present = bool(syntax.get("warnings"))
            if warns_present and bool(cfg.get("use_llm_oracle", False)):
                try:
                    from lbopb.src.rlsac.kernel.common.llm_oracle import call_llm, build_pathfinder_prompt
                    # 可选: 将算子参数化与取值一并提交给 LLM
                    ops_det = None
                    extra_meta = None
                    try:
                        if bool(cfg.get("llm_include_params", False)):
                            from lbopb.src.rlsac.kernel.rlsac_pathfinder.op_space_utils import load_op_space, param_grid_of, params_from_grid  # type: ignore
                            # 推导该域的空间定义文件（v1）
                            _mod_dir = Path(__file__).resolve().parent
                            _space_ref = _mod_dir / "operator_spaces" / f"{domain}_op_space.v1.json"
                            if _space_ref.exists():
                                _space = load_op_space(str(_space_ref))
                                # 为每个算子选择一个“默认”索引（网格中位数），反查 params
                                steps = []
                                for nm in list(seq):
                                    try:
                                        names, grids = param_grid_of(_space, nm)
                                    except Exception:
                                        # 无参数定义则跳过参数化
                                        steps.append({"name": nm})
                                        continue
                                    gi: list[int] = []
                                    for g in grids:
                                        L = len(g)
                                        gi.append(max(0, (L - 1) // 2))
                                    prs = params_from_grid(_space, nm, gi)
                                    steps.append({"name": nm, "grid_index": gi, "params": prs})
                                ops_det = steps
                                extra_meta = {
                                    "op_space_id": _space.get("space_id", None),
                                    "op_space_ref": str(_space_ref).replace("\\", "/"),
                                }
                    except Exception:
                        ops_det = None
                        extra_meta = None
                    llm_prompt = build_pathfinder_prompt(domain, list(seq), ops_det, extra_meta)
                    if debug:
                        # 分割线与请求前打印
                        logc("========== LLM REQUEST BEGIN ==========", ANSI_CYAN)
                        # 优先打印序列与参数段，便于快速定位关键信息
                        logc(f"[SEQ] {seq}", ANSI_YELLOW)
                        if ops_det:
                            try:
                                import json as _json
                                _ops_preview = _json.dumps(ops_det, ensure_ascii=False)[:240]
                                logc(f"[PARAMS] {_ops_preview}", ANSI_YELLOW)
                            except Exception:
                                pass
                        # 其后再打印 LLM 启动信息与提示词预览
                        _msg0 = (
                            f"[LLM] start: provider=gemini domain={domain} seq_len={len(seq)} "
                            f"warnings={len(syntax.get('warnings', []))} heur_ok={heur_ok}"
                        )
                        logc(_msg0, ANSI_CYAN)
                        try:
                            _pv = str(llm_prompt)
                            # 打印预览时压成单行：换行→\n，制表→空格，连续空白压缩
                            _pv_oneline = _pv.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace("\n", "\\n")
                            # 适当截断，避免刷屏
                            _pv2 = _pv_oneline[:240] + ("..." if len(_pv_oneline) > 240 else "")
                            _msg1 = f"[LLM] prompt preview: {_pv2}"
                            logc(_msg1, ANSI_CYAN)
                        except Exception:
                            pass
                    # 节流：若设置了请求间隔，则在两次请求之间等待
                    nonlocal last_llm_time
                    if llm_req_interval > 0.0 and last_llm_time > 0.0:
                        now = _pytime.time()
                        wait = (last_llm_time + llm_req_interval) - now
                        if wait > 0:
                            if debug:
                                logc(f"[LLM] throttle: sleep {wait:.2f}s", ANSI_CYAN)
                            _pytime.sleep(wait)
                    llm_attempted = True
                    _t0 = _pytime.time()
                    txt = call_llm(llm_prompt, model=gemini_model)
                    _dt = _pytime.time() - _t0
                    last_llm_time = _pytime.time()
                    llm_raw = str(txt) if txt is not None else None
                    _is_err = isinstance(txt, str) and (
                        txt.startswith("[Gemini Error]") or txt.startswith("[Gemini HTTPError]")
                    )
                    if _is_err:
                        llm_used = False
                        llm_status = "skipped_exception"
                        # LLM 失败：不纳入训练
                        train_include = False
                        train_reason = "llm_failed"
                        # 回退使用启发式作为标签占位
                        label = int(bool(heur_ok))
                        # 若为 429 (RESOURCE_EXHAUSTED)，解析建议的 Retry-After 并休眠以缓解配额限制
                        try:
                            import re as _re
                            s = llm_raw or ""
                            # 方式1：从错误提示文本中提取 "Please retry in Xs."
                            m1 = _re.search(r"Please retry in\s+([0-9.]+)s", s)
                            # 方式2：从 JSON 字段提取 \"retryDelay\": \"Xs\"
                            m2 = _re.search(r"\"retryDelay\"\s*:\s*\"([0-9.]+)s\"", s)
                            sec = None
                            if m1:
                                sec = float(m1.group(1))
                            elif m2:
                                sec = float(m2.group(1))
                            if sec and sec > 0:
                                if debug:
                                    logc(f"[LLM] 429 quota: retry-after {sec:.2f}s -> sleeping", ANSI_CYAN)
                                _pytime.sleep(sec)
                                # 更新节流时间基准，避免紧接着再次触发
                                last_llm_time = _pytime.time()
                        except Exception:
                            pass
                    else:
                        llm_used = True
                        llm_status = "used"
                        if isinstance(txt, str):
                            _s = txt.strip()
                            if _s == "1" or ("1" in _s and "0" not in _s):
                                label = 1
                                train_include = True
                                train_reason = "llm_1"
                            elif _s == "0" or ("0" in _s and "1" not in _s):
                                label = 0
                                train_include = True
                                train_reason = "llm_0"
                            else:
                                # 无法解析：当作失败，不纳入
                                train_include = False
                                train_reason = "llm_unparsed"
                                # 回退使用启发式作为标签占位
                                label = int(bool(heur_ok))
                    if debug:
                        # 返回结果打印
                        try:
                            _rprev = llm_raw if isinstance(llm_raw, str) else "<None>"
                            _rprev2 = _rprev[:240] + (
                                "..." if isinstance(_rprev, str) and len(_rprev) > 240 else ""
                            )
                            _msg2 = (
                                f"[LLM] done: used={llm_used}, status={llm_status}, dt={_dt:.2f}s, "
                                f"result_len={len(llm_raw) if isinstance(llm_raw, str) else 0}, result_preview={_rprev2}"
                            )
                            logc(_msg2, ANSI_CYAN)
                            # 结果高亮（含中文解释）
                            if _is_err:
                                logc("[RESULT] LLM 请求失败/未解析 → 不纳入训练", ANSI_RED)
                            else:
                                try:
                                    _s = (llm_raw or "").strip()
                                except Exception:
                                    _s = ""
                                if _s == "1" or ("1" in _s and "0" not in _s):
                                    logc("[RESULT] 1 → 符合公理系统（正样本，纳入训练）", ANSI_GREEN)
                                elif _s == "0" or ("0" in _s and "1" not in _s):
                                    logc("[RESULT] 0 → 不符合公理系统（负样本，纳入训练）", ANSI_RED)
                                else:
                                    logc("[RESULT] 未能解析为 1/0 → 不纳入训练", ANSI_MAGENTA)
                            logc("========== LLM REQUEST END ==========", ANSI_CYAN)
                        except Exception:
                            pass
                except Exception as e:
                    llm_attempted = True
                    llm_used = False
                    llm_status = "skipped_exception"
                    llm_raw = None
                    # LLM 异常：不纳入训练
                    train_include = False
                    train_reason = "llm_exception"
                    if debug:
                        try:
                            _msgE = f"[LLM] exception: {e}"
                            logc(_msgE, ANSI_RED)
                            logc("[RESULT] LLM 异常 → 不纳入训练", ANSI_RED)
                            logc("========== LLM REQUEST END ==========", ANSI_CYAN)
                        except Exception:
                            pass
            elif warns_present and not bool(cfg.get("use_llm_oracle", False)):
                # 未启用 LLM：不纳入训练；标签回退为启发式
                llm_status = "skipped_use_llm_false"
                train_include = False
                train_reason = "warnings_unresolved"
                label = int(bool(heur_ok))
            else:
                # 无错误无警告：明确正确
                llm_status = "skipped_no_warn"
                label = 1
                train_include = True
                train_reason = "syntax_ok"
        # 控制台调试输出
        if debug:
            print(
                "[SAMPLE] seq={seq} label={label} syntax_errors={e} warnings={w} "
                "heur_ok={heur} llm_attempted={la} llm_used={lu} llm_status={ls} "
                "train_include={ti} train_reason={tr}".format(
                    seq=seq,
                    label=label,
                    e=len(syntax["errors"]),
                    w=len(syntax["warnings"]),
                    heur=heur_ok,
                    la=llm_attempted,
                    lu=llm_used,
                    ls=llm_status,
                    ti=train_include,
                    tr=train_reason,
                )
            )
        return {
            "label": int(label),
            "syntax": syntax,
            "heur_ok": bool(heur_ok),
            "delta_risk": float(dr),
            "cost": float(c),
            "llm_used": bool(llm_used),
            "llm_raw": llm_raw,
            "llm_prompt": llm_prompt,
            "llm_status": llm_status,
            "llm_attempted": bool(llm_attempted),
            "train_include": bool(train_include),
            "train_reason": train_reason,
        }

    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    for i in range(n_samples):
        seq = sample_random_package(spec, min_len=min_len, max_len=max_len, no_consecutive_duplicate=no_dup)
        # 特征：op 计数
        cnts = [float(seq.count(nm)) for nm in op_names]
        # 作用效果特征
        _, dr, c = apply_sequence(this_domain, init_state, seq)
        length = float(len(seq))
        vec = cnts + [length, float(dr), float(c)]
        judge = evaluate_with_details(this_domain, seq, init_state)
        label_i = int(judge["label"])
        if bool(judge.get("train_include", False)):
            X_rows.append([float(x) for x in vec])
            y_rows.append(float(label_i))
        if debug_dump:
            dataset_dump.append({
                "index": int(i),
                "sequence": list(seq),
                "features": {
                    "bag_of_ops": {op_names[j]: float(cnts[j]) for j in range(len(op_names))},
                    "length": float(length),
                    "delta_risk": float(dr),
                    "cost": float(c)
                },
                "judge": judge,
                "label": int(label_i),
                "train_include": bool(judge.get("train_include", False)),
                "train_reason": str(judge.get("train_reason", ""))
            })
    # 构造训练张量
    if len(X_rows) > 0:
        X_t = torch.tensor(X_rows, dtype=torch.float32)
        y_t = torch.tensor(y_rows, dtype=torch.float32)
    else:
        X_t = torch.zeros((0, feat_dim), dtype=torch.float32)
        y_t = torch.zeros((0,), dtype=torch.float32)

    model = PackageScorer(feat_dim).to(device)
    if X_t.shape[0] > 0:
        train_scorer(model, X_t.to(device), y_t.to(device), epochs=epochs, batch_size=batch_size,
                     lr=float(cfg.get("learning_rate_actor", 3e-4)))

    # 校验并可选迭代直至完美匹配
    fit_until_perfect = bool(cfg.get("fit_until_perfect", True))
    max_refit_rounds = int(cfg.get("max_refit_rounds", 10))
    refit_epochs = int(cfg.get("refit_epochs", epochs))
    rounds = 0
    if X_t.shape[0] > 0:
        with torch.no_grad():
            preds = (model(X_t.to(device)).cpu().view(-1) >= 0.5).to(torch.float32)
            acc = float((preds == y_t).to(torch.float32).mean().item())
        while fit_until_perfect and acc < 1.0 and rounds < max_refit_rounds:
            train_scorer(model, X_t.to(device), y_t.to(device), epochs=refit_epochs, batch_size=batch_size,
                         lr=float(cfg.get("learning_rate_actor", 3e-4)))
            with torch.no_grad():
                preds = (model(X_t.to(device)).cpu().view(-1) >= 0.5).to(torch.float32)
                acc = float((preds == y_t).to(torch.float32).mean().item())
            rounds += 1
    else:
        acc = 1.0

    # 训练完生成大量候选，打分选 Top-K，写入辞海
    cand_n = int(cfg.get("candidate_generate", 1000))
    scored: list[tuple[float, List[str], float, float]] = []
    with torch.no_grad():
        for _ in range(cand_n):
            seq = sample_random_package(spec, min_len=min_len, max_len=max_len, no_consecutive_duplicate=no_dup)
            cnts = [float(seq.count(nm)) for nm in op_names]
            _, dr, c = apply_sequence(this_domain, init_state, seq)
            length = float(len(seq))
            vec = torch.tensor([*(cnts), length, float(dr), float(c)], dtype=torch.float32).unsqueeze(0).to(device)
            score = float(model(vec).item())
            scored.append((score, seq, float(dr), float(c)))
    scored.sort(key=lambda t: t[0], reverse=True)
    if debug_dump:
        try:
            # 统计 warnings / llm 覆盖率与训练纳入情况
            total = len(dataset_dump)
            err_cnt = 0;
            warn_cnt = 0;
            llm_cnt = 0;
            syntax_ok_cnt = 0;
            llm_attempted_cnt = 0
            llm_used_cnt = 0;
            llm_skip_no_warn = 0;
            llm_skip_use_llm_false = 0;
            llm_skip_errors = 0;
            llm_skip_exception = 0
            train_included_cnt = 0
            train_reason_cnt: Dict[str, int] = {}
            for item in dataset_dump:
                j = item.get("judge", {})
                syn = j.get("syntax", {}) if isinstance(j, dict) else {}
                errors = syn.get("errors", []) or []
                warns = syn.get("warnings", []) or []
                if len(errors) > 0:
                    err_cnt += 1
                if len(warns) > 0:
                    warn_cnt += 1
                if len(errors) == 0 and len(warns) == 0:
                    syntax_ok_cnt += 1
                if bool(j.get("llm_used", False)):
                    llm_cnt += 1
                if bool(j.get("llm_attempted", False)):
                    llm_attempted_cnt += 1
                # 细分 llm 状态
                ls = str(j.get("llm_status", ""))
                if ls == "used":
                    llm_used_cnt += 1
                elif ls == "skipped_no_warn":
                    llm_skip_no_warn += 1
                elif ls == "skipped_use_llm_false":
                    llm_skip_use_llm_false += 1
                elif ls == "skipped_errors":
                    llm_skip_errors += 1
                elif ls == "skipped_exception":
                    llm_skip_exception += 1
                # 训练纳入统计
                if bool(item.get("train_include", False)):
                    train_included_cnt += 1
                tr = str(item.get("train_reason", ""))
                if tr:
                    train_reason_cnt[tr] = int(train_reason_cnt.get(tr, 0)) + 1
            # 控制台与日志打印
            msg_stats = (
                f"[STATS] total={total} error_samples={err_cnt} warning_samples={warn_cnt} "
                f"syntax_ok_samples={syntax_ok_cnt} llm_attempted_samples={llm_attempted_cnt} llm_used_samples={llm_cnt} "
                f"train_included_samples={train_included_cnt} train_excluded_samples={total - train_included_cnt}"
            )
            print(msg_stats)
            if log_to_file and 'logger' in locals() and logger:
                logger.write_line(msg_stats)
            # 打印 LLM 细分统计
            msg_llm = (
                f"[LLM_STATS] attempted={llm_attempted_cnt} used={llm_used_cnt} skipped_no_warn={llm_skip_no_warn} "
                f"skipped_use_llm_false={llm_skip_use_llm_false} skipped_errors={llm_skip_errors} "
                f"skipped_exception={llm_skip_exception}"
            )
            print(msg_llm)
            if log_to_file and 'logger' in locals() and logger:
                logger.write_line(msg_llm)
            # 打印训练纳入原因统计
            try:
                _reasons = ", ".join([f"{k}={v}" for k, v in train_reason_cnt.items()])
            except Exception:
                _reasons = ""
            msg_train = f"[TRAIN_STATS] included={train_included_cnt} reasons: {_reasons}"
            print(msg_train)
            if log_to_file and 'logger' in locals() and logger:
                logger.write_line(msg_train)

            debug_ds_path = run_dir / "debug_dataset.json"
            debug_cand_path = run_dir / "debug_candidates.json"
            with open(debug_ds_path, 'w', encoding='utf-8', newline='') as f:
                json.dump({
                    "domain": this_domain,
                    "op_names": op_names,
                    "samples": dataset_dump,
                    "fit_rounds": rounds if fit_until_perfect else 0,
                    "train_acc": acc,
                    "stats": {
                        "total": total,
                        "error_samples": err_cnt,
                        "warning_samples": warn_cnt,
                        "syntax_ok_samples": syntax_ok_cnt,
                        "llm_attempted_samples": llm_attempted_cnt,
                        "llm_used_samples": llm_cnt,
                        "train": {
                            "included": train_included_cnt,
                            "excluded": total - train_included_cnt,
                            "reasons": train_reason_cnt
                        },
                        "llm_detail": {
                            "used": llm_used_cnt,
                            "skipped_no_warn": llm_skip_no_warn,
                            "skipped_use_llm_false": llm_skip_use_llm_false,
                            "skipped_errors": llm_skip_errors,
                            "skipped_exception": llm_skip_exception
                        }
                    }
                }, f, ensure_ascii=False, indent=2)
            with open(debug_cand_path, 'w', encoding='utf-8', newline='') as f:
                json.dump([
                    {"score": float(s), "sequence": seq, "delta_risk": float(dr), "cost": float(c)}
                    for (s, seq, dr, c) in scored[: int(cfg.get("debug_candidates_top", 50))]
                ], f, ensure_ascii=False, indent=2)
            # 将正确样本（label=1）的算子包去重纳入专用目录，并按 score 重新排序
            try:
                try:
                    from .package_store import ingest_from_debug_dataset  # type: ignore
                except Exception:
                    from lbopb.src.rlsac.kernel.rlsac_pathfinder.package_store import \
                        ingest_from_debug_dataset  # type: ignore
                ingest_from_debug_dataset(debug_ds_path, domain=this_domain, cost_lambda=cost_lambda)
            except Exception:
                pass
        except Exception:
            pass

    # 选择 Top-K 并写入本域辞海（run_dir 下与全局）
    unique = []
    seen = set()
    for score, seq, dr, c in scored:
        t = tuple(seq)
        if t in seen:
            continue
        seen.add(t)
        unique.append((score, seq, dr, c))
        if len(unique) >= topk:
            break

    # 保存模型权重（可选）
    torch.save(model.state_dict(), run_dir / "scorer.pt")

    # 训练结束后，自动提取算子包并写入训练目录
    try:
        pkg = extract_operator_package(run_dir, cfg_path, domain_override=this_domain)
        run_dict_path = run_dir / f"{this_domain}_operator_packages.json"
        arr: List[Dict[str, Any]] = []
        if run_dict_path.exists():
            try:
                arr = json.loads(run_dict_path.read_text(encoding="utf-8"))
            except Exception:
                arr = []
        arr.append(pkg)
        text = json.dumps(arr, ensure_ascii=False, indent=2)
        text = text.replace("\r\n", "\n")
        run_dict_path.write_text(text, encoding="utf-8")
    except Exception:
        pass

    print(f"Training finished. Artifacts at: {run_dir}")
    return run_dir


def train_all(config_path: str | Path | None = None, domains: list[str] | None = None) -> list[Path]:
    """按顺序训练多个域（默认七域），并分别提取算子包。

    返回各域的训练输出目录列表。
    """
    doms = domains or ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]
    out: list[Path] = []
    for d in doms:
        try:
            rd = train(config_path, domain_override=d)
            out.append(rd)
        except Exception:
            continue
    return out


def extract_operator_package(run_dir: str | Path, config_path: str | Path | None = None, max_len: int | None = None) -> \
        Dict[str, Any]:
    """贪心解码提取算子包，写入对应域的 operator_packages.json。"""
    mod_dir = Path(__file__).resolve().parent
    cfg_path = Path(config_path) if config_path else (mod_dir / "config.json")
    cfg = _load_config(cfg_path)
    domain = _resolve_domain(cfg, None)
    env = build_env_from_config(cfg_path, domain_override=domain)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.high - env.action_space.low

    # 加载策略
    run_dir = Path(run_dir)
    global DiscretePolicy
    if DiscretePolicy is None:
        import torch.nn as nn
        import torch.nn.functional as F  # noqa: F401
        class _FallbackDiscretePolicy(nn.Module):
            def __init__(self, od: int, na: int, hidden=(128, 128)):
                super().__init__()
                layers = []
                last = od
                for h in hidden:
                    layers.append(nn.Linear(last, h))
                    layers.append(nn.ReLU())
                    last = h
                layers.append(nn.Linear(last, na))
                self.net = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor):
                logits = self.net(x)
                probs = torch.softmax(logits, dim=-1)
                return logits, probs

        DiscretePolicy = _FallbackDiscretePolicy  # type: ignore
    policy = DiscretePolicy(obs_dim, n_actions)
    policy.load_state_dict(torch.load(run_dir / "policy.pt", map_location="cpu"))
    policy.eval()

    # 贪心 rollout
    names = {v: k for k, v in env.op2idx.items()}
    seq: List[str] = []
    s = env.reset().unsqueeze(0)
    steps = 0
    max_len = int(max_len or env._max_steps)
    with torch.no_grad():
        while steps < max_len:
            logits, probs = policy(s)
            a = int(torch.argmax(probs, dim=-1).item())
            seq.append(names.get(a, str(a)))
            s2, r, d, _ = env.step(a)
            s = s2.unsqueeze(0)
            steps += 1
            if d:
                break

    # 写入 package 字典
    pkg = {
        "id": f"pkg_{domain}_{int(_pytime.time())}",
        "domain": domain,
        "initial": {
            "b": env._init_state_cfg.b,
            "n_comp": env._init_state_cfg.n_comp,
            "perim": env._init_state_cfg.perim,
            "fidelity": env._init_state_cfg.fidelity,
        },
        "target": {
            "b": env._goal.target.b,
            "n_comp": env._goal.target.n_comp,
            "perim": env._goal.target.perim,
            "fidelity": env._goal.target.fidelity,
        },
        "sequence": seq,
        "max_steps": max_len,
        "created_at": int(_pytime.time()),
    }

    dict_path = mod_dir / get_domain_spec(domain).dict_filename
    arr: List[Dict[str, Any]] = []
    if dict_path.exists():
        try:
            arr = json.loads(dict_path.read_text(encoding="utf-8"))
        except Exception:
            arr = []
    arr.append(pkg)
    text = json.dumps(arr, ensure_ascii=False, indent=2)
    text = text.replace("\r\n", "\n")
    dict_path.write_text(text, encoding="utf-8")
    return pkg


if __name__ == "__main__":
    out = train()
    print(out)

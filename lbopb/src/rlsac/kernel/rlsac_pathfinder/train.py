# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _pytime
from pathlib import Path
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


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def build_env_from_config(cfg_path: Path, domain_override: str | None = None) -> DomainPathfinderEnv:
    cfg = _load_config(cfg_path)
    domain = str(domain_override if domain_override is not None else cfg.get("domain", "pem")).lower()
    spec = get_domain_spec(domain)
    s0_cfg = cfg.get("initial_state", {"b": 3.0, "n_comp": 3, "perim": 5.0, "fidelity": 0.4})
    st_cfg = cfg.get("target_state", {"b": 0.5, "n_comp": 1, "perim": 2.0, "fidelity": 0.9})
    tol_cfg = cfg.get("tolerance", {"b": 0.1, "n": 0.5, "p": 0.2, "f": 0.05})
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

    class RunLogger:
        def __init__(self, path: Path, append: bool = False):
            self.path = path
            self.f = open(path, 'a' if append else 'w', encoding='utf-8', newline='')

        def write_line(self, text: str) -> None:
            try:
                self.f.write(text + "\r\n")
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

    # 环境
    # 确定本次训练的域
    this_domain = str(domain_override if domain_override is not None else cfg.get("domain", "pem")).lower()

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

    X = torch.zeros((n_samples, feat_dim), dtype=torch.float32)
    y = torch.zeros((n_samples,), dtype=torch.float32)
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
        # heuristics
        _, dr, c = apply_sequence(domain, s0, seq)
        heur_ok = bool((dr - cost_lambda * c) > 0.0)
        # LLM only when warnings present and enabled
        llm_used = False
        llm_attempted = False
        llm_raw: str | None = None
        llm_prompt: str | None = None
        llm_status: str = "skipped_no_warn"  # used | skipped_no_warn | skipped_use_llm_false | skipped_errors | skipped_exception
        if syntax["errors"]:
            label = 0
            llm_status = "skipped_errors"
        else:
            label = int(heur_ok)
            warns_present = bool(syntax.get("warnings"))
            if warns_present and bool(cfg.get("use_llm_oracle", False)):
                try:
                    from lbopb.src.rlsac.kernel.common.llm_oracle import call_llm, build_pathfinder_prompt
                    llm_prompt = build_pathfinder_prompt(domain, list(seq))
                    llm_attempted = True
                    txt = call_llm(llm_prompt)
                    llm_used = True
                    llm_status = "used"
                    llm_raw = str(txt) if txt is not None else None
                    if isinstance(txt, str):
                        label = int(heur_ok and (("1" in txt and "0" not in txt) or (txt.strip() == "1")))
                except Exception:
                    llm_attempted = True
                    llm_used = False
                    llm_status = "skipped_exception"
                    llm_raw = None
            elif warns_present and not bool(cfg.get("use_llm_oracle", False)):
                llm_status = "skipped_use_llm_false"
            else:
                llm_status = "skipped_no_warn"
        # 控制台调试输出
        if debug:
            print(
                "[SAMPLE] seq={seq} label={label} syntax_errors={e} warnings={w} "
                "heur_ok={heur} llm_attempted={la} llm_used={lu} llm_status={ls}".format(
                    seq=seq,
                    label=label,
                    e=len(syntax["errors"]),
                    w=len(syntax["warnings"]),
                    heur=heur_ok,
                    la=llm_attempted,
                    lu=llm_used,
                    ls=llm_status,
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
        }
    for i in range(n_samples):
        seq = sample_random_package(spec, min_len=min_len, max_len=max_len, no_consecutive_duplicate=no_dup)
        # 特征：op 计数
        cnts = [float(seq.count(nm)) for nm in op_names]
        # 作用效果特征
        _, dr, c = apply_sequence(this_domain, init_state, seq)
        length = float(len(seq))
        vec = cnts + [length, float(dr), float(c)]
        X[i] = torch.tensor(vec, dtype=torch.float32)
        judge = evaluate_with_details(this_domain, seq, init_state)
        label_i = int(judge["label"])
        y[i] = float(label_i)
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
                "label": int(label_i)
            })

    model = PackageScorer(X.shape[1]).to(device)
    train_scorer(model, X.to(device), y.to(device), epochs=epochs, batch_size=batch_size,
                 lr=float(cfg.get("learning_rate_actor", 3e-4)))

    # 校验并可选迭代直至完美匹配
    fit_until_perfect = bool(cfg.get("fit_until_perfect", True))
    max_refit_rounds = int(cfg.get("max_refit_rounds", 10))
    refit_epochs = int(cfg.get("refit_epochs", epochs))
    rounds = 0
    with torch.no_grad():
        preds = (model(X.to(device)).cpu().view(-1) >= 0.5).to(torch.float32)
        acc = float((preds == y).to(torch.float32).mean().item())
    while fit_until_perfect and acc < 1.0 and rounds < max_refit_rounds:
        train_scorer(model, X.to(device), y.to(device), epochs=refit_epochs, batch_size=batch_size,
                     lr=float(cfg.get("learning_rate_actor", 3e-4)))
        with torch.no_grad():
            preds = (model(X.to(device)).cpu().view(-1) >= 0.5).to(torch.float32)
            acc = float((preds == y).to(torch.float32).mean().item())
        rounds += 1

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
            # 统计 warnings / llm 覆盖率
            total = len(dataset_dump)
            err_cnt = 0; warn_cnt = 0; llm_cnt = 0; syntax_ok_cnt = 0; llm_attempted_cnt = 0
            llm_used_cnt = 0; llm_skip_no_warn = 0; llm_skip_use_llm_false = 0; llm_skip_errors = 0; llm_skip_exception = 0
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
            # 控制台与日志打印
            msg_stats = (
                f"[STATS] total={total} error_samples={err_cnt} warning_samples={warn_cnt} "
                f"syntax_ok_samples={syntax_ok_cnt} llm_attempted_samples={llm_attempted_cnt} llm_used_samples={llm_cnt}"
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
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")
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
    domain = str(cfg.get("domain", "pem")).lower()
    env = build_env_from_config(cfg_path)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.high - env.action_space.low

    # 加载策略
    run_dir = Path(run_dir)
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
    text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    dict_path.write_text(text, encoding="utf-8")
    return pkg


if __name__ == "__main__":
    out = train()
    print(out)

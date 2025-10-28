# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _pytime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# 支持脚本直跑与包内运行的双路径导入
try:
    from .sampler import load_domain_packages, sample_random_connection  # type: ignore
    from .oracle import ConnectorAxiomOracle, MODULES  # type: ignore
except Exception:
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))
        from lbopb.src.rlsac.kernel.rlsac_connector.sampler import load_domain_packages, sample_random_connection  # type: ignore
        from lbopb.src.rlsac.kernel.rlsac_connector.oracle import ConnectorAxiomOracle, MODULES  # type: ignore
    except Exception as _e:
        raise _e

from lbopb.src.rlsac.kernel.rlsac_pathfinder.scorer import PackageScorer, train_scorer  # type: ignore
from lbopb.src.rlsac.application.rlsac_nsclc.utils import select_device_from_config  # type: ignore


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


def _ops_detailed_of(pkg: Dict[str, Any]) -> List[Dict[str, Any]] | None:
    steps = pkg.get("ops_detailed")
    if isinstance(steps, list) and steps:
        return steps
    seq = list(pkg.get("sequence", []) or [])
    if not seq:
        return None
    return [{"name": str(nm)} for nm in seq]


def _dump_pairwise_dataset(cfg: Dict[str, Any], *, pkg_map: Dict[str, List[Dict]]) -> Path:
    """生成两两域算子包联络数据集（含参数 steps），写入 out/rlsac_connector/dataset_<ts>。"""
    import hashlib
    import importlib
    import random as _rnd

    repo_root = _repo_root()
    out_root = repo_root / "out"
    base_out = out_root / "rlsac_connector"
    base_out.mkdir(parents=True, exist_ok=True)
    run_dir = base_out / ("dataset_" + str(int(_pytime.time())))
    run_dir.mkdir(parents=True, exist_ok=True)

    per_pair = int(cfg.get("dataset_per_pair", 50))
    use_llm = bool(cfg.get("use_llm_oracle", False))

    samples: List[Dict[str, Any]] = []
    modules = MODULES
    for i, a in enumerate(modules):
        arr_a = pkg_map.get(a) or []
        if not arr_a:
            continue
        for b in modules[i + 1 :]:
            arr_b = pkg_map.get(b) or []
            if not arr_b:
                continue
            for _ in range(min(per_pair, max(len(arr_a), len(arr_b)))):
                pa = _rnd.choice(arr_a)
                pb = _rnd.choice(arr_b)
                seq_a = list(pa.get("sequence", []) or [])
                seq_b = list(pb.get("sequence", []) or [])
                steps_a = _ops_detailed_of(pa)
                steps_b = _ops_detailed_of(pb)
                payload = {
                    "pair": [a, b],
                    "src": {"domain": a, "id": pa.get("id", ""), "sequence": seq_a, "ops_detailed": steps_a or []},
                    "dst": {"domain": b, "id": pb.get("id", ""), "sequence": seq_b, "ops_detailed": steps_b or []},
                }
                blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                hid = hashlib.sha256(blob).hexdigest()

                # pair 级语法与可选 LLM 判定
                syn_res: Dict[str, Any] | None = None
                llm_status = "skipped_no_warn"
                llm_used = False
                try:
                    _ensure_repo_in_sys_path()
                    mod = importlib.import_module(f"lbopb.src.common.{a}_{b}_syntax_checker")
                    syn_fn = getattr(mod, "check", None)
                    if callable(syn_fn):
                        syn_res = syn_fn(seq_a, seq_b)
                except Exception:
                    syn_res = None

                if use_llm and syn_res and int(syn_res.get("warnings", 0)) > 0:
                    try:
                        from lbopb.src.rlsac.kernel.common.llm_oracle import build_connector_prompt, call_llm  # type: ignore
                        txt = call_llm(build_connector_prompt({a: seq_a, b: seq_b}))
                        if isinstance(txt, str):
                            llm_used = True
                            ok = ("1" in txt and "0" not in txt) or (txt.strip() == "1")
                            llm_status = "used" if ok else "used_neg"
                    except Exception:
                        llm_status = "skipped_exception"

                rec = {
                    "id": f"conn_{a}_{b}_{hid}",
                    "pair": {"a": a, "b": b},
                    "package_a": payload["src"],
                    "package_b": payload["dst"],
                    "features": {
                        "length_a": int(len(seq_a)),
                        "length_b": int(len(seq_b)),
                        "length": int(len(seq_a) + len(seq_b)),
                    },
                    "judge": {
                        "syntax": syn_res or {"result": "未知", "errors": [], "warnings": []},
                        "llm_used": bool(llm_used),
                        "llm_status": llm_status,
                    },
                    "created_at": int(_pytime.time()),
                    "updated_at": int(_pytime.time()),
                    "source": "pair_from_packages",
                }
                samples.append(rec)

    # 写文件：debug_dataset.json + 简要统计
    ds_path = run_dir / "debug_dataset.json"
    with open(ds_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "pairs": MODULES,
            "samples": samples,
        }, f, ensure_ascii=False, indent=2)

    # 统计
    try:
        total = len(samples)
        by_pair: Dict[str, int] = {}
        for it in samples:
            pr = f"{it['pair']['a']}_{it['pair']['b']}"
            by_pair[pr] = int(by_pair.get(pr, 0)) + 1
        stats = {
            "updated_at": int(_pytime.time()),
            "total": total,
            "pairs": by_pair,
        }
        with (run_dir / "debug_dataset.stats.json").open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(stats, ensure_ascii=False, indent=2))
    except Exception:
        pass

    print(f"[dataset] written: {ds_path} items={len(samples)}")
    return run_dir




def _load_config(cfg_path: Path) -> Dict[str, Any]:
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _feat_from_meta(lens: List[float], meta: Dict[str, float]) -> torch.Tensor:
    return torch.tensor([
        *lens,
        float(meta.get("delta_risk_sum", 0.0)),
        float(meta.get("cost", 0.0)),
        float(meta.get("consistency", 0.0)),
    ], dtype=torch.float32)


def train(config_path: str | Path | None = None) -> Path:
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
                self.f.write(text + "\n")
                self.f.flush()
            except Exception:
                pass

        def close(self) -> None:
            try:
                self.f.flush();
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

    # 采样-监督设置
    samples = int(cfg.get("samples", 2000))
    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch_size", 64))
    topk = int(cfg.get("topk", 50))
    cand_gen = int(cfg.get("candidate_generate", 1000))
    cost_lambda = float(cfg.get("cost_lambda", 0.2))
    eps_change = float(cfg.get("eps_change", 1e-3))
    pk_dir = Path(cfg.get("packages_dir", "lbopb/src/rlsac/kernel/rlsac_pathfinder"))
    # 兼容 monoid_packages 子目录
    if not (pk_dir / "pem_operator_packages.json").exists():
        pk_dir = pk_dir / "monoid_packages"
    pkg_map = load_domain_packages(pk_dir)
    oracle = ConnectorAxiomOracle(cost_lambda=cost_lambda, eps_change=eps_change,
                                  use_llm=bool(cfg.get('use_llm_oracle', False)))

    # explore_only: 仅生成 out/rlsac_connector/dataset_<ts>
    if bool(cfg.get('explore_only', True)):
        return _dump_pairwise_dataset(cfg, pkg_map=pkg_map)

    repo_root = Path(__file__).resolve().parents[5]
    out_root = repo_root / 'out'
    base_out = out_root / Path(cfg.get('output_dir', 'out_connector'))
    base_out.mkdir(parents=True, exist_ok=True)
    run_dir = base_out / ("train_" + str(int(_pytime.time())))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    logger: RunLogger | None = None
    if log_to_file:
        logger = RunLogger(log_path, append=False)
        logger.write_line(f"# TRAIN START {_pytime.strftime('%Y-%m-%d %H:%M:%S', _pytime.localtime())}")
        logger.write_line(f"device={device}")
        logger.write_line(f"config={json.dumps(cfg, ensure_ascii=False)}")

    # 构造监督数据
    # 特征维度与监督数据容器
    feat_dim = 10
    X = torch.zeros((samples, feat_dim), dtype=torch.float32)
    y = torch.zeros((samples,), dtype=torch.float32)

    for i in range(samples):
        choice = sample_random_connection(pkg_map)
        lens = [float(len((choice[m].get("sequence", []) or []))) for m in MODULES]
        label, meta = oracle.judge({m: (choice[m].get("sequence", []) or []) for m in MODULES})
        X[i] = _feat_from_meta(lens, meta)
        y[i] = float(label)
        if logger and (i % max(1, samples // 10) == 0):
            logger.write_line(f"[SAMPLE {i}] lens={lens} meta={meta} label={int(label)}")

    # 训练打分器
    model = PackageScorer(feat_dim).to(device)
    train_scorer(model, X.to(device), y.to(device), epochs=epochs, batch_size=batch_size,
                 lr=float(cfg.get("learning_rate_actor", 3e-4)))
    torch.save(model.state_dict(), run_dir / "scorer.pt")

    # 生成候选并评分，取 Top-K
    cand: List[tuple[float, Dict[str, Dict], Dict[str, float]]] = []
    with torch.no_grad():
        for _ in range(cand_gen):
            choice = sample_random_connection(pkg_map)
            lens = [float(len((choice[m].get("sequence", []) or []))) for m in MODULES]
            label, meta = oracle.judge({m: (choice[m].get("sequence", []) or []) for m in MODULES})
            vec = _feat_from_meta(lens, meta).unsqueeze(0).to(device)
            score = float(model(vec).item())
            cand.append((score, choice, meta))
    cand.sort(key=lambda t: t[0], reverse=True)

    # 写入联络辞海（模块目录与运行目录）
    law_entries: List[Dict[str, Any]] = []
    for score, choice, meta in cand[:topk]:
        entry = {
            "id": f"conn_{int(_pytime.time())}",
            "chosen": {m: choice[m].get("id", "") for m in MODULES},
            "meta": meta,
            "score": float(score),
        }
        law_entries.append(entry)

    # 追加写入模块目录 law_connections.json
    mod_law = mod_dir / "law_connections.json"
    mod_arr: List[Dict[str, Any]] = []
    if mod_law.exists():
        try:
            mod_arr = json.loads(mod_law.read_text(encoding='utf-8'))
        except Exception:
            mod_arr = []
    mod_arr.extend(law_entries)
    text = json.dumps(mod_arr, ensure_ascii=False, indent=2).replace("\r\n", "\n")
    with mod_law.open("w", encoding='utf-8', newline='\n') as f:
        f.write(text)

    # 写入运行目录 law_connections.json
    run_law = run_dir / "law_connections.json"
    run_arr: List[Dict[str, Any]] = []
    if run_law.exists():
        try:
            run_arr = json.loads(run_law.read_text(encoding='utf-8'))
        except Exception:
            run_arr = []
    run_arr.extend(law_entries)
    text2 = json.dumps(run_arr, ensure_ascii=False, indent=2).replace("\r\n", "\n")
    with run_law.open("w", encoding='utf-8', newline='\n') as f:
        f.write(text2)

    print(f"Training finished. Artifacts at: {run_dir}")
    return run_dir


def extract_connection(run_dir: str | Path, config_path: str | Path | None = None) -> Dict[str, Any]:
    """使用已训练的打分器随机采样并选出最高分的联络候选，写入 law_connections.json。"""
    mod_dir = Path(__file__).resolve().parent
    cfg_path = Path(config_path) if config_path else (mod_dir / "config.json")
    cfg = _load_config(cfg_path)
    run_dir = Path(run_dir)

    # 加载打分器
    feat_dim = 10
    model = PackageScorer(feat_dim)
    try:
        model.load_state_dict(torch.load(run_dir / "scorer.pt", map_location="cpu"))
    except Exception:
        pass
    model.eval()

    pk_dir = Path(cfg.get("packages_dir", "lbopb/src/rlsac/kernel/rlsac_pathfinder"))
    pkg_map = load_domain_packages(pk_dir)
    oracle = ConnectorAxiomOracle(cost_lambda=cost_lambda, eps_change=eps_change,
                                  use_llm=bool(cfg.get('use_llm_oracle', False)))

    # explore_only: 仅生成 out/rlsac_connector/dataset_<ts>
    if bool(cfg.get('explore_only', True)):
        return _dump_pairwise_dataset(cfg, pkg_map=pkg_map)

    repo_root = Path(__file__).resolve().parents[5]
    out_root = repo_root / 'out'
    base_out = out_root / Path(cfg.get('output_dir', 'out_connector'))
    best = None
    best_choice = None
    best_meta = None
    with torch.no_grad():
        for _ in range(int(cfg.get("candidate_generate", 1000))):
            choice = sample_random_connection(pkg_map)
            lens = [float(len((choice[m].get("sequence", []) or []))) for m in MODULES]
            label, meta = oracle.judge({m: (choice[m].get("sequence", []) or []) for m in MODULES})
            vec = _feat_from_meta(lens, meta).unsqueeze(0)
            score = float(model(vec).item())
            if best is None or score > best:
                best = score;
                best_choice = choice;
                best_meta = meta

    conn = {
        "id": f"conn_{int(_pytime.time())}",
        "chosen": {m: best_choice[m].get("id", "") for m in MODULES} if best_choice else {},
        "meta": best_meta or {},
        "score": float(best or 0.0),
    }

    law_path = mod_dir / "law_connections.json"
    arr: List[Dict[str, Any]] = []
    if law_path.exists():
        try:
            arr = json.loads(law_path.read_text(encoding="utf-8"))
        except Exception:
            arr = []
    arr.append(conn)
    text = json.dumps(arr, ensure_ascii=False, indent=2).replace("\r\n", "\n")
    with law_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return conn


if __name__ == "__main__":
    out = train()
    print(out)

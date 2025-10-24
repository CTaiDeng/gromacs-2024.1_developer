# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _pytime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .sampler import load_domain_packages, sample_random_connection
from .oracle import ConnectorAxiomOracle, MODULES
from lbopb.src.rlsac.kernel.rlsac_pathfinder.scorer import PackageScorer, train_scorer
from lbopb.src.rlsac.application.rlsac_nsclc.utils import select_device_from_config


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
                self.f.write(text + "\r\n")
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
    pkg_map = load_domain_packages(pk_dir)
    oracle = ConnectorAxiomOracle(cost_lambda=cost_lambda, eps_change=eps_change,
                                  use_llm=bool(cfg.get("use_llm_oracle", False)))

    # 特征：每域长度 + 全局统计（ΣΔrisk, Σcost, consistency）= 7 + 3 = 10
    feat_dim = 10
    X = torch.zeros((samples, feat_dim), dtype=torch.float32)
    y = torch.zeros((samples,), dtype=torch.float32)

    repo_root = Path(__file__).resolve().parents[5]
    out_root = repo_root / "out"
    base_out = out_root / Path(cfg.get("output_dir", "out_connector"))
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
    text = json.dumps(mod_arr, ensure_ascii=False, indent=2)
    text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    mod_law.write_text(text, encoding='utf-8')

    # 写入运行目录 law_connections.json
    run_law = run_dir / "law_connections.json"
    run_arr: List[Dict[str, Any]] = []
    if run_law.exists():
        try:
            run_arr = json.loads(run_law.read_text(encoding='utf-8'))
        except Exception:
            run_arr = []
    run_arr.extend(law_entries)
    text2 = json.dumps(run_arr, ensure_ascii=False, indent=2)
    text2 = text2.replace("\r\n", "\n").replace("\n", "\r\n")
    run_law.write_text(text2, encoding='utf-8')

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
    oracle = ConnectorAxiomOracle(cost_lambda=float(cfg.get("cost_lambda", 0.2)),
                                  eps_change=float(cfg.get("eps_change", 1e-3)))

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
    text = json.dumps(arr, ensure_ascii=False, indent=2)
    text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    law_path.write_text(text, encoding="utf-8")
    return conn


if __name__ == "__main__":
    out = train()
    print(out)

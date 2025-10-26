# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _t
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    from .domain import get_domain_spec
    from .scorer import PackageScorer, train_scorer
except Exception:
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.domain import get_domain_spec  # type: ignore
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.scorer import PackageScorer, train_scorer  # type: ignore


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_domain(cfg: Dict[str, Any]) -> str:
    d = cfg.get("domain")
    if isinstance(d, str) and d:
        return d.strip().lower()
    try:
        dom_num = int(d)
        mapping = cfg.get("domain_choose", {}) or {}
        for k, v in mapping.items():
            try:
                if int(v) == dom_num:
                    return str(k).strip().lower()
            except Exception:
                continue
    except Exception:
        pass
    return "pem"

def _all_domains(cfg: Dict[str, Any]) -> List[str]:
    mapping = cfg.get("domain_choose", {}) or {}
    if isinstance(mapping, dict) and mapping:
        try:
            return [k for k, _ in sorted(mapping.items(), key=lambda kv: int(kv[1]))]
        except Exception:
            return list(mapping.keys())
    return ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]


def _reward_from_record(rec: Dict[str, Any]) -> float:
    # 优先使用 labeled 的 validation
    val = rec.get("validation")
    if isinstance(val, dict):
        # Gemini 结果优先（若存在）
        gem = val.get("gemini") or {}
        if isinstance(gem, dict) and gem.get("used") and isinstance(gem.get("result"), str):
            r = str(gem.get("result")).strip()
            if r == "正确":
                return 1.0
            if r == "错误":
                return 0.0
        # 其次看 syntax
        syn = val.get("syntax") or {}
        if isinstance(syn, dict) and isinstance(syn.get("result"), str):
            r = str(syn.get("result")).strip()
            if r == "正确":
                return 1.0
            if r == "警告":
                return 0.5
            if r == "错误":
                return 0.0
    # 回退：使用 judge 结构
    j = rec.get("judge", {}) or {}
    syn = j.get("syntax", {}) or {}
    try:
        if (syn.get("errors") or []) or False:
            return 0.0
        if (syn.get("warnings") or []):
            return 0.5
        return 1.0
    except Exception:
        pass
    # 最后回退 label
    try:
        return 1.0 if int(rec.get("label", 0)) == 1 else 0.0
    except Exception:
        return 0.0


def train_from_debug_dataset(config_path: str | Path | None = None, domain_override: str | None = None) -> Path:
    mod_dir = Path(__file__).resolve().parent
    cfg_path = Path(config_path) if config_path else (mod_dir / "config.json")
    cfg = _load_json(cfg_path) or {}
    domains: List[str] = [str(domain_override).lower()] if domain_override else _all_domains(cfg)

    # 输出目录
    repo_root = Path(__file__).resolve().parents[5]
    out_root = repo_root / "out" / (cfg.get("output_dir", "out_pathfinder"))
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / ("train_" + str(int(_t.time())))
    run_dir.mkdir(parents=True, exist_ok=True)

    # 读取聚合数据
    ds_path = mod_dir / "train_datas" / "debug_dataset.json"
    data = _load_json(ds_path)
    if not isinstance(data, list):
        data = []
    # 特征构造：bag-of-ops + length（基于域的 op 集）
    op_set = set()
    for dom in domains:
        try:
            spec = get_domain_spec(dom)
            for cls in spec.op_classes:
                try:
                    nm = cls().name
                except Exception:
                    nm = cls.__name__
                op_set.add(str(nm))
        except Exception:
            continue
    op_names: List[str] = sorted(op_set)
    dom_index = {d: i for i, d in enumerate(domains)}

    # Debug 配置与颜色
    debug = bool(cfg.get("debug", True))
    ANSI_RESET = "\x1b[0m"
    ANSI_RED = "\x1b[31;1m"
    ANSI_GREEN = "\x1b[32;1m"
    ANSI_YELLOW = "\x1b[33;1m"
    ANSI_CYAN = "\x1b[36;1m"

    if debug:
        print(f"{ANSI_CYAN}[train_rl] domains={domains} ds_path={ds_path}{ANSI_RESET}")

    X: List[List[float]] = []
    y: List[float] = []
    for rec in data:
        try:
            dom = str(rec.get("domain", "")).lower()
            if domain_override and dom != domains[0]:
                continue
            seq = list(rec.get("sequence", []) or [])
            length = float(len(seq))
            cnts = [float(seq.count(nm)) for nm in op_names]
            onehot = [0.0] * len(domains)
            if dom in dom_index:
                onehot[dom_index[dom]] = 1.0
            X.append([*cnts, length, *onehot])
            y.append(float(_reward_from_record(rec)))
        except Exception:
            continue

    # 训练
    feat_dim = len(op_names) + 1 + len(domains)
    model = PackageScorer(feat_dim)
    if len(X) > 0:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        train_scorer(model, X_t, y_t, epochs=int(cfg.get("epochs", 20)), batch_size=int(cfg.get("batch_size", 64)),
                     lr=float(cfg.get("learning_rate_actor", 3e-4)))
        # 追加多轮训练
        train_rounds = int(cfg.get("train_rounds", 3))
        epochs_per_round = int(cfg.get("epochs_per_round", cfg.get("epochs", 20)))
        batch_size = int(cfg.get("batch_size", 64))
        lr = float(cfg.get("learning_rate_actor", 3e-4))
        if debug:
            print(f"{ANSI_CYAN}[train_rl] samples={len(X)} feat_dim={len(op_names)+1+len(domains)} ops={len(op_names)} rounds={train_rounds} epochs/round={epochs_per_round}{ANSI_RESET}")
        for r in range(2, train_rounds + 1):
            if debug:
                print(f"{ANSI_YELLOW}[train_rl] round {r}/{train_rounds} begin{ANSI_RESET}")
            train_scorer(model, X_t, y_t, epochs=epochs_per_round, batch_size=batch_size, lr=lr)
            with torch.no_grad():
                preds = model(X_t).cpu().view(-1)
                mse = float(torch.mean((preds - y_t) ** 2).item())
                acc = float(((preds >= 0.5).to(torch.float32) == y_t).to(torch.float32).mean().item())
            if debug:
                print(f"{ANSI_GREEN}[train_rl] round {r} done mse={mse:.4f} acc={acc:.4f}{ANSI_RESET}")

    torch.save(model.state_dict(), run_dir / "scorer.pt")
    # 写入简单元数据
    meta = {
        "domains": domains,
        "samples": len(X),
        "op_names": op_names,
        "feat_dim": feat_dim,
        "source": str(ds_path),
        "train_rounds": int(cfg.get("train_rounds", 3)),
        "epochs_per_round": int(cfg.get("epochs_per_round", cfg.get("epochs", 20)))
    }
    (run_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{ANSI_GREEN}[train_rl] saved scorer to: {run_dir / 'scorer.pt'}{ANSI_RESET}")
    # 不再生成内嵌 apply_model.py；请使用 CLI：
    # python lbopb/src/rlsac/kernel/rlsac_pathfinder/apply_model_cli.py <run_dir> [infile] [outfile]
    # 自动生成 demo 样本（含参数），便于后续使用 CLI 推理
    try:
        ops = meta.get("op_names", []) or []
        doms = meta.get("domains", []) or []
        d0 = (doms[0] if doms else "pem")
        try:
            spec0 = get_domain_spec(d0)
            dom_ops = []
            for cls in spec0.op_classes:
                try:
                    nm = cls().name
                except Exception:
                    nm = cls.__name__
                dom_ops.append(str(nm))
        except Exception:
            dom_ops = ops
        seq1 = [dom_ops[0]] if len(dom_ops) > 0 else []
        seq2 = [dom_ops[1], dom_ops[2]] if len(dom_ops) > 2 else seq1
        steps1 = []
        steps2 = []
        try:
            from lbopb.src.rlsac.kernel.rlsac_pathfinder.op_space_utils import load_op_space, param_grid_of, params_from_grid  # type: ignore
            space_ref = mod_dir / "operator_spaces" / f"{d0}_op_space.v1.json"
            if space_ref.exists():
                space = load_op_space(str(space_ref))
                def _synth(seq):
                    steps = []
                    for nm in seq:
                        try:
                            _, grids = param_grid_of(space, nm)
                            gi = [max(0, (len(g) - 1) // 2) for g in grids]
                            prs = params_from_grid(space, nm, gi)
                            steps.append({"name": nm, "grid_index": gi, "params": prs})
                        except Exception:
                            steps.append({"name": nm})
                    return steps
                steps1 = _synth(seq1)
                steps2 = _synth(seq2)
        except Exception:
            pass
        if not steps1:
            steps1 = [{"name": nm} for nm in seq1]
        if not steps2:
            steps2 = [{"name": nm} for nm in seq2]
        samples = [
            {"id": "demo_1", "domain": d0, "sequence": seq1, "ops_detailed": steps1},
            {"id": "demo_2", "domain": d0, "sequence": seq2, "ops_detailed": steps2},
        ]
        (run_dir / "samples.input.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{ANSI_YELLOW}[train_rl] sample written: {run_dir / 'samples.input.json'}")
        print("使用 CLI 推理: python lbopb/src/rlsac/kernel/rlsac_pathfinder/apply_model_cli.py <run_dir> [infile] [outfile]")
    except Exception:
        pass
    return run_dir

    # 生成应用脚本与样本输入
    try:
        apply_py = run_dir / "apply_model.py"
        apply_src = """
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.scorer import PackageScorer  # type: ignore
except Exception:
    # 兜底：最小定义（与训练时相同结构）
    import torch.nn as nn  # type: ignore
    class PackageScorer(nn.Module):  # type: ignore
        def __init__(self, in_dim: int, hidden=(128, 64)) -> None:
            super().__init__()
            h1, h2 = hidden
            self.net = nn.Sequential(
                nn.Linear(in_dim, h1), nn.ReLU(),
                nn.Linear(h1, h2), nn.ReLU(),
                nn.Linear(h2, 1)
            )
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            import torch.nn.functional as F  # type: ignore
            return torch.sigmoid(self.net(x)).view(-1)


ANSI_RESET = "\x1b[0m"; ANSI_CYAN = "\x1b[36;1m"; ANSI_GREEN = "\x1b[32;1m"; ANSI_YELLOW = "\x1b[33;1m"


def _truth_from_item(it: Dict[str, Any]) -> str | None:
    val = it.get("validation") or {}
    if isinstance(val, dict):
        gem = val.get("gemini") or {}
        if isinstance(gem, dict) and gem.get("used") and isinstance(gem.get("result"), str):
            r = str(gem.get("result")).strip()
            if r in ("正确", "错误"):
                return r
        syn = val.get("syntax") or {}
        if isinstance(syn, dict) and isinstance(syn.get("result"), str):
            r = str(syn.get("result")).strip()
            if r in ("正确", "警告", "错误"):
                return r
    if "label" in it:
        try:
            return "正确" if int(it.get("label", 0)) == 1 else "错误"
        except Exception:
            return None
    return None


def main() -> None:
    run_dir = Path(__file__).resolve().parent
    meta = json.loads((run_dir / "train_meta.json").read_text(encoding="utf-8"))
    model = PackageScorer(int(meta["feat_dim"]))
    model.load_state_dict(torch.load(run_dir / "scorer.pt", map_location="cpu"))
    model.eval()

    # IO（支持绝对/相对路径参数）
    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        _ip = Path(args[0])
        in_path = _ip if _ip.is_absolute() else (run_dir / _ip)
    else:
        in_path = run_dir / "samples.input.json"
    if len(args) > 1 and not args[1].startswith("-"):
        _op = Path(args[1])
        out_path = _op if _op.is_absolute() else (run_dir / _op)
    else:
        out_path = run_dir / "samples.output.json"
    print(f"{ANSI_CYAN}[apply] model={run_dir/'scorer.pt'} in={in_path} out={out_path}{ANSI_RESET}")
    try:
        data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    except Exception:
        data = []
    if not isinstance(data, list):
        data = []

    doms: List[str] = list(meta.get("domains", []))
    ops: List[str] = list(meta.get("op_names", []))
    def _feat_of(item: Dict[str, Any]) -> torch.Tensor:
        dom = str(item.get("domain", "")).lower()
        seq = list(item.get("sequence", []) or [])
        cnts = [float(seq.count(nm)) for nm in ops]
        length = float(len(seq))
        onehot = [0.0] * len(doms)
        try:
            idx = doms.index(dom)
            onehot[idx] = 1.0
        except Exception:
            pass
        return torch.tensor([*cnts, length, *onehot], dtype=torch.float32).unsqueeze(0)

    out: List[Dict[str, Any]] = []
    for it in data:
        try:
            x = _feat_of(it)
            with torch.no_grad():
                s = float(model(x).item())
            s3 = round(s * 2.0) / 2.0\n            label3 = ("正确" if s3 >= 0.75 else ("警告" if s3 >= 0.25 else "错误"))\n            reward = 1.0 if label3 == "正确" else (0.5 if label3 == "警告" else 0.0)\n            pred = "正确" if s >= 0.5 else "错误"
            truth = _truth_from_item(it)
            corr = (pred == truth) if truth in ("正确", "错误") else None
            out.append({
                "id": it.get("id"),
                "domain": it.get("domain"),
                "sequence": it.get("sequence"),
                "score": s,
                "pred": pred,
                **({"truth": truth} if truth is not None else {}),\n                "ops_detailed": it.get("ops_detailed"),\n                "score_tri": s3,\n                "label3": label3,\n                "reward": reward,
                **({"correct": bool(corr)} if corr is not None else {}),
                "threshold": 0.5,
            })
        except Exception:
            continue

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{ANSI_GREEN}[apply] written: {out_path}{ANSI_RESET}")


if __name__ == "__main__":
    main()
"""
        apply_py.write_text(apply_src.replace("\r\n", "\n"), encoding="utf-8")

        # 样本输入（基于元数据提供 2 条最小样例）
        ops = meta.get("op_names", []) or []
        doms = meta.get("domains", []) or []
        d0 = (doms[0] if doms else "pem")
        # 以域内算子名生成样本序列
        try:
            spec0 = get_domain_spec(d0)
            dom_ops = []
            for cls in spec0.op_classes:
                try:
                    nm = cls().name
                except Exception:
                    nm = cls.__name__
                dom_ops.append(str(nm))
        except Exception:
            dom_ops = ops
        seq1 = [dom_ops[0]] if len(dom_ops) > 0 else []
        seq2 = [dom_ops[1], dom_ops[2]] if len(dom_ops) > 2 else seq1
        # 合成中位参数
        steps1 = []
        steps2 = []
        try:
            from lbopb.src.rlsac.kernel.rlsac_pathfinder.op_space_utils import load_op_space, param_grid_of, params_from_grid  # type: ignore
            space_ref = mod_dir / "operator_spaces" / f"{d0}_op_space.v1.json"
            if space_ref.exists():
                space = load_op_space(str(space_ref))
                def _synth(seq):
                    steps = []
                    for nm in seq:
                        try:
                            _, grids = param_grid_of(space, nm)
                            gi = [max(0, (len(g) - 1) // 2) for g in grids]
                            prs = params_from_grid(space, nm, gi)
                            steps.append({"name": nm, "grid_index": gi, "params": prs})
                        except Exception:
                            steps.append({"name": nm})
                    return steps
                steps1 = _synth(seq1)
                steps2 = _synth(seq2)
        except Exception:
            pass
        if not steps1:
            steps1 = [{"name": nm} for nm in seq1]
        if not steps2:
            steps2 = [{"name": nm} for nm in seq2]
        samples = [
            {"id": "demo_1", "domain": d0, "sequence": seq1, "ops_detailed": steps1},
            {"id": "demo_2", "domain": d0, "sequence": seq2, "ops_detailed": steps2},
        ]
        (run_dir / "samples.input.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{ANSI_YELLOW}[train_rl] helper scripts written: {apply_py.name}, samples.input.json{ANSI_RESET}")
    except Exception:
        pass
    return run_dir


if __name__ == "__main__":
    train_from_debug_dataset()


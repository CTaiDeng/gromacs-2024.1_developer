# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _t
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for anc in [p.parent] + list(p.parents):
        try:
            if (anc / ".git").exists():
                return anc
        except Exception:
            continue
    return p.parents[-1]


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _label_from_validation(rec: Dict[str, Any]) -> int:
    val = rec.get("validation") or {}
    if isinstance(val, dict):
        syn = val.get("syntax") or {}
        try:
            errs = int(syn.get("errors", 0)) if isinstance(syn.get("errors"), int) else (len(syn.get("errors", []) or []))
            return 1 if errs == 0 else 0
        except Exception:
            pass
    try:
        return 1 if int(rec.get("label", 0)) == 1 else 0
    except Exception:
        return 0


def _gather_ops_and_pairs(data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    ops: List[str] = []
    op_set = set()
    pairs: List[str] = []
    pair_set = set()
    for it in data:
        pr = str(it.get("pair", "")).lower()
        if pr and pr not in pair_set:
            pair_set.add(pr); pairs.append(pr)
        # 优先从 ops_detailed 汇总操作名（两个域）
        od = it.get("ops_detailed") or {}
        if isinstance(od, dict) and od:
            for steps in od.values():
                try:
                    for st in (steps or []):
                        nm = str(st.get("name", ""))
                        if nm and nm not in op_set:
                            op_set.add(nm); ops.append(nm)
                except Exception:
                    continue
        else:
            seqs = it.get("sequences") or {}
            if isinstance(seqs, dict):
                for seq in seqs.values():
                    for nm in (seq or []):
                        nm = str(nm)
                        if nm and nm not in op_set:
                            op_set.add(nm); ops.append(nm)
    return ops, pairs


def _feat_of(rec: Dict[str, Any], ops: List[str], pairs: List[str]) -> List[float]:
    # bag-of-ops（汇总两个域）+ length_a + length_b + pair onehot
    cnts = [0.0] * len(ops)
    od = rec.get("ops_detailed") or {}
    if isinstance(od, dict) and od:
        for steps in od.values():
            try:
                for st in (steps or []):
                    nm = str(st.get("name", ""))
                    if nm in ops:
                        cnts[ops.index(nm)] += 1.0
            except Exception:
                continue
    else:
        seqs = rec.get("sequences") or {}
        if isinstance(seqs, dict):
            for seq in seqs.values():
                for nm in (seq or []):
                    nm = str(nm)
                    if nm in ops:
                        cnts[ops.index(nm)] += 1.0
    # lengths
    seqs = rec.get("sequences") or {}
    if isinstance(seqs, dict):
        vals = list(seqs.values())
        la = float(len(vals[0])) if len(vals) >= 1 else 0.0
        lb = float(len(vals[1])) if len(vals) >= 2 else 0.0
    else:
        la = float(rec.get("length_a", 0))
        lb = float(rec.get("length_b", 0))
    # pair onehot
    pr = str(rec.get("pair", "")).lower()
    pair_oh = [0.0] * len(pairs)
    try:
        idx = pairs.index(pr)
        pair_oh[idx] = 1.0
    except Exception:
        pass
    return [*cnts, la, lb, la + lb, *pair_oh]


class ConnectorClassifier(nn.Module):  # type: ignore
    def __init__(self, in_dim: int, hidden: Tuple[int, int] = (256, 128)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.sigmoid(self.net(x)).view(-1)


def train_from_connector_debug(debug_path: str | Path | None = None, *, epochs: int = 20, batch_size: int = 64, lr: float = 3e-4) -> Path:
    if torch is None:  # pragma: no cover
        raise RuntimeError("未检测到 PyTorch，请先安装 torch 后再运行训练。")

    mod_dir = Path(__file__).resolve().parent
    ds_path = Path(debug_path) if debug_path else (mod_dir / "train_datas" / "debug_dataset.json")
    data = _load_json(ds_path)
    if not isinstance(data, list):
        data = []
    # 构建词表与特征
    op_names, pair_names = _gather_ops_and_pairs(data)
    X: List[List[float]] = []
    y: List[float] = []
    for it in data:
        try:
            X.append(_feat_of(it, op_names, pair_names))
            y.append(float(_label_from_validation(it)))
        except Exception:
            continue
    if not X:
        raise RuntimeError(f"数据集中没有有效样本：{ds_path}")

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # 输出目录 out/out_connector/train_<ts>
    repo_root = _repo_root()
    out_root = repo_root / "out" / "out_connector"
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / ("train_" + str(int(_t.time())))
    run_dir.mkdir(parents=True, exist_ok=True)

    # 记录元信息，便于预测/复现
    meta = {
        "feat_dim": int(len(X_t[0])),
        "op_names": op_names,
        "pairs": pair_names,
        "dataset": str(ds_path),
        "samples": int(len(X)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
    }
    (run_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConnectorClassifier(len(X_t[0])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    log_path = run_dir / "train.log"
    with log_path.open("w", encoding="utf-8", newline="\n") as flog:
        flog.write(f"# TRAIN START {_t.strftime('%Y-%m-%d %H:%M:%S', _t.localtime())}\n")
        flog.write(f"device={device}\n")
        flog.write(f"meta={json.dumps(meta, ensure_ascii=False)}\n")
        for ep in range(int(epochs)):
            model.train(); total = 0.0; n = 0
            for bx, by in loader:
                bx = bx.to(device); by = by.to(device)
                opt.zero_grad()
                pred = model(bx)
                loss = loss_fn(pred, by)
                loss.backward(); opt.step()
                total += float(loss.item()) * int(bx.size(0)); n += int(bx.size(0))
            avg = total / max(1, n)
            print(f"[train] epoch={ep+1} loss={avg:.6f}")
            flog.write(f"[train] epoch={ep+1} loss={avg:.6f}\n")
            flog.flush()

    torch.save(model.state_dict(), run_dir / "scorer.pt")
    print(f"[train] finished. artifacts={run_dir}")
    return run_dir


if __name__ == "__main__":  # pragma: no cover
    train_from_connector_debug()


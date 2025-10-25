# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import json
import time as _pytime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .domain import get_domain_spec  # type: ignore
except Exception:
    # 直接脚本运行的兜底导入
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))
    from lbopb.src.rlsac.kernel.rlsac_pathfinder.domain import get_domain_spec  # type: ignore


def _store_dir() -> Path:
    """返回用于存放各幺半群算子包的目录路径。"""
    return Path(__file__).resolve().parent / "monoid_packages"


def ensure_store_dir() -> Path:
    p = _store_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _store_file_for_domain(domain: str) -> Path:
    spec = get_domain_spec(domain)
    # 复用规范化文件名，但归档于专用目录下
    return ensure_store_dir() / spec.dict_filename


def _compute_score(delta_risk: float, cost: float, cost_lambda: float) -> float:
    try:
        return float(delta_risk) - float(cost_lambda) * float(cost)
    except Exception:
        return float(delta_risk)


def _normalize_lf(text: str) -> str:
    # 统一为 LF 行尾
    return text.replace("\r\n", "\n")


def ingest_from_debug_dataset(debug_dataset_path: str | Path, *, domain: str | None = None,
                              cost_lambda: float = 0.2) -> Path:
    """
    从 debug_dataset.json 中提取 label=1 的“正确”算子包，按序列去重纳入，并按 score 重新排序后落盘。

    - 排序规则：score(desc) -> length(asc) -> sequence(字典序)
    - score = delta_risk - cost_lambda * cost

    返回：该幺半群对应的汇总 JSON 文件路径。
    """
    debug_dataset_path = Path(debug_dataset_path)
    if not debug_dataset_path.exists():
        raise FileNotFoundError(f"debug dataset not found: {debug_dataset_path}")

    data = json.loads(debug_dataset_path.read_text(encoding="utf-8"))
    ds_domain = str((domain or data.get("domain") or "")).strip().lower()
    if not ds_domain:
        raise ValueError("domain is required (either from file or argument)")

    samples: List[Dict[str, Any]] = list(data.get("samples", []) or [])
    out_path = _store_file_for_domain(ds_domain)

    # 读取现有聚合
    existing: List[Dict[str, Any]] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    # 基于 sequence 去重的 map
    def _seq_key(seq: List[str]) -> Tuple[str, ...]:
        return tuple(str(x) for x in (seq or []))

    by_seq: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for item in existing:
        seq = list(item.get("sequence", []) or [])
        by_seq[_seq_key(seq)] = item

    # 纳入新样本（仅 label==1）
    ts_now = int(_pytime.time())
    for it in samples:
        try:
            if int(it.get("label", 0)) != 1:
                continue
            seq = list(it.get("sequence", []) or [])
            feats = it.get("features", {}) if isinstance(it.get("features", {}), dict) else {}
            judge = it.get("judge", {}) if isinstance(it.get("judge", {}), dict) else {}
            # 两处择优取数
            dr = float(feats.get("delta_risk", judge.get("delta_risk", 0.0)))
            cost = float(feats.get("cost", judge.get("cost", 0.0)))
            length = int(feats.get("length", len(seq)))
            score = _compute_score(dr, cost, cost_lambda)

            key = _seq_key(seq)
            prev = by_seq.get(key)
            if prev is None:
                by_seq[key] = {
                    "id": f"pkg_{ds_domain}_{abs(hash(key)) % (10**10)}",
                    "domain": ds_domain,
                    "sequence": seq,
                    "length": int(length),
                    "delta_risk": float(dr),
                    "cost": float(cost),
                    "score": float(score),
                    "created_at": ts_now,
                    "updated_at": ts_now,
                    "source": "debug_dataset",
                }
            else:
                # 若重复，则保留分数更高的一条，并更新统计时间
                try:
                    if float(score) > float(prev.get("score", -1e9)):
                        prev["delta_risk"] = float(dr)
                        prev["cost"] = float(cost)
                        prev["length"] = int(length)
                        prev["score"] = float(score)
                except Exception:
                    pass
                prev["updated_at"] = ts_now
        except Exception:
            # 忽略异常样本，继续处理
            continue

    # 重新整理排序
    items = list(by_seq.values())
    items.sort(key=lambda d: (
        -float(d.get("score", 0.0)),
        int(d.get("length", 0)),
        tuple(str(x) for x in d.get("sequence", []))
    ))

    text = json.dumps(items, ensure_ascii=False, indent=2)
    text = _normalize_lf(text)
    out_path.write_text(text, encoding="utf-8")
    return out_path


__all__ = [
    "ensure_store_dir",
    "ingest_from_debug_dataset",
]

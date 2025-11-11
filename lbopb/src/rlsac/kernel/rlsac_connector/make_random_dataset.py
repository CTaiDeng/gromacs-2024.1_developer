# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 only.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# --- 著作权独立性声明 (Copyright Independence Declaration) ---
# 本文件（“载荷”）是作者 (GaoZheng) 的原创著作物，其知识产权
# 独立于其运行平台 GROMACS（“宿主”）。
# 本文件的授权遵循上述 SPDX 标识，不受“宿主”许可证的管辖。
# 详情参见项目文档 "my_docs/project_docs/1762636780_🚩🚩gromacs-2024.1_developer项目的著作权设计策略：“宿主-载荷”与“双轨制”复合架构.md"。
# ------------------------------------------------------------------

from __future__ import annotations

import json
import random
import time as _t
from pathlib import Path
from typing import Any, Dict, List


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def main() -> None:
    # 输出到 out/out_connector/dataset_<ts>/debug_dataset.json
    root = _repo_root()
    out_root = root / "out" / "out_connector"
    ts = int(_t.time())
    run = out_root / f"dataset_{ts}"
    run.mkdir(parents=True, exist_ok=True)

    # 读取成对辞海，合并样本并随机抽样
    base = Path(__file__).resolve().parent / "monoid_packages"
    items: List[Dict[str, Any]] = []
    for f in sorted(base.glob("*_operator_packages.json")):
        arr = _read_json(f) or []
        for it in (arr or []):
            try:
                pair = it.get("pair") or {}
                a = str(pair.get("a")).lower()
                b = str(pair.get("b")).lower()
                seqs = it.get("sequences") or {}
                seq_a = list(seqs.get(a, []) or [])
                seq_b = list(seqs.get(b, []) or [])
                items.append(
                    {
                        "id": it.get("id"),
                        "pair": f"{a}_{b}",
                        "sequences": {a: seq_a, b: seq_b},
                        "length": int(len(seq_a) + len(seq_b)),
                        "created_at": int(it.get("created_at", ts)),
                        "updated_at": int(it.get("updated_at", ts)),
                        "source": "pair_monoid",
                    }
                )
            except Exception:
                continue
    random.shuffle(items)
    with (run / "debug_dataset.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(items, ensure_ascii=False, indent=2))
    print(f"[dataset] written: {run / 'debug_dataset.json'} items={len(items)}")


if __name__ == "__main__":
    main()

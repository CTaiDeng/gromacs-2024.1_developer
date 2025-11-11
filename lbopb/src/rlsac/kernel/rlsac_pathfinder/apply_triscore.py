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
from pathlib import Path
from typing import Any, Dict, List


def discretize(p: float) -> float:
    # round to nearest 0.5
    try:
        return round(float(p) * 2.0) / 2.0
    except Exception:
        return 0.0


def label_of(s3: float) -> str:
    if s3 >= 0.75:
        return "正确"
    if s3 >= 0.25:
        return "警告"
    return "错误"


def reward_of(lbl: str) -> float:
    return 1.0 if lbl == "正确" else (0.5 if lbl == "警告" else 0.0)


def main() -> None:
    # usage: python apply_triscore.py <run_dir> [infile] [outfile]
    import sys
    args = sys.argv[1:]
    if not args:
        print("usage: python apply_triscore.py <run_dir> [infile] [outfile]")
        return
    run_dir = Path(args[0]).resolve()
    in_path = run_dir / (args[1] if len(args) > 1 else "samples.output.json")
    out_path = run_dir / (args[2] if len(args) > 2 else "samples.output.tri.json")
    try:
        data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    except Exception:
        print(f"[apply_triscore] read failed: {in_path}")
        return
    if not isinstance(data, list):
        print("[apply_triscore] input not a list")
        return
    out: List[Dict[str, Any]] = []
    for it in data:
        try:
            s = float(it.get("score", 0.0))
            s3 = discretize(s)
            lbl = label_of(s3)
            rwd = reward_of(lbl)
            o = dict(it)
            o.update({"score_tri": s3, "label3": lbl, "reward": rwd})
            out.append(o)
        except Exception:
            out.append(dict(it))
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[apply_triscore] written: {out_path}")


if __name__ == "__main__":
    main()

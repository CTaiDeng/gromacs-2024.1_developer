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
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List


ANSI_RESET = "\x1b[0m"
ANSI_RED = "\x1b[31;1m"
ANSI_GREEN = "\x1b[32;1m"
ANSI_YELLOW = "\x1b[33;1m"
ANSI_CYAN = "\x1b[36;1m"


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


def _oneline(s: str, n: int = 200) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", " ").replace("\n", "\\n")
    return s[:n] + ("..." if len(s) > n else "")


def _hash_suffix(domain: str, sequence: List[str], ops_detailed: List[Dict[str, Any]] | None) -> str:
    payload = {
        "domain": str(domain).lower(),
        "sequence": list(sequence or []),
        "ops_detailed": list(ops_detailed or []),
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(s).hexdigest()  # full hex, no decimal


def _normalize_space_ref(v: Any) -> Any:
    # 将绝对路径统一为仓库相对（使用正斜杠）；若不是字符串/不存在则原样返回
    try:
        if isinstance(v, str) and v:
            p = Path(v)
            if p.is_absolute() and p.exists():
                rel = str(p.relative_to(_repo_root())).replace("\\", "/")
                return rel
            return v.replace("\\", "/")
    except Exception:
        pass
    return v


def rebuild_ids_for_file(pack_file: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    domain = pack_file.name.split("_")[0].lower()
    try:
        arr = json.loads(pack_file.read_text(encoding="utf-8"))
    except Exception:
        arr = []
    changed = 0
    total = 0
    out: List[Dict[str, Any]] = []
    for it in (arr or []):
        if not isinstance(it, dict):
            continue
        total += 1
        seq = list(it.get("sequence", []) or [])
        ops = it.get("ops_detailed") if isinstance(it.get("ops_detailed"), list) else []
        # 归一化 space 引用，避免绝对路径
        if "op_space_ref" in it:
            it["op_space_ref"] = _normalize_space_ref(it.get("op_space_ref"))
        suf = _hash_suffix(domain, seq, ops)
        new_id = f"pkg_{domain}_{suf}"
        old_id = str(it.get("id", ""))
        if old_id != new_id:
            changed += 1
            it["id"] = new_id
        out.append(it)

    if not dry_run:
        text = json.dumps(out, ensure_ascii=False, indent=2)
        text = text.replace("\r\n", "\n")
        pack_file.write_text(text, encoding="utf-8")
    return {"file": str(pack_file), "domain": domain, "total": total, "changed": changed}


def main() -> None:
    base = Path(__file__).resolve().parent
    files = [
        base / "pem_operator_packages.json",
        base / "pdem_operator_packages.json",
        base / "pktm_operator_packages.json",
        base / "pgom_operator_packages.json",
        base / "tem_operator_packages.json",
        base / "prm_operator_packages.json",
        base / "iem_operator_packages.json",
    ]
    args = sys.argv[1:]
    dry = ("--dry-run" in args) or ("-n" in args)
    # 可选：指定单文件
    only: List[Path] = []
    for a in args:
        if a.endswith("_operator_packages.json"):
            only.append(Path(a))
    if only:
        files = only

    print(f"{ANSI_CYAN}[rebuild] dry_run={dry} files={len(files)}{ANSI_RESET}")
    summary: List[Dict[str, Any]] = []
    for f in files:
        if not f.exists():
            continue
        r = rebuild_ids_for_file(f, dry_run=dry)
        summary.append(r)
        print(f"{ANSI_YELLOW}[rebuild] file={f.name} domain={r['domain']} total={r['total']} changed={r['changed']}{ANSI_RESET}")

    total = sum(int(x.get("total", 0)) for x in summary)
    changed = sum(int(x.get("changed", 0)) for x in summary)
    color = ANSI_GREEN if changed >= 0 else ANSI_RED
    print(f"{color}[rebuild] done: files={len(summary)} total={total} changed={changed}{ANSI_RESET}")


if __name__ == "__main__":
    main()

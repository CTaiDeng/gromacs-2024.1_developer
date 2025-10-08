#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Derivation guard: enforce fork/derivation compliance before commit.

Checks (fail -> block commit):
1) Required files exist: COPYING, CITATION.cff, README, NOTICE.
2) README top contains Chinese-communication note and non-official derivation disclaimer.
3) NOTICE mentions non-official derivation and LGPL.
4) Heuristics on staged changes: discourage new binary blobs; flag 'official' claims.
5) my_docs/project_docs 前缀唯一性：10位数字前缀视为文档ID，必须唯一（排除 kernel_reference）。

Optional LLM assist: if GEMINI/GOOGLEAI key set and DERIVATION_GUARD_USE_LLM=1,
summarize staged diff and ask model for PASS/FAIL with reasoning. Lack of key
does not fail the check.

Exit codes: 0 pass; non-zero fail.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> str:
    # Use bytes + robust decode to avoid locale decode errors on Windows (e.g., GBK).
    out = subprocess.check_output(cmd, cwd=ROOT)
    try:
        return out.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return out.decode(errors="ignore")
        except Exception:
            return ""


def staged_files() -> list[str]:
    out = run(["git", "diff", "--cached", "--name-only"])
    return [x for x in out.splitlines() if x.strip()]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(errors="ignore")
        except Exception:
            return ""


def check_required_files() -> list[str]:
    req = ["COPYING", "CITATION.cff", "README", "NOTICE"]
    missing = [p for p in req if not (ROOT / p).exists()]
    return [f"缺失关键文件：{','.join(missing)}"] if missing else []


def check_readme_headers() -> list[str]:
    readme = ROOT / "README"
    if not readme.exists():
        return ["README 不存在"]
    text = read_text(readme)
    head = "\n".join(text.splitlines()[:30])
    comm_ok = (
        any(k in head for k in ("简体中文", "中文", "沟通", "交流"))
        or re.search(r"Simplified\s+Chinese", head, re.I) is not None
    )
    deriv_ok = (
        any(k in head for k in ("非官方", "派生", "衍生"))
        or (
            re.search(r"(non[-\s]*official|unofficial)", head, re.I) is not None
            and re.search(r"(derivative|fork)", head, re.I) is not None
        )
    )
    if not comm_ok or not deriv_ok:
        return [
            "README 顶部缺少沟通语言或非官方派生声明（支持中文或等价英文表述）",
            "修复建议：在 README 顶部加入（中文或英文）：",
            " - Note: This repository communicates in Simplified Chinese by default.",
            " - Disclaimer: This is a non-official derivative/fork of GROMACS; see COPYING (LGPL-2.1). Unless otherwise noted, added/modified parts follow LGPL-2.1.",
        ]
    return []


def check_notice() -> list[str]:
    path = ROOT / "NOTICE"
    if not path.exists():
        return ["NOTICE 不存在"]
    text = read_text(path)
    ok = ("非官方" in text or "派生" in text or "衍生" in text) and ("LGPL" in text or "LGPL-2.1" in text)
    return [] if ok else ["NOTICE 中缺少派生/许可说明（应包含‘非官方/派生/衍生’与‘LGPL’ 关键词）"]


BIN_PAT = re.compile(r"\.(?:exe|dll|so|a|lib|pyd|o|obj|jar|zip|7z|tgz|gz|xz)$", re.I)


def check_staged_content() -> list[str]:
    files = staged_files()
    msgs: list[str] = []
    # 1) discourage new binary blobs
    for f in files:
        if BIN_PAT.search(f) and (ROOT / f).exists():
            msgs.append(f"暂存包含二进制/压缩制品：{f}（建议不要纳入仓库；若确因演示需要，请在 NOTICE 说明）")
    # 2) grep 'official' claims in changed text files
    try:
        diff = run(["git", "diff", "--cached", "--unified=0"])
        lines = diff.splitlines()
        cur_file = ""
        def should_check(path: str) -> bool:
            # Ignore our guard and policy files to avoid self-matching.
            if path.endswith("my_scripts/check_derivation_guard.py"):
                return False
            if Path(path).name.lower() in {"agents.md"}:
                return False
            return True

        flagged = False
        for ln in lines:
            if ln.startswith("+++ b/"):
                cur_file = ln[6:].strip()
                continue
            if not (ln.startswith("+") and not ln.startswith("+++")):
                continue
            if not should_check(cur_file):
                continue
            content = ln[1:]
            if re.search(r"official\s+gromacs", content, re.I):
                flagged = True
                break
        if flagged:
            msgs.append("检测到新增文本包含 ‘official gromacs’ 字样，请避免官方身份表述或在 README 顶部提供清晰非官方声明（已存在则可忽略）")
    except subprocess.CalledProcessError:
        pass
    return msgs


def check_project_docs_unique_prefix() -> list[str]:
    """Ensure timestamp prefixes under my_docs/project_docs are unique (excluding kernel_reference)."""
    proj = ROOT / "my_docs" / "project_docs"
    if not proj.exists():
        return []
    duplicates: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for p in sorted(proj.glob("*")):
        if not p.is_file():
            continue
        # Exclude kernel_reference subtree if any files appear (symlinked dir typically)
        try:
            rel = p.resolve().relative_to(ROOT.resolve()).as_posix()
        except Exception:
            rel = p.as_posix().replace("\\", "/")
        if rel.startswith("my_docs/project_docs/kernel_reference/"):
            continue
        name = p.name
        if "_" not in name:
            continue
        pref = name.split("_", 1)[0]
        if not (len(pref) == 10 and pref.isdigit()):
            continue
        if pref in seen and seen[pref] != name:
            duplicates.setdefault(pref, [seen[pref]]).append(name)
        else:
            seen[pref] = name
    if not duplicates:
        return []
    msgs = ["my_docs/project_docs 存在重复的10位前缀（文档ID冲突）："]
    for k, lst in duplicates.items():
        files = ", ".join(lst)
        first = seen.get(k, "?")
        msgs.append(f" - {k} -> {first}, {files}")
    msgs.append("修复建议：运行 my_scripts/align_my_documents.py（将自动对冲突项回退1秒直至唯一），或手工重命名并同步‘日期：’行。")
    return msgs


def llm_assist_if_enabled(summary: str) -> list[str]:
    if os.environ.get("DERIVATION_GUARD_USE_LLM", "0") != "1":
        return []
    # Soft dependency on API key
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLEAI_API_KEY")
    )
    if not api_key:
        return ["LLM 辅助检查：未检测到 API Key，已跳过（设置 DERIVATION_GUARD_USE_LLM=1 且配置 GEMINI_API_KEY 可启用）"]
    # Minimal stub: we do not hard-depend on external libs; provide informative note.
    # 可按需扩展为真实 HTTP 调用。
    return ["LLM 辅助检查：已启用占位（未集成外部依赖）；请后续根据需要扩展为实际远程校验。"]


def main() -> int:
    problems: list[str] = []
    problems += check_required_files()
    problems += check_readme_headers()
    problems += check_notice()
    problems += check_staged_content()
    problems += check_project_docs_unique_prefix()

    # Short summary for optional LLM
    try:
        names = staged_files()
        stat = run(["git", "diff", "--cached", "--shortstat"]).strip()
        summary = f"FILES: {names}\nSTAT: {stat}"
    except Exception:
        summary = ""
    problems += llm_assist_if_enabled(summary)

    if problems:
        sys.stderr.write("[Derivation Guard] 提交被阻止：\n")
        for m in problems:
            sys.stderr.write(f" - {m}\n")
        sys.stderr.write("\n可临时跳过：git commit --no-verify（不推荐）。\n")
        return 1
    print("[Derivation Guard] OK — 合规检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

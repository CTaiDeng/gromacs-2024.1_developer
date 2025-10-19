#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2025 GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.

"""
使用 Google Generative AI（Gemini，可选）对已暂存改动生成提交信息摘要。

环境变量
- GEMINI_API_KEY 或 GOOGLE_API_KEY：Google AI Studio API Key（可选）

注意
- 提交信息生成会忽略外部知识参考目录：my_docs/project_docs/kernel_reference（由 my_scripts/docs_whitelist.json 配置）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional, List, Tuple
import urllib.request
import urllib.error
import ssl


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "my_scripts" / "docs_whitelist.json"


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        return e.output.decode("utf-8", errors="replace")


def _load_doc_excludes() -> List[str]:
    try:
        if CFG_PATH.exists():
            data = json.loads(CFG_PATH.read_text(encoding="utf-8"))
            return [str(x).replace("\\", "/").rstrip("/") for x in data.get("doc_write_exclude", [])]
    except Exception:
        pass
    return []


EX_DOCS = _load_doc_excludes()
EXTRA_EXCLUDED_FILES = [
    # Do not include whitelist changes in commit message generation
    "my_scripts/docs_whitelist.json",
]


def _is_excluded_path(path_rel: str) -> bool:
    rp = path_rel.replace("\\", "/").lstrip("./")
    for e in EX_DOCS:
        if rp == e or rp.startswith(e + "/"):
            return True
    for f in EXTRA_EXCLUDED_FILES:
        if rp == f:
            return True
    return False


def collect_diff_filtered(max_patch_chars: int = 8000) -> tuple[str, str]:
    names_raw = run(["git", "diff", "--staged", "--name-only"]).strip().splitlines()
    names = [n for n in (x.strip() for x in names_raw) if n and not _is_excluded_path(n)]
    if names:
        stat = run(["git", "diff", "--staged", "--name-status", "--", *names]).strip()
        patch = run(["git", "diff", "--staged", "--unified=0", "--", *names]).strip()
    else:
        stat = ""
        patch = ""
    if len(patch) > max_patch_chars:
        patch = patch[: max_patch_chars - 1] + "\n…(truncated)"
    return stat, patch


PROMPT_TMPL = (
    "请阅读已暂存的 Git 变更并生成提交信息：\n"
    "- 第一行不超过 60 字，形如 `type: subject`，type ∈ [feat, fix, docs, chore, refactor, test, perf, build, ci]\n"
    "- 然后列出 1-3 条要点，每条一行，以 `- ` 开头\n\n"
    "变更清单（name-status）：\n{stat}\n\n"
    "差异片段（可能已截断）：\n{patch}\n"
)


def build_prompt(stat: str, patch: str, lang: str) -> str:
    lang = (lang or "zh").lower()
    # 始终要求以简体中文回答，并在提示前加入中文说明
    chs_instr = (
        "请使用简体中文输出提交信息，不要使用繁体字或英文。\n"
        "首行≤60个字符，格式 `type: subject`，type ∈ {feat, fix, docs, chore, refactor, test, perf, build, ci}；"
        "随后列出 1-3 条要点，每条以 `- ` 开头。\n\n"
    )
    return chs_instr + PROMPT_TMPL.format(stat=stat, patch=patch)


def _debug(msg: str) -> None:
    if os.environ.get("COMMIT_MSG_DEBUG", "0") == "1":
        try:
            sys.stderr.write(f"[commit-msg] {msg}\n")
        except Exception:
            pass


def _generate_with_gemini_rest(prompt: str) -> Optional[str]:
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLEAI_API_KEY")
    )
    if not api_key:
        _debug("REST fallback: no API key")
        return None
    # REST endpoint
    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        # Allow default SSL context
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
    except urllib.error.HTTPError as e:
        _debug(f"REST HTTPError: {e.code}")
        return None
    except Exception as e:
        _debug(f"REST exception: {e}")
        return None
    # Parse candidates
    try:
        cands = obj.get("candidates") or []
        texts: List[str] = []
        for c in cands:
            content = c.get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                t = part.get("text")
                if t:
                    texts.append(t)
        text = "\n".join([t for t in texts if t]).strip()
        return text or None
    except Exception as e:
        _debug(f"REST parse exception: {e}")
        return None


def generate_with_gemini(prompt: str) -> Optional[str]:
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLEAI_API_KEY")
    )
    if not api_key:
        _debug("no API key in GEMINI_API_KEY/GOOGLE_API_KEY/GOOGLEAI_API_KEY; fallback to 'update'")
        return None
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        _debug("google-generativeai package not installed; trying REST fallback")
        return _generate_with_gemini_rest(prompt)
    try:
        genai.configure(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text and hasattr(resp, "candidates") and resp.candidates:
            parts = []
            for c in resp.candidates:
                try:
                    parts.append(c.content.parts[0].text)
                except Exception:
                    continue
            text = "\n".join([p for p in parts if p])
        if text:
            return text.strip()
        _debug("empty response text from SDK; trying REST fallback")
        return _generate_with_gemini_rest(prompt)
    except Exception as e:
        _debug(f"SDK exception: {e}; trying REST fallback")
        return _generate_with_gemini_rest(prompt)


# ------------------------------
# Offline fallback (no network)
# ------------------------------

CODE_EXTS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx",
    ".cu", ".cuh", ".py", ".rs", ".go", ".js", ".ts", ".java",
    ".cs", ".m", ".mm", ".kt", ".swift", ".sh", ".ps1",
}
DOC_EXTS = {".md", ".rst", ".adoc", ".txt"}
TEST_HINTS = {"test", "tests", "_test.", "-test.", "spec"}


def _collect_names_filtered() -> List[str]:
    names_raw = run(["git", "diff", "--staged", "--name-only"]).strip().splitlines()
    return [n for n in (x.strip() for x in names_raw) if n and not _is_excluded_path(n)]


def _collect_numstat(names: List[str]) -> List[Tuple[int, int, str]]:
    if not names:
        return []
    raw = run(["git", "diff", "--staged", "--numstat", "--", *names]).strip()
    rows: List[Tuple[int, int, str]] = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                add = int(parts[0]) if parts[0].isdigit() else 0
                dele = int(parts[1]) if parts[1].isdigit() else 0
            except Exception:
                add, dele = 0, 0
            path = parts[2]
            if not _is_excluded_path(path):
                rows.append((add, dele, path))
    return rows


def _infer_type(names: List[str], stat: str, patch: str, numstat: List[Tuple[int, int, str]]) -> str:
    lowered = [n.lower() for n in names]
    has_doc = any(Path(n).suffix.lower() in DOC_EXTS or "/docs/" in n.replace("\\", "/") for n in lowered)
    has_code = any(Path(n).suffix.lower() in CODE_EXTS for n in lowered)
    has_test = any(any(h in n for h in TEST_HINTS) for n in lowered)
    added_code = False
    for s in stat.splitlines():
        s = s.strip()
        if not s:
            continue
        if s.startswith("A\t"):
            n = s.split("\t", 1)[1]
            if Path(n).suffix.lower() in CODE_EXTS:
                added_code = True
                break
    if has_doc and not has_code:
        return "docs"
    if added_code:
        return "feat"
    patch_l = patch.lower()
    if "fix" in patch_l or "bug" in patch_l:
        return "fix"
    if has_test and has_code:
        return "test"
    if has_code:
        return "refactor"
    return "chore"


def _shorten(text: str, limit: int = 60) -> str:
    t = text.strip()
    return t if len(t) <= limit else t[: limit - 1] + "…"


def generate_offline_summary(stat: str, patch: str) -> Optional[str]:
    names = _collect_names_filtered()
    if not names:
        return None
    numstat = _collect_numstat(names)
    adds = sum(a for a, _, _ in numstat)
    dels = sum(d for _, d, _ in numstat)
    changed = len(names)
    t = _infer_type(names, stat, patch, numstat)

    # subject in Chinese
    if t == "docs" and changed == 1:
        subject = f"文档更新：{Path(names[0]).name}"
    elif t == "docs":
        subject = f"文档更新（{changed} 个文件）"
    elif t == "feat":
        subject = f"新增/扩展：{changed} 个文件"
    elif t == "fix":
        subject = f"修复与调整（{changed} 个文件）"
    elif t == "test":
        subject = f"测试用例更新（{changed} 个文件）"
    elif t == "refactor":
        subject = f"代码重构与清理（{changed} 个文件）"
    else:
        subject = f"杂项维护（{changed} 个文件）"

    header = f"{t}: {_shorten(subject)}"

    # top files by churn
    top = sorted(numstat, key=lambda r: (r[0] + r[1]), reverse=True)[:3]
    top_paths = [p for _, _, p in top]

    bullets: List[str] = []
    bullets.append(f"- 变更统计：+{adds} / -{dels} 行，涉及 {changed} 个文件")
    if top_paths:
        bullets.append("- 主要文件：" + ", ".join(Path(p).as_posix() for p in top_paths))
    if any("my_scripts/" in p.replace("\\", "/") for p in names):
        bullets.append("- 脚本调整：my_scripts 下的工具更新")
    if any(p.lower().endswith(".md") for p in names):
        bullets.append("- 文档同步：Markdown 说明更新")

    return "\n".join([header] + bullets)


def is_comment_only_patch(patch: str) -> bool:
    if not patch.strip():
        return False
    has_change = False
    for raw in patch.splitlines():
        if not raw:
            continue
        if raw.startswith(('diff --git', 'index ', '--- ', '+++ ', '@@')):
            continue
        if raw[0] not in ['+', '-']:
            continue
        if raw.startswith('+++') or raw.startswith('---'):
            continue
        has_change = True
        line = raw[1:].lstrip()
        if line == '':
            continue
        comment_prefixes = (
            '#', '//', '/*', '*', '*/', '--', ';', "'''", '"""', 'REM ', 'rem '
        )
        if line.startswith(comment_prefixes) or line.startswith('!'):
            continue
        return False
    return has_change


def main() -> int:
    stat, patch = collect_diff_filtered()
    # Always try to generate, even for comment-only patches
    if not stat and not patch:
        print('update')
        return 0
    lang = os.environ.get("COMMIT_MSG_LANG", "zh").lower()
    prompt = build_prompt(stat, patch, lang)
    text = generate_with_gemini(prompt)
    if not text:
        offline = generate_offline_summary(stat, patch)
        if offline and offline.strip():
            print(offline.strip())
            return 0
        print('update')
        return 0
    print(text.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())

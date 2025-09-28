#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 Google Generative AI（Gemini）基于已暂存改动生成提交信息摘要。

环境变量：
- GEMINI_API_KEY 或 GOOGLE_API_KEY：Google AI Studio API 密钥

依赖：
- python -m pip install google-generativeai

用法：
- python scripts/gen_commit_msg_googleai.py            # 针对已暂存改动生成提交信息
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        return e.output.decode("utf-8", errors="replace")


def collect_diff(max_patch_chars: int = 8000) -> tuple[str, str]:
    stat = run(["git", "diff", "--staged", "--name-status"]).strip()
    patch = run(["git", "diff", "--staged", "--unified=0"]).strip()
    if len(patch) > max_patch_chars:
        patch = patch[: max_patch_chars - 1] + "\n…(truncated)"
    return stat, patch


PROMPT_TMPL = (
    "请根据以下 Git 已暂存改动，生成简洁的提交信息：\n"
    "- 第一行不超过 60 字，风格建议 `type: subject`，type ∈ [feat, fix, docs, chore, refactor, test, perf, build, ci]；\n"
    "- 如有必要，再给出 1–3 条要点，每条一行，以 `- ` 开头；\n"
    "- 保持客观、具体，避免冗余与口语化。\n\n"
    "【变更文件】\n{stat}\n\n"
    "【差异片段（精简）】\n{patch}\n"
)


def build_prompt(stat: str, patch: str, lang: str) -> str:
    lang = (lang or "zh").lower()
    if lang == "en":
        return (
            "Please read the staged Git changes and produce a concise commit message.\n"
            "- First line <= 60 chars in `type: subject`, type ∈ {{feat, fix, docs, chore, refactor, test, perf, build, ci}}.\n"
            "- Then list 1–3 bullet points, each one line starting with `- `.\n"
            "- Output in English only.\n\n"
            "Name-status list:\n{stat}\n\n"
            "Diff patch (may be truncated):\n{patch}\n"
        ).format(stat=stat, patch=patch)
    # default zh
    return (
        "请根据以下 Git 已暂存改动，生成简洁的提交信息。\n"
        "- 第一行不超过 60 字，形如 `type: subject`，type ∈ {{feat, fix, docs, chore, refactor, test, perf, build, ci}}。\n"
        "- 其后最多列出 1–3 条要点，每条一行，以 `- ` 开头。\n"
        "- 必须仅用简体中文输出，不要夹杂英文。\n\n"
        "变更列表（name-status）：\n{stat}\n\n"
        "差异补丁（可能已截断）：\n{patch}\n"
    ).format(stat=stat, patch=patch)


def generate_with_gemini(prompt: str) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return None
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
        return None
    except Exception:
        return None


def is_comment_only_patch(patch: str) -> bool:
    """Heuristically detect if changes are comment/whitespace-only.
    Looks at added/removed lines and checks for common comment markers.
    """
    if not patch.strip():
        return False
    has_change = False
    for raw in patch.splitlines():
        if not raw:
            continue
        # skip headers
        if raw.startswith(('diff --git', 'index ', '--- ', '+++ ', '@@')):
            continue
        if raw[0] not in ['+', '-']:
            continue
        if raw.startswith('+++') or raw.startswith('---'):
            continue
        has_change = True
        line = raw[1:].lstrip()
        if line == '':
            # pure whitespace add/remove
            continue
        # common comment markers across languages
        comment_prefixes = (
            '#', '//', '/*', '*', '*/', '--', ';', "'''", '"""', 'REM ', 'rem '
        )
        # shebang counts as comment
        if line.startswith(comment_prefixes) or line.startswith('!'):
            continue
        # treat only closing braces or semicolons as non-functional? keep simple: count as code
        return False
    return has_change


def main() -> int:
    stat, patch = collect_diff()
    # 注释（仅修改注释/空白）提交：不调用 AI，直接使用 update
    if is_comment_only_patch(patch):
        print('update')
        return 0
    # 无变更：直接 update
    if not stat and not patch:
        print('update')
        return 0
    lang = os.environ.get("COMMIT_MSG_LANG", "zh").lower()
    prompt = build_prompt(stat, patch, lang)
    text = generate_with_gemini(prompt)
    # 无法访问 API 或生成失败：按要求使用 update
    if not text:
        print('update')
        return 0
    print(text.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())

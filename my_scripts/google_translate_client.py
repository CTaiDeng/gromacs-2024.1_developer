# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
基于 Gemini 的翻译最小封装（复用 my_scripts.gemini_client）。

环境变量：
- GEMINI_API_KEY 或 GOOGLE_API_KEY（二选一）；
- 可选：GEMINI_MODEL（未设置时走默认）。

用法示例：
    from my_scripts.google_translate_client import translate_text, build_markdown_with_translation
    zh = translate_text("Hello world", target="zh-CN")
    md = build_markdown_with_translation(original_json_text, zh)
"""

from __future__ import annotations

import json
import os
from typing import Optional

try:
    from my_scripts.gemini_client import generate_gemini_content as _gemini_generate
except Exception as _e:  # 回退占位（避免导入期错误）
    _gemini_generate = None  # type: ignore


def translate_text(
    text: str,
    *,
    target: str = "zh-CN",
    source: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: float = 20.0,
) -> str:
    """
    使用 Gemini 将 `text` 翻译为 `target` 语言（仅返回译文）。
    - 保持 Markdown/JSON/代码块等格式，尽量不改变结构与标点。
    - 失败时返回以 [Translate Error] 开头的错误字符串。
    """
    if _gemini_generate is None:
        return "[Translate Error] my_scripts.gemini_client not available"

    # 依据目标语言组装提示词
    if (target or "").lower().startswith("zh"):
        sys_prompt = (
            "请将以下内容翻译为简体中文，仅输出译文，不添加任何解释。"
            "保持原有 Markdown/JSON/代码块与格式、标点、数字、术语一致。\n\n"
            "【待翻译内容】\n"
        )
    else:
        sys_prompt = (
            f"Please translate the following text into {target}. "
            "Output translation only, no extra explanations. "
            "Keep original Markdown/JSON/code blocks and formatting.\n\n"
            "[TEXT]\n"
        )

    prompt = sys_prompt + str(text)
    # 使用集中封装：env 取 GEMINI_MODEL / GEMINI_API_KEY；这里透传 api_key 与 timeout
    try:
        out = _gemini_generate(prompt, api_key=api_key, model=None, timeout_sec=timeout_sec)  # type: ignore
    except TypeError:
        # 兼容较老签名（无 timeout_sec 参数）
        out = _gemini_generate(prompt, api_key=api_key, model=None)  # type: ignore
    except Exception as e:
        return f"[Translate Error] {e}"

    return out


def build_markdown_with_translation(original_json_text: str, translated_text: str, *, title: str = "LLM 评价（英文/中文）") -> str:
    """\r
    生成包含“原始 JSON（英文）”与“中文翻译（Gemini）”两段的 Markdown 字符串。
    调用方负责写入文件（UTF-8 + CRLF）。
    """
    lines: list[str] = []
    lines.append(f"# {title}\r\n\r\n")
    lines.append("## 英文原始结果 (JSON)\r\n\r\n")
    # 直接包裹为代码块，避免渲染歧义
    lines.append("```json\r\n")
    lines.append(original_json_text.rstrip("\r\n") + "\r\n")
    lines.append("```\r\n\r\n")
    lines.append("## 中文翻译 (Gemini)\r\n\r\n")
    lines.append(translated_text.rstrip("\r\n") + "\r\n")
    return "".join(lines)


__all__ = [
    "translate_text",
    "build_markdown_with_translation",
]

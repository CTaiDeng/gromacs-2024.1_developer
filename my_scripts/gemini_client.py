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
通用 Gemini 客户端（REST 直连，无外部依赖）。
- 默认从环境变量 `GEMINI_API_KEY`（或 `GOOGLE_API_KEY` 兜底）读取密钥；
- 模型优先使用环境变量 `GEMINI_MODEL`，未设置时回落到默认；
- 提供简单的一次性文本生成接口 `generate_gemini_content(text)`；
- 失败时返回错误字符串（前缀含 [Gemini Error]/[Gemini HTTPError]）。

注意：本模块不自动写入文件；调用方如需落盘，应以 encoding='utf-8' 且 LF 行尾写入。
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional
import time as _t

# 默认模型（当未提供且未设置 GEMINI_MODEL 时生效）
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _extract_texts_from_response(obj: dict) -> str:
    """尽量鲁棒地从 Gemini 响应 JSON 中提取文本；否则返回原始 JSON 串。"""
    try:
        cands = obj.get("candidates", [])
        if not cands:
            return json.dumps(obj, ensure_ascii=False)
        parts = cands[0].get("content", {}).get("parts", [])
        texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("text")]
        return "\n".join(texts) if texts else json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps(obj, ensure_ascii=False)


def generate_gemini_content(
    prompt_text: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    endpoint_template: str = DEFAULT_ENDPOINT_TMPL,
    timeout_sec: float = 180.0,
) -> str:
    """
    调用 Gemini 进行一次性文本生成。

    参数：
    - prompt_text: 纯文本提示词；内部将封装为 contents.parts[].text。
    - api_key: 显式密钥；为 None 时从环境变量 GEMINI_API_KEY/GOOGLE_API_KEY 读取。
    - model: 使用的 Gemini 模型；None 时从 GEMINI_MODEL 读取，未设置则回落默认。
    - endpoint_template: 端点模板；通常无需修改。
    - timeout_sec: 请求超时时间。

    返回：
    - 成功：提取的文本（多段合并 '\n'）。
    - 失败：以 "[Gemini Error]" 或 "[Gemini HTTPError]" 开头的错误字符串。
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    verbose = bool(os.environ.get("LBOPB_GEMINI_DEBUG") or os.environ.get("GEMINI_DEBUG"))
    if not key:
        msg = "[Gemini Error] missing GEMINI_API_KEY/GOOGLE_API_KEY in environment"
        if verbose:
            print(f"[Gemini] {msg}")
        return msg

    # 解析模型：env 优先，其次参数默认
    model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    url = endpoint_template.format(model=model)
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "x-goog-api-key": key,
    }
    body = {"contents": [{"parts": [{"text": prompt_text}]}]}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    if verbose:
        try:
            print(
                f"[Gemini] prepare: model={model} url={url} prompt_len={len(prompt_text)} bytes={len(data)} t={int(_t.time())}"
            )
        except Exception:
            print("[Gemini] prepare: model/url/prompt info unavailable")
    try:
        # 允许通过环境变量覆盖超时（优先级：参数 < 环境变量）
        eff_timeout = timeout_sec
        try:
            _env_to = os.environ.get("LBOPB_GEMINI_TIMEOUT_SEC") or os.environ.get("GEMINI_TIMEOUT_SEC")
            if _env_to:
                eff_timeout = float(_env_to)
        except Exception:
            pass
        if verbose:
            print(f"[Gemini] sending request... timeout={eff_timeout}s")
        with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
            if verbose:
                try:
                    print(f"[Gemini] response: code={resp.getcode()} t={int(_t.time())}")
                except Exception:
                    print("[Gemini] response: received")
            raw = resp.read().decode("utf-8", errors="replace")
            jo = json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            err_text = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_text = str(e)
        msg = f"[Gemini HTTPError] {e.code}: {err_text}"
        if verbose:
            print(msg)
        return msg
    except Exception as e:
        msg = f"[Gemini Error] {e}"
        if verbose:
            print(msg)
        return msg
    res = _extract_texts_from_response(jo)
    if verbose:
        try:
            _h = res[:240] + ("..." if len(res) > 240 else "")
            print(f"[Gemini] parsed: text_len={len(res)} head={_h}")
        except Exception:
            print("[Gemini] parsed: <unprintable>")
    return res


__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "generate_gemini_content",
]




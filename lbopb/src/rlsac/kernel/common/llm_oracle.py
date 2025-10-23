# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from typing import Optional


def call_llm(prompt: str, *, provider: str = "gemini", model: str | None = None, api_key: str | None = None) -> \
Optional[str]:
    """占位封装：调用外部 LLM（如 Gemini），返回文本判定结果。

    出于可移植与安全考虑，本函数不内置具体 HTTP 调用逻辑；
    项目集成时可按运行环境在此处接入实际 API 调用并返回字符串结果。
    返回 None 表示未调用或调用失败。
    """
    return None

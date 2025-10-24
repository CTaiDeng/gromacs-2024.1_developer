# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from typing import Optional, Any, Dict, List


def call_llm(prompt: str, *, provider: str = "gemini", model: str | None = None, api_key: str | None = None) -> \
Optional[str]:
    """占位封装：调用外部 LLM（如 Gemini），返回文本判定结果。

    出于可移植与安全考虑，本函数不内置具体 HTTP 调用逻辑；
    项目集成时可按运行环境在此处接入实际 API 调用并返回字符串结果。
    返回 None 表示未调用或调用失败。
    """
    try:
        if provider.lower() == "gemini":
            import importlib
            cli = importlib.import_module("my_scripts.gemini_client")
            for fname in ("ask", "generate", "chat", "gemini_text", "query", "run"):
                fn = getattr(cli, fname, None)
                if callable(fn):
                    try:
                        res: Any = fn(prompt)
                        if isinstance(res, str):
                            return res
                        if hasattr(res, "text"):
                            return str(res.text)
                    except Exception:
                        continue
    except Exception:
        pass
    return None


# ---- Axiom prompt templates ----

DOC_MAP: Dict[str, str] = {
    "pem": "my_docs/project_docs/1761062400_病理演化幺半群 (PEM) 公理系统.md",
    "prm": "my_docs/project_docs/1761062401_生理调控幺半群 (PRM) 公理系统.md",
    "tem": "my_docs/project_docs/1761062403_毒理学效应幺半群 (TEM) 公理系统.md",
    "pktm": "my_docs/project_docs/1761062404_药代转运幺半群 (PKTM) 公理系统.md",
    "pgom": "my_docs/project_docs/1761062405_药理基因组幺半群 (PGOM) 公理系统.md",
    "pdem": "my_docs/project_docs/1761062406_药效效应幺半群 (PDEM) 公理系统.md",
    "iem": "my_docs/project_docs/1761062407_免疫效应幺半群 (IEM) 公理系统.md",
}

# 在联络一致性中使用的“对偶域”检查对
PAIRWISE: List[tuple[str, str]] = [
    ("pdem", "pktm"),
    ("pgom", "pem"),
    ("tem", "pktm"),
    ("prm", "pem"),
    ("iem", "pem"),
]


def build_pathfinder_prompt(domain: str, sequence: List[str]) -> str:
    d = (domain or "").lower()
    doc = DOC_MAP.get(d, "<axiom-doc-not-found>")
    return (
        "你是一名严格的形式系统审查器。\n"
        f"幺半群域: {d}\n"
        f"公理文档: {doc}\n"
        "任务: 判断以下基本算子序列(算子包)是否严格符合该域的公理系统(所有必要约束: 方向性/不可交换/序次/阈值/停机等)。\n"
        f"算子序列: {sequence}\n"
        "要求: 只返回单个字符 '1' 或 '0'。1 表示符合公理系统, 0 表示不符合。不得输出其他字符或解释。\n"
    )


def build_connector_prompt(conn: Dict[str, List[str]]) -> str:
    """两两整合注入模板: 针对对偶域逐一判定联络是否成立, 最后整体判定。"""
    parts = []
    for a, b in PAIRWISE:
        da = DOC_MAP.get(a, "<doc-na>")
        db = DOC_MAP.get(b, "<doc-na>")
        sa = conn.get(a, [])
        sb = conn.get(b, [])
        parts.append({
            "pair": f"{a}<->{b}",
            "doc_a": da,
            "doc_b": db,
            "seq_a": sa,
            "seq_b": sb,
        })
    header = (
        "你是一名严格的形式系统审查器, 要判断跨七域的联络候选体是否成立。\n"
        "请基于各域公理系统, 对以下对偶域进行逐一一致性判定(因果/量纲/时序/阈值), 最后给出整体判定: \n"
    )
    body = "\n".join(
        [
            (
                f"对偶域 {p['pair']}\n"
                f"文档A: {p['doc_a']}\n文档B: {p['doc_b']}\n"
                f"序列A: {p['seq_a']}\n序列B: {p['seq_b']}\n"
            )
            for p in parts
        ]
    )
    tail = (
        "请严格依公理判定整体联络是否成立: 若所有对偶域均一致且不违背任何域公理, 返回 '1'; 否则返回 '0'。\n"
        "只返回单个字符 '1' 或 '0'。不得输出其他文本。\n"
    )
    return header + body + "\n" + tail

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

from typing import Optional, Any, Dict, List
from pathlib import Path
import os
import time as _t
import json


def call_llm(prompt: str, *, provider: str = "gemini", model: str | None = None, api_key: str | None = None) -> Optional[str]:
    """轻量的 LLM 调用封装（默认使用外部 my_scripts.gemini_client）。

    - 返回值为字符串时表示成功；返回 None 表示失败或未调用成功。
    - 具体 HTTP 逻辑在外部客户端中实现，这里仅做动态导入与容错。
    """
    try:
        dbg = bool(os.environ.get("LBOPB_GEMINI_DEBUG") or os.environ.get("GEMINI_DEBUG"))
        if provider.lower() == "gemini":
            import importlib
            if dbg:
                print(f"[LLM] call_llm enter: provider=gemini, t={int(_t.time())}")
            cli = importlib.import_module("my_scripts.gemini_client")
            try:
                if model:
                    os.environ["LBOPB_GEMINI_MODEL"] = str(model)
                    os.environ.setdefault("GEMINI_MODEL", str(model))
            except Exception:
                pass
            for fname in ("ask", "generate", "chat", "gemini_text", "query", "run", "generate_gemini_content"):
                fn = getattr(cli, fname, None)
                if callable(fn):
                    if dbg:
                        print(f"[LLM] call_llm try func={fname}")
                    try:
                        try:
                            res: Any = fn(prompt, model=model)  # type: ignore[call-arg]
                        except TypeError:
                            res = fn(prompt)
                        if isinstance(res, str):
                            if dbg:
                                _h = res[:120] + ("..." if len(res) > 120 else "")
                                print(f"[LLM] call_llm ok via {fname}, len={len(res)}, head={_h}")
                            return res
                        if hasattr(res, "text"):
                            s = str(getattr(res, "text"))
                            if dbg:
                                _h = s[:120] + ("..." if len(s) > 120 else "")
                                print(f"[LLM] call_llm ok via {fname}.text, len={len(s)}, head={_h}")
                            return s
                    except Exception as e:
                        if dbg:
                            print(f"[LLM] call_llm func={fname} exception: {e}")
                        continue
    except Exception as e:
        if bool(os.environ.get("LBOPB_GEMINI_DEBUG") or os.environ.get("GEMINI_DEBUG")):
            print(f"[LLM] call_llm outer exception: {e}")
        pass
    return None


# ---- 公理文档路径映射（默认兜底；优先用同目录 JSON 覆盖） ----

DOC_MAP: Dict[str, str] = {
    "pem": "my_docs/project_docs/1761062400_病理演化幺半群 (PEM) 公理系统.md",
    "prm": "my_docs/project_docs/1761062401_生理调控幺半群 (PRM) 公理系统.md",
    "tem": "my_docs/project_docs/1761062402_毒理学效应幺半群 (TEM) 公理系统.md",
    "pktm": "my_docs/project_docs/1761062403_药代转运幺半群 (PKTM) 公理系统.md",
    "pgom": "my_docs/project_docs/1761062404_药理基因组幺半群 (PGOM) 公理系统.md",
    "pdem": "my_docs/project_docs/1761062405_药效效应幺半群 (PDEM) 公理系统.md",
    "iem": "my_docs/project_docs/1761062406_免疫效应幺半群 (IEM) 公理系统.md",
}


# 在联络一致性中使用的“对偶域”检查对
PAIRWISE: List[tuple[str, str]] = [
    ("pdem", "pktm"),
    ("pgom", "pem"),
    ("tem", "pktm"),
    ("prm", "pem"),
    ("iem", "pem"),
]


def _load_doc_map() -> Dict[str, str]:
    """从同目录 axiom_docs.json 加载覆盖映射；键统一小写。"""
    m = dict(DOC_MAP)
    try:
        here = Path(__file__).resolve().parent
        cfg = here / "axiom_docs.json"
        if cfg.exists():
            data = json.loads(cfg.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    try:
                        m[str(k).strip().lower()] = str(v)
                    except Exception:
                        continue
    except Exception:
        pass
    return m


def build_pathfinder_prompt(domain: str, sequence: List[str],
                            ops_detailed: Optional[List[Dict[str, Any]]] = None,
                            extra: Optional[Dict[str, Any]] = None) -> str:
    d = (domain or "").lower()
    doc_rel = _load_doc_map().get(d, "<axiom-doc-not-found>")
    # 读取公理文档
    doc_text = "<axiom-doc-content-not-found>"
    try:
        repo_root = Path(__file__).resolve().parents[5]
        doc_path = repo_root / doc_rel
        if doc_path.exists():
            doc_text = doc_path.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"Axiom document not found for domain '{d}': {doc_path}")
    except Exception:
        raise

    # 可选：附带参数/网格索引供 LLM 参考
    param_block = ""
    if isinstance(ops_detailed, list) and len(ops_detailed) > 0:
        try:
            import json as _json
            simplified: List[Dict[str, Any]] = []
            for st in ops_detailed:
                row: Dict[str, Any] = {"name": st.get("name"), "params": st.get("params")}
                if st.get("grid_index") is not None:
                    row["grid_index"] = st.get("grid_index")
                simplified.append(row)
            meta: Dict[str, Any] = {}
            if isinstance(extra, dict):
                for k in ("op_space_id", "op_space_ref"):
                    if k in extra:
                        meta[k] = extra[k]
            param_block = (
                "\n（以下为可选补充）:\n"
                + (f"meta={_json.dumps(meta, ensure_ascii=False)}\n" if meta else "")
                + f"steps={_json.dumps(simplified, ensure_ascii=False)}\n"
            )
        except Exception:
            param_block = ""

    return (
        "你是一名严格的形式系统审查器。\n"
        f"幺半群域: {d}\n"
        "以下是该域的公理文档：\n"
        "-----BEGIN AXIOM DOC-----\n"
        f"{doc_text}\n"
        "-----END AXIOM DOC-----\n"
        "任务: 严格依据以上公理文档内容, 判断以下基本算子序列(算子包)是否严格符合该域的公理系统(所有必要约束: 方向性/不可交换/序次/阈值/停机等)。\n"
        f"算子序列: {sequence}\n"
        f"{param_block}"
        "要求: 仅返回字符 '1' 或 '0'。1 表示符合公理系统, 0 表示不符合。\n"
    )


def build_connector_prompt(conn: Dict[str, List[str]]) -> str:
    """联络一致性检验提示词：跨域对偶检查。"""
    parts = []
    doc_map = _load_doc_map()
    for a, b in PAIRWISE:
        da = doc_map.get(a, "<doc-na>")
        db = doc_map.get(b, "<doc-na>")
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
        "你是一名严格的形式系统审查器，需判断跨域对偶的连通性/一致性。\n"
        "请在各自公理系统下，对下列对偶域序列逐一检查（方向/次序/时序/阈值）。\n"
    )
    body = "\n".join([
        (
            f"对偶对 {p['pair']}\n"
            f"文档A: {p['doc_a']}\n文档B: {p['doc_b']}\n"
            f"序列A: {p['seq_a']}\n序列B: {p['seq_b']}\n"
        ) for p in parts
    ])
    tail = (
        "严格判断对偶对是否一致：若所有对偶对均不违反任何公理，返回 '1'；否则返回 '0'。\n"
        "仅返回字符 '1' 或 '0'。\n"
    )
    return header + body + "\n" + tail

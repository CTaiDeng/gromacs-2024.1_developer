# 使用 axiom_docs.json 作为权威映射，内置 DOC_MAP 为兜底\r\n
def _load_doc_map() -> Dict[str, str]:
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

PAIRWISE: List[tuple[str, str]] = [
    ("pdem", "pktm"),
    ("pgom", "pem"),
    ("tem", "pktm"),
    ("prm", "pem"),
    ("iem", "pem"),
]


def build_pathfinder_prompt(domain: str, sequence: List[str],
                            ops_detailed: Optional[List[Dict[str, Any]]] = None,
                            extra: Optional[Dict[str, Any]] = None) -> str:
    d = (domain or "").lower()
    doc_rel = _load_doc_map().get(d, "<axiom-doc-not-found>")
    # 读取并注入公理文档内容（以仓库根为基准解析路径）。不在提示词中暴露具体路径或说明性标注。
    doc_text = "<axiom-doc-content-not-found>"
    try:
        from pathlib import Path as _Path
        repo_root = _Path(__file__).resolve().parents[5]
        doc_path = repo_root / doc_rel
        if doc_path.exists():
            doc_text = doc_path.read_text(encoding="utf-8")
        else:
            # 未找到公理文档，抛出致命错误以便上层退出
            raise FileNotFoundError(f"Axiom document not found for domain '{d}': {doc_path}")
    except Exception:
        # 显式升级为错误：确保调用方感知并中止
        raise

    # 可选参数化动作描述（如提供）。为减少长度，仅包含 name + params (+ grid_index 可选)。
    param_block = ""
    if isinstance(ops_detailed, list) and len(ops_detailed) > 0:
        try:
            import json as _json
            simplified: List[Dict[str, Any]] = []
            for st in ops_detailed:
                simplified.append({
                    "name": st.get("name"),
                    "params": st.get("params"),
                    **({"grid_index": st.get("grid_index")} if st.get("grid_index") is not None else {})
                })
            meta = {}
            if isinstance(extra, dict):
                for k in ("op_space_id", "op_space_ref"):
                    if k in extra:
                        meta[k] = extra[k]
            param_block = (
                "\n参数化动作（如提供）:\n"
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


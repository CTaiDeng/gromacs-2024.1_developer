from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Dict

from .ms_tts import Segment


_ZERO_WIDTH_PATTERN = re.compile("[\u200B\u200C\u200D\u2060\uFEFF]")


def normalize_text(text: str, *, form: str = "NFC", strip_zero_width: bool = True, normalize_newlines: bool = True) -> str:
    """
    文本规范化：
    - Unicode 规范化（默认 NFC；亦可传 NFKC）
    - 去除零宽字符（ZWSP/ZWNJ/ZWJ/WWJ/BOM）
    - 统一换行为 LF
    """
    t = text or ""
    if normalize_newlines:
        t = t.replace("\r\n", "\n").replace("\r", "\n")
    if strip_zero_width:
        t = _ZERO_WIDTH_PATTERN.sub("", t)
    try:
        t = unicodedata.normalize(form, t)
    except Exception:
        t = unicodedata.normalize("NFC", t)
    return t


def validate_segments(segments: List[Segment], source_texts: Iterable[str], *, normalize: bool = True) -> Dict:
    expected_list = [str(s) for s in source_texts]
    mismatches: List[Dict] = []  # type: ignore[type-arg]
    for i, (exp, seg) in enumerate(zip(expected_list, segments)):
        a = normalize_text(exp) if normalize else exp
        b = normalize_text(seg.text) if normalize else seg.text
        if a != b:
            mismatches.append({"index": i, "expected": a, "actual": b})
    return {
        "ok": len(mismatches) == 0 and len(expected_list) == len(segments),
        "total": min(len(expected_list), len(segments)),
        "mismatches": mismatches,
        "len_expected": len(expected_list),
        "len_segments": len(segments),
    }


def correct_segments_text(segments: List[Segment], source_texts: Iterable[str], *, normalize: bool = True) -> List[Segment]:
    expected_list = [str(s) for s in source_texts]
    n = min(len(expected_list), len(segments))
    out: List[Segment] = []
    for i in range(n):
        txt = normalize_text(expected_list[i]) if normalize else expected_list[i]
        seg = segments[i]
        out.append(Segment(text=txt, start_ms=seg.start_ms, end_ms=seg.end_ms))
    for i in range(n, len(segments)):
        out.append(segments[i])
    return out


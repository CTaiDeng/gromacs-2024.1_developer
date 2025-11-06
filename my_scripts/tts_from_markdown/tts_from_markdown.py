from __future__ import annotations

"""
从 Markdown 生成旁白音频（基于 Azure TTS）。

参考：
- test/video_production/video_production_3_generate_ssml_from_sample.py
- test/video_production/video_production_4_tts_from_ssml.py

功能：
- 读取指定 Markdown（默认：docs/project_docs/1759075200_*.md），解析为语音段落（标题/段落/列表项），
- 使用 Azure 语音合成（zh-CN-YunyeNeural）生成 WAV 音频，
- 可选生成 SRT 字幕（基于书签分段回收时间戳）。

环境变量：
- AZURE_SPEECH_KEY
- AZURE_SPEECH_REGION

示例：
  # 方式一：显式参数
  python test/video_production/tts_from_markdown/tts_from_markdown.py \
    --md docs/project_docs/1759075200_Git提交信息AI助手钩子.md \
    --wav out/tts/1759075200_Git提交信息AI助手钩子.wav \
    --srt out/tts/1759075200_Git提交信息AI助手钩子.srt

  # 方式二：位置参数（自动推导输出文件名）
  python test/video_production/tts_from_markdown/tts_from_markdown.py \
    docs/project_docs/kernel_reference/1752417025_从全局流变到局域刚性：论流变景观中刚性截面的逻辑守恒.md
  # 等价于：
  # python test/video_production/tts_from_markdown/tts_from_markdown.py \
  #   --md  docs/project_docs/kernel_reference/1752417025_从全局流变到局域刚性：论流变景观中刚性截面的逻辑守恒.md \
  #   --wav out/tts/1752417025_从全局流变到局域刚性：论流变景观中刚性截面的逻辑守恒.wav \
  #   --srt out/tts/1752417025_从全局流变到局域刚性：论流变景观中刚性截面的逻辑守恒.srt
"""

import argparse
import os
import re
import sys
import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

# 让仓库根目录可导入本地包等（本文件位于 script/tts_from_markdown/）
# 在不同目录层级下运行时，稳健定位仓库根（包含 docs/ 与 video_production/）
_here = Path(__file__).resolve()
_repo_root: Path
try:
    _repo_root = next(
        cand for cand in _here.parents
        if (cand / "docs").exists() and (cand / "video_production").exists()
    )
except StopIteration:
    # 回退：常见层级（script/tts_from_markdown/ -> repo_root）
    _repo_root = _here.parents[2] if len(_here.parents) >= 3 else _here.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from video_production_part.tts import AzureSynthOptions, synthesize_with_azure
from video_production_part.console import cprint as _cprint


def print(*args, **kwargs):  # type: ignore[override]
    kwargs.setdefault("flush", True)
    return _cprint(*args, **kwargs)


@dataclass
class MdParseOptions:
    # 最大段落字符数，超出则按句号/换行尝试进一步切分
    max_chars_per_segment: int = 500
    # 是否把标题作为独立片段
    keep_headings: bool = True
    # 是否保留无内容的空行片段（通常为 False）
    keep_empty: bool = False


# --- 配置（可选）：test/video_production/tts_from_markdown/tts_from_markdown.config.json ---
def _load_local_config() -> dict:
    """从同目录 tts_from_markdown.config.json 读取配置。"""
    cfg_path = Path(__file__).resolve().with_name("tts_from_markdown.config.json")
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _voice_name_from_cfg(cfg: dict, default_name: str = "zh-CN-YunyeNeural") -> str:
    """读取配置中的 voice：
    - 若 voice 为数字（1/2），则在 voice_src 中查找编号对应的名称；
    - 若 voice 为字符串，直接返回；
    - 否则返回默认名称。
    """
    v = cfg.get("voice")
    if isinstance(v, int):
        # invert mapping: id -> name
        vsrc = cfg.get("voice_src") or {}
        try:
            inv = {int(val): str(key) for key, val in vsrc.items()}
        except Exception:
            inv = {}
        name = inv.get(v)
        if isinstance(name, str) and name:
            return name
        # 兜底的默认映射
        if v == 1:
            return "zh-CN-XiaomengNeural"
        if v == 2:
            return "zh-CN-YunyeNeural"
        return default_name
    if isinstance(v, str) and v:
        return v
    return default_name


# --- 环境变量预检与脱敏打印（参考 script/print_env_ai.ps1） ---
def _mask(s: Optional[str]) -> str:
    if not s:
        return ""
    l = len(s)
    head = min(4, l)
    tail = min(2, max(0, l - head))
    prefix = s[:head]
    suffix = s[-tail:] if tail > 0 else ""
    return f"{prefix}***{suffix}"


def _print_env_summary() -> None:
    ak = os.getenv("AZURE_SPEECH_KEY")
    ar = os.getenv("AZURE_SPEECH_REGION")
    gk = os.getenv("GEMINI_API_KEY")
    gm = os.getenv("GEMINI_MODEL")

    def _row(name: str, is_set: bool, value: str) -> str:
        status = "SET" if is_set else "MISSING"
        tag = "[OK]" if is_set else "[WARN]"
        return f"  {tag} {name:<22} : {status:<8} {value}"

    print("[DEBUG] === AI Environment Variables ===")
    print(_row("AZURE_SPEECH_KEY", bool(ak), _mask(ak)))
    print(_row("AZURE_SPEECH_REGION", bool(ar), str(ar or "")))
    print(_row("GEMINI_API_KEY", bool(gk), _mask(gk)))
    print(_row("GEMINI_MODEL", bool(gm), str(gm or "")))


def _normalize_filename(s: str) -> str:
    # 去除常见引号，便于宽松匹配（适配用户未输入 “ ” 的情况）
    return str(s).replace("\u201c", "").replace("\u201d", "").replace('"', "").replace("'", "")


def _resolve_md_path(p: Path) -> Optional[Path]:
    """更稳健地解析 Markdown 路径：
    - 支持相对/绝对路径；
    - 若文件不存在，尝试在同一目录进行“去引号”名匹配；
    - 若存在形如 1234567890_ 前缀，尝试用前缀模糊匹配。
    """
    # 1) 原路径（原样/加仓库根）
    if p.exists():
        return p
    rp = (_repo_root / p) if not p.is_absolute() else p
    if rp.exists():
        return rp
    # 2) 同目录宽松匹配
    parent = rp.parent if rp.is_absolute() else p.parent
    if not parent.exists():
        # 若父目录不存在但显然是项目内相对路径，则拼到仓库根后再判断（不创建目录）
        parent = (_repo_root / parent)
    if parent.exists():
        target_name_norm = _normalize_filename(p.name)
        # 2.1 去引号精确匹配
        for f in parent.glob("*.md"):
            if _normalize_filename(f.name) == target_name_norm:
                return f
        # 2.2 数字前缀匹配（如 1752417024_...）
        import re as _re
        m = _re.match(r"^(\d{8,})_", p.name)
        if m:
            prefix = m.group(1) + "_"
            cand = sorted([f for f in parent.glob(f"{prefix}*.md")])
            if cand:
                return cand[0]
    return None


# --- 大小写读法规则与字幕换行 ---
_UPPER_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")


def _uppercase_letters_to_spelled(text: str) -> str:
    """将纯大写英文单词转为按字母拼读（例：CPU -> C P U）。保留其他内容不变。"""
    def repl(m: re.Match[str]) -> str:
        w = m.group(0)
        return " ".join(list(w))
    return _UPPER_WORD_RE.sub(repl, text)


def _apply_tts_case_rules(segments: List[str]) -> List[str]:
    return [_uppercase_letters_to_spelled(s) for s in segments]


def _split_chunks_by_chars(text: str, max_chars: int) -> List[str]:
    """按最大字符数切割为多条独立字幕，保持原文字符不变（不插入换行）。

    示例：max_chars=10 时，将文本切成每段至多 10 字符的若干段。
    """
    if not text:
        return []
    if max_chars is None or max_chars <= 0:
        return [text]
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    out: List[str] = []
    i = 0
    L = len(s)
    while i < L:
        j = min(i + max_chars, L)
        out.append(s[i:j])
        i = j
    return out


def _format_srt_timestamp(ms: float) -> str:
    ms_int = int(round(ms))
    h = ms_int // 3600000
    m = (ms_int % 3600000) // 60000
    s = (ms_int % 60000) // 1000
    ms_part = ms_int % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms_part:03d}"


def _write_srt_custom(segments_with_time: List[tuple[str, float, float]], path: Path, max_chars: int) -> None:
    """将每个原始分段进一步按 max_chars 切割为多个字幕条目，并等分时间。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = 1
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for text, start, end in segments_with_time:
            # 切分成若干块
            chunks = _split_chunks_by_chars(str(text or ""), max_chars)
            if not chunks:
                continue
            dur = max(0.0, float(end) - float(start))
            step = dur / float(len(chunks)) if len(chunks) > 0 else 0.0
            for k, chunk in enumerate(chunks):
                s0 = float(start) + k * step
                e0 = float(start) + (k + 1) * step if k + 1 < len(chunks) else float(end)
                if e0 < s0:
                    e0 = s0
                f.write(str(idx) + "\n")
                f.write(_format_srt_timestamp(s0) + " --> " + _format_srt_timestamp(e0) + "\n")
                f.write((chunk or "") + "\n\n")
                idx += 1


def _concat_wavs(wav_paths: List[Path], out_path: Path) -> float:
    """按顺序拼接多段 WAV（需相同参数），返回总时长（毫秒）。"""
    if not wav_paths:
        raise ValueError("no wavs to concat")
    params = None
    total_frames = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf_out:
        for i, p in enumerate(wav_paths):
            with wave.open(str(p), "rb") as wf_in:
                if params is None:
                    params = (wf_in.getnchannels(), wf_in.getsampwidth(), wf_in.getframerate())
                    wf_out.setnchannels(params[0])
                    wf_out.setsampwidth(params[1])
                    wf_out.setframerate(params[2])
                else:
                    if (
                        wf_in.getnchannels() != params[0]
                        or wf_in.getsampwidth() != params[1]
                        or wf_in.getframerate() != params[2]
                    ):
                        raise RuntimeError("WAV parameters mismatch when concatenating")
                frames = wf_in.readframes(wf_in.getnframes())
                wf_out.writeframes(frames)
                total_frames += wf_in.getnframes()
    rate = params[2] if params else 48000
    return total_frames * 1000.0 / float(rate)


def _batch_by_char_limit(texts: List[str], limit: int) -> List[List[int]]:
    batches: List[List[int]] = []
    cur: List[int] = []
    total = 0
    for i, s in enumerate(texts):
        l = len(s)
        if cur and total + l > limit:
            batches.append(cur)
            cur = [i]
            total = l
        else:
            cur.append(i)
            total += l
    if cur:
        batches.append(cur)
    return batches


def _synthesize_with_batching(
    segments_for_tts: List[str],
    segments_original: List[str],
    tts_opts: AzureSynthOptions,
    out_wav: Path,
    out_srt: Optional[Path],
    srt_max_chars: int,
    char_limit: int,
    debug: bool = False,
) -> None:
    try:
        batches = _batch_by_char_limit(segments_for_tts, char_limit)
    except Exception:
        batches = [list(range(len(segments_for_tts)))]

    tmp_dir = Path("out/tmp/tts_from_md_batches")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_wavs: List[Path] = []
    all_pairs: List[tuple[str, float, float]] = []
    offset_ms = 0.0
    try:
        for bi, idxs in enumerate(batches):
            sub_tts = [segments_for_tts[i] for i in idxs]
            sub_org = [segments_original[i] for i in idxs]
            tmp_wav = tmp_dir / f"batch_{bi:03d}.wav"
            # 合成当前批；不写 SRT，保留分段时间；若失败则自动降级更小批次
            def _synth_one_batch(tts_list: List[str], org_list: List[str], wav_path: Path) -> None:
                nonlocal offset_ms
                res = synthesize_with_azure(tts_list, wav_path, options=tts_opts, srt_path=None)
                tmp_wavs.append(wav_path)
                n = min(len(res.segments), len(org_list))
                for i in range(n):
                    st = float(res.segments[i].start_ms) + offset_ms
                    ed = float(res.segments[i].end_ms) + offset_ms
                    all_pairs.append((str(org_list[i]), st, ed))
                # 更新偏移：使用实际音频总时长，避免累计误差
                try:
                    with wave.open(str(wav_path), "rb") as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate() or 48000
                        offset_ms += frames * 1000.0 / float(rate)
                except Exception:
                    if res.segments:
                        offset_ms = float(all_pairs[-1][2])

            try:
                print(f"[INFO] 批次 {bi+1}/{len(batches)}：开始合成（条目 {len(sub_tts)}）…")
                _synth_one_batch(sub_tts, sub_org, tmp_wav)
                print(f"[OK] 批次 {bi+1}/{len(batches)} 完成，累计时长约 {int(offset_ms)} ms")
            except Exception as e:
                msg = str(e)
                if debug:
                    print(f"[WARN] 批次 {bi+1} 合成失败：{msg}\n       尝试减小每批字符数并重试……")
                # 递减限制并切更小子批
                sub_limit = max(300, int(char_limit // 2))
                # 基于 idxs 切分：按 segments_for_tts[i] 的长度累计
                def _split_indices_by_limit(indices: List[int], limit: int) -> List[List[int]]:
                    groups: List[List[int]] = []
                    cur: List[int] = []
                    total = 0
                    for ii in indices:
                        l = len(segments_for_tts[ii])
                        if cur and total + l > limit:
                            groups.append(cur)
                            cur = [ii]
                            total = l
                        else:
                            cur.append(ii)
                            total += l
                    if cur:
                        groups.append(cur)
                    return groups

                small_groups = _split_indices_by_limit(idxs, sub_limit)
                gi = 0
                for g in small_groups:
                    gi += 1
                    tts_g = [segments_for_tts[i] for i in g]
                    org_g = [segments_original[i] for i in g]
                    tmp_wav_g = tmp_dir / f"batch_{bi:03d}_part_{gi:02d}.wav"
                    # 若仍失败，进一步降级为逐条调用，尽量避免整体中断
                    try:
                        _synth_one_batch(tts_g, org_g, tmp_wav_g)
                    except Exception as e2:
                        if debug:
                            print(f"[WARN] 子批 {bi+1}-{gi} 合成失败：{e2}；降级为逐条调用……")
                        # 逐条调用，必要时再次按更小长度切分
                        item_idx = 0
                        for i_item, (t_item, o_item) in enumerate(zip(tts_g, org_g)):
                            item_idx += 1
                            try:
                                tmp_wav_item = tmp_dir / f"batch_{bi:03d}_part_{gi:02d}_item_{item_idx:03d}.wav"
                                _synth_one_batch([t_item], [o_item], tmp_wav_item)
                            except Exception as e3:
                                # 最后兜底：对该条再次按较小上限拆分再逐条调用（例如 120 字）
                                if debug:
                                    print(f"[WARN] 单条失败，继续细分：{e3}")
                                tiny_parts = _split_long_text(t_item, 120)
                                tiny_orgs = _split_long_text(o_item, 120)
                                for j, (t_small, o_small) in enumerate(zip(tiny_parts, tiny_orgs), start=1):
                                    tmp_wav_small = tmp_dir / (
                                        f"batch_{bi:03d}_part_{gi:02d}_item_{item_idx:03d}_seg_{j:03d}.wav"
                                    )
                                    _synth_one_batch([t_small], [o_small], tmp_wav_small)
                if debug:
                    print(f"[DEBUG] 批次 {bi+1} 已拆分为 {len(small_groups)} 个子批并完成。累计时长约 {int(offset_ms)} ms")

        # 拼接音频
        _concat_wavs(tmp_wavs, out_wav)
        # 写 SRT（按原文+切割）
        if out_srt is not None:
            _write_srt_custom(all_pairs, out_srt, srt_max_chars)
    finally:
        # 清理临时 wav
        for p in tmp_wavs:
            try:
                p.unlink()
            except Exception:
                pass


_MD_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_MD_LIST_RE = re.compile(r"^\s{0,3}([\-*+]|\d+[\.)])\s+(.*)$")


def _strip_md_inline(text: str) -> str:
    """去除 Markdown 内联标记，保留可读文本。"""
    t = text or ""
    # 图片: ![alt](url) -> alt
    t = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", t)
    # 链接: [text](url) -> text
    t = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", t)
    # 粗体/斜体: **x** / *x* / __x__ / _x_ -> x
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = re.sub(r"_([^_]+)_", r"\1", t)
    # 行内代码: `x` -> x
    t = re.sub(r"`([^`]+)`", r"\1", t)
    # HTML 标签简单移除
    t = re.sub(r"<[^>]+>", "", t)
    return t.strip()


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """把过长文本按句号/顿号/逗号/分号/换行优先切分到不超过上限。"""
    t = text.strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    # 先按换行切
    parts: List[str] = []
    for para in t.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            parts.append(para)
            continue
        # 再按句读符号切
        buf = ""
        for ch in re.split(r"([。！？!?；;，,])", para):
            if not ch:
                continue
            new_buf = buf + ch
            if len(new_buf) > max_chars and buf:
                parts.append(buf)
                buf = ch
            else:
                buf = new_buf
        if buf:
            parts.append(buf)
    return [p.strip() for p in parts if p.strip()]


def markdown_to_segments(md_text: str, *, options: Optional[MdParseOptions] = None) -> List[str]:
    """将 Markdown 文本解析为朗读片段列表。"""
    opts = options or MdParseOptions()
    lines = md_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    in_code = False
    buffer: List[str] = []
    segments: List[str] = []

    def flush_buffer():
        nonlocal buffer
        if not buffer:
            return
        text = _strip_md_inline(" ".join(buffer).strip())
        buffer = []
        for p in _split_long_text(text, opts.max_chars_per_segment):
            if p or opts.keep_empty:
                segments.append(p)

    for raw in lines:
        ln = raw.rstrip("\n")
        if _MD_FENCE_RE.match(ln):
            in_code = not in_code
            continue
        if in_code:
            continue  # 跳过代码块
        if not ln.strip():
            flush_buffer()
            continue
        m_h = _MD_HEADING_RE.match(ln)
        if m_h:
            flush_buffer()
            if opts.keep_headings:
                seg = _strip_md_inline(m_h.group(2))
                if seg:
                    # 标题末尾补句号，提高听感
                    if not re.search(r"[。.!?！？]$", seg):
                        seg += "。"
                    segments.append(seg)
            continue
        m_l = _MD_LIST_RE.match(ln)
        if m_l:
            item = _strip_md_inline(m_l.group(2))
            if item:
                buffer.append(item)
            continue
        # 普通段落行
        buffer.append(_strip_md_inline(ln))

    flush_buffer()
    # 过滤过短/无意义段落
    segments = [s for s in segments if s and re.sub(r"\s+", "", s) != "-"]
    return segments


def _maybe_transform_with_gemini(
    segments: List[str], *,
    enabled: bool,
    model_name: str = "gemini-2.5-flash",
    debug: bool = False,
) -> List[str]:
    """
    可选：调用 Gemini 将数学符号/LaTeX 等转为中文可朗读文本。
    - 若未设置 GOOGLE_API_KEY 或未安装 google-generativeai，直接返回原分段。
    - 失败时保底返回原分段。
    """
    if not enabled:
        return segments
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        if debug:
            print("[GEMINI] 未检测到 GOOGLE_API_KEY，跳过数学朗读增强。")
        return segments
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        if debug:
            print("[GEMINI] 未安装 google-generativeai，跳过数学朗读增强。")
        return segments

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or "gemini-2.5-flash")
    except Exception as e:
        if debug:
            print(f"[GEMINI] 初始化失败: {e}")
        return segments

    # 组合输入，要求严格 JSON 输出，以保持分段对应关系
    payload = {"segments": [{"index": i, "text": s} for i, s in enumerate(segments)]}
    sys_prompt = (
        "你是中文朗读优化助手。请将输入列表中的每个段落改写为适合中文TTS朗读的文本，"
        "仅做朗读友好化改写，不改变技术含义与叙述顺序。\n"
        "关键要求：\n"
        "1) 将数学公式与符号（含 LaTeX 与 Unicode：如 ∑, ∫, ≤, ≥, ≈, →, α, β 等）改写为中文读法；\n"
        "2) 希腊字母中文读法示例：α→阿尔法，β→贝塔，γ→伽马，δ→德尔塔，θ→西塔，λ→拉姆达，μ→缪，ν→纽，ξ→克西，π→派，ρ→柔，σ→西格玛，τ→套，φ→斐，χ→卡伊，ψ→普赛，ω→欧米伽；\n"
        "3) 常见运算符读法：=等于，≠不等于，≈约等于，≤小于等于，≥大于等于，→趋向于/指向，↦映射为，∑求和，∏连乘，∫积分，∬二重积分，∭三重积分，∂偏导，∇梯度/纳布拉，⋅点乘，×乘号，/除以，·中点，∘复合；\n"
        "4) 指数：x^2 读作“x 的二次方”，x^n 读作“x 的 n 次方”；\n"
        "5) 分式：a/b 读作“a 除以 b”；\n"
        "6) 绝对值与范数：|x| 读作“x 的绝对值”，∥x∥ 读作“x 的范数”；\n"
        "7) 导数与微分：f'(x) 读作“f 在 x 处的一阶导数”，d/dx 读作“对 x 求导”，∂f/∂x 读作“f 对 x 的偏导”；\n"
        "8) 矢量/矩阵：带粗体或箭头可读作“向量/矩阵”；\n"
        "9) 单位与符号尽量给出常用中文读法；\n"
        "10) 必须保留每段原有顺序与数量，不合并/不增删。\n\n"
        "严格按照 JSON 返回，结构为：{\"segments\":[{\"index\":i,\"text\":\"...\"}, ...]}，"
        "不得输出任何多余说明或标注。"
    )

    # 控制长度：若过长则分批处理
    def _batches(src: List[str], limit_chars: int = 8000) -> List[List[int]]:
        batches: List[List[int]] = []
        cur: List[int] = []
        total = 0
        for i, s in enumerate(src):
            sl = len(s)
            if cur and total + sl > limit_chars:
                batches.append(cur)
                cur = [i]
                total = sl
            else:
                cur.append(i)
                total += sl
        if cur:
            batches.append(cur)
        return batches

    out = segments[:]
    for idxs in _batches(segments):
        sub_payload = {"segments": [{"index": i, "text": segments[i]} for i in idxs]}
        try:
            resp = model.generate_content([
                {"role": "user", "parts": [
                    {"text": sys_prompt},
                    {"text": json.dumps(sub_payload, ensure_ascii=False)}
                ]}
            ])
            text = getattr(resp, "text", None) or ""
            # 去除可能包裹的代码块围栏
            text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE|re.MULTILINE)
            data = json.loads(text)
            items = data.get("segments") or []
            for item in items:
                try:
                    i = int(item.get("index"))
                    t = str(item.get("text") or "").strip()
                except Exception:
                    continue
                if 0 <= i < len(out) and t:
                    out[i] = t
        except Exception as e:
            if debug:
                print(f"[GEMINI] 子批处理失败，跳过：{e}")
            continue
    return out


def _default_md_file() -> Optional[Path]:
    # 首选固定示例；若因编码差异不存在，则使用通配匹配 1759075200_*.md
    fixed = Path("docs/project_docs/1759075200_Git提交信息AI助手钩子.md")
    if fixed.exists():
        return fixed
    cand = sorted(Path("docs/project_docs").glob("1759075200_*.md"))
    return cand[0] if cand else None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="从 Markdown 生成旁白音频（Azure TTS）")
    # 位置参数：直接传入 Markdown 路径时，自动推导输出 WAV/SRT 名称
    ap.add_argument("md_path", nargs="?", type=Path, help="Markdown 路径（可选，亦可用 --md 指定）")
    ap.add_argument("--md", type=Path, default=None, help="输入 Markdown 文件路径")
    ap.add_argument("--wav", type=Path, default=None, help="输出 WAV 路径，不传则根据文件名生成")
    ap.add_argument("--srt", type=Path, default=None, help="可选 SRT 输出路径")
    ap.add_argument("--language", default="zh-CN")
    # 由配置提供默认音色；未命令行覆盖时，读取 JSON
    ap.add_argument("--voice", default=None)
    ap.add_argument("--max-chars", type=int, default=500, help="每段最大字符数")
    # Gemini 数学符号朗读增强
    ap.add_argument("--gemini-math", dest="gemini_math", action="store_true", default=True, help="启用 Gemini 数学朗读增强")
    ap.add_argument("--no-gemini-math", dest="gemini_math", action="store_false", help="关闭 Gemini 数学朗读增强")
    ap.add_argument("--gemini-model", default=None, help="Gemini 模型名（不传则读取环境或配置）")
    ap.add_argument("--debug", action="store_true", default=True)
    args = ap.parse_args(argv)

    # 读取本地配置（若存在）
    cfg = _load_local_config()

    # 环境变量回退：系统环境变量优先，缺失时从配置补齐
    def _ensure_env_from_cfg(cfg: dict) -> dict:
        sources = {}
        def set_if_missing(name: str, value: str | None):
            cur = os.getenv(name)
            if cur:
                sources[name] = "env"
                return cur
            if value not in (None, ""):
                os.environ[name] = str(value)
                sources[name] = "cfg"
                return os.environ[name]
            sources[name] = "missing"
            return None

        set_if_missing("AZURE_SPEECH_KEY", cfg.get("AZURE_SPEECH_KEY"))
        set_if_missing("AZURE_SPEECH_REGION", cfg.get("AZURE_SPEECH_REGION") or "eastasia")
        set_if_missing("GEMINI_API_KEY", cfg.get("GEMINI_API_KEY"))
        set_if_missing("GEMINI_MODEL", cfg.get("GEMINI_MODEL"))
        return sources

    sources = _ensure_env_from_cfg(cfg)

    # 环境变量预检与打印（脱敏，彩色）
    _print_env_summary()
    ak = os.getenv("AZURE_SPEECH_KEY")
    ar = os.getenv("AZURE_SPEECH_REGION")
    if not (ak and ar):
        missing = []
        if not ak:
            missing.append("AZURE_SPEECH_KEY")
        if not ar:
            missing.append("AZURE_SPEECH_REGION")
        print("[ERROR] 缺少必要环境变量：" + ", ".join(missing))
        print("[READ] 提示：")
        print("[READ]   Linux/macOS: export AZURE_SPEECH_KEY='...' && export AZURE_SPEECH_REGION='...' ")
        print("[READ]   Windows CMD: set AZURE_SPEECH_KEY=... & set AZURE_SPEECH_REGION=... ")
        print("[READ]   PowerShell : $env:AZURE_SPEECH_KEY='...'; $env:AZURE_SPEECH_REGION='...'")
        return 1

    # 优先 --md，其次位置参数 md_path，否则回退默认样例
    raw_md: Optional[Path] = args.md or args.md_path or _default_md_file()
    md_file: Optional[Path] = None
    if raw_md is not None:
        md_file = _resolve_md_path(raw_md)
    if not md_file or not md_file.exists():
        print("[ERROR] 未找到 Markdown：", str(raw_md or "docs/project_docs/1759075200_*.md"))
        # 若用户提供的是目录或文件名前缀，给出提示
        if raw_md is not None:
            print("[WARN] 提示：若文件名包含中文引号（“”），可直接输入不含引号版本，脚本会自动匹配。")
        return 2

    base_name = md_file.stem  # 不含后缀
    out_wav = args.wav or Path("out/tts") / f"{base_name}.wav"
    cfg_voice = _voice_name_from_cfg(cfg, default_name="zh-CN-YunyeNeural")
    cfg_generate_srt = bool(cfg.get("generate_srt", True))
    # srt_max_chars: 0 或负数表示不切割，直接使用原分段作为整行字幕
    _srt_val = cfg.get("srt_max_chars")
    try:
        cfg_srt_max_chars = int(_srt_val) if _srt_val is not None else 0
    except Exception:
        cfg_srt_max_chars = 0
    # srt 生成策略：命令行 --srt 优先；未传则依据配置决定是否生成到默认路径
    out_srt: Optional[Path]
    if args.srt is not None:
        out_srt = args.srt  # 显式传入
    else:
        out_srt = (Path("out/tts") / f"{base_name}.srt") if cfg_generate_srt else None
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_srt is not None:
        out_srt.parent.mkdir(parents=True, exist_ok=True)

    opts = MdParseOptions(max_chars_per_segment=max(50, int(args.max_chars)))
    md_text = md_file.read_text(encoding="utf-8")
    segments_original = markdown_to_segments(md_text, options=opts)
    # Gemini 数学朗读增强
    # 选择 Gemini 模型优先级：命令行 > 环境变量 > 配置 > 默认
    effective_gemini_model = args.gemini_model or os.getenv("GEMINI_MODEL") or str(cfg.get("GEMINI_MODEL") or "gemini-2.5-flash")
    segments_for_tts = _maybe_transform_with_gemini(
        segments_original[:],
        enabled=bool(args.gemini_math),
        model_name=str(effective_gemini_model),
        debug=bool(args.debug),
    )
    # 合成语音时应用大小写读法规则（CPU -> C P U；OpenAI/word 保持英文）
    segments_for_tts = _apply_tts_case_rules(segments_for_tts)
    if args.debug:
        print(f"[DEBUG] 段落数: {len(segments_for_tts)} | 文件: {md_file}")
        for i, s in enumerate(segments_for_tts[:5], 1):
            s_show = s if len(s) <= 80 else (s[:77] + "...")
            print(f"[READ] - [{i}] {s_show}")
        if len(segments_for_tts) > 5:
            print(f"[READ] ... 其余 {len(segments_for_tts)-5} 段")

    voice = args.voice or cfg_voice
    tts_opts = AzureSynthOptions(language=args.language, voice=voice)
    # 启用 SDK 级合成流事件打印（若外部未显式关闭）
    if os.getenv("AZURE_TTS_DEBUG") in (None, ""):
        os.environ["AZURE_TTS_DEBUG"] = "1"
    # 根据长度决定是否分批避免 Azure 超时
    total_chars = sum(len(s) for s in segments_for_tts)
    env_limit = os.getenv("AZURE_TTS_BATCH_CHARS")
    # 允许通过环境变量覆盖阈值；否则使用配置值；再否则使用安全默认值 600
    try:
        limit = int(env_limit) if env_limit not in (None, "") else int(cfg.get("azure_tts_batch_chars", 600))
    except Exception:
        limit = 600
    start_ts = time.time()
    if total_chars <= limit:
        # 单次合成
        print(f"[INFO] 开始合成：段落={len(segments_for_tts)}，总字数={total_chars}，语音={voice}")
        print("[INFO] 正在请求 Azure TTS（一次提交）…")
        res = synthesize_with_azure(segments_for_tts, out_wav, options=tts_opts, srt_path=None)
        print("[OK] 音频:", res.audio_path)
        if out_srt is not None:
            print("[INFO] 生成字幕…")
            pairs = []
            n = min(len(res.segments), len(segments_original))
            for i in range(n):
                st = float(res.segments[i].start_ms)
                ed = float(res.segments[i].end_ms)
                txt = str(segments_original[i])
                pairs.append((txt, st, ed))
            _write_srt_custom(pairs, out_srt, cfg_srt_max_chars)
            print("[OK] 字幕:", out_srt)
        dur_ms = 0.0
        try:
            dur_ms = float(res.segments[-1].end_ms) if res.segments else 0.0
        except Exception:
            dur_ms = 0.0
        cost = time.time() - start_ts
        print(f"[DONE] 合成完成：音频时长≈{int(dur_ms/1000)}s，用时{cost:.1f}s")
    else:
        # 文本较长
        batch_enabled = bool(cfg.get("azure_tts_batch_enable", True))
        if batch_enabled:
            if args.debug:
                print(f"[OK] 文本较长（{total_chars} chars），按 {limit} chars/批 次进行分段合成以避免 Azure 超时。")
            print(f"[INFO] 开始合成：段落={len(segments_for_tts)}，总字数={total_chars}，语音={voice}")
            print(f"[INFO] 将分批合成（约 {limit} 字/批）…")
            _synthesize_with_batching(
                segments_for_tts,
                segments_original,
                tts_opts,
                out_wav,
                out_srt,
                cfg_srt_max_chars,
                limit,
                debug=bool(args.debug),
            )
            print("[OK] 音频:", out_wav)
            if out_srt is not None:
                print("[OK] 字幕:", out_srt)
            # 统计耗时与时长
            dur_ms = 0.0
            try:
                with wave.open(str(out_wav), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate() or 48000
                    dur_ms = frames * 1000.0 / float(rate)
            except Exception:
                dur_ms = 0.0
            cost = time.time() - start_ts
            print(f"[DONE] 合成完成：音频时长≈{int(dur_ms/1000)}s，用时{cost:.1f}s")
        else:
            # 不分批：直接一次提交
            print(f"[INFO] 开始合成：段落={len(segments_for_tts)}，总字数={total_chars}，语音={voice}")
            print("[INFO] 正在请求 Azure TTS（一次提交）…")
            res = synthesize_with_azure(segments_for_tts, out_wav, options=tts_opts, srt_path=None)
            print("[OK] 音频:", res.audio_path)
            if out_srt is not None:
                print("[INFO] 生成字幕…")
                pairs = []
                n = min(len(res.segments), len(segments_original))
                for i in range(n):
                    st = float(res.segments[i].start_ms)
                    ed = float(res.segments[i].end_ms)
                    txt = str(segments_original[i])
                    pairs.append((txt, st, ed))
                _write_srt_custom(pairs, out_srt, cfg_srt_max_chars)
                print("[OK] 字幕:", out_srt)
            dur_ms = 0.0
            try:
                dur_ms = float(res.segments[-1].end_ms) if res.segments else 0.0
            except Exception:
                dur_ms = 0.0
            cost = time.time() - start_ts
            print(f"[DONE] 合成完成：音频时长≈{int(dur_ms/1000)}s，用时{cost:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

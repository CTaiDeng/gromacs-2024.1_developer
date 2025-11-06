from __future__ import annotations

import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import azure.cognitiveservices.speech as speechsdk
from ..console import cprint as _cprint


@dataclass(frozen=True)
class AzureSynthOptions:
    language: str = "zh-CN"
    voice: str = "zh-CN-YunyeNeural"
    rate: str = "+0%"   # e.g. "+10%", "-20%"
    pitch: str = "+0%"  # e.g. "+2%"
    volume: str = "+0%" # e.g. "+0%", "+50%"


@dataclass(frozen=True)
class Segment:
    text: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class AzureTTSResult:
    audio_path: Path
    segments: List[Segment]
    srt_path: Optional[Path]


def _audio_duration_ms(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames * 1000.0 / float(rate or 1)


def split_text(text: str, granularity: str = "char") -> List[str]:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    if granularity == "char":
        return [c for c in t if c]
    if granularity == "word":
        return [w for w in t.split() if w]
    if granularity == "line":
        return [ln for ln in t.split("\n") if ln]
    return [text]


def _make_ssml(opts: AzureSynthOptions, texts: List[str]) -> str:
    # Insert bookmarks before each segment to capture offsets
    parts: List[str] = []
    for i, t in enumerate(texts):
        safe = str(t or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(
            f"<bookmark mark='mark{i}'/><prosody rate='{opts.rate}' pitch='{opts.pitch}' volume='{opts.volume}'>{safe}</prosody>"
        )
    text_xml = "".join(parts)
    ssml = (
        "<speak version='1.0' xml:lang='{}' "
        "xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts'>"
        "<voice name='{}'>{}"  # voice wraps the prosody parts
        "</voice></speak>"
    ).format(opts.language, opts.voice, text_xml)
    return ssml


def _write_srt(segments: List[Segment], path: Path) -> None:
    def fmt(ms: float) -> str:
        ms = max(0.0, ms)
        s, ms_rem = divmod(int(round(ms)), 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms_rem:03d}"

    lines: List[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{fmt(seg.start_ms)} --> {fmt(seg.end_ms)}")
        lines.append(seg.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def synthesize_with_azure(
    texts: str | Iterable[str],
    out_wav: Path,
    *,
    options: Optional[AzureSynthOptions] = None,
    srt_path: Optional[Path] = None,
) -> AzureTTSResult:
    """
    使用 Azure 语音合成，将文本或文本段列表合成为 WAV，并返回段级时间信息。

    - texts 可为字符串或字符串列表；列表将通过 SSML bookmark 标记以恢复分段时间。
    - 需要环境变量：AZURE_SPEECH_KEY、AZURE_SPEECH_REGION。
    """
    key = os.getenv("AZURE_SPEECH_KEY") or os.getenv("AZURE_SPEECH_KEY".lower())
    region = os.getenv("AZURE_SPEECH_REGION") or os.getenv("AZURE_SPEECH_REGION".lower())
    if not key or not region:
        raise RuntimeError("Missing AZURE_SPEECH_KEY/AZURE_SPEECH_REGION in environment")

    opts = options or AzureSynthOptions()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(texts, str):
        texts_list = [texts]
    else:
        texts_list = [str(t) for t in texts]
    if not texts_list:
        raise ValueError("texts is empty")

    ssml = _make_ssml(opts, texts_list)

    # 输出 48kHz 单声道 PCM WAV
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff48Khz16BitMonoPcm
    )

    temp_wav = out_wav.with_suffix(out_wav.suffix + ".generate.wav")
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False, filename=str(temp_wav))
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    bookmarks: List[int] = []

    def _on_bookmark(evt):
        # Azure offset in 100-ns ticks; convert to milliseconds
        bookmarks.append(int(evt.audio_offset / 10000))

    if len(texts_list) > 1:
        synthesizer.bookmark_reached.connect(_on_bookmark)  # type: ignore[attr-defined]

    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        raise RuntimeError(f"Azure TTS canceled: {details.reason} | {details.error_details or ''}")
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"Azure TTS failed: reason={result.reason}")

    # 写入目标文件（部分平台需等待系统句柄释放，这里做稳健重试）
    for i in range(10):
        try:
            temp_wav.replace(out_wav)
            break
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    if not out_wav.exists():
        # 退化：尝试复制
        with open(temp_wav, "rb") as rf, open(out_wav, "wb") as wf:
            while True:
                chunk = rf.read(1024 * 1024)
                if not chunk:
                    break
                wf.write(chunk)
            wf.flush()
        try:
            temp_wav.unlink(missing_ok=True)  # type: ignore[call-arg]
        except Exception:
            pass

    # 构造段落时间信息
    segs: List[Segment] = []
    total = _audio_duration_ms(out_wav)
    if len(texts_list) <= 1 or not bookmarks:
        segs.append(Segment(text=texts_list[0], start_ms=0.0, end_ms=total))
    else:
        # 若 bookmark 数量与段落数量不一致，按平均切分兜底
        if len(bookmarks) != len(texts_list):
            step = total / float(len(texts_list))
            for i, t in enumerate(texts_list):
                s = i * step
                e = total if i == len(texts_list) - 1 else (i + 1) * step
                segs.append(Segment(text=t, start_ms=s, end_ms=e))
        else:
            for i, t in enumerate(texts_list):
                s = float(bookmarks[i])
                e = float(bookmarks[i + 1]) if i + 1 < len(bookmarks) else float(total)
                e = max(e, s)
                segs.append(Segment(text=t, start_ms=s, end_ms=e))

    out_srt: Optional[Path] = None
    if srt_path is not None:
        out_srt = Path(srt_path)
        out_srt.parent.mkdir(parents=True, exist_ok=True)
        _write_srt(segs, out_srt)

    return AzureTTSResult(audio_path=out_wav, segments=segs, srt_path=out_srt)


def synthesize_with_azure_ssml(
    ssml: str,
    out_wav: Path,
    *,
    options: Optional[AzureSynthOptions] = None,
    srt_path: Optional[Path] = None,
) -> AzureTTSResult:
    """直接接受 SSML 进行合成，支持 bookmark 恢复分段。"""
    opts = options or AzureSynthOptions()
    # 简单将 SSML 包裹到目标语音配置中（若已有 <voice> 则直接使用给定 SSML）
    import re as _re
    if not _re.search(r"<\s*voice\b", ssml, flags=_re.IGNORECASE):
        ssml = (
            f"<speak version='1.0' xml:lang='{opts.language}' "
            "xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{opts.voice}'>{ssml}</voice>"
            "</speak>"
        )

    # 为了复用上面的实现，简单拆成单段调用（bookmark 在传入 SSML 中）
    # 这里直接调用 Azure SDK 一次，复用 bookmarks 逻辑与结果生成流程
    key = os.getenv("AZURE_SPEECH_KEY") or os.getenv("AZURE_SPEECH_KEY".lower())
    region = os.getenv("AZURE_SPEECH_REGION") or os.getenv("AZURE_SPEECH_REGION".lower())
    if not key or not region:
        raise RuntimeError("Missing AZURE_SPEECH_KEY/AZURE_SPEECH_REGION in environment")

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff48Khz16BitMonoPcm
    )

    temp_wav = out_wav.with_suffix(out_wav.suffix + ".generate.wav")
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False, filename=str(temp_wav))
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    bookmarks: List[int] = []
    def _on_bookmark(evt):
        bookmarks.append(int(evt.audio_offset / 10000))
    synthesizer.bookmark_reached.connect(_on_bookmark)  # type: ignore[attr-defined]

    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        raise RuntimeError(f"Azure TTS canceled: {details.reason} | {details.error_details or ''}")
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"Azure TTS failed: reason={result.reason}")

    # 写入目标文件
    for i in range(10):
        try:
            temp_wav.replace(out_wav)
            break
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    if not out_wav.exists():
        with open(temp_wav, "rb") as rf, open(out_wav, "wb") as wf:
            while True:
                chunk = rf.read(1024 * 1024)
                if not chunk:
                    break
                wf.write(chunk)
            wf.flush()
        try:
            temp_wav.unlink(missing_ok=True)  # type: ignore[call-arg]
        except Exception:
            pass

    # 根据 bookmark 恢复分段（若无 bookmark，退化为单段）
    # 简化实现：去除 SSML 标签时不严格解析，仅做基本替换即可
    def _strip_ssml_tags(text: str) -> str:
        t = _re.sub(r"<[^>]+>", "", text or "")
        t = _re.sub(r"\s+", " ", t).strip()
        return t

    import re as _re
    parts: List[str] = []
    bm_pat = _re.compile(r"<\s*bookmark\s+[^>]*mark=\s*['\"]mark(\d+)['\"][^>]*/\s*>", _re.IGNORECASE)
    idxs = [(m.start(), m.end(), int(m.group(1))) for m in bm_pat.finditer(ssml)]
    if not idxs:
        texts_list = [_strip_ssml_tags(ssml)]
    else:
        s = ssml
        idxs.sort(key=lambda x: x[0])
        for i, (s0, e0, n0) in enumerate(idxs):
            start = e0
            end = idxs[i + 1][0] if i + 1 < len(idxs) else len(s)
            chunk = s[start:end]
            parts.append(_strip_ssml_tags(chunk))
        texts_list = [p for p in parts if p]
        if not texts_list:
            texts_list = [_strip_ssml_tags(ssml)]

    total = _audio_duration_ms(out_wav)
    segs: List[Segment] = []
    if len(texts_list) <= 1 or not bookmarks:
        txt = texts_list[0] if texts_list else ""
        segs.append(Segment(text=txt, start_ms=0.0, end_ms=total))
    else:
        if len(bookmarks) != len(texts_list):
            step = total / float(len(texts_list))
            for i, t in enumerate(texts_list):
                s = i * step
                e = total if i == len(texts_list) - 1 else (i + 1) * step
                segs.append(Segment(text=t, start_ms=s, end_ms=e))
        else:
            for i, t in enumerate(texts_list):
                s = float(bookmarks[i])
                e = float(bookmarks[i + 1]) if i + 1 < len(bookmarks) else float(total)
                e = max(e, s)
                segs.append(Segment(text=t, start_ms=s, end_ms=e))

    out_srt: Optional[Path] = None
    if srt_path is not None:
        out_srt = Path(srt_path)
        out_srt.parent.mkdir(parents=True, exist_ok=True)
        _write_srt(segs, out_srt)

    return AzureTTSResult(audio_path=out_wav, segments=segs, srt_path=out_srt)


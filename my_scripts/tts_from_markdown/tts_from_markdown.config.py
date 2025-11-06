from __future__ import annotations

"""
配置编辑器（交互式）：用于读取/修改 tts_from_markdown 的配置 JSON。

位置：test/video_production/tts_from_markdown/tts_from_markdown.config.py
配置：test/video_production/tts_from_markdown/tts_from_markdown.config.json

提供能力：
- 查看配置：--show
- 交互式配置：直接运行（逐项打印并输入/选择，回车保留当前值）

示例：
  # 查看配置
  python test/video_production/tts_from_markdown/tts_from_markdown.config.py --show

  # 交互式逐项配置（默认行为）
  python test/video_production/tts_from_markdown/tts_from_markdown.config.py
"""

import argparse
import json
import sys
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULTS = {
    "generate_srt": True,
    # 语音：支持两种配置方式：
    # - voice_src: { name -> id }
    # - voice: 1|2（数字，引用 voice_src）或 直接 name 字符串（兼容旧版）
    "voice_src": {
        "zh-CN-XiaomengNeural": 1,
        "zh-CN-YunyeNeural": 2,
    },
    "voice": 1,
    "AZURE_SPEECH_KEY": "",
    "AZURE_SPEECH_REGION": "eastasia",
    "GEMINI_API_KEY": "",
    "GEMINI_MODEL": "gemini-2.5-flash",
    # srt_max_chars: 0 表示不对字幕进行切割（恢复长字幕）
    "srt_max_chars": 0,
    # Azure TTS 分批控制：默认开启，且从较小的每批字符数开始以降低超时概率
    "azure_tts_batch_chars": 600,
    "azure_tts_batch_enable": True,
}


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return dict(DEFAULTS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(DEFAULTS)
        # 合并默认值，保底
        out = dict(DEFAULTS)
        out.update({k: v for k, v in data.items() if k in DEFAULTS})
        return out
    except Exception:
        return dict(DEFAULTS)


def save_config(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mask(s: str) -> str:
    if not s:
        return ""
    l = len(s)
    head = s[: min(4, l)]
    tail = s[-min(2, max(0, l - 4)) :] if l > 4 else ""
    return f"{head}***{tail}"


def _prompt_bool(prompt: str, cur: bool) -> bool:
    dv = "y" if cur else "n"
    while True:
        ans = input(f"{prompt} (y/n) [default: {dv}]: ").strip().lower()
        if ans == "":
            return cur
        if ans in ("y", "yes", "1", "true"):  # type: ignore[return-value]
            return True
        if ans in ("n", "no", "0", "false"):
            return False
        print("请输入 y/n 或回车保留。")


def _prompt_int(prompt: str, cur: int, min_value: int = 1) -> int:
    while True:
        ans = input(f"{prompt} [default: {cur}]: ").strip()
        if ans == "":
            return cur
        try:
            v = int(ans)
            if v < min_value:
                print(f"必须 >= {min_value}")
                continue
            return v
        except Exception:
            print("请输入整数或回车保留。")


def _prompt_secret(prompt: str, cur_masked: str) -> str:
    print(f"{prompt} [当前: {cur_masked or '(空)'}]")
    print("- 回车：保留现有值；输入 '-'：清空；其余：更新为新值。")
    ans = getpass("")
    if ans.strip() == "":
        return None  # type: ignore[return-value]
    if ans.strip() == "-":
        return ""
    return ans


def _prompt_choice(prompt: str, current_label: str, options: list[tuple[str, int]]) -> int:
    print(prompt)
    for name, vid in options:
        print(f"  {vid}) {name}")
    while True:
        ans = input(f"选择编号或名称 [当前: {current_label}]: ").strip()
        if ans == "":
            # 返回当前编号
            for name, vid in options:
                if name == current_label:
                    return vid
            try:
                return int(current_label)  # 允许当前就是数字
            except Exception:
                # 默认第一个
                return options[0][1]
        # 数字
        try:
            vid = int(ans)
            if any(v == vid for _, v in options):
                return vid
        except Exception:
            pass
        # 名称
        for name, vid in options:
            if ans == name:
                return vid
        print("无效输入，请输入列表中的编号或名称，或回车保留。")


def interactive_update(data: Dict[str, Any]) -> Dict[str, Any]:
    print("=== 交互式配置（回车保留，按提示输入） ===")
    # 1) generate_srt
    cur_gen = bool(data.get("generate_srt", DEFAULTS["generate_srt"]))
    data["generate_srt"] = _prompt_bool("是否生成字幕 (generate_srt)", cur_gen)

    # 2) voice (编号)
    vsrc = data.get("voice_src") or DEFAULTS["voice_src"]
    try:
        options = sorted([(str(k), int(v)) for k, v in vsrc.items()], key=lambda x: x[1])
    except Exception:
        options = [("zh-CN-XiaomengNeural", 1), ("zh-CN-YunyeNeural", 2)]
    cur_voice = data.get("voice", DEFAULTS["voice"])  # may be int or str
    # 解析当前 label
    cur_label = None
    if isinstance(cur_voice, int):
        for name, vid in options:
            if vid == cur_voice:
                cur_label = name
                break
        if cur_label is None:
            cur_label = str(cur_voice)
    else:
        cur_label = str(cur_voice)
    vid = _prompt_choice("选择音色 (voice)", cur_label, options)
    data["voice"] = vid

    # 3) AZURE_SPEECH_KEY（可空）
    cur_azure_key = str(data.get("AZURE_SPEECH_KEY", ""))
    ans = _prompt_secret("AZURE_SPEECH_KEY（密钥，建议留空走环境变量）", _mask(cur_azure_key))
    if ans is not None:
        data["AZURE_SPEECH_KEY"] = ans

    # 4) AZURE_SPEECH_REGION
    cur_region = str(data.get("AZURE_SPEECH_REGION", DEFAULTS["AZURE_SPEECH_REGION"]))
    ans = input(f"AZURE_SPEECH_REGION [default: {cur_region}]: ").strip()
    if ans != "":
        data["AZURE_SPEECH_REGION"] = ans

    # 5) GEMINI_API_KEY（可空）
    cur_gk = str(data.get("GEMINI_API_KEY", ""))
    ans = _prompt_secret("GEMINI_API_KEY（密钥，建议留空走环境变量）", _mask(cur_gk))
    if ans is not None:
        data["GEMINI_API_KEY"] = ans

    # 6) GEMINI_MODEL（两项任选）
    cur_model = str(data.get("GEMINI_MODEL", DEFAULTS["GEMINI_MODEL"]))
    model = _prompt_choice(
        "选择 GEMINI_MODEL",
        cur_model,
        [("gemini-2.5-flash", 1), ("gemini-2.5-pro", 2)],
    )
    data["GEMINI_MODEL"] = "gemini-2.5-flash" if model == 1 else "gemini-2.5-pro"

    # 7) srt_max_chars（0 表示不切割字幕）
    try:
        cur_srt = int(data.get("srt_max_chars", DEFAULTS["srt_max_chars"]))
    except Exception:
        cur_srt = DEFAULTS["srt_max_chars"]  # type: ignore[index]
    # 允许用户输入 0 代表不切割；最小值设为 0
    data["srt_max_chars"] = _prompt_int("字幕每行最大字符数 (0 表示不切割)", cur_srt, 0)

    # 8) azure_tts_batch_chars（每批最大字符数，用于分批合成避免超时）
    try:
        cur_batch = int(data.get("azure_tts_batch_chars", DEFAULTS["azure_tts_batch_chars"]))
    except Exception:
        cur_batch = DEFAULTS["azure_tts_batch_chars"]  # type: ignore[index]
    data["azure_tts_batch_chars"] = _prompt_int("每批最大字符数 (azure_tts_batch_chars)", cur_batch, 100)

    # 9) azure_tts_batch_enable（是否启用分批合成）
    cur_enable = bool(data.get("azure_tts_batch_enable", DEFAULTS["azure_tts_batch_enable"]))
    data["azure_tts_batch_enable"] = _prompt_bool("是否启用分批合成 (azure_tts_batch_enable)", cur_enable)

    return data


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="编辑 tts_from_markdown 配置")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("tts_from_markdown.config.json"),
        help="配置文件路径（默认：同目录 tts_from_markdown.config.json）",
    )
    ap.add_argument("--show", action="store_true", help="仅显示配置，不写入")
    args = ap.parse_args(argv)

    cfg_path: Path = args.config
    data = load_config(cfg_path)

    # 合并默认 voice_src，保持编号可用
    vsrc = dict(DEFAULTS["voice_src"])  # type: ignore[index]
    vcfg = data.get("voice_src")
    if isinstance(vcfg, dict):
        try:
            vsrc.update({str(k): int(v) for k, v in vcfg.items()})
        except Exception:
            pass
    data["voice_src"] = vsrc

    if args.show:
        # 仅查看
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    # 交互式（默认）
    data = interactive_update(data)
    save_config(cfg_path, data)
    print(f"[OK] 已写入：{cfg_path}")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

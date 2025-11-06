from __future__ import annotations

# 轻量本地依赖包，仅供 test/video_production/tts_from_markdown 使用。
# 暴露与上游 video_production 包一致的 API 子集，避免直接依赖仓库根的包结构。

__all__ = [
    "console",
    "tts",
]


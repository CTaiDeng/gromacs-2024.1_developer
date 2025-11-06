# 从 Markdown 生成旁白（Azure TTS）

本目录提供基于 Azure 语音合成的 Markdown 旁白制作脚本与配置：

- 脚本：`test/video_production/tts_from_markdown/tts_from_markdown.py`
- 配置：`test/video_production/tts_from_markdown/tts_from_markdown.config.json`
- 配置编辑器：`test/video_production/tts_from_markdown/tts_from_markdown.config.py`

功能：
- 解析 Markdown 为朗读片段（标题/段落/列表项），可选使用 Gemini 做数学符号读法优化；
- 调用 Azure TTS 合成 WAV 音频；
- 可选生成与音频对齐的 SRT 字幕（基于书签时间戳）。

先决条件：
- 环境变量：`AZURE_SPEECH_KEY`、`AZURE_SPEECH_REGION`
- 依赖：`azure-cognitiveservices-speech`（已在项目 `requirement.txt` 中）

用法：
- 方式一（显式参数）：
  ```bash
  python test/video_production/tts_from_markdown/tts_from_markdown.py \
    --md docs/project_docs/1759075200_Git提交信息AI助手钩子.md \
    --wav out/tts/1759075200_Git提交信息AI助手钩子.wav \
    --srt out/tts/1759075200_Git提交信息AI助手钩子.srt
  ```
- 方式二（仅给 Markdown，自动推导输出名）：
  ```bash
  python test/video_production/tts_from_markdown/tts_from_markdown.py \
    docs/project_docs/kernel_reference/1752417025_从全局流变到局域刚性：论流变景观中刚性截面的逻辑守恒.md
  ```

配置（可选）：`tts_from_markdown.config.json`
- `generate_srt`：是否默认生成 SRT（命令行 `--srt` 显式传入时优先生效）
- `voice`：Azure TTS 音色（如 `zh-CN-YunyeNeural`）
- `srt_max_chars`：字幕每行最大字符数；设置为 `0` 表示不切割（恢复长字幕）。

查看/修改配置：
```bash
# 查看
python test/video_production/tts_from_markdown/tts_from_markdown.config.py --show

# 设置音色并关闭字幕
python test/video_production/tts_from_markdown/tts_from_markdown.config.py \
  --voice zh-CN-YunxiNeural --no-generate-srt
```

说明：本目录自带使用说明，已脱离 `script/README.md` 统一登记。
 
环境变量（可选）：
- `AZURE_TTS_OUTPUT_FORMAT`：优先选择输出格式以降低超时（详见 `video_production/README.md`）。
- `AZURE_TTS_DEBUG=1`：输出 SDK 事件日志。
- `AZURE_TTS_SDK_EVENTS`：控制 SDK 事件日志类别（逗号分隔）。可选：`bookmark,start,progress,done,cancel`；默认仅 `bookmark`。
- `AZURE_TTS_ATTEMPT_LOG`：控制合成失败重试的日志级别：`none`（默认，不打印）、`compact`（简要提示）、`full`（详细错误）。

超时处理策略：
- 当单批次合成超时时，脚本会自动将该批拆半重试；若子批仍失败，会继续降级为逐条调用；若单条仍失败，会进一步按约 120 字符再细分确保完成。

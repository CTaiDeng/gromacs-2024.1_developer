# 开发协议（Codex 协作）

本仓库与智能助手（Codex CLI 等）协作时的约定与提示。

## 交流语言

- 默认使用简体中文进行交流与答复。

## 提交信息自动生成

- 推荐使用脚本 `scripts/gen_commit_msg.py` 自动生成提交信息。
- Git 钩子模板：已提供 `admin/hooks/prepare-commit-msg`，可按以下方式启用：
  - 方式 A（推荐，仓库级 hooks 目录）：
    - `git config core.hooksPath admin/hooks`
  - 方式 B（复制到默认 hooks）：
    - Windows: `copy admin\hooks\prepare-commit-msg .git\hooks\prepare-commit-msg`
    - Linux/macOS: `cp admin/hooks/prepare-commit-msg .git/hooks/prepare-commit-msg && chmod +x .git/hooks/prepare-commit-msg`

### 生成策略

- 优先使用 Gemini API（需设置环境变量 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`/`GOOGLEAI_API_KEY`）。
- 若无可用 API Key，脚本会基于暂存变更本地生成简洁摘要（文件数/增删统计或增改删汇总），保证无网或无密钥场景也能生成可用的一行提交信息。

### 环境变量（Windows 示例）

- `setx GEMINI_API_KEY "<your_key>"`（设置后需重新打开终端/IDE 生效）。
- 可选：`setx GEMINI_MODEL "gemini-1.5-flash-latest"`（未设置则用默认）。

## 其他

- 如需临时跳过钩子，可使用 `git commit --no-verify`。
- 代码风格、命名等遵循既有代码库风格，变更尽量聚焦、最小化。


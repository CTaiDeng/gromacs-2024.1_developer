# 开发协议（Codex 协作）

本仓库与智能助手（如 Codex CLI）协作时的约定与提示。

## 交流语言

- 默认使用简体中文进行交流与答复。

## 优先执行（系统提示词）

- 当与 Codex 对话且代理读取到本开发协议时，优先执行：在 `README` 或 `README.md` 中补充一条“默认中文简体沟通”的说明。
- 幂等要求：若 `README`/`README.md` 已包含等效说明（含“中文/简体/沟通/交流”等关键词），则跳过，不重复写入。
- 建议文案（可直接插入到 README 顶部标题或欢迎语后的一行）：
  - `说明：本仓库默认使用简体中文进行沟通与答复。`

## 提交信息自动生成

- 推荐使用脚本 `scripts/gen_commit_msg_googleai.py` 自动生成提交信息。
- Git 钩子目录：统一使用 `.githooks`。
  - 启用：`git config core.hooksPath .githooks`
  - 钩子脚本：`.githooks/prepare-commit-msg`
  - 提交模板：`git config commit.template .githooks/.git-commit-template.txt`

### 生成策略

- 优先使用 Gemini API（需设置环境变量 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`/`GOOGLEAI_API_KEY`）。
- 若无可用 API Key，脚本会基于暂存变更本地生成简洁摘要（文件数/增删统计或增改删汇总），保证无网或无密钥场景也能生成可用的一行提交信息。

### 环境变量（Windows 示例）

- `setx GEMINI_API_KEY "<your_key>"`（设置后需重新打开终端/IDE 生效）。
- 可选：`setx GEMINI_MODEL "gemini-1.5-flash-latest"`（未设置则用默认）。

## 其他

- 如需临时跳过钩子，可使用 `git commit --no-verify`。
- 代码风格、命名等遵循既有代码库风格，变更尽量聚焦、最小化。

# my_docs 文档与知识库对齐协议

当触发“文档对齐指令”时，执行以下两项整理以保持知识库一致性与可检索性：

1) 文件名时间戳标准化
- 目标：将 `my_docs/**` 下文件名以 `<数字>_` 开头的文档，其时间戳前缀改为该文档的 Git 首次入库时间（Unix 秒 `%at`）。
- 做法：使用 `git mv` 重命名，以保留历史。

2) 标题下方日期补齐
- 目标：对 Markdown 文档，在首个 `# ` 标题下插入/规范日期行为 `日期：YYYY年MM月DD日`（来源为文件名中的时间戳前缀）。
- 若已有“日期”行，统一为上述格式；若无标题，则以文件名（去前缀与扩展名）生成一级标题并写入日期。

执行指令（文档对齐指令）
- PowerShell（Windows / PowerShell 7）：
  - `pwsh -File my_scripts/align_my_documents.ps1`
- 或 Python 直接调用：
  - `python3 my_scripts/align_my_documents.py`

脚本说明
- 路径：`my_scripts/align_my_documents.py`
- 包装器：`my_scripts/align_my_documents.ps1`
- 幂等：多次执行无副作用；仅在需要时重命名/写入。
- 重命名仅作用于以 `<数字>_` 命名的文件；未跟踪文件不会被重命名。

注意事项
- 请在干净工作区（`git status` 无未保存改动）下运行，便于审阅本次对齐的差异与提交。
- 若文件此前从未提交过，无法从 Git 获得时间戳，脚本将跳过该文件。

目录约定
- `my_docs/dev_docs`：项目开发文档（开发说明、环境、依赖、流程、安装脚本说明等）。
- `my_docs/project_docs`：项目知识库（论文笔记、方案、研究记录等，一般以“时间戳_标题.md”命名）。
- `my_project/`：存档的案例项目（归档从 `out/` 迁移的完整案例目录，内含该案例的 Markdown 报告与相关文件）。

命名与维护（my_project/**/docs）
- `my_project/**/docs` 下的文档命名与维护方式与 `my_docs/project_docs` 保持一致：
  - 文件名使用“<秒级时间戳>_标题.md”
  - 文档首行为标题（`# `），标题下一行添加日期行（`日期：YYYY年MM月DD日`）
  - 可使用对齐脚本（`my_scripts/align_my_documents.py`/`.ps1`）进行幂等对齐

临时文件清理
- 任务执行过程中如需创建临时探测/占位文件（例如 `my_project/.write_test`），在任务完成后必须删除。
- 临时文件不得提交到版本库；如确需短暂存在，应添加到 `.gitignore` 或显式在任务收尾阶段清理。

通用脚本约定
- 自定义/自动化脚本统一放在 `my_scripts/` 目录（支持 bash/ps1/py 等）。
- 新增脚本请放入 `my_scripts/`，避免存放到 `scripts/` 目录。
- 示例：`my_scripts/install_cmake_wsl.sh`。

## my_docs 职责分工（索引与规范）

- 规范与约束：`my_docs/AGENTS.md`
  - 定义 `my_docs/**` 子树的文档创建约束与维护协议（目录与命名规范、内容规范、时间戳规则与豁免、O3 注释自动插入、自动化脚本）。
  - 作为 my_docs 子树的权威规范入口。
- 索引与摘要：`my_docs/README.md`
  - 仅负责“文档路径枚举 + 摘要介绍”的索引页，不承载规范。
  - 可通过 `python3 my_scripts/gen_my_docs_index.py` 自动生成/更新；不会改动原文档内容。
  - 追加说明：`my_docs/README.md` 索引中的摘要为完整呈现（不截断）。
- 对齐脚本：
  - `python3 my_scripts/align_my_documents.py` 或 `pwsh -File my_scripts/align_my_documents.ps1`
  - 功能：按 Git 首次入库时间戳重命名 `<数字>_*.md`、在标题下补齐/规范 `日期：YYYY年MM月DD日`；当正文命中 “O3理论/O3元数学理论/主纤维丛版广义非交换李代数/PFB-GNLA” 任一关键词时，在日期下方自动插入统一注释（幂等去重）。


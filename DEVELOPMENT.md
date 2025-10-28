# 开发协同（重点约束补充）—统一 UTF-8 + LF

本文明确本仓库在“文档与源码”的编码/换行规范：所有生成与编辑一律使用 UTF‑8（无 BOM）+ LF。该条为最高优先；当与上游风格冲突时，以本条为准（可在 README 顶部保留“非官方派生”声明对冲）。更多总纲见根目录 `AGENTS.md`。

## 适用范围
- 项目根部文档：`README`、`README.md`、`CITATION.cff` 等
- 仓库内所有 Markdown/纯文本文档：`*.md`、`*.txt`、`*.json`、`*.yml`/`*.yaml`、`*.toml`、`*.ini`
- 项目知识库：
  - `my_docs/project_docs/**`
  - `my_project/gmx_split_20250924_011827/docs/**`

说明：为覆盖多平台/多语言工具链，统一 LF 便于差异对比、脚本处理与持续集成。

## 基线与仓库配置
- 根目录 `.gitattributes` 已固定范围：`text eol=lf working-tree-encoding=UTF-8`；常见二进制扩展已标记为 `binary`，避免文本化写入。

## 脚本写入规范（强制）
- Python 文本写入统一使用：
  - `open(path, 'w', encoding='utf-8', newline='\n')`
  - 如使用 Pathlib，显式 `open(..., newline='\n')`；不推荐直接 `Path.write_text()`（Windows 上可能写出 CRLF）。
- 生成文本在写盘前统一替换：`text = text.replace('\r\n', '\n')`。
- Markdown/JSON/CMake/脚本等文本类文件均以 LF 结尾；必要时补齐结尾 `\n`。

## 迁移与校验（可选手动执行）
- 对既有文件批量对齐规范（不会更改二进制）：
  - `git add --renormalize .`
  - `git commit -m "chore: renormalize docs to UTF-8+LF"`
- 编辑器层面建议添加 `.editorconfig`：
  - `[*] charset = utf-8`
  - `[*.md] end_of_line = lf`、`[*.txt] end_of_line = lf`

## 批量修复与验证脚本
- 使用根目录脚本执行统一修复/校验：
  - `pwsh ./convert_to_utf8_lf.ps1 -ConfigPath convert_to_utf8_lf_config_whitelist.json`
- 行为说明：
  - 按白名单目录/文件遍历，二进制扩展自动跳过，文本扩展参与转换。
  - 去除 UTF‑8 BOM，统一 CRLF/CR 为 LF，并以 UTF‑8（无 BOM）落盘。
  - 输出 `changed=0` 表示当前工作区已满足 UTF‑8+LF 规范。

## 与 AGENTS 的关系
- 本文为开发协作的“执行细则”，与根目录 `AGENTS.md` 的“最高规范”一致；若出现冲突，以 `AGENTS.md` 为准。

## 时间与时区统一（说明）
- 与“时间戳/头部格式”相关的自动化，统一使用 Asia/Shanghai（UTC+8）。
- 建议：
  - Python 使用 `zoneinfo.ZoneInfo('Asia/Shanghai')`，避免跨平台时区差异。
  - CI/本地可设置 `TZ=Asia/Shanghai`，保证一致性。


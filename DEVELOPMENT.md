# 开发协议（最高优先级补充）：文档一律 UTF-8 + LF

本补充条款确立：仓库内“文档类文件”（README/CITATION.cff/Markdown/纯文本/项目知识库与说明性文档）一律采用 UTF-8 编码与 LF 换行。本条为最高优先级补充；如与其他说明发生冲突，以本条为准（建议后续并入根 `AGENTS.md` 以长期固化）。

## 适用范围
- 顶层文档：`README`、`README.md`、`CITATION.cff`
- 全仓库的 Markdown/纯文本：`*.md`、`*.txt`
- 项目知识库与文档目录：
  - `my_docs/project_docs/**`
  - `my_project/gmx_split_20250924_011827/docs/**`

（说明）为避免影响跨平台构建，源码与脚本文件不在本强制规则内，仍按各自既有约定（通常 LF）维护；如需扩展到更多文件类型，请在评估构建/工具链兼容性后再行更新。

## 工程落地（已完成）
- 在根目录新增 `.gitattributes`，对上述范围启用：`text eol=lf working-tree-encoding=UTF-8`；并显式标记常见二进制为 `binary`，避免任何换行处理。

## 迁移与校验（可选手动执行）
- 将现有文件按规则重规范化（可能产生大量行尾更新，请在专用提交中完成）：
  - `git add --renormalize .`
  - `git commit -m "chore: renormalize docs to UTF-8+LF"`
- 编辑器层面可配合 `.editorconfig`（非必需）：
  - `[*] charset = utf-8`
  - `[*.md] end_of_line = lf`、`[*.txt] end_of_line = lf`

## 与现有协作规范的关系
- 本条款为对现在的协作规范（AGENTS 指南）的“最高优先级补充”。
- 不改变“知识库只读/白名单”等既有合规约束；本变更仅通过 Git 属性控制检出/提交时的行尾与编码，不直接修改知识库内容。


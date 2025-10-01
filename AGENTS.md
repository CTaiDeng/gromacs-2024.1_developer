# AGENTS 指南（派生约束｜最高规范）

本文件为本仓库（GROMACS 派生版）的最高级协作规范（“最高宪法”）。其约束适用于本目录树内的所有工作（人类与智能助手）。更深层目录如另有 `AGENTS.md`，以更深层为准覆盖本文件冲突处。

## 沟通与语言
- 默认使用简体中文沟通与答复（README 已注明）。

## 派生与合规模块（必须遵守）
- 非官方声明：本仓库为 GROMACS 的非官方派生版，与上游无隶属或担保关系。
- 许可遵循：保留上游 `COPYING`（LGPL-2.1）；除另有声明外，新增/修改部分遵循 LGPL-2.1。
- 文件要求：根目录必须存在 `COPYING`、`CITATION.cff`、`NOTICE`、`README`；`README` 顶部需包含：
  - 沟通语言说明（中文或等价英文；中文包含“中文/简体/沟通/交流”等关键词其一；英文等价如 “Simplified Chinese”/“communicates by default” 等）
  - 派生/非官方声明（中文或等价英文；中文包含“派生/衍生/非官方”等关键词其一；英文等价如 “non‑official”、“unofficial”、“derivative/fork” 等）
- 禁止“官方”误导：文档与提交信息不得宣称“官方/official”身份；若难以避免（如上游原文），需在显著位置（README 顶部）加以非官方声明进行冲抵。
- 二进制分发合规：若对外分发二进制，需满足 LGPL 的“可重新链接”义务（推荐动态链接或同时分发可重链接目标文件）。

## Git 提交流程约束（强制）
- 已配置 `.githooks/pre-commit` 在提交前执行程序化合规检查：
  1) 关键文件存在性与 README 顶部声明；
  2) 禁止“官方”误导性语句；
  3) `NOTICE` 中含派生与许可说明；
  4) 对暂存改动进行启发式扫描（例如新增二进制、疑似版权移除）。
- 可选 LLM 辅助审查：如设置 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY/GOOGLEAI_API_KEY`，将附加触发一次大模型审查（仅作加分与风险提示，不泄露秘钥内容；无 Key 时自动跳过）。
- 任何检查未通过将阻断提交；可使用 `git commit --no-verify` 临时跳过，但不建议在长期分支中使用。

## 开发者操作建议
- 启用钩子与模板：
  - `git config core.hooksPath .githooks`
  - `git config commit.template .githooks/.git-commit-template.txt`
- 提交信息建议使用 `my_scripts/gen_commit_msg_googleai.py` 自动生成（无 Key 也可离线摘要）。

## 文档与目录约定（知识库/外部参考）
- `my_docs/project_docs`：项目知识库（受仓库脚本维护，允许写入）。
- `my_docs/project_docs/kernel_reference`：外部知识参考（只读）。约束如下：
  - 不纳入 `my_scripts/gen_my_docs_index.py` 的索引输出；
  - 不参与写入型自动化脚本（如 `align_my_documents.py`、`ensure_summaries.py`）的遍历与改写；
  - 提交信息生成脚本 `my_scripts/gen_commit_msg_googleai.py` 在生成摘要时忽略该路径的改动；
  - 统一由 `my_scripts/docs_whitelist.json` 管理白名单/排除项，默认已包含该排除路径。

- AI 助手知识库引用范围：当未明确指定路径而笼统提及“知识库”时，默认包含 `my_docs/project_docs` 及其全部递归子目录（包含只读子目录 `kernel_reference`）；但仍须遵守其只读属性与上述索引/写入排除规则。如需临时排除此子目录，请在指令或脚本中显式注明。

## 例外与裁量
- 若确有合理理由（例如上游文档保留），可在 README 顶部保持非官方声明以对冲相关表述。
- 对合规检查有误报时，优先提交修正 PR 至 `my_scripts/check_derivation_guard.py`；紧急情况下可临时 `--no-verify`，并在后续补齐整改。

# 开发协议（my_docs 子树）

- 本协议适用于 `my_docs/**` 目录及其子目录。
- `my_docs/README.md` 为知识库“总文档”，作为索引与规范入口；新增/调整规范时需同步更新。

## 目录与命名规范
- `my_docs/dev_docs`：开发类文档。命名使用“英文大写，以下划线分隔”，扩展名 `.md`（示例：`BUILD_GUIDE.md`）。
- `my_docs/project_docs`：项目论文/知识库文档。命名格式：`{时间戳}_{文件名}.md`
  - `{时间戳}`：10 位 Unix 秒级时间戳，表示首次入库时间；
  - `{文件名}`：等于文档标题，建议仅使用中英文、数字、下划线或短横线；
- 详细规则：`my_docs/dev_docs/DOC_NAMING_RULES.md`

## 内容规范
- Markdown 文档：
  1) 第一行为一级标题：`# {标题}`
  2) 第二行为日期：`日期：YYYY年MM月DD日`（由文件名时间戳换算；脚本可自动补齐）

## 对齐与自动化
- PowerShell：`pwsh -File my_scripts/align_my_documents.ps1`
- Python：`python3 my_scripts/align_my_documents.py`
- 脚本功能：
  - 基于 Git 首次入库时间戳重命名 `<数字>_*.md` 文件；
  - 在标题下方补齐/规范日期行；
  - 命中文内关键词（“O3理论/…/PFB-GNLA”）时，在日期下方自动插入统一注释（幂等去重）。


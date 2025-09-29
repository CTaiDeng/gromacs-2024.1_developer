# DOC_NAMING_RULES

本文件规定 `my_docs/dev_docs` 与 `my_docs/project_docs` 的文档命名与内容规范，并作为新建文档的唯一依据。

## project_docs（项目论文/研究文档）

- 存放位置：`my_docs/project_docs`
- 命名格式：`{时间戳}_{文件名}.md`
  - `{时间戳}`：10 位 Unix 秒级时间戳（例如 `1758809069`），表示文档首次入库时间；
  - `{文件名}`：等于文档标题（Title），建议仅包含中英文、数字、下划线 `_` 或短横线 `-`；
- 文件内容：
  1. 第一行：`# {标题}`（Markdown 一级标题，与 `{文件名}` 对齐）
  2. 第二行：`日期：yyyy年MM月dd日`（由文件名时间戳换算的本地日期，脚本可自动补齐）

示例：

```
my_docs/project_docs/1758809069_示例标题.md

# 示例标题
日期：2025年02月25日

（正文）
```

## dev_docs（开发类说明文档）

- 存放位置：`my_docs/dev_docs`
- 命名格式：英文大写，单词以下划线分隔，扩展名为 `.md`
  - 例如：`BUILD_GUIDE.md`、`ENV_SETUP.md`、`DOC_NAMING_RULES.md`
- 建议内容：
  - 第一行：`# {标题}`；
  - 第二行：可选 `日期：yyyy年MM月dd日`；

## 对齐与自动化

- 运行脚本自动对齐命名与日期：
  - PowerShell：`pwsh -File my_scripts/align_my_documents.ps1`
  - Python：`python3 my_scripts/align_my_documents.py`
- 幂等：脚本可多次运行，仅在需要时执行重命名/写入。
- 注意：对齐前请确保工作区干净（`git status` 无未提交改动）。

## 附：O3 相关注释自动插入

当文档正文出现以下任意关键词时：
- “O3理论”、“O3元数学理论”、“主纤维丛版广义非交换李代数”、“PFB-GNLA”。

脚本会在“日期：YYYY年MM月DD日”下一行自动插入统一注释（幂等去重）：

#### ***注：“O3理论/O3元数学理论/主纤维丛版广义非交换李代数(PFB-GNLA)”相关理论参见： [作者（GaoZheng）网盘分享](https://drive.google.com/drive/folders/1lrgVtvhEq8cNal0Aa0AjeCNQaRA8WERu?usp=sharing) 或 [作者（GaoZheng）主页](https://mymetamathematics.blogspot.com)***

适用范围：
- `my_docs/**` 下的 Markdown 文档
- `my_project/**/docs/**` 下的 Markdown 文档

使用方法：
- PowerShell：`pwsh -File my_scripts/align_my_documents.ps1`
- Python：`python3 my_scripts/align_my_documents.py`

注意：若文档缺少标题或日期行，脚本会先补齐后再插入注释；若已有等效注释，脚本会规范位置并去重。

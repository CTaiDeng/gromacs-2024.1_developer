# 开发协议（my_docs 子树）
日期：2025年09月29日
#### ***注：“O3理论/O3元数学理论/主纤维丛版广义非交换李代数(PFB-GNLA)”相关理论参见： [作者（GaoZheng）网盘分享](https://drive.google.com/drive/folders/1lrgVtvhEq8cNal0Aa0AjeCNQaRA8WERu?usp=sharing) 或 [作者（GaoZheng）主页](https://mymetamathematics.blogspot.com)***

## 职责分工
- `my_docs/AGENTS.md`：知识库文档的“创建约束与维护协议”（权威规范）。
- `my_docs/README.md`：仅负责“文档路径枚举 + 摘要介绍”的索引页；通过 `my_scripts/gen_my_docs_index.py` 自动生成/更新；索引中的“摘要”为文档内“## 摘要”段的完整呈现（不截断）。

## 目录与命名规范
- `my_docs/dev_docs`：开发类文档。命名使用“英文大写，以下划线分隔”，扩展名 `.md`（示例：`BUILD_GUIDE.md`、`ENV_SETUP.md`）。
- `my_docs/project_docs`：项目论文/知识库文档。命名格式：`{时间戳}_{文件名}.md`
  - `{时间戳}`：10 位 Unix 秒级时间戳，表示首次入库时间；
  - `{文件名}`：等于文档标题，建议仅使用中英文、数字、下划线或短横线；

## 内容规范
- Markdown 文档：
  1) 第一行为一级标题：`# {标题}`
  2) 第二行为日期：`日期：YYYY年MM月DD日`（由文件名时间戳换算；脚本可自动补齐）

## 时间戳规则与豁免
- 新增文档：时间戳一律取“Git 首次入库时间戳（Unix 秒）”，脚本会在重命名时对齐，并将标题下第二行的日期同步为该时间戳对应的本地日期。
- 既有专项文档（豁免重命名，日期取文件名时间戳）：
  - `my_docs/project_docs/1759156359_药理学的代数形式化：一个作用于基因组的算子体系.md`
  - `my_docs/project_docs/1759156360_论药理学干预的代数结构：药理基因组算子幺半群（PGOM）的形式化.md`
  - `my_docs/project_docs/1759156361_从物理实在到代数干预：论PFB-GNLA向PGOM的平庸化退化.md`
  - `my_docs/project_docs/1759156362_O3理论下的本体论退化：从流变实在到刚性干预——论PFB-GNLA向PGOM的逻辑截面投影.md`
  - `my_docs/project_docs/1759156363_论O3理论的自相似动力学：作为递归性子系统 GRL 路径积分的 PGOM.md`
  - `my_docs/project_docs/1759156364_PGOM作为元理论：生命科学各分支的算子幺半群构造.md`
  - `my_docs/project_docs/1759156365_论六大生命科学代数结构的幺半群完备性：基于元素与子集拓扑的双重视角分析.md`
  - `my_docs/project_docs/1759156366_LBOPB的全息宇宙：生命系统在六个不同观测参考系下的完整论述.md`
  - `my_docs/project_docs/1759156367_同一路径，六重宇宙：论HIV感染与治疗的GRL路径积分在LBOPB六大参考系下的全息解释.md`
  - `my_docs/project_docs/1759156368_计算化学的O3理论重构：作为PDEM幺半群内在动力学的多尺度计算引擎.md`

说明：上述文件的“日期”行直接使用其文件名中的 10 位时间戳换算的日期；其他新文档遵循“入库时间戳”规则。

## O3 相关注释自动插入
- 当文档正文出现以下任意关键词时：
  - “O3理论”、“O3元数学理论”、“主纤维丛版广义非交换李代数”、“PFB-GNLA”。
- 对齐脚本会在“日期：YYYY年MM月DD日”下一行自动插入统一注释（幂等去重）：

#### ***注：“O3理论/O3元数学理论/主纤维丛版广义非交换李代数(PFB-GNLA)”相关理论参见： [作者（GaoZheng）网盘分享](https://drive.google.com/drive/folders/1lrgVtvhEq8cNal0Aa0AjeCNQaRA8WERu?usp=sharing) 或 [作者（GaoZheng）主页](https://mymetamathematics.blogspot.com)***

## 自动化脚本
- 对齐脚本：
  - PowerShell：`pwsh -File my_scripts/align_my_documents.ps1`
  - Python：`python3 my_scripts/align_my_documents.py`
  - 功能：基于 Git 首次入库时间戳重命名 `<数字>_*.md`，在标题下方补齐/规范日期行，并在命中 O3 关键词时插入统一注释。
- 索引生成脚本：
  - Python：`python3 my_scripts/gen_my_docs_index.py`
  - 作用：生成/更新 `my_docs/README.md`，枚举文档路径并抽取“## 摘要”段的完整文本（不截断），不改变任何原文档内容。

## 摘要规范与自动生成
- 要求：`my_docs/dev_docs` 与 `my_docs/project_docs` 下所有 Markdown 文档均应包含“## 摘要”段。
  - 位置：紧随“日期：YYYY年MM月DD日”之后；若存在 O3 注释，则位于 O3 注释之后。
  - 内容：建议 180–240 字的简要说明（脚本会从正文首段或“摘要/简介”附近段落抽取）。
- 自动生成：
  - Python：`python3 my_scripts/ensure_summaries.py`
  - 行为：为缺少“摘要”段的文档自动插入“## 摘要”与生成的摘要文本；幂等，重复执行不会重复插入。


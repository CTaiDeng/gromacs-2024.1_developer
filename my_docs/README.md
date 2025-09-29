# my_docs 知识库总文档

说明：`my_docs` 目录用于存放本项目的知识库与文档索引，包含开发类文档（dev_docs）与项目论文/研究记录（project_docs）。本文档作为总入口与规范索引。

## 目录结构
- `my_docs/dev_docs`：项目开发类文档（开发说明、环境、依赖、流程、脚本说明等）。
- `my_docs/project_docs`：项目知识库/论文与研究记录（通常以“时间戳_标题.md”命名）。
- 规范详情见：`my_docs/dev_docs/DOC_NAMING_RULES.md`

## 命名与内容规范（摘要）
- project_docs
  - 命名：`{时间戳}_{文件名}.md`
    - `{时间戳}`：10 位 Unix 秒级时间戳（如 `1758809069`），表示文档入库时间。
    - `{文件名}`：等于文档标题，建议仅用中英文、数字、下划线或短横线。
  - 内容：
    1. 第一行：`# {标题}`（Markdown 一级标题）
    2. 第二行：`日期：yyyy年MM月dd日`（由文件名时间戳换算本地日期）

- dev_docs
  - 命名：英文全大写，单词以下划线分隔，扩展名为 `.md`（示例：`BUILD_GUIDE.md`、`ENV_SETUP.md`）。

## 文档对齐脚本
- Python：`python3 my_scripts/align_my_documents.py`
- PowerShell：`pwsh -File my_scripts/align_my_documents.ps1`

对齐内容：
- 将 `my_docs/**` 与 `my_project/**/docs/**` 中以 `<数字>_` 开头的文件重命名为“首个 Git 入库时间戳 + 原名”；
- 对 Markdown 文档，在首个 `# ` 标题下补齐/规范日期行为 `日期：YYYY年MM月DD日`；
- 当文档正文出现“O3理论 / O3元数学理论 / 主纤维丛版广义非交换李代数 / PFB-GNLA”任一关键词时，在日期下方自动插入统一注释引用（幂等去重）。

更多说明：`my_docs/dev_docs/知识库对齐-O3注释规则.md`

## 时间戳规则与豁免

- 新增文档：时间戳一律取“Git 首次入库时间戳（Unix 秒）”，脚本会在重命名时对齐，并将标题下第二行的日期同步为该时间戳对应的本地日期（`日期：YYYY年MM月DD日`）。
- 既有专项文档（豁免重命名，日期取文件名时间戳）：
  - `my_docs/project_docs/1752417159_药理学的代数形式化：一个作用于基因组的算子体系.md`
  - `my_docs/project_docs/1752417160_论药理学干预的代数结构：药理基因组算子幺半群（PGOM）的形式化.md`
  - `my_docs/project_docs/1752417161_从物理实在到代数干预：论PFB-GNLA向PGOM的平庸化退化.md`
  - `my_docs/project_docs/1752417162_O3理论下的本体论退化：从流变实在到刚性干预——论PFB-GNLA向PGOM的逻辑截面投影.md`
  - `my_docs/project_docs/1752417163_论O3理论的自相似动力学：作为递归性子系统 GRL 路径积分的 PGOM.md`
  - `my_docs/project_docs/1752417164_PGOM作为元理论：生命科学各分支的算子幺半群构造.md`
  - `my_docs/project_docs/1752417165_论六大生命科学代数结构的幺半群完备性：基于元素与子集拓扑的双重视角分析.md`
  - `my_docs/project_docs/1752417166_LBOPB的全息宇宙：生命系统在六个不同观测参考系下的完整论述.md`
  - `my_docs/project_docs/1752417167_同一路径，六重宇宙：论HIV感染与治疗的GRL路径积分在LBOPB六大参考系下的全息解释.md`
  - `my_docs/project_docs/1752417168_计算化学的O3理论重构：作为PDEM幺半群内在动力学的多尺度计算引擎.md`

说明：上述文件的“日期”行将直接使用其文件名中的 10 位时间戳换算的日期；其他新文档遵循“入库时间戳”规则。

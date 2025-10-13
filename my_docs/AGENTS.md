# AGENTS（my_docs 文档规范）

- 作者：GaoZheng
- 日期：2025-10-13

#### ***注：O3 理论/O3 元理论/多维宇宙生成论（PFB-GNLA）相关参考请见 [作者 GaoZheng 的资料库](https://drive.google.com/drive/folders/1lrgVtvhEq8cNal0Aa0AjeCNQaRA8WERu?usp=sharing) 与 [作者 GaoZheng 主页](https://mymetamathematics.blogspot.com)***

## 目标与范围
- 目录范围：`my_docs` 下的 `dev_docs` 与 `project_docs`。
- 目标：统一文档结构与命名，支持自动化脚本处理，提升一致性与可读性。

## 文档结构规范（MUST）
- 固定头部，顺序不可缺：
  1) `# {标题}`（H1）
  2) `- 作者：{姓名}`（默认 GaoZheng）
  3) `- 日期：YYYY-MM-DD`（ISO 日期）
  4) O3 参考提示块：`#### ***注：...***`
  5) `## 摘要`（建议 180–240 字，脚本可自动生成/归一化）
- 头部各块之间与正文之间，保持单空行。
 - 摘要段落结束后必须加入水平分隔线 `---`；
   - 摘要内容与 `---` 之间仅保留 1 个空行；
   - `---` 与后续正文之间仅保留 1 个空行。

## 摘要处理优先级（MUST/SHOULD）
- [MUST] 若原文已有摘要，标准化为 `## 摘要` 标题，不改动内容。
- [SHOULD] 若存在“#### 摘要/摘要：”等非标准写法，脚本自动合并为 `## 摘要`。
- [SHOULD] 若缺失摘要，脚本自动生成一段简要摘要（约 180–240 字）。

## 命名与时间戳（MUST/SHOULD）
- [MUST] 文件名：`{10位Unix时间戳}_{中文标题}.md`，10 位前缀同时作为文档 ID。
- [MUST] `project_docs` 下若出现同秒冲突，按 `-1s` 依次回退直至唯一（由脚本保障）。
- [MUST] H1 标题中不得保留“时间戳_”前缀（脚本自动去除）。
- [SHOULD] 提交前由脚本自动对齐文件日期行与 ID 来源，保持稳定。

## 索引与展示（MUST/SHOULD）
- [MUST] `my_docs/README.md` 展示 `project_docs` 文件，默认按 10 位时间戳升序。
- [SHOULD] 摘要过长时在索引中换行展示；自动化由 `gen_my_docs_index.py` 完成。

## 临时文件管理（MUST）
- 临时文件示例：`tmp_gen_idx_before.txt`、`tmp_*.txt`、`*.tmp`、`*.bak` 等。
- 约束：
  - 不纳入提交；开发时可临时使用，用后删除。
  - 若需长期保留，应转为正式文档（含头部/结构/摘要）。
  - `.gitignore` 保持通配排除（如 `tmp_*`, `*.tmp`, `*.bak`）。

## 自动化脚本
- `align_my_documents.py`
  - 统一文件名时间戳唯一性（`project_docs`）。
  - 规范化头部：作者、ISO 日期、O3 提示、去除 H1 时间戳前缀。
  - 清理重复的“### 标题/#### 摘要”等，合并为标准结构；
  - 强制“摘要后 → 空行 → `---` → 空行 → 正文”的分隔线规范（自动插入或修复）。
- `gen_my_docs_index.py`
  - 生成/刷新 `my_docs/README.md`（UTF-8 with BOM），兼容 Windows 控制台。
- `ensure_summaries.py`
  - 在缺失时补全 `## 摘要` 区块。

## 目录与权限
- `my_docs/project_docs`：项目知识库（可写）。
- `my_docs/project_docs/kernel_reference`：外部参考（只读，排除写入型脚本遍历）。

## 编码规范（重点）
- 为兼容 Windows 传统控制台编码，`my_docs/AGENTS.md` 与 `my_docs/README.md` 使用 UTF-8 with BOM 保存。
- 其他 Markdown 文件建议使用 UTF-8；如需在控制台直接阅读，也可手动转换为 UTF-8 with BOM。

## 写入协议（重要）
- 当遇到如下指令时：
  - “将下文基于文档命名规范写入 my_docs\project_docs，先保持内容不变写入创建的文件，然后应用全部文档规范调整”
- 协议解释：
  - “保持内容不变”不包括对“标题（H1）与摘要（## 摘要）”的规范化调整。
  - 允许并优先执行：
    - 标题归一（去除时间戳前缀、确保置顶 H1）；
    - 摘要标准化为 `## 摘要`，并去除其标题后的空行；
    - 头部顺序与空行规范（H1 下一空行；作者与日期相邻无空行；日期与 O3 提示之间恰一空行）。

以上规范在仓库脚本持续演进下保持兼容与更新。





# lbopb 子目录协作规范（子 AGENTS）

本文件为子目录 AGENTS，作用域为 `lbopb/` 及其全部递归子目录（包含 `lbopb/lbopb_examples/`）。如与仓库根级 `AGENTS.md` 存在冲突，以本文件对本作用域内的条目为准。

## 许可与头注
- 源码文件统一采用 GPL-3.0-only 许可头注（SPDX）。
- 版权与声明遵循仓库根级 AGENTS.md 的“源代码头注规范（MUST）”。
- 建议使用仓库提供的脚本进行自动化对齐，保持一致性。

## 注释语言
- 源码与包文档字符串默认使用简体中文，必要时可附英文补充。
- 包级文件需指明：模块职责、主要算子/指标、知识库文档引用路径。

## 联络与幂集（配置与产物）
- 联络与幂集配置统一收敛于 `lbopb/src/operator_crosswalk.json`：
  - `basic_ops`：各模块基本算子与语义标签；
  - `crosswalk_by_tag`：标签到跨模块基本算子的映射；
  - `canonical_packages`/`canonical_packages_desc`：规范化算子包与说明；
  - `powersets`：幂集（约束/常用家族/生成器及说明）。
- 预览文档通过脚本自动生成：
  - `python -m lbopb.src.gen_operator_crosswalk_md`
  - 产物：`lbopb/src/operator_crosswalk.md`（请勿手改）。

### 文档样式规范（operator_crosswalk.md）
- “说明”段落使用四级标题样式：`#### 说明：`（而非 Markdown 引用 `>`）。
- 生成器展示包含：
  - “链式模式”表格；
  - “示例（前N条）”表格（默认 N=5）。
  - 如需调整 N 值，请修改脚本内 `SAMPLE_N` 常量后重新生成。

## 头注同步脚本
- 提供 `lbopb/scripts/sync_headers.py` 用于本目录头注检查与修复：
  - 移除不符合本子目录规范的头注；
  - 确保 SPDX 与版权行齐备。

## 贡献流程（建议）
1. 修改或扩展 `lbopb/src/operator_crosswalk.json`；
2. 运行 `python -m lbopb.src.gen_operator_crosswalk_md` 同步 Markdown；
3. 运行 `python lbopb/scripts/sync_headers.py` 检查头注；
4. 添加/更新示例至 `lbopb/lbopb_examples/`（可选）。


# lbopb 开发协议（注释与联络）

本协议适用于 `lbopb/`、`lbopb_examples/` 目录：

## 许可与头注

- 统一采用 GPL-3.0-only 许可头注（SPDX）。
- 版权行仅包含：
  - `Copyright (C) 2025 GaoZheng`
- 不包含 `Copyright (C) 2010- The GROMACS Authors`。
- 若需要在其它目录继承上游头注，请遵循仓库根级规范（不影响本目录协议）。

## 注释语言

- 源码与包文档字符串默认使用简体中文，必要时可附英文补充。
- 包级文件需指明：模块职责、主要算子/指标、知识库文档引用路径。

## 联络与幂集

- 联络与幂集配置统一收敛至 `lbopb/src/operator_crosswalk.json`：
  - `basic_ops`：各模块基本算子 → 语义标签；
  - `crosswalk_by_tag`：标签 → 跨模块基本算子；
  - `canonical_packages`/`canonical_packages_desc`：规范化算子包与说明；
  - `powersets`：幂集（约束/常用家族/常用生成器及说明）。
- 预览文档通过脚本自动生成：
  - `python -m lbopb.src.gen_operator_crosswalk_md`
  - 产物：`lbopb/src/operator_crosswalk.md`（请勿手改）。

### 文档样式规范（operator_crosswalk.md）

- “说明”段落使用四级标题样式：`#### 说明：`（而非 Markdown 引用 `>`）。
- 生成器展示包含：
  - “链式模式”表格；
  - “示例（前N条）”表格（默认 N=5）。
  - 需要调整 N 值时，请修改脚本内 `SAMPLE_N` 常量并重新生成。

## 头注同步脚本

- 提供 `lbopb/scripts/sync_headers.py` 用于本目录头注检查与修复：
  - 移除上游版权行；
  - 确保 SPDX 与本地版权行齐备。

## 贡献流程（建议）

1. 修改或扩展 `operator_crosswalk.json`；
2. 运行 `python -m lbopb.src.gen_operator_crosswalk_md` 同步 Markdown；
3. 运行 `python lbopb/scripts/sync_headers.py` 检查头注；
4. 添加/更新示例至 `lbopb_examples/`（可选）。

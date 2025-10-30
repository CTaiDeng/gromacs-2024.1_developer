# git_search：Git 历史全文搜索工具

`git_search` 是一个仓库内置的 Python 工具，用于在整个 Git 历史进行全文搜索，可同时搜索“文件内容快照（跨所有提交）”与“提交信息（subject/body）”，并将报告输出到 `out/git_search`（Markdown 与 JSON）。

- 依赖：已安装 Git、Python 3.8+（仓库自带运行，无需单独安装包）
- 输出结构：
  - 目录：`out/git_search/report_<时间戳秒>/`
  - 文件：`out/git_search/report_<时间戳秒>/report_<时间戳秒>.md`
  - 文件：`out/git_search/report_<时间戳秒>/report_<时间戳秒>.json`
- 适用场景：
  - 回溯关键符号/常量/接口在历史中的演化
  - 甄别某段文案或许可证文本在不同版本中的分布
  - 基于提交信息的变更主题检索

## 快速开始

- 同时搜索“文件内容 + 提交信息”：

  ```bash
  python -m git_search "关键字"
  ```

- 仅搜索“提交信息”（较快）：

  ```bash
  python -m git_search "关键字" --no-blobs
  ```

- 仅搜索“文件内容”（遍历所有提交快照，可能耗时）：

  ```bash
  python -m git_search "关键字" --no-messages
  ```

- 常见示例：

  ```bash
  # 提交信息中查 5 条示例
  python -m git_search "GROMACS" --no-blobs --limit-messages 5

  # 从 2025-10-01 起，仅在 Markdown 中查文件内容
  python -m git_search "GPL-3.0" --no-messages --since "2025-10-01" --path-glob "*.md"

  # 正则 + 忽略大小写 + 限定 src 目录
  python -m git_search "pattern.*here" --regex -i --path-glob "src/**"
  ```

## 参数说明

- `query`：要搜索的字符串；默认按字面量匹配。
- `--regex`：将查询解释为正则表达式（默认字面量）。
- `-i, --ignore-case`：大小写不敏感匹配。
- `--no-blobs`：不搜索文件内容，仅搜索提交信息。
- `--no-messages`：不搜索提交信息，仅搜索文件内容。
- `--since <date>`：仅搜索此时间之后的提交（Git 支持的任意日期格式）。
- `--until <date>`：仅搜索此时间之前的提交。
- `--author <pattern>`：按作者过滤“提交信息”搜索。
- `--path-glob <glob>`：限制“文件内容”搜索的路径通配，可多次指定；例：`--path-glob "*.c" --path-glob "src/**"`。
- `--limit-blobs <n>`：文件内容匹配最大条数（-1 为不限）。
- `--limit-messages <n>`：提交信息匹配最大条数（-1 为不限）。
- `--output-dir <dir>`：输出目录（默认 `out/git_search`）。
- `--output-prefix <name>`：输出文件名前缀（默认 `report`）。

## 输出内容

- Markdown 报告：适合快速浏览。包含：
  - 查询参数概要（字面量/正则、是否忽略大小写、时间/作者/路径过滤）
  - 文件内容匹配统计：匹配行数、唯一提交数、唯一文件数；示例最多 100 条（每条行内容在 MD 中最多显示 200 字符）
  - 提交信息匹配统计：匹配提交数；示例最多 100 条
  - 完整 JSON 文件路径
- JSON 报告：结构化明细，适合后处理。包含所有匹配结果（受 `--limit-*` 约束），并附计数信息。

## 性能与实现要点

- “文件内容”搜索通过 `git rev-list --all` 获取提交集合，并分批对每批提交使用 `git grep -n -I`；历史庞大时会显著耗时，建议配合 `--since/--until/--path-glob` 限定范围。
- “提交信息”搜索通过 `git log --all --grep` 实现，通常更快。
- `--regex`：
  - 对“文件内容”搜索使用 `git grep -E`；对“提交信息”搜索传给 `git log --grep`。
  - 若不加 `--regex`，则“文件内容”使用 `-F` 字面量匹配；“提交信息”将对查询做转义后再以 `-E` 提交给 `--grep`。
- 输出文件一律 UTF‑8（无 BOM）+ LF 行尾；Windows 控制台可能出现中文显示为乱码，建议在编辑器中打开报告文件查看。

## 作为 Python 模块调用

```python
from git_search import search_git_history

md_path, json_path = search_git_history(
    "关键词",
    include_blobs=False,      # 仅提交信息
    limit_messages=10,
)
print("MD:", md_path)
print("JSON:", json_path)
```

返回值：二元组 `(markdown_report_path, json_report_path)`，均为 `pathlib.Path`。

## 已知限制

- 对每个匹配行，Markdown 报告会对“行内容”做 200 字符的显示截断；JSON 不截断。
- 二进制文件已通过 `git grep -I` 排除；极个别文本编码异常的文件可能在历史中被 Git 视为二进制而跳过。
- 若仓库包含大量子模块或大型二进制历史，建议结合路径与时间过滤，以避免长时间扫描。

## 许可

本工具与仓库一致，遵循 GPL-3.0；上游版权与致谢保留。

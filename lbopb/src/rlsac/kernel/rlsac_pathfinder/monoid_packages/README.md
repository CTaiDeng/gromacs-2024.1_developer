该目录用于存放各幺半群在训练结束后，从每次运行的 `debug_dataset.json` 中筛选出的“正确（label=1）”算子包。

规范：
- 纳入策略：仅包含 label=1 的样本，按序列去重；同一序列多次出现时保留分数更高者。
- 排序规则：score(desc) -> length(asc) -> sequence（字典序）。
- 评分定义：score = delta_risk - cost_lambda * cost。
- 文件命名：沿用 `<domain>_operator_packages.json`，但统一归档在本目录下。

写入方式：
- 由 `lbopb.src.rlsac.kernel.rlsac_pathfinder.package_store.ingest_from_debug_dataset()` 自动创建/更新。

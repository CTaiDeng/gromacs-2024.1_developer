# debug_dataset.stats.json 配套说明（自动生成）

- 生成时间：2025-10-26 11:24:53
- 样本总数：19

## 域分布（domains）
- pem: 19

## 标签统计（labels）
- 正样本(1)：1
- 负样本(0)：18
- 未知(unknown)：0

## 数值汇总（min / max / avg）
- length: min=1.000 max=4.000 avg=2.789
- score: min=-5.710 max=-0.292 avg=-2.637
- delta_risk: min=-4.953 max=0.045 avg=-2.238
- cost: min=0.152 max=5.733 avg=1.996

## 按域统计（per_domain）
### 域：pem
- 样本数：19
- 标签：1=1 0=18 unknown=0
- 均值：score=-2.637, delta_risk=-2.238, cost=1.996, length=2.789

> 本文件由 collect_debug_dataset.py 每次运行自动覆盖生成，用于配合 debug_dataset.stats.json 的可读化展示。
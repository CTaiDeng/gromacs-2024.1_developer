# debug_dataset.stats.json 配套说明（自动生成）

- 生成时间：2025-10-26 12:46:45
- 样本总数：17

## 域分布（domains）
- pem: 17

## 标签统计（labels）
- 正样本(1)：13
- 负样本(0)：4
- 未知(unknown)：0

## 数值汇总（min / max / avg）
- length: min=1.000 max=4.000 avg=2.765
- score: min=-3.760 max=0.300 avg=-1.960
- delta_risk: min=-3.162 max=0.300 avg=-1.620
- cost: min=0.000 max=3.476 avg=1.700

## 指标解读（含举例）
- length：算子包序列的长度（操作步数）。一般越短越精简，但需结合得分与收益综合评估。
- score：综合得分，用于训练与筛选。定义为 score = delta_risk − cost_lambda × cost（当前 cost_lambda=0.200）。
- delta_risk：收益项（越大越好），可理解为风险下降/效用提升的度量；为负表示变差。
- cost：代价/资源消耗项（越小越好），是多因素的抽象，例如时间、药物毒性/不良反应、价格、操作难度或临床风险等。
- 校核：约有 avg_score ≈ avg_delta_risk − cost_lambda × avg_cost ≈ -1.620 − 0.200 × 1.700 ≈ -1.960（当前统计 avg_score=-1.960）

## 按域统计（per_domain）
### 域：pem
- 样本数：17
- 标签：1=13 0=4 unknown=0
- 均值：score=-1.960, delta_risk=-1.620, cost=1.700, length=2.765

> 本文件由 collect_debug_dataset.py 每次运行自动覆盖生成，用于配合 debug_dataset.stats.json 的可读化展示。
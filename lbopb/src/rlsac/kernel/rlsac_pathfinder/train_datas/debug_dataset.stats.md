# debug_dataset.stats.json 说明（自动生成）

- 生成时间：2025-10-26 16:27:36
- 样本总数：18

## 分布（domains）
- pem: 18

## 标签统计（labels）
- 正确(1)：15
- 错误(0)：3
- 未知(unknown)：0

## 数值概览（min / max / avg）
- length: min=1.000 max=4.000 avg=2.833
- score: min=-4.205 max=0.300 avg=-1.966
- delta_risk: min=-3.922 max=0.300 avg=-1.657
- cost: min=0.000 max=5.021 avg=1.545

## 指标解读与示例
- length：算子包序列长度。一般越短越优，但需综合评分考量。
- score：综合评分，当前筛选规则为 score = delta_risk − cost_lambda × cost（cost_lambda=0.200）。
- delta_risk：收益指标，越大越好。可理解为病灶负担下降/疗效提升等抽象。
- cost：成本（越小越好），综合抽象了时间、药物毒性/不良反应、价格、操作难度或临床风险等。
- 校验：约有 avg_score ≈ avg_delta_risk − cost_lambda × avg_cost = -1.657 − 0.200 × 1.545 = -1.966；当前统计 avg_score=-1.966。

## 分域统计（per_domain）
### 域 pem
- 样本数：18
- 标签：1=0 0=0 unknown=0
- 均值：score=-1.966, delta_risk=-1.657, cost=1.545, length=2.833

> 本文件由 collect_debug_dataset.py 每次运行自动覆盖生成，配合 debug_dataset.json 的可读呈现。
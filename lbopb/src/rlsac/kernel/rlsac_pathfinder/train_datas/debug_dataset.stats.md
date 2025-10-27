# debug_dataset.stats.json 说明（自动生成）

- 生成时间：2025-10-28 00:58:16
- 样本总数：57

## 分布（domains）
- tem: 5
- prm: 5
- pdem: 10
- pgom: 5
- pem: 22
- iem: 5
- pktm: 5

## 标签统计（labels）
- 正确(1)：41
- 错误(0)：16
- 未知(unknown)：0

## 数值概览（min / max / avg）
- length: min=1.000 max=4.000 avg=2.667
- score: min=-4.783 max=0.986 avg=-1.082
- delta_risk: min=-3.922 max=1.103 avg=-0.835
- cost: min=0.000 max=5.733 avg=1.237

## 指标解读与示例
- length：算子包序列长度。一般越短越优，但需综合评分考量。
- score：综合评分，当前筛选规则为 score = delta_risk − cost_lambda × cost（cost_lambda=0.200）。
- delta_risk：收益指标，越大越好。可理解为病灶负担下降/疗效提升等抽象。
- cost：成本（越小越好），综合抽象了时间、药物毒性/不良反应、价格、操作难度或临床风险等。
- 校验：约有 avg_score ≈ avg_delta_risk − cost_lambda × avg_cost = -0.835 − 0.200 × 1.237 = -1.082；当前统计 avg_score=-1.082。

## 分域统计（per_domain）
### 域 tem
- 样本数：5
- 标签：1=0 0=0 unknown=0
- 均值：score=-1.084, delta_risk=-0.653, cost=2.155, length=2.800
### 域 prm
- 样本数：5
- 标签：1=0 0=0 unknown=0
- 均值：score=-0.548, delta_risk=-0.182, cost=1.831, length=2.400
### 域 pdem
- 样本数：10
- 标签：1=0 0=0 unknown=0
- 均值：score=-0.089, delta_risk=-0.018, cost=0.359, length=2.500
### 域 pgom
- 样本数：5
- 标签：1=0 0=0 unknown=0
- 均值：score=0.063, delta_risk=0.086, cost=0.112, length=1.800
### 域 pem
- 样本数：22
- 标签：1=0 0=0 unknown=0
- 均值：score=-2.194, delta_risk=-1.845, cost=1.745, length=2.909
### 域 iem
- 样本数：5
- 标签：1=0 0=0 unknown=0
- 均值：score=-0.650, delta_risk=-0.431, cost=1.096, length=3.200
### 域 pktm
- 样本数：5
- 标签：1=0 0=0 unknown=0
- 均值：score=-0.282, delta_risk=-0.179, cost=0.513, length=2.400

> 本文件由 collect_debug_dataset.py 每次运行自动覆盖生成，配合 debug_dataset.json 的可读呈现。
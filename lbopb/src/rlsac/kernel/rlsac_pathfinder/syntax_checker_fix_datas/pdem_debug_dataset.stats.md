# pdem_debug_dataset.stats.json 说明（自动生成）

- 更新时间：2025-10-28 00:58:20
- 样本总数：10

## 分布（domains）
- pdem: 10

## 标签统计（labels）
- 正确(1)：5
- 可疑(0)：5
- 未知(unknown)：0

## 数值指标（min / max / avg）
- length: min=2.000 max=4.000 avg=2.500
- score: min=-0.777 max=0.343 avg=-0.089
- delta_risk: min=-0.650 max=0.366 avg=-0.018
- cost: min=0.092 max=0.721 avg=0.359

## 指标解释
- length：操作序列长度。一般越长越难，且综合评分可能更低。
- score：综合评分；约定义为 score = delta_risk − cost_lambda × cost，cost_lambda=0.200。
- delta_risk：风险净减少，越大越好；体现为治疗有效度或负效应降低等。
- cost：代价，越小越好；可综合时长、药物用量/副作用、工程复杂度或二次验证成本等。
- 校验：约有 avg_score ≈ avg_delta_risk − cost_lambda × avg_cost = -0.018 − 0.200 × 0.359 = -0.089；当前统计 avg_score=-0.089。

> 本文件由 make_syntax_checker_fix_datas.py 自动生成，基于 train_datas/debug_dataset.json 的可读摘要。
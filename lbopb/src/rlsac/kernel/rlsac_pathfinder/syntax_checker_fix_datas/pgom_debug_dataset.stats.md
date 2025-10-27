# pgom_debug_dataset.stats.json 说明（自动生成）

- 更新时间：2025-10-28 00:58:20
- 样本总数：5

## 分布（domains）
- pgom: 5

## 标签统计（labels）
- 正确(1)：4
- 可疑(0)：1
- 未知(unknown)：0

## 数值指标（min / max / avg）
- length: min=1.000 max=3.000 avg=1.800
- score: min=-0.108 max=0.335 avg=0.063
- delta_risk: min=-0.075 max=0.349 avg=0.086
- cost: min=0.011 max=0.280 avg=0.112

## 指标解释
- length：操作序列长度。一般越长越难，且综合评分可能更低。
- score：综合评分；约定义为 score = delta_risk − cost_lambda × cost，cost_lambda=0.200。
- delta_risk：风险净减少，越大越好；体现为治疗有效度或负效应降低等。
- cost：代价，越小越好；可综合时长、药物用量/副作用、工程复杂度或二次验证成本等。
- 校验：约有 avg_score ≈ avg_delta_risk − cost_lambda × avg_cost = 0.086 − 0.200 × 0.112 = 0.063；当前统计 avg_score=0.063。

> 本文件由 make_syntax_checker_fix_datas.py 自动生成，基于 train_datas/debug_dataset.json 的可读摘要。
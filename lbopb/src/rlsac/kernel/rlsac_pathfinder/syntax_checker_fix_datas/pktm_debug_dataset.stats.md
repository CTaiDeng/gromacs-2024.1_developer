# pktm_debug_dataset.stats.json 说明（自动生成）

- 更新时间：2025-10-28 00:58:20
- 样本总数：5

## 分布（domains）
- pktm: 5

## 标签统计（labels）
- 正确(1)：5
- 可疑(0)：0
- 未知(unknown)：0

## 数值指标（min / max / avg）
- length: min=1.000 max=4.000 avg=2.400
- score: min=-0.961 max=0.100 avg=-0.282
- delta_risk: min=-0.905 max=0.100 avg=-0.179
- cost: min=0.000 max=2.282 avg=0.513

## 指标解释
- length：操作序列长度。一般越长越难，且综合评分可能更低。
- score：综合评分；约定义为 score = delta_risk − cost_lambda × cost，cost_lambda=0.200。
- delta_risk：风险净减少，越大越好；体现为治疗有效度或负效应降低等。
- cost：代价，越小越好；可综合时长、药物用量/副作用、工程复杂度或二次验证成本等。
- 校验：约有 avg_score ≈ avg_delta_risk − cost_lambda × avg_cost = -0.179 − 0.200 × 0.513 = -0.282；当前统计 avg_score=-0.282。

> 本文件由 make_syntax_checker_fix_datas.py 自动生成，基于 train_datas/debug_dataset.json 的可读摘要。
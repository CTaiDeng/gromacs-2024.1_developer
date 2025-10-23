# rlsac_connector（跨领域“法则联络”映射发现｜SAC 版）

- 目标：基于七本“领域辞海”（各域的算子包集合），在统一的 LBOPB 全息状态上，同时应用七域算子包并评估“全局逻辑自洽性”，发现高分的“法则联络”。
- 背景：对应《O3理论的自举之路》第二阶段（The Connector）。

## 使用方法

- 训练：
  - `python -m lbopb.src.rlsac.kernel.rlsac_connector.train`
  - 配置：`lbopb/src/rlsac/kernel/rlsac_connector/config.json`（指定 `packages_dir` 指向第一阶段输出的辞海目录）
- 提取联络：
  - 训练后调用 `extract_connection(run_dir)`，在本目录生成/更新 `law_connections.json`，写入联络七元组与评分元信息。

## Observation / Action / Step

- Observation（35 维）= concat_{mod∈[pem,pdem,pktm,pgom,tem,prm,iem]} `[B, P, F, N, risk]`
- Action：单一离散 id，解码为七域各选一个“算子包”的七元组（混合基数展开）
- Step：一步评估后立即 `done=True`（一次性候选体提案与验证）

## 奖励（简化实现）

- 基础项：`sum(Δrisk_mod)`（七域风险下降之和）
- 一致性：
  - 对若干对偶域（pdem↔pktm、pgom↔pem、tem↔pktm、prm↔pem、iem↔pem），若双方均有显著变化（>eps），奖励 +1；若一方显著而另一方近乎不变，惩罚 -1。
  - 若七域中 ≥5 域均发生非零变化，额外 +1。
- 成本：`- λ * sum(cost_mod)`（各域序列的作用成本）

以上权重可在 `config.json` 中调节（`consistency_bonus`/`inconsistency_penalty`/`cost_lambda`/`eps_change`）。

## 依赖数据（七本“领域辞海”）

- 目录：`lbopb/src/rlsac/kernel/rlsac_pathfinder/`
- 文件：`<domain>_operator_packages.json`（由第一阶段 rlsac_pathfinder 生成；元素包含 `id` 与 `sequence`）

## 产物

- `out/out_connector/train_*/`: 训练日志与权重（policy/q1/q2），`action_space_meta.json`（总动作数与各域基数）
- `lbopb/src/rlsac/kernel/rlsac_connector/law_connections.json`: 联络候选体（七元组）与评分记录

> 说明：为保持与现有离散 SAC 训练脚本一致性，本实现将“七域联动选择”编码为单一离散动作。
> 实际可替换为多头策略（Multi-Discrete）以降低组合空间复杂度。





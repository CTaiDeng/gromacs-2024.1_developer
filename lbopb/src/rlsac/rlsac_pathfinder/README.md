# rlsac_pathfinder（PEM 单域“算子包”路径探索）

- 目标：在单个幺半群（pem/pdem/pktm/pgom/tem/prm/iem）内，基于离散 SAC 自主探索从 `S_initial` 到 `S_target` 的“有效算子序列（算子包）”，并记录到 `<domain>_operator_packages.json`。
- 背景：对应《O3理论的自举之路》第一阶段（路径探索者）。

## 使用方法

- 训练：
  - `python -m lbopb.src.rlsac.rlsac_pathfinder.train`（或在代码内调用 `train()`）。
  - 配置文件：`lbopb/src/rlsac/rlsac_pathfinder/config.json`，通过 `domain` 指定目标域（默认 `pem`）
- 提取算子包（贪心解码）：
  - 训练结束后：`extract_operator_package(run_dir)` 会在同目录生成/更新 `<domain>_operator_packages.json`。

## Observation / Action（统一）

- Observation: `[b, n_comp, perim, fidelity]`（float32，shape=(4,)）。
- Action: 离散基本算子集的索引（`0..N-1`），各域对应的基本算子自动装配；可通过 `include_identity` 加入 `Identity`。

## 奖励与终止条件（简化实现）

- 距离改进奖励：`reward += (dist_prev - dist_cur)`（按特征设定权重并求加权 L1 距离）。
- 目标奖励：若状态进入目标容差（`tol_*`）范围，额外奖励 `+5.0`，并终止回合。
- 步长惩罚：每步 `-0.01`（鼓励更短路径）。
- 回合上限：`episode_max_steps` 步。

## 产物

- `out_pathfinder/train_*/`：训练日志、权重（`policy.pt`、`q1.pt`、`q2.pt`）、`op_index.json`（动作索引映射）。
- `lbopb/src/rlsac/rlsac_pathfinder/<domain>_operator_packages.json`：辞海式存储的算子包条目数组。

## 配置关键项（示例）

- `initial_state/target_state/tolerance`：四个量 `b/n_comp/perim/fidelity` 的目标与容差。
- `include_identity`：是否将 `Identity` 纳入动作集合（默认 false）。
- RL 超参：`learning_rate_* / gamma / tau / total_steps / *` 与 `out_pathfinder` 输出目录名等。

> 注：本实现pem/pdem/pktm/pgom/tem/prm/iem；扩展到其他幺半群时，可复用训练框架并替换环境定义与基本算子集。





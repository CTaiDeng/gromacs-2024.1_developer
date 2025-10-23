# rlsac_pathfinder（算子包打分与辞海生成｜支持七大幺半群）

- 目标：在单个幺半群（pem/pdem/pktm/pgom/tem/prm/iem）内，随机生成“算子包”（由基本算子序列构成），用“内置公理系统规则引擎”打
  0/1 标签进行监督，训练一个神经网络打分器（PackageScorer）。训练完成后，让网络对大量候选算子包打分，选出 Top‑K，生成本域“辞海”
  `<domain>_operator_packages.json`。最终目标是用网络近似并替代规则引擎的判定。
- 背景：对应《O3理论的自举之路》第一阶段（路径探索者）。

## 使用方法

- 训练（单域）：
    - `python -m lbopb.src.rlsac.kernel.rlsac_pathfinder.train`（或在代码内调用 `train()`）。
    - 配置：`lbopb/src/rlsac/kernel/rlsac_pathfinder/config.json`，通过 `domain` 指定域（默认 `pem`）。
- 一次训练七域：
    - 代码调用：`from lbopb.src.rlsac.kernel.rlsac_pathfinder import train_all; train_all()`

## 算法流程（新）

- 随机生成算子包：从各域基本算子中随机采样长度 `[min_len, max_len]` 的序列（可禁止相邻重复）。
- 公理系统打分：使用内置规则引擎（各模块 syntax_checker.py 和 pathfinder/oracle.py 中的规则计算）进行 0/1 判定。
- 监督训练打分器：将“算子计数 + 长度 + Δrisk + cost”作为特征，训练 MLP 输出 `p(correct)`。
- Top‑K 生成辞海：训练后随机生成大量候选，按打分排序取 Top‑K 写入辞海。

## 奖励与终止条件（简化实现）

- 距离改进奖励：`reward += (dist_prev - dist_cur)`（按特征设定权重并求加权 L1 距离）。
- 目标奖励：若状态进入目标容差（`tol_*`）范围，额外奖励 `+5.0`，并终止回合。
- 步长惩罚：每步 `-0.01`（鼓励更短路径）。
- 回合上限：`episode_max_steps` 步。

## 产物

- `out/out_pathfinder/train_*/`：训练产物（`scorer.pt` 权重、提取的 `<domain>_operator_packages.json`）
- `lbopb/src/rlsac/kernel/rlsac_pathfinder/<domain>_operator_packages.json`：辞海式存储的算子包条目数组。

## 配置关键项（示例）

- `initial_state/target_state/tolerance`：四个量 `b/n_comp/perim/fidelity` 的目标与容差。
- `include_identity`：是否将 `Identity` 纳入动作集合（默认 false）。
- RL 超参：`learning_rate_* / gamma / tau / total_steps / *` 与 `out_pathfinder` 输出目录名等。

> 注：如需更严谨的校验，可扩展各模块 syntax_checker.py 中的规则引擎（上下文/阈值/停机/不可交换/次序模式等）。






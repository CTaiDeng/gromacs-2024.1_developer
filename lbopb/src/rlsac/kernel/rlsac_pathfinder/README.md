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

## 调试步骤（建议）

1) 最小化训练配置
- 将 `samples/epochs/candidate_generate/topk` 设为较小值（如 `samples=50, epochs=3, candidate_generate=100, topk=10`）。
- 开启 `debug_dump=true`（默认已开启）。
- 如需 LLM 辅助，仅当 `use_llm_oracle=true` 且出现 warnings 时才会请求。

2) 运行并查看运行目录
- 命令：`python -m lbopb.src.rlsac.kernel.rlsac_pathfinder.train`
- 运行目录：`out/out_pathfinder/train_<ts>_<domain>/`
- 重点文件：
  - `debug_dataset.json`：包含每个样本的 `sequence/特征(label/Δrisk/cost)`；
  - `debug_candidates.json`：按网络评分排序的候选 Top‑N（由 `debug_candidates_top` 控制，默认 50）。

3) 逐条核对标签生成
- 对 `debug_dataset.json` 某条样本的 `sequence` 手动调用：
  - `python lbopb/src/<domain>/syntax_checker.py <op1> <op2> ...`
- 确认 `errors/warnings` 与 `oracle` 中记录的一致；有 `warnings` 时再比对 LLM 输出（如启用）。

4) 特征与判定联动
- 检查特征的 `delta_risk/cost/length` 是否与 `apply_sequence` 计算一致；
- 如需进一步解释，可在 scorer 处同时输出中间层激活或引入“违规类型”多标签头。

5) 日志开关
- 环境变量：`RLSAC_DEBUG=1`、`RLSAC_LOG_EVERY_STEP=1` 可加大日志量；
- 配置中的 `log_to_file=true` 将日志写入 `train.log`；
- 直接脚本执行时的导入问题已做回退，推荐使用包方式：`python -m lbopb.src.rlsac.kernel.rlsac_pathfinder.train`。

## Network I/O And Dual Validation (可落地方案)

一、网络要学什么（输出）

- 输出语义：y^ = p(valid | package, domain) ∈ (0,1)，即单域“算子包是否合法/可采纳”的概率。
- 推断：用 y^ 排序候选算子包，Top‑K 写入该域辞海；训练：BCE(y^, y) 监督。
- 标签 y（双重判定来源）：
  - 强规则（syntax_checker）优先：若 errors 非空 → y=0 直接否决；
  - 轻违规（warnings）时才启用 LLM（Gemini）辅助：warnings 非空 且 use_llm_oracle=true → 要求 LLM 返回 1/0，与启发式共同决策；
  - 启发式（路径度量）补充：Δrisk − λ·cost > 0 视为 1，否则 0；
  - 组合逻辑：
    - 有 errors → 返回 0（不调用 LLM）；
    - 无 errors 且 warnings 存在 且 use_llm_oracle=true → 返回 (启发式 AND LLM)；
    - 无 warnings → 返回 启发式。

二、如何把“算子包”表达为输入（X）

- 基础方案（当前已实现）：
  - 对域内基本算子建立顺序（op2idx），长度 M；
  - 特征 X = [bag‑of‑ops(M), len, Δrisk, cost] ∈ R^(M+3)；
  - 模型：PackageScorer(in_dim=M+3) → y^；
  - 优点：简单稳定、可解释，与启发式/规则引擎兼容。
- 增强方案（可选）：
  - 序列编码（RNN/Transformer/1D‑CNN）+ 统计特征融合；
  - 也可输出“违规类型”多标签用于解释。

三、训练数据结构（建议）

```
{
  "domain": "pem",
  "sequence": ["Inflammation", "Apoptosis"],
  "x": {
    "op_index": { "Inflammation": 0, "Apoptosis": 1, ... },
    "bag_of_ops": [1,1,0,...],
    "length": 2,
    "delta_risk": 0.42,
    "cost": 0.05
  },
  "oracle": {
    "syntax": { "errors": [], "warnings": ["Step 0: ..."], "valid": true },
    "heur": { "score": 0.42-λ·0.05, "lambda": 0.2 },
    "llm_used": true,
    "llm_raw": "1"
  },
  "y": 1
}
```

四、训练/推理流程（伪代码）

标签 y 生成（训练时）

```
for seq in candidate_sequences:
    res = syntax_checker.check_sequence(seq, init_state)
    if res.errors:
        y = 0
    else:
        heur_ok = (delta_risk(seq) - λ*cost(seq) > 0)
        if res.warnings and use_llm:
            prompt = build_pathfinder_prompt(domain, seq)  // llm_oracle.py
            llm_out = call_llm(prompt)  // '1' or '0'
            y = int(heur_ok and (llm_out == '1'))
        else:
            y = int(heur_ok)
```

网络训练

```
for (X, y) in dataset:
    y_hat = scorer(X)  // PackageScorer
    loss = BCE(y_hat, y)
    update(loss)
```

推理（写辞海）

```
candidates = random_generate(...)
ranked = sort_by(scorer(X(candidate)), desc=True)
top_k = select_top_k(ranked, K)
write_to_<domain>_operator_packages.json(top_k)
```

五、Pathfinder 与 Connector 的 I/O 对比

- Pathfinder（单域）
  - 输入 X：基于 sequence 的向量编码（Bag‑of‑ops + 统计）；
  - 输出 y^：是否合法 p(valid)，监督来自“syntax_checker 优先 +（仅警告启用 LLM）+ 启发式”；
  - 产物：<domain>_operator_packages.json。
- Connector（七域）
  - 输入 X：7 域包长度 + 全局统计（ΣΔrisk, Σcost, consistency）→ 长度 10；
  - 输出 y^：联络合法 p(valid)，同样遵循“显著错误→直接 0、仅警告→按配置启用 LLM”的监督生成；
  - 产物：law_connections.json。

六、与现有实现对齐

- 已实现：Bag‑of‑ops + 统计特征（scorer.py），并按上面方式生成标签；双重判定逻辑在 oracle.py 中落实（errors→0，warnings→可配 LLM）。
- 可选增强：替换为序列模型或加入不交换矩阵、位置编码等（与现框架兼容）。






# rlsac_connector（纤维丛“联络”打分与辞海生成）

- 目标：基于七本“领域辞海”（各域的算子包集合），随机生成“纤维丛算子包向量”（七域各取一包）作为联络候选体，使用“七个域的公理系统
  Oracle”打 0/1 标签对其联络进行检验（可替换为 Gemini），据此训练一个神经网络打分器；训练完成后，大量采样候选体并由网络打分，选出高分联络写入联络辞海。最终目标是用网络替代外部
  Oracle（如 Gemini 基于公理系统的判定）。
- 背景：对应《O3理论的自举之路》第二阶段（The Connector），七个模型可按“不同纤维丛/主视角”分别训练与应用。

## 使用方法

- 训练：
    - `python -m lbopb.src.rlsac.kernel.rlsac_connector.train`
    - 配置：`lbopb/src/rlsac/kernel/rlsac_connector/config.json`（指定 `packages_dir` 指向第一阶段 `rlsac_pathfinder`
      输出的辞海目录）
- 提取联络（额外采样挑选最佳）：
    - 训练后调用 `extract_connection(run_dir)`，在模块目录与运行目录生成/更新 `law_connections.json`，写入联络七元组与评分元信息。

## 算法流程（新）

- 随机生成联络候选体：从每个域的辞海中各抽取一个算子包（id/sequence）。
- 公理系统 Oracle 打分：当前实现为启发式（ΣΔrisk + 一致性 − λ·Σcost），>0 判定为 1；可替换为外部 LLM（Gemini）实现严谨的公理检验（见
  `oracle.py`）。
- 监督训练打分器：以“每域序列长度 + 全局统计（ΣΔrisk, Σcost, consistency）”为特征，训练 MLP 输出 `p(correct)`。
- Top‑K 生成联络辞海：训练后随机生成大量候选体，按打分排序取 Top‑K 写入联络辞海。

> 一致性：对若干对偶域（pdem↔pktm、pgom↔pem、tem↔pktm、prm↔pem、iem↔pem），若双方均有显著变化（>eps）加分，一方显著而另一方近乎不变减分；若七域中
> ≥5 域均发生非零变化，额外加分。权重在 `config.json` 中可调（`cost_lambda/eps_change`）。

## 依赖数据（七本“领域辞海”）

- 目录：`lbopb/src/rlsac/kernel/rlsac_pathfinder/`
- 文件：`<domain>_operator_packages.json`（由第一阶段 rlsac_pathfinder 生成；元素包含 `id` 与 `sequence`）

## 产物

- `out/out_connector/train_*/`: 训练日志（`train.log`）与 `scorer.pt` 模型权重，`law_connections.json`（本次 Top‑K 联络）
- `lbopb/src/rlsac/kernel/rlsac_connector/law_connections.json`: 全局联络辞海（累积记录）

> 说明：默认判定为内置一致性与度量启发式；如需更严格校验，可结合各模块 syntax_checker 的结果。

## Network I/O And Dual Validation（可落地方案）

一、网络要学什么（输出）

- 输出语义：y^ = p(valid | connection) ∈ (0,1)，即“七域联络是否合法/可采纳”的概率。
- 推断：用 y^ 排序候选联络，Top‑K 写入联络辞海；训练：BCE(y^, y) 监督。
- 标签 y（双重判定来源）：
  - 单域显著错误（syntax_checker.errors 非空）→ 直接判 0；
  - 仅有警告（任何域 warnings 非空）且 use_llm_oracle=true 时，调用 LLM（Gemini）两两整合模板判定，结果与启发式共同决策；
  - 启发式一致性：ΣΔrisk + consistency − λ·Σcost > 0；
  - 组合逻辑：
    - 若任何域有 errors → 0（不调用 LLM）；
    - 若存在 warnings 且启用 LLM → y = 启发式 AND LLM；
    - 无 warnings → y = 启发式。

二、如何把“联络候选体”表达为输入（X）

- 基础方案（当前已实现）：
  - 向量 X = [len(seq_pem), len(seq_pdem), ..., len(seq_iem), ΣΔrisk, Σcost, consistency] ∈ R^10；
  - 模型：PackageScorer(in_dim=10) → y^；
  - 优点：结构简单稳定，适合快速验证一致性度量的效果。
- 增强方案（可选）：
  - 拼接 7 域的“序列编码”与全局统计；或加入“单域违规分布”多标签头，提升解释能力。

三、训练数据结构（建议）

```
{
  "conn": {
    "pem": ["Apoptosis", ...], "pdem": [...], ..., "iem": [...]
  },
  "x": {
    "lengths": [L_pem, L_pdem, ..., L_iem],
    "delta_risk_sum": 1.25,
    "cost_sum": 0.8,
    "consistency": 2.0
  },
  "oracle": {
    "syntax": { "fatals": false, "warnings": true },
    "heur": { "score": 1.25 + 2.0 − λ·0.8, "lambda": 0.2 },
    "llm_used": true, "llm_raw": "1"
  },
  "y": 1
}
```

四、训练/推理流程（伪代码）

标签 y 生成（训练时）

```
for conn in random_connections:
    fatals, warns = per_domain_syntax_checks(conn)
    if fatals: y=0; continue
    heur_ok = (risk_sum(conn) + consistency(conn) - λ*cost_sum(conn) > 0)
    if warns and use_llm:
        llm_out = call_llm(build_connector_prompt(conn))  // '1' or '0'
        y = int(heur_ok and (llm_out=='1'))
    else:
        y = int(heur_ok)
```

网络训练/推理同 Pathfinder，区别在 X 的定义与联络辞海输出。

五、与 Pathfinder 的 I/O 对比

- Pathfinder（单域）：X = Bag‑of‑ops + [len, Δrisk, cost] → y^；产物为 <domain>_operator_packages.json。
- Connector（联络）：X = [七域长度, ΣΔrisk, Σcost, consistency] → y^；产物为 law_connections.json。

六、与现有实现对齐

- 已实现：10 维输入的打分器、双重判定与（仅警告时）LLM 辅助；输出 law_connections.json。
- 可选增强：把单域序列嵌入/违规类型预测接入，使模型更善于“解释”联络为何不成立。




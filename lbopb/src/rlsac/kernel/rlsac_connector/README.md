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





# rlsac_nsclc（PEM 联络环境训练）

- 训练目标：以整体生命体状态为输入，输出一个 PEM 域的“算子包”（来自 rlsac_pathfinder 的辞海），并以 PEM 为基底、补齐其它 6
  域包形成联络候选体，以联络一致性评分作为奖励进行强化学习。
- 运行：
    - `python lbopb/src/rlsac/application/rlsac_nsclc/train.py`（配置见同目录 `config.json`）
- 产物：`out/out_pathfinder/` 或 `out/out_connector/` 下的训练日志与权重。

说明：本包使用 PEM 联络环境，不再使用旧的 SequenceEnv/目标路径指针推进逻辑。

结论（更准确的表述）：

- rlsac_nsclc 并非直接“用 NSCLC 的 JSON 当监督样本训练”，而是“把 `NSCLC_Therapy_Path` 作为动作空间与目标干预路径的定义”，在交互式环境中在线生成训练样本
  `(s, a, r, s')` 进行强化学习。

## 深入解析（验证）

下面解读在 `lbopb/src/rlsac/application/rlsac_nsclc` 框架中，`operator_crosswalk_train.json` 如何被使用，以及输入/输出设计与训练数据的来源。

### 1. PEM 算子包来源

- 来自 `lbopb/src/rlsac/kernel/rlsac_pathfinder/pem_operator_packages.json`。

### 2. Observation

- 7 域 [B, P, F, N, risk] 拼接，合计 35 维。

### 3. Action

- 选择一个 PEM 算子包（来自辞海）作为动作输出。

### 4. Reward 与判定

- ΣΔrisk + consistency − λ·Σcost；显著错误（syntax_checker.errors）直接判 0；仅警告时可启用 LLM（Gemini）辅助判定（config:
  use_llm_oracle）。

### 5. 样本生成

- 与环境交互在线生成 `(s, a, r, s')`；PEM 包来自辞海，其它 6 域包随机补齐。

### 6. 奖励与指针推进（rlsac_nsclc 的语义）

- 奖励：`Δrisk − λ·cost`（risk/cost 来自各模块 `metrics.py`；`λ` 可在 `config.json` 的 `reward_lambda` 调整）。
- 指针推进：只有当选中的动作与“该模块目标序列中的下一应选算子”一致时，才推进指针与模块状态，以体现“模仿/预热”的属性；否则不推进，但不会报错。

### 7. 流程图（概念）

```mermaid
graph TD
    A[Observation 向量] --> B{策略网络 π(a|s)}
    B --> C[选择 PEM 算子包]
    C --> D{环境：PEM 联络评分}
    D --> E[随机补齐其它 6 域包 → 联络候选体]
    E --> F[ΣΔrisk + consistency − λ·Σcost]
    F --> G[奖励 r]
    subgraph 回放池
        J[(s, a, r, s')]
    end
    J --> B
```

小结：`operator_crosswalk_train.json` 是“做什么”的菜单；Observation 是“当前情况”的报告；Action
是基于报告从菜单中做出的“最佳选择”的预测，目标是使长期累积奖励最大化。





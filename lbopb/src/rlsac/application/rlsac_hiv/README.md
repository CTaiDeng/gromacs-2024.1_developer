# rlsac_hiv（PEM 联络环境训练）

- 训练目标：在“生命体”整体状态为输入的前提下，输出一个 PEM 域的“算子包”（来自第一阶段 rlsac_pathfinder 的辞海），并以 PEM
  为基底、补齐其它 6 域的联络构成候选体，由联络一致性评分作为奖励进行强化学习。
- 运行：
    - `python lbopb/src/rlsac/application/rlsac_hiv/train.py`（配置见同目录 `config.json`）
- 产物：`out/out_pathfinder/` 或 `out/out_connector/` 下的训练日志和权重（`policy.pt` 等）。

结论：本环境不是在“固定目标序列”下模仿推进行为，而是在“联络一致性评分”下学习输出最优的 PEM
算子包，以最大化跨域一致性（ΣΔrisk + consistency − λ·Σcost）。

具体说明：

- 动作空间（PEM 算子包）：来自 `lbopb/src/rlsac/kernel/rlsac_pathfinder/pem_operator_packages.json` 的 PEM 算子包集合（第一阶段产物）。
- 环境：`lbopb/src/rlsac/application/rlsac_hiv/pem_connector_env.py`。
    - Observation：7 域的 [B, P, F, N, risk] 拼接，合计 35 维。
    - Action：选择一个 PEM 算子包（离散动作 id）。
    - Reward：将所选 PEM 包与其它 6 域随机补齐的包组成联络候选体，使用 `ConnectorAxiomOracle` 计算 ΣΔrisk + consistency −
      λ·Σcost。
    - 语法/公理校验：若单域存在显著错误（syntax_checker.errors），直接判 0；仅存在警告时，可按配置启用 LLM（Gemini）辅助判定。
    - 关键配置（config.json）：`packages_dir/cost_lambda/eps_change/use_llm_oracle`。

说明：本包使用 PEM 联络环境，不再使用旧的 SequenceEnv/目标路径指针推进逻辑。

## 深入解析（验证）

下面解读在 `lbopb/src/rlsac/application/rlsac_hiv` 框架中，PEM 联络环境如何定义 Observation/Action/Reward，以及训练数据的生成与判定依据。

### 1. PEM 算子包来源

- 来自第一阶段 rlsac_pathfinder 生成的 `pem_operator_packages.json`，作为本环境的离散动作空间（选择 PEM 包 id）。

### 2. Observation

- 7 域依次拼接的 [B, P, F, N, risk]，合计 35 维，用于表征当前生命体全息状态。

它是系统“当前状态”的数值化快照，供策略网络作决策。

### 3. Action

- 选择一个 PEM 算子包（离散 id，对应 `pem_operator_packages.json` 中的包）。

### 4. Reward 与判定

- Reward = ΣΔrisk + consistency − λ·Σcost。
- 显著错误（syntax_checker.errors）直接判 0；仅有警告（syntax_checker.warnings）时可启用 LLM（Gemini）辅助判定（config:
  use_llm_oracle）。

### 5. 样本生成

- 训练样本 `(s, a, r, s')` 在与联络环境交互过程中动态生成并写入回放池，PEM 包来自辞海，其它 6 域包随机补齐。

### 6. 其他

- 不再使用“目标路径指针推进”逻辑。

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






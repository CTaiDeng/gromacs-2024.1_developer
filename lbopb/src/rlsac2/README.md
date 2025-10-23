# rlsac2（SequenceEnv 版）

- 独立的离散 SAC 实现，使用 `LBOPBSequenceEnv` 将 `operator_crosswalk_train.json` 的“模块→算子序列”转为训练样本。
- 运行：
  - `python lbopb/src/rlsac2/train.py`（配置见同目录 `config.json`）
- 产物：`out/train_*/` 下生成 `policy.pt` 等权重与日志；自动导出 `op_index.json`（op→id 对照表）。

结论（更准确的表述）：

- rlsac2 并非直接“用 HIV_Therapy_Path 的 JSON 当监督样本训练”，而是“把 HIV_Therapy_Path 作为动作空间与目标干预路径的定义”，在交互式环境中在线生成训练样本 `(s, a, r, s')` 进行强化学习。

具体说明：
- 动作空间与目标路径：来自 `lbopb/src/rlsac2/operator_crosswalk_train.json` 的 `HIV_Therapy_Path` 定义了全局干预算子集合（动作空间）与各模块的目标干预序列。训练时在运行目录导出 `op_index.json`，把整数动作 id 映射到具体干预算子名（便于解释策略输出）。
- 环境与样本：环境为 `lbopb/src/rlsac2/sequence_env.py`（SequenceEnv）。Observation 拼接为 `[module_onehot(7) | per‑module(B/P/F/N/risk)*7 | next_op_onehot(M) | pos]`；奖励为 `Δrisk − λ·cost`（risk/cost 来自各模块 metrics；`λ` 可在 `config.json` 调整）；推进规则为仅当选中动作等于“该模块目标序列的下一应选算子”时推进指针（相当于用目标路径做“教师/引导”）。样本来源是与环境交互在线产生的 `(state, action, reward, next_state, done)`，并存入回放池，JSON 本身不是监督样本。
- 若“作为样本”指“用目标路径引导训练”：是的，SequenceEnv 把 `HIV_Therapy_Path` 的目标序列作为“样本模式/教师信号”来约束推进与奖励，但仍是 RL 的在线采样学习，不是静态标签的监督学习。
- 离线示例（可选）：`lbopb/src/rlsac2/train_data/` 提供了 JSON 结构的 Demo 样本（演示用）；如需预填回放池，可扩展 `train.py` 读取这些样本进行 warm‑start。

说明：本包不依赖 `lbopb/src/rlsac`，内部自洽（sequence_env/models/utils/replay_buffer/train）。

## 深入解析（验证）

下面解读在 `lbopb/src/rlsac2` 框架中，`operator_crosswalk_train.json` 如何被使用，以及输入/输出设计与训练数据的来源。

### 1. `operator_crosswalk_train.json` 是什么？

- 它不是训练样本，而是“字典/指令集”。用于定义智能体可以执行的“干预算子（Operator）”全集，以及各模块的目标序列（Case Package）。
- 本包会在初始化时读取该 JSON，构建全局“干预算子词表（vocabulary）”，并在训练目录导出 `op_index.json`（op→id 对照），用于将整数动作 id 反查为具体算子名。

### 2. 输入 Observation 代表什么？

在 rlsac2 中，Observation 被设计为一个拼接向量：

- `[module_onehot(7) | per-module(B,P,F,N,risk)*7 | next_op_onehot(M) | pos]`
  - `module_onehot(7)`: 当前正在推进的模块（pem/pdem/pktm/pgom/tem/prm/iem）。
  - 每模块 5 维数值特征：`B、P、F、N、risk`（来自各模块 `metrics.py`）。
  - `next_op_onehot(M)`: 在该模块目标序列中“下一应选算子”的 one-hot 提示（M 为全局干预算子个数）。
  - `pos`: 当前指针在该模块目标序列的归一化位置。

它是系统“当前状态”的数值化快照，供策略网络作决策。

### 3. 输出 Action 代表什么？如何映射到算子？

- 输出是一个离散动作 id（整数，范围 `0..M-1`），对应全局“干预算子词表”中的某个算子。
- 训练开始时，系统会把 `env.op2idx` 写成 `op_index.json`（训练目录），用于把整型动作反查为算子名；在报告或推理解释时直接引用即可。

### 4. 输出对输入的“预测”含义

- 策略网络输出的是 `π(a|s)`：给定当前 Observation，选择每个干预算子的概率分布。这是一种“对最优行为的预测”。
- 当某个动作概率高时，本质上是在“预测”此动作对当前状态更可能带来长期更高的累积回报。

### 5. 训练数据从哪里来？

- 本框架不使用离线监督样本；训练样本 `(s, a, r, s')` 在与环境交互过程中动态生成并写入回放池（ReplayBuffer）。
- `operator_crosswalk_train.json` 提供的是“动作空间与目标序列”，不是 `(s,a,r,s')` 样本。

### 6. 奖励与指针推进（rlsac2 的语义）

- 奖励：`Δrisk − λ·cost`（risk/cost 来自各模块 `metrics.py`；`λ` 可在 `config.json` 的 `reward_lambda` 调整）。
- 指针推进：只有当选中的动作与“该模块目标序列中的下一应选算子”一致时，才推进指针与模块状态，以体现“模仿/预热”的属性；否则不推进，但不会报错。

### 7. 流程图（概念）

```mermaid
graph TD
    A[Observation 向量] --> B{策略网络 π(a|s)}
    B --> C[动作分布 over 全局干预算子]
    subgraph 指令集
        D[op_index.json: id→operator]
    end
    C --> E[采样/选最大概率 → 动作id]
    E --> D
    D --> F[具体算子]
    F --> G{环境：SequenceEnv}
    G --> H[下一 Observation]
    G --> I[奖励 r = Δrisk − λ·cost]
    subgraph 回放池
        J[(s, a, r, s')]
    end
    J --> B
```

小结：`operator_crosswalk_train.json` 是“做什么”的菜单；Observation 是“当前情况”的报告；Action 是基于报告从菜单中做出的“最佳选择”的预测，目标是使长期累积奖励最大化。

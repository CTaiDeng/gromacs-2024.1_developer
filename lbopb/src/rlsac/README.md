# RLSAC 子模块说明（LBOPB 离散 SAC）

- 沟通语言：中文（简体）
- 本仓库为 GROMACS 的非官方派生版，说明文档仅用于方法学与技术演示，不构成医学建议或临床诊断/治疗方案；亦不用于任何实际诊疗决策或药物使用指导。

## 实现概要

- 新增环境 `LBOPBSequenceEnv`：将 `operator_crosswalk_train.json` 的“模块→算子序列”直接转为离散 SAC 的训练样本，已接入训练主程序。
  - 新文件：`lbopb/src/rlsac/sequence_env.py`
  - 选择开关：在 `lbopb/src/rlsac/config.json` 中设置 `"env": "sequence"` 即启用；否则默认使用 `DummyEnv`。
  - 接入点：`lbopb/src/rlsac/train.py`（环境选择与日志输出）。

## 输入/输出设计

- 输入 observation（向量拼接，长度 = 7 + M + 1）：
  - 模块 one‑hot 7 维：顺序固定为 `pem / pdem / pktm / pgom / tem / prm / iem`。
  - 下一步应当执行的算子 one‑hot M 维：M 为本案例包出现的基本算子去重后的总数（如 `Inflammation/Bind/Dose/...`）。
  - 位置标记 1 维：当前指针在本模块序列中的归一化位置 `pos = ptr / len(seq)`。
- 输出 action（离散标量）：
  - 全局算子字典的索引（`0..M‑1`）。例如 `Inflammation -> 0, Bind -> 1, ...`（具体映射见训练日志或在环境内 `op2idx`）。

## 输出相对输入的意义

- 策略网络输出对每个离散算子的概率分布 `π(a|s)`，表示“在当前 observation 下，预测应选择哪个算子作为下一步”的置信度；并非对输入状态向量的重建。
- 评论家网络输出 `Q(s,:)`，是“对每个算子在当前 observation 下的长期回报”的预测（例如还可完成多少正确步骤的期望）。

## 训练信号与转移

- 奖励：选择的 `action` 与 observation 中“下一步应当的算子”一致 → 奖励 `+1`；否则 `0`（可按需调为负值）。
- 转移：正确则推进当前模块序列指针 `ptr += 1`；完成当前模块序列后 `done = True`，切换到下一个非空模块（或由上层 `reset`）。
- 动作空间：与案例包出现的算子全集一致，便于统一学习“跨模块的基本算子选择”。

## 如何启用

- 修改配置文件：`lbopb/src/rlsac/config.json` 增加或设置
  - `"env": "sequence"`
- 运行训练：
  - `python lbopb/src/rlsac/train.py --new`
- 观察日志：首次会打印环境选择与 `obs_dim / n_actions`，并按奖励收敛；可用运行目录内的 `run_infer.py` 进行推理与 CSV 记录。

## 观察与输出的“预测”关系（直观说明）

- 例：在 `pem` 模块，序列为 `[Inflammation, Apoptosis]`，当前处于 `ptr = 0`
  - `observation = [pem-onehot=1, next_op=Inflammation-onehot=1, pos=0]`
  - 策略输出 `π(a|s)` 中，`Inflammation` 的概率应最高（学习正确“预测下一步算子”）。
  - 选择 `Inflammation` → 奖励 `+1`，指针进入 `ptr = 1`，下一步 observation 指示 `Apoptosis`。
  - 评论家 `Q(s,:)` 同时学习“若现在选某个算子，后续还能拿到多少奖励”的预期值。
- 因此，输出并不重建输入向量；它“预测的是：在当前输入（模块/所需下一算子/位置）下，哪个动作最优”。

## 与需求文档对齐要点（摘）

- 状态向量是对“立体状态/上下文”的标准化描述，这里最小可用地编码为“当前模块、该模块下一算子、相对位置”；属于可扩展的 `vectorize_state(s)` 版本。
- 动作为“全局基本算子”的离散选择；策略输出为概率、评论家输出为各动作的价值向量。
- 回报与转移机制可按文档进一步增强（引入风险/代价 `Δrisk − λ·cost` 等），当前实现用于把案例 JSON 直接变为可用训练样本（模仿/预热）。

## 可选增强（更贴近物理/药理目标）

- 扩展 observation：为每个模块追加数值特征（如 `B/P/F/N`、风险/代价估计），形成更丰富的 `vectorize_state`。
- 奖励函数：由 “是否命中下一算子” 改为 “`Δrisk − λ·cost`”，结合 `lbopb/src/*/metrics.py` 中的指标函数。
- 数据管线：保留当前 `sequence` 环境用于预填与模仿学习，再切换到物理驱动环境做微调。

## op→id 对照表（导出建议）

- 训练时 `LBOPBSequenceEnv` 内部维护 `op2idx`（全局算子到索引的映射）。为便于解释模型输出，可在训练开始后将其写入运行目录：
  - 参考示例（伪码）：
    - 在 `train()` 初始化 `env` 后：
      - 若 `hasattr(env, "op2idx")`，则将 `env.op2idx` 序列化为 `op_index.json` 写入当前 `run_dir`。
- 该映射可用于将策略输出向量的第 `i` 维，反查为具体的算子名，便于可视化与报表生成。

---

- 免责声明：本说明仅用于方法学与技术演示，不构成医学建议或临床诊断/治疗方案；亦不用于任何实际诊疗决策或药物使用指导。

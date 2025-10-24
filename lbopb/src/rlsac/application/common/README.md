# 观测量整数化映射（Observation Quantization）与公共环境说明

本目录提供将“生命体全息状态”的连续观测量（7 域 × [B, perim, fidelity, n, risk]）量化为“整数索引”的通用机制与环境，便于在强化学习中统一
Observation 的维度与取值域。

- 量化配置：`observation_map.json`
- 量化工具：`observation.py`（`ObservationQuantizer`）
- PEM 联络环境（整数观测版）：`pem_connector_env.py`

## 文件与职责

- `observation_map.json`
    - `modules`：为 7 个域（pem/pdem/pktm/pgom/tem/prm/iem）分别给出 5 个量（`b/perim/fidelity/n/risk`
      ）的“可接受离散值序列”（档位/粒度）。例如：
        - `perim: [0, 0.25, 0.5, 1, 2]` 表示该域的 perim 维度被量化为 5 个档位（索引 0..4）。
    - `quantize_mode`：量化策略，当前仅支持 `nearest`（将连续数值映射到离散序列中“最近”的一个值的索引）。
    - `dim_index`：35 维观测的维度编号（顺序），约定为 7 域依次拼接的 `[b, perim, fidelity, n, risk]`，例如：
        - ["pem.b","pem.perim","pem.fidelity","pem.n","pem.risk", ..., "iem.risk"]（共 35 项）。

- `observation.py`
    - `ObservationQuantizer(map_path)`：载入 `observation_map.json`；
    - `quantize_state(domain, state)`：将某一域的连续状态（需含 `b/perim/fidelity/n_comp` 字段）量化为 4 个整数索引（不包含
      risk）；
    - `quantize_full(states_by_domain, risk_by_domain)`：对 7 域分别量化并附加每域 `risk` 的索引，得到长度 35 的整型观测向量。

- `pem_connector_env.py`
    - 以 PEM 为动作空间、以“联络一致性评分”为奖励的统一环境（Observation 为整型索引，Action 为 PEM 算子包 id）：
        - ObservationSpace：`int32[35]`；
        - ActionSpace：`int32[1]`，范围 `0..(pem_package_count-1)`；
        - Reward：`ΣΔrisk + consistency − λ·Σcost`（显著错误 → 0；仅警告时可按配置启用 Gemini 辅助判定）。

## 量化映射示例

以 pem 域为例，`observation_map.json` 中（节选）：

```
"pem": {
  "b":       [0, 1, 2, 3, 4, 6, 8, 10, 12],
  "perim":   [0, 0.5, 1, 1.5, 2, 3, 5],
  "fidelity": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "n":       [0, 1, 2, 3, 5, 8],
  "risk":    [0, 1, 2, 3, 5, 8, 13]
}
```

- 若某时刻 pem 域连续状态为：`b=1.1, perim=0.6, fidelity=0.58, n_comp=2`，在 `nearest` 模式下对应索引：
    - `b_idx = 1`（1.1 最近的是 1）
    - `perim_idx = 2`（0.6 最近的是 0.5 或 1，任选最近者，这里 0.5 对应索引 2）
    - `fidelity_idx = 3`（0.58 最近的是 0.6）
    - `n_idx = 2`（2 最近就是 2）
- 对应 risk 的离散序列为 `[0,1,2,3,5,8,13]`，若 risk 连续值为 `2.7`，则 `risk_idx = 3`（最近为 3）。
- 将 7 域依次量化并拼接（每域 5 维），得到长度 35 的整型 Observation。

## 与训练的对接

- 应用侧（hiv/nsclc）的训练脚本已切换为使用公共环境：
    - `from lbopb.src.rlsac.application.common.pem_connector_env import PemConnectorEnv`
    - 环境构造示例：

```
packages_dir = "lbopb/src/rlsac/kernel/rlsac_pathfinder"
observation_map = "lbopb/src/rlsac/application/common/observation_map.json"
env = PemConnectorEnv(
  packages_dir=packages_dir,
  observation_map=observation_map,
  cost_lambda=0.2,
  eps_change=1e-3,
  use_llm_oracle=False,
)
```

- 说明：
    - `packages_dir` 指向第一阶段 `rlsac_pathfinder` 输出的辞海目录（至少包含 `pem_operator_packages.json`）。
    - `observation_map` 为本目录的量化配置文件，可针对项目需求调整粒度。

## Demo 样本（与维度编号）

- `lbopb/src/rlsac/application/rlsac_hiv/train_data/samples.demo.json`
- `lbopb/src/rlsac/application/rlsac_nsclc/train_data/samples.demo.json`

两者结构一致：

```
{
  "dim_index": ["pem.b","pem.perim",...,"iem.risk"],
  "samples": [
    {
      "state": [35个整数索引],
      "action": 0,
      "reward": 1.0,
      "next_state": [35个整数索引],
      "done": false
    },
    ...
  ]
}
```

- `dim_index` 给出了 35 维观测的维度编号顺序，便于解释与可视化。
- `state/next_state` 为量化后的整型索引；`reward` 依然是标量浮点（来自联络一致性评分）。

## 自定义与扩展

- 粒度调整：
    - 在 `observation_map.json` 中修改对应域、对应量的离散值数组（保持升序）即可，越多值粒度越细。
- 多模式支持：
    - 目前仅 `nearest`，如需 `bucket/linear` 等，可在 `ObservationQuantizer` 中扩展。
- 维度顺序：
    - 若需改变 35 维的拼接顺序，请同步更新 `dim_index` 与量化逻辑（不建议频繁更改）。

## 与 LLM 判定的关系

- 本目录只负责 Observation 的整数化映射与公共环境；
- 单域/联络判定逻辑（含 syntax_checker 与可选 Gemini 调用）位于：
    - Pathfinder: `lbopb/src/rlsac/kernel/rlsac_pathfinder/oracle.py`
    - Connector: `lbopb/src/rlsac/kernel/rlsac_connector/oracle.py`
- 配置 `use_llm_oracle` 为 true 且仅在语法检查出现“警告”时才会调用 LLM 进行辅助判定（显著错误直接判 0）。

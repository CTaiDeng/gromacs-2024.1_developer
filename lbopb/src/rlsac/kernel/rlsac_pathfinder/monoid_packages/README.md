该目录用于存放各幺半群在训练结束后，从每次运行的 `debug_dataset.json` 中筛选出的“正确（label=1）”算子包。

规范：
- 纳入策略：仅包含 label=1 的样本，按序列去重；同一序列多次出现时保留分数更高者。
- 排序规则：score(desc) -> length(asc) -> sequence（字典序）。
- 评分定义：score = delta_risk - cost_lambda * cost。
- 文件命名：沿用 `<domain>_operator_packages.json`，但统一归档在本目录下。

写入方式：
- 由 `lbopb.src.rlsac.kernel.rlsac_pathfinder.package_store.ingest_from_debug_dataset()` 自动创建/更新。

扩展字段（可选，向后兼容）：
- 背景：基础 JSON 仅记录算子名称序列与聚合指标，无法复现“参数化动作”。为确保可操作性、可复现与可审计，允许追加以下字段（不影响既有消费者）：
- `op_space_id`：算子空间版本号，如 `pem.v1`。
- `op_space_ref`：空间定义文件相对路径，如 `lbopb/src/rlsac/kernel/rlsac_pathfinder/operator_spaces/pem_op_space.v1.json`。
- `ops_detailed`：逐步细化序列（与 `sequence` 一一对应），每步项包含：
  - `name`：算子名。
  - `params`：该步使用的离散化参数取值（显式数值）。
  - `grid_index`：可选；在空间网格中的索引（例如 `[i0, i1, ...]`）。
  - `op_idx`：可选；当前环境内的动作索引（便于对齐 `env_pem.op2idx`）。
- 轻量别名（可选）：
  - `op_index_seq`：动作索引序列（整型，与 `sequence` 对齐）。
  - 注意：不建议提交 `op_param_seq`（数值序列）；应以 `grid_index` + `op_space_ref` 表达参数，服务端可据此反查并填充 `params`。
- 复现实验上下文（可选）：
  - `env_state`：`init_state`、`goal`（含容差）、`max_steps`、`seed`、`code_commit`。
  - `trace`：逐步 `PEMState` 快照与每步指标（如 `reward`/`dist`）。

最小示例（节选）：

```json
{
  "id": "pkg_pem_6590899514",
  "domain": "pem",
  "sequence": ["Apoptosis", "Carcinogenesis"],
  "length": 2,
  "delta_risk": 0.045,
  "cost": 1.686,
  "score": -0.2922,
  "created_at": 1761348557,
  "updated_at": 1761348557,
  "source": "debug_dataset",

  "op_space_id": "pem.v1",
  "op_space_ref": "lbopb/src/rlsac/kernel/rlsac_pathfinder/operator_spaces/pem_op_space.v1.json",
  "ops_detailed": [
    {
      "name": "Apoptosis",
      "params": { "gamma_b": 0.2, "gamma_n": 0.1, "gamma_p": 0.15, "delta_f": 0.1 },
      "grid_index": [1,1,1,2],
      "op_idx": 0
    },
    {
      "name": "Carcinogenesis",
      "params": { "k_b": 0.25, "k_p": 0.15, "k_f": 0.1, "dn": 0 },
      "grid_index": [2,1,1,0],
      "op_idx": 3
    }
  ]
}

双重验证（syntax_checker + Gemini）
- 配置：
  - `config.json` 中设置：
    - `use_llm_oracle: true`（启用 LLM）
    - `llm_include_params: true`（提示携带参数化动作）
    - `llm_force_dual_validation: true`（即使无 warnings 也提交 LLM 复核）
    - `llm_request_interval_sec`（请求间隔秒，用于配额节流）
    - `gemini_model_choose` + `GEMINI_MODEL`（模型选择）
- 独立验证脚本：`verify_with_gemini.py`
  - 位置：本目录下。
  - 功能：对七个 `*_operator_packages.json` 逐条执行“语法检查 + LLM 校验”，输出 `verify_<domain>.json` 与 `verify_summary.json`。
  - 规则：
    - 语法检查：若条目含 `ops_detailed`，调用 `check_package(pkg)`；否则调用 `check_sequence(sequence)`。
    - LLM 校验：使用 `build_pathfinder_prompt(domain, sequence, ops_detailed, {op_space_id/op_space_ref})` 并调用 `call_llm`，返回 `1/0`。
    - 双重通过：两者均通过才计入 `both_ok`。
  - 用法：
    - `python lbopb/src/rlsac/kernel/rlsac_pathfinder/monoid_packages/verify_with_gemini.py`
    - 依赖 `config.json` 的 LLM 配置与请求间隔；若条目未含 `ops_detailed`，脚本将按 v1 空间自动补全中位参数进行提示。

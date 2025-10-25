# 算子空间（Operator Spaces）

本目录维护各幺半群（七域）的离散化“算子参数网格”定义文件（v1）。

目的：
- 统一用 `ops_detailed[*].grid_index` + `op_space_ref` 反查得到每步算子的 `params`；
- 避免在算子包中直接提交“数值的序列”（`op_param_seq`），确保可复现、可审计且易于版本化；
- 支持逐版本扩展粒度与维度。

## 规范要点
- 算子空间定义（建议）：
  - 文件：如 `lbopb/src/rlsac/kernel/rlsac_pathfinder/operator_spaces/pem_op_space.v1.json`
  - 内容：对每个基本算子的参数提供有限离散网格；消费者可据此用 `grid_index` 反查数值或校验 `params` 合法性。
- 索引规则：
  - `ops_detailed[*].grid_index` 的下标顺序，与空间文件中该算子 `params` 字段的“键顺序”保持一致。
- 版本化：
  - 文件命名 `{domain}_op_space.v{n}.json`；`space_id` 同步为 `{domain}.v{n}`；
  - 非兼容更新必须提升主版本；兼容性扩展（仅追加更细刻度）建议在尾部追加以保持既有下标不变。

## 文件清单（v1）
- PEM：`operator_spaces/pem_op_space.v1.json`（`space_id: pem.v1`）
- PDEM：`operator_spaces/pdem_op_space.v1.json`（`space_id: pdem.v1`）
- PKTM：`operator_spaces/pktm_op_space.v1.json`（`space_id: pktm.v1`）
- PGOM：`operator_spaces/pgom_op_space.v1.json`（`space_id: pgom.v1`）
- TEM：`operator_spaces/tem_op_space.v1.json`（`space_id: tem.v1`）
- PRM：`operator_spaces/prm_op_space.v1.json`（`space_id: prm.v1`）
- IEM：`operator_spaces/iem_op_space.v1.json`（`space_id: iem.v1`）

## 整体维度分类与统计（v1）
- 域与算子数：
  - PEM：4 个（Apoptosis / Metastasis / Inflammation / Carcinogenesis）
  - PDEM：6 个（Bind / Signal / Desensitization / Antagonist / Potentiation / InverseAgonist）
  - PKTM：7 个（Dose / Absorb / Distribute / Metabolize / Excrete / Bind / Transport）
  - PGOM：7 个（Activate / Repress / Mutation / RepairGenome / EpigeneticMod / PathwayInduction / PathwayInhibition）
  - TEM：7 个（Exposure / Absorption / Distribution / Lesion / Inflammation / Detox / Repair）
  - PRM：6 个（Ingest / Exercise / Hormone / Proliferation / Adaptation / Stimulus）
  - IEM：6 个（Activate / Suppress / Proliferate / Differentiate / CytokineRelease / Memory）
- 细化维度（参数网格）来源：各域 `operator_spaces/{domain}_op_space.v1.json`。

## 细化维度业务解释（按域）
- PEM
  - Apoptosis：`gamma_b`（负荷下降比例）、`gamma_n`（组分收敛比例）、`gamma_p`（边界下降）、`delta_f`（保真上升）。
  - Metastasis：`alpha_n`（组分增加量）、`alpha_p`（边界上升）、`beta_b`（负荷稀释）、`beta_f`（保真下降）。
  - Inflammation：`eta_b/eta_p/eta_f`（负荷/边界上升、保真下降）、`dn`（组分微增）。
  - Carcinogenesis：`k_b/k_p/k_f`（负荷/边界上升、保真下降）、`dn`（可微增）。
- PDEM
  - Bind：`alpha_b/alpha_p`（占有提升）、`delta_f`（保真上升）、`dn`（位点增量）。
  - Signal：`beta_b/beta_p`（效应增强）、`delta_f`（保真上升）。
  - Desensitization：`gamma_*`（效应/边界/保真/位点比例下降）。
  - Antagonist：`k_*`（抑制通路，效应/边界/保真下降）。
  - Potentiation：`xi_b/xi_p`（增强）、`delta_f`（保真上升）。
  - InverseAgonist：`rho_*`（自发活性与边界下降、保真下降）。
- PKTM
  - Dose：`delta_b`（加性剂量）、`alpha_p/alpha_f`（边界上升/保真下降）。
  - Absorb：`alpha_*`（负荷/边界上升、保真下降）。
  - Distribute：`alpha_n/p/b`（隔室/边界/负荷上升）。
  - Metabolize/Excrete：`gamma_*` 或 `rho_*`（负荷/边界下降）、`delta_f`（保真上升）。
  - Bind/Transport：`theta_*`/`xi_*`（比例调节，正负可配）。
- PGOM
  - Activate：`alpha_b/alpha_p` 上升、`delta_f` 上升、`dn` 模块增量。
  - Repress：`gamma_*` 下降、`dn` 可微调。
  - Mutation：`alpha_n/p` 上升、`beta_b` 变化、`beta_f` 下降。
  - RepairGenome：`rho_*` 下降、`delta_f` 上升。
  - EpigeneticMod：`theta_*`（表达/保真/边界可正可负）、`dn` 微调。
  - PathwayInduction/Inhibition：`alpha_*` 上升 / `gamma_*` 下降、`delta_f` 上升/下降。
- TEM
  - Exposure/Absorption：`alpha_*`/`beta_*`（负荷/边界上升、保真下降）、`dn` 灶数增量。
  - Distribution：`alpha_n/p/b/f`（扩散相关比例）。
  - Lesion/Inflammation：`k_*`/`eta_*`（损伤/炎症加剧）。
  - Detox/Repair：`gamma_*`/`rho_*` 下降、`delta_f` 上升、`rho_n` 组分收敛。
- PRM
  - Ingest/Stimulus：`alpha_*`/`xi_*` 上升、`dn` 可增。
  - Exercise/Adaptation：`gamma_*`/`eta_*` 下降、`delta_f` 上升、`eta_n` 收敛。
  - Hormone：`theta_*` 比例调谐，可正可负。
- IEM
  - Activate/Proliferate：`alpha_*` 上升、`theta_f` 调节、`dn` 可增。
  - Suppress/Memory：`gamma_*`/`rho_*` 下降、`delta_f` 上升、`gamma_n/rho_n` 收敛。
  - Differentiate：`delta_f` 上升、`theta_p/b` 微调、`dn` 微调。
  - CytokineRelease：`xi_*` 上升、`dn` 增量。

## 粒度解释与扩展策略
- 粒度（Granularity）= 参数网格每个维度的离散刻度数量（例如 `gamma_b` 取 3 档即粒度=3）。
- 扩展方式：
  - 增加现有参数的取值刻度（更细粒度）；
  - 引入新的参数键（更高维度）；
  - 新增算子（扩充动作空间）。
- 版本化：
  - 扩展需在 `operator_spaces/{domain}_op_space.v{n}.json` 中完成，`space_id` 递增；
  - 算子包可继续引用老版本以保持可复现性。

## 维度包的可扩展维度空间
- 每条算子包可指定：
  - `op_space_id` 与 `op_space_ref` 指向某一版本的空间；
  - `ops_detailed[*].grid_index` 可在新版本空间下取更细的索引；
  - 未变更历史包时，服务端仍可通过老版本空间反查，保持回放稳定。
- 建议：
  - 新版本扩展尽量向后兼容（在尾部追加刻度，保持既有下标不变）。
  - 对非兼容更新，务必提升 `space_id` 并在提交说明中标注迁移策略。


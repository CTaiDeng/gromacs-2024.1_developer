# LBOPB 幺半群算子联络预览（自动生成）

本文件由脚本自动生成（请勿手工编辑）。数据源：`lbopb/src/operator_crosswalk.json`。

## 概念与术语

- 幺半群（Monoid）：带有结合律与单位元的代数结构；本文中各模块（PEM/PRM/TEM/PKTM/PGOM/PDEM/IEM）均是非交换幺半群。
- 基本算子（Basic Operator）：各模块的最小过程单元（如 Dose/Absorb/Activate/Repair 等）。
- 规范化算子包（Canonical Package）：仅由基本算子构成、能代表通用时序的序列（如 ADME 管线、损伤-修复链）。
- 联络（Crosswalk）：以语义标签为桥，建立跨模块基本算子的类比映射与包的对齐规则。
- 幂集（Powerset）：在约束下枚举仅基本算子构成的序列（自由幺半群），并可结合常用序列与生成器。

## 对齐原则与使用建议

- 对齐原则：以 B（负荷）、P（边界/接口）、N_comp（单元/克隆/隔室）、F（保真/质量）的变化方向作为一阶语义对齐基准。
- 语义标签：activate/repress/proliferate/repair 等用于抽象过程本质，跨模块类比时优先匹配标签。
- 联络使用：按标签 crosswalk 选择目标模块基本算子，保持序列顺序（非交换性重要）。
- 幂集生成：仅基本算子构成的自由幺半群序列，遵循各模块 powersets 的约束与常用家族/生成器。
- 风险与代价：评估跨模块路径时，可结合各模块 metrics（如 tox_risk/eff_risk/topo_risk、action_cost）进行筛选。

## 基本信息

| 键 | 值 |
| :--- | :--- |
| 版本 | 0.1.0 |
| 模块 | pem, prm, tem, pktm, pgom, pdem, iem |

## 知识库文档引用

- `my_docs/project_docs/1761062400_病理演化幺半群 (PEM) 公理系统.md`
- `my_docs/project_docs/1761062401_生理调控幺半群 (PRM) 公理系统.md`
- `my_docs/project_docs/1761062403_毒理学效应幺半群 (TEM) 公理系统.md`
- `my_docs/project_docs/1761062404_药代转运幺半群 (PKTM) 公理系统.md`
- `my_docs/project_docs/1761062405_药理基因组幺半群 (PGOM) 公理系统.md`
- `my_docs/project_docs/1761062406_药效效应幺半群 (PDEM) 公理系统.md`
- `my_docs/project_docs/1761062407_免疫效应幺半群 (IEM) 公理系统.md`
- `my_docs/project_docs/1761062408_《病理演化幺半群》的核心构造及理论完备性.md`
- `my_docs/project_docs/1761062409_《生理调控幺半群》的核心构造及理论完备性.md`
- `my_docs/project_docs/1761062410_《毒理学效应幺半群》的核心构造及理论完备性.md`
- `my_docs/project_docs/1761062411_《药代转运幺半群》的核心构造及理论完备性.md`
- `my_docs/project_docs/1761062412_《药理基因组算子幺半群》的核心构造及理论完备性.md`
- `my_docs/project_docs/1761062413_《药效效应幺半群》的核心构造及理论完备性.md`
- `my_docs/project_docs/1761062414_《免疫效应幺半群》的核心构造及理论完备性.md`

## 语义标签（统一字典）

| 标签 | 说明 |
| :--- | :--- |
| absorb | 吸收 |
| activate | 激活/诱导 |
| antagonist | 拮抗/阻断 |
| bind | 结合/占有 |
| boundary_down | P↓（边界/接口减少） |
| boundary_up | P↑（边界/接口增加） |
| component_down | N_comp↓（单元/克隆/隔室减少） |
| component_up | N_comp↑（单元/克隆/隔室增加） |
| cytokine | 细胞因子释放/风暴 |
| damage | 损伤/病灶形成 |
| desens | 去敏/耐受 |
| differentiate | 分化/成熟 |
| distribute | 分布/迁移 |
| dose | 给药/外源注入 |
| excrete | 排泄 |
| fidelity_down | F↓（保真/功能下降） |
| fidelity_up | F↑（保真/功能提升） |
| identity | 单位元/不改变状态 |
| inflammation | 炎症/反应增强 |
| inverse_agonist | 反向激动 |
| load_down | B↓（负荷/量/强度减少） |
| load_up | B↑（负荷/量/强度增加） |
| memory | 免疫记忆/长期稳态 |
| metabolize | 代谢 |
| mutation | 突变/致癌转化 |
| potentiate | 增效/正变构 |
| proliferate | 增殖/扩增 |
| repair | 修复/解毒/愈合 |
| repress | 抑制/抑制通路 |
| signal | 信号传导 |
| transport | 跨膜/转运体介导转运 |

## 各模块基本算子与标签

### pem

| 基本算子 | 语义标签 |
| :--- | :--- |
| Apoptosis | load_down, component_down, boundary_down, fidelity_up |
| Carcinogenesis | mutation, load_up, boundary_up, fidelity_down |
| Identity | identity |
| Inflammation | inflammation, load_up, boundary_up, fidelity_down |
| Metastasis | component_up, boundary_up, fidelity_down, load_down |

### prm

| 基本算子 | 语义标签 |
| :--- | :--- |
| Adaptation | repair, load_down, boundary_down, fidelity_up, component_down |
| Exercise | load_down, boundary_down, fidelity_up |
| Hormone | activate, repress |
| Identity | identity |
| Ingest | load_up, boundary_up, fidelity_up, component_up |
| Proliferation | proliferate, component_up, boundary_up, load_up, fidelity_down |
| Stimulus | activate, load_up, boundary_up, fidelity_down |

### tem

| 基本算子 | 语义标签 |
| :--- | :--- |
| Absorption | absorb, load_up, boundary_up, fidelity_down |
| Detox | repair, load_down, boundary_down, fidelity_up |
| Distribution | distribute, component_up, boundary_up, load_up, fidelity_down |
| Exposure | dose, damage, load_up, boundary_up, fidelity_down |
| Identity | identity |
| Inflammation | inflammation, load_up, boundary_up, fidelity_down |
| Lesion | damage, component_up, boundary_up, load_up, fidelity_down |
| Repair | repair, load_down, boundary_down, fidelity_up, component_down |

### pktm

| 基本算子 | 语义标签 |
| :--- | :--- |
| Absorb | absorb, load_up, boundary_up, fidelity_down |
| Bind | bind, fidelity_up |
| Distribute | distribute, component_up, boundary_up, load_up |
| Dose | dose, load_up, boundary_up, fidelity_down |
| Excrete | excrete, load_down, boundary_down, fidelity_up |
| Identity | identity |
| Metabolize | metabolize, load_down, boundary_down, fidelity_up |
| Transport | transport, boundary_up |

### pgom

| 基本算子 | 语义标签 |
| :--- | :--- |
| Activate | activate, load_up, boundary_up, fidelity_up |
| EpigeneticMod | activate, repress |
| Identity | identity |
| Mutation | mutation, component_up, boundary_up, fidelity_down |
| PathwayInduction | activate, load_up, boundary_up, fidelity_up |
| PathwayInhibition | repress, load_down, boundary_down, fidelity_down |
| RepairGenome | repair, load_down, boundary_down, fidelity_up, component_down |
| Repress | repress, load_down, boundary_down, fidelity_down |

### pdem

| 基本算子 | 语义标签 |
| :--- | :--- |
| Antagonist | antagonist, load_down, boundary_down, fidelity_down |
| Bind | bind, load_up, boundary_up, fidelity_up |
| Desensitization | desens, load_down, boundary_down, fidelity_down |
| Identity | identity |
| InverseAgonist | inverse_agonist, load_down, boundary_down, fidelity_down |
| Potentiation | potentiate, load_up, boundary_up, fidelity_up |
| Signal | signal, load_up, boundary_up, fidelity_up |

### iem

| 基本算子 | 语义标签 |
| :--- | :--- |
| Activate | activate, load_up, boundary_up, fidelity_up |
| CytokineRelease | cytokine, load_up, boundary_up, fidelity_down, component_up |
| Differentiate | differentiate, fidelity_up, boundary_up |
| Identity | identity |
| Memory | memory, repair, load_down, boundary_down, fidelity_up, component_down |
| Proliferate | proliferate, component_up, boundary_up, load_up |
| Suppress | repress, load_down, boundary_down, fidelity_down |

## 跨模块联络（按语义标签）

### activate

| 模块 | 基本算子 |
| :--- | :--- |
| prm | Stimulus、Hormone |
| pgom | Activate、PathwayInduction |
| pdem | Signal、Potentiation |
| iem | Activate |

### bind

| 模块 | 基本算子 |
| :--- | :--- |
| pktm | Bind |
| pdem | Bind |

### damage

| 模块 | 基本算子 |
| :--- | :--- |
| pem | Carcinogenesis |
| tem | Lesion、Exposure |
| pktm | Dose |
| pgom | Mutation |
| pdem | InverseAgonist |
| iem | CytokineRelease |

### desens

| 模块 | 基本算子 |
| :--- | :--- |
| prm | Adaptation |
| pdem | Desensitization |
| iem | Suppress |

### dose

| 模块 | 基本算子 |
| :--- | :--- |
| prm | Ingest |
| tem | Exposure |
| pktm | Dose |

### inflammation

| 模块 | 基本算子 |
| :--- | :--- |
| pem | Inflammation |
| prm | Stimulus |
| tem | Inflammation |
| iem | CytokineRelease |

### proliferate

| 模块 | 基本算子 |
| :--- | :--- |
| pem | Metastasis |
| prm | Proliferation |
| tem | Distribution、Lesion |
| pktm | Distribute |
| pgom | Mutation |
| iem | Proliferate |

### repair

| 模块 | 基本算子 |
| :--- | :--- |
| pem | Apoptosis |
| prm | Adaptation |
| tem | Detox、Repair |
| pktm | Metabolize、Excrete |
| pgom | RepairGenome |
| iem | Memory |

### repress

| 模块 | 基本算子 |
| :--- | :--- |
| prm | Hormone |
| pgom | Repress、PathwayInhibition |
| pdem | Antagonist |
| iem | Suppress |

### signal

| 模块 | 基本算子 |
| :--- | :--- |
| pgom | PathwayInduction |
| pdem | Signal |
| iem | Activate |

## 规范化算子包（仅基本算子）

### activation_signal

#### 说明：

激活-信号-去敏（或拮抗）效应链：适用于受体结合-信号放大-耐受/阻断的动力学。

| 模块 | 算子序列 |
| :--- | :--- |
| pem | Inflammation → Apoptosis |
| prm | Stimulus → Adaptation |
| tem | Inflammation → Detox |
| pktm | Bind → Transport |
| pgom | Activate → PathwayInduction → PathwayInhibition |
| pdem | Bind → Signal → Desensitization |
| iem | Activate → Differentiate → Suppress |

### adme_pipeline

#### 说明：

药代学 ADME 管线：从给药/暴露到清除，兼容 TEM/PKTM/PDEM/PGOM/IEM/PRM/PEM 的跨视角投影。

| 模块 | 算子序列 |
| :--- | :--- |
| pem | Inflammation → Apoptosis |
| prm | Stimulus → Adaptation |
| tem | Exposure → Absorption → Distribution → Detox |
| pktm | Dose → Absorb → Distribute → Metabolize → Excrete |
| pgom | Activate → PathwayInduction → RepairGenome |
| pdem | Bind → Signal → Desensitization |
| iem | Activate → Proliferate → Memory |

### damage_heal

#### 说明：

损伤-炎症-修复序列：适用于组织损伤后的炎症反应与恢复过程的统一抽象。

| 模块 | 算子序列 |
| :--- | :--- |
| pem | Inflammation → Apoptosis |
| prm | Stimulus → Adaptation |
| tem | Lesion → Inflammation → Detox → Repair |
| pktm | Dose → Metabolize → Excrete |
| pgom | Mutation → RepairGenome |
| pdem | InverseAgonist → Potentiation |
| iem | CytokineRelease → Suppress → Memory |

## 幂集算法配置与常用幂集（仅基本算子）

### pem

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Inflammation, Apoptosis, Metastasis, Carcinogenesis, Identity |
| 最大长度 | 3 |
| 约束 | 禁止相邻重复, 跳过Identity |

#### 说明：

PEM 以病理负荷/边界/单元/保真为核心观测。常用序列体现炎症驱动的损伤进展与凋亡型修复。

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| progression | Inflammation → Carcinogenesis → Metastasis |
| repair_path | Inflammation → Apoptosis |

| 族名 | 说明 |
| :--- | :--- |
| progression | 炎症—致癌—转移的进展链，负荷/边界/单元多上升，F 下降。 |
| repair_path | 炎症后凋亡与清除，偏向负向负荷/边界收敛，F 提升。 |

| 生成器名 | 链式模式 |
| :--- | :--- |
| progress_heal | Inflammation → (Carcinogenesis | Metastasis | Identity) → (Apoptosis | Identity) |

| 生成器名 | 说明 |
| :--- | :--- |
| progress_heal | 从炎症出发，允许进入进展分支或保持原状，随后进入修复/保持。 |

| 生成器名 | 示例（前5条） |
| :--- | :--- |
| progress_heal | Inflammation → Carcinogenesis → Apoptosis；Inflammation → Carcinogenesis → Identity；Inflammation → Metastasis → Apoptosis；Inflammation → Metastasis → Identity；Inflammation → Identity → Apoptosis |

### prm

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Ingest, Exercise, Hormone, Proliferation, Adaptation, Stimulus, Identity |
| 最大长度 | 3 |
| 约束 | 禁止相邻重复, 跳过Identity |

#### 说明：

PRM 强调能量/边界/保真的协同调控，常用链路体现‘摄入-运动-适应’与激素调谐。

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| hormonal_mod | Hormone → Adaptation |
| performance | Ingest → Exercise → Adaptation |
| recovery | Stimulus → Adaptation |

| 族名 | 说明 |
| :--- | :--- |
| hormonal_mod | 激素短期调谐后以适应实现稳态回归。 |
| performance | 摄入补给后训练并以适应收束，提升 F，降低 B/P。 |
| recovery | 刺激激发后通过适应回归稳态。 |

| 生成器名 | 链式模式 |
| :--- | :--- |
| train_recover | Ingest → Exercise{1..3} → Adaptation |

| 生成器名 | 说明 |
| :--- | :--- |
| train_recover | 典型‘补给-训练-恢复’循环，训练强度通过重复次数控制。 |

| 生成器名 | 示例（前5条） |
| :--- | :--- |
| train_recover | Ingest → Exercise → Adaptation；Ingest → Exercise → Exercise → Adaptation；Ingest → Exercise → Exercise → Exercise → Adaptation |

### tem

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Exposure, Absorption, Distribution, Lesion, Inflammation, Detox, Repair, Identity |
| 最大长度 | 3 |
| 约束 | 禁止相邻重复, 跳过Identity |

#### 说明：

TEM 关注外源损伤与内源反应，常用链路体现急性损伤后的炎症与解毒/修复。

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| detox_repair | Detox → Repair |
| insult | Exposure → Lesion → Inflammation |

| 族名 | 说明 |
| :--- | :--- |
| detox_repair | 解毒与组织修复阶段，B/P 下降，F 回升。 |
| insult | 外源暴露/病灶诱发炎症反应，B/P 上升明显。 |

| 生成器名 | 链式模式 |
| :--- | :--- |
| insult_detox | (Exposure | Lesion) → Inflammation → (Detox | Repair) |

| 生成器名 | 说明 |
| :--- | :--- |
| insult_detox | 损伤路径（暴露/病灶）后进入炎症期，随后选择解毒或修复。 |

| 生成器名 | 示例（前5条） |
| :--- | :--- |
| insult_detox | Exposure → Inflammation → Detox；Exposure → Inflammation → Repair；Lesion → Inflammation → Detox；Lesion → Inflammation → Repair |

### pktm

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Dose, Absorb, Distribute, Metabolize, Excrete, Bind, Transport, Identity |
| 最大长度 | 4 |
| 约束 | 禁止相邻重复, 跳过Identity |

#### 说明：

PKTM 为 ADME/转运视角，常用链路为 ADME 管线与结合-转运。

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| adme | Dose → Absorb → Distribute → Metabolize → Excrete |
| binding_transport | Bind → Transport |

| 族名 | 说明 |
| :--- | :--- |
| adme | 标准 ADME 流：给药→吸收→分布→代谢→排泄。 |
| binding_transport | 结合后经转运体跨膜转运。 |

| 生成器名 | 链式模式 |
| :--- | :--- |
| adme_chain | Dose → Absorb → Distribute → (Metabolize | Identity) → (Excrete | Identity) |

| 生成器名 | 说明 |
| :--- | :--- |
| adme_chain | ADME 主链，允许在代谢/排泄处保持（Identity）以模拟滞留。 |

| 生成器名 | 示例（前5条） |
| :--- | :--- |
| adme_chain | Dose → Absorb → Distribute → Metabolize → Excrete；Dose → Absorb → Distribute → Metabolize → Identity；Dose → Absorb → Distribute → Identity → Excrete；Dose → Absorb → Distribute → Identity → Identity |

### pgom

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Activate, Repress, Mutation, RepairGenome, EpigeneticMod, PathwayInduction, PathwayInhibition, Identity |
| 最大长度 | 3 |
| 约束 | 禁止相邻重复, 跳过Identity |

#### 说明：

PGOM 聚焦基因/通路调控，常用链路体现激活-诱导与修复/抑制。

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| activation | Activate → PathwayInduction |
| inhibition | PathwayInhibition |
| repair | RepairGenome |

| 族名 | 说明 |
| :--- | :--- |
| activation | 基因/通路激活与诱导，B/P/F 同步上行。 |
| inhibition | 通路抑制，B/P/F 下行。 |
| repair | 基因修复，F 提升，B/P 下降。 |

| 生成器名 | 链式模式 |
| :--- | :--- |
| activation_cycle | Activate → (PathwayInduction | EpigeneticMod) → (RepairGenome | Identity) |

| 生成器名 | 说明 |
| :--- | :--- |
| activation_cycle | 激活后选择诱导或表观修饰，随后进入修复或保持。 |

| 生成器名 | 示例（前5条） |
| :--- | :--- |
| activation_cycle | Activate → PathwayInduction → RepairGenome；Activate → PathwayInduction → Identity；Activate → EpigeneticMod → RepairGenome；Activate → EpigeneticMod → Identity |

### pdem

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Bind, Signal, Desensitization, Antagonist, Potentiation, InverseAgonist, Identity |
| 最大长度 | 3 |
| 约束 | 禁止相邻重复, 跳过Identity |

#### 说明：

PDEM 对应药效动力学，常用链路体现结合-信号-去敏/拮抗的效应链。

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| activation_signal | Bind → Signal |
| block | Antagonist |
| tolerance | Desensitization |

| 族名 | 说明 |
| :--- | :--- |
| activation_signal | 结合后引发信号放大，B/P/F 上行。 |
| block | 拮抗阻断效应。 |
| tolerance | 去敏/耐受，F 下行并收敛。 |

| 生成器名 | 链式模式 |
| :--- | :--- |
| activation_tolerance | Bind → Signal → (Desensitization | Antagonist) |

| 生成器名 | 说明 |
| :--- | :--- |
| activation_tolerance | 结合-信号后进入去敏或拮抗分支。 |

| 生成器名 | 示例（前5条） |
| :--- | :--- |
| activation_tolerance | Bind → Signal → Desensitization；Bind → Signal → Antagonist |

### iem

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Activate, Suppress, Proliferate, Differentiate, CytokineRelease, Memory, Identity |
| 最大长度 | 3 |
| 约束 | 禁止相邻重复, 跳过Identity |

#### 说明：

IEM 聚焦免疫识别-效应-分辨-记忆的全链路过程。

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| immune_response | Activate → Proliferate → CytokineRelease |
| maturation | Activate → Differentiate → Memory |
| resolution | Suppress → Memory |

| 族名 | 说明 |
| :--- | :--- |
| immune_response | 激活扩增并释放细胞因子，B/P 上行，F 可能下降。 |
| maturation | 分化成熟并进入记忆。 |
| resolution | 抑制与记忆形成，B/P 下降，F 上行。 |

| 生成器名 | 链式模式 |
| :--- | :--- |
| response_resolution | Activate → Proliferate → (CytokineRelease | Identity) → (Memory | Suppress) |

| 生成器名 | 说明 |
| :--- | :--- |
| response_resolution | 应答后进入释放或保持，再进入记忆/抑制以实现分辨。 |

| 生成器名 | 示例（前5条） |
| :--- | :--- |
| response_resolution | Activate → Proliferate → CytokineRelease → Memory；Activate → Proliferate → CytokineRelease → Suppress；Activate → Proliferate → Identity → Memory；Activate → Proliferate → Identity → Suppress |

## 同步规范（重要）

- 本文件由 `python -m lbopb.src.gen_operator_crosswalk_md` 自动生成；源为 `operator_crosswalk.json`。
- 修改 JSON 后，请重新执行上述命令同步更新本文件。
- 仓库 Git Hooks 不执行自动改写（遵循 AGENTS 规范）；需人工手动运行脚本。


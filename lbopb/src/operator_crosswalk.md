# LBOPB 幺半群算子联络预览（自动生成）

本文件由脚本自动生成（请勿手工编辑）。数据源：`lbopb/src/operator_crosswalk.json`。

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

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| progression | Inflammation → Carcinogenesis → Metastasis |
| repair_path | Inflammation → Apoptosis |

| 生成器名 | 链式模式 |
| :--- | :--- |
| progress_heal | Inflammation → (Carcinogenesis | Metastasis | Identity) → (Apoptosis | Identity) |

### prm

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Ingest, Exercise, Hormone, Proliferation, Adaptation, Stimulus, Identity |
| 最大长度 | 3 |
| 约束 | 禁止相邻重复, 跳过Identity |

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| hormonal_mod | Hormone → Adaptation |
| performance | Ingest → Exercise → Adaptation |
| recovery | Stimulus → Adaptation |

| 生成器名 | 链式模式 |
| :--- | :--- |
| train_recover | Ingest → Exercise{1..3} → Adaptation |

### tem

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Exposure, Absorption, Distribution, Lesion, Inflammation, Detox, Repair, Identity |
| 最大长度 | 3 |
| 约束 | 禁止相邻重复, 跳过Identity |

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| detox_repair | Detox → Repair |
| insult | Exposure → Lesion → Inflammation |

| 生成器名 | 链式模式 |
| :--- | :--- |
| insult_detox | (Exposure | Lesion) → Inflammation → (Detox | Repair) |

### pktm

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Dose, Absorb, Distribute, Metabolize, Excrete, Bind, Transport, Identity |
| 最大长度 | 4 |
| 约束 | 禁止相邻重复, 跳过Identity |

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| adme | Dose → Absorb → Distribute → Metabolize → Excrete |
| binding_transport | Bind → Transport |

| 生成器名 | 链式模式 |
| :--- | :--- |
| adme_chain | Dose → Absorb → Distribute → (Metabolize | Identity) → (Excrete | Identity) |

### pgom

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Activate, Repress, Mutation, RepairGenome, EpigeneticMod, PathwayInduction, PathwayInhibition, Identity |
| 最大长度 | 3 |
| 约束 | 禁止相邻重复, 跳过Identity |

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| activation | Activate → PathwayInduction |
| inhibition | PathwayInhibition |
| repair | RepairGenome |

| 生成器名 | 链式模式 |
| :--- | :--- |
| activation_cycle | Activate → (PathwayInduction | EpigeneticMod) → (RepairGenome | Identity) |

### pdem

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Bind, Signal, Desensitization, Antagonist, Potentiation, InverseAgonist, Identity |
| 最大长度 | 3 |
| 约束 | 禁止相邻重复, 跳过Identity |

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| activation_signal | Bind → Signal |
| block | Antagonist |
| tolerance | Desensitization |

| 生成器名 | 链式模式 |
| :--- | :--- |
| activation_tolerance | Bind → Signal → (Desensitization | Antagonist) |

### iem

| 键 | 值 |
| :--- | :--- |
| 基本算子集 | Activate, Suppress, Proliferate, Differentiate, CytokineRelease, Memory, Identity |
| 最大长度 | 3 |
| 约束 | 禁止相邻重复, 跳过Identity |

| 常用幂集族名 | 算子序列组 |
| :--- | :--- |
| immune_response | Activate → Proliferate → CytokineRelease |
| maturation | Activate → Differentiate → Memory |
| resolution | Suppress → Memory |

| 生成器名 | 链式模式 |
| :--- | :--- |
| response_resolution | Activate → Proliferate → (CytokineRelease | Identity) → (Memory | Suppress) |

## 同步规范（重要）

- 本文件由 `python -m lbopb.src.gen_operator_crosswalk_md` 自动生成；源为 `operator_crosswalk.json`。
- 修改 JSON 后，请重新执行上述命令同步更新本文件。
- 仓库 Git Hooks 不执行自动改写（遵循 AGENTS 规范）；需人工手动运行脚本。


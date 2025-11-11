<!-- @guard-begin: lbopb-legal-guard (DO NOT EDIT BY SCRIPTS) -->
# LBOPB 子项目：著作权与独立性声明

**重要法律声明：**

本项目（`lbopb`，下称“载荷”）是作者 GaoZheng 的原创著作物，其知识产权源自“高政 G 框架（渊源：PL-PI 元数学理论）”。

“载荷”被设计为在 GROMACS 平台（下称“宿主”）上运行，但**在法律上是独立于“宿主”的著作物**。

* **宿主（GROMACS）**：受其原始许可证（GPL/LGPL）约束。
* **载荷（本项目 `lbopb`）**：
    * **理论文档**（如 `my_docs` 及相关理论文件）：遵循 `CC-BY-NC-ND 4.0` 许可及“思想即表达”原则。
    * **源代码**（如 `lbopb/src` 内文件）：遵循 `GPL-3.0-only` 许可，著作权由 GaoZheng 独立持有。

本“载荷”的著作权与授权策略（“双轨制”）独立于“宿主”的许可证，不受其“传染”。详细的法律架构请参见《[🚩🚩gromacs-2024.1_developer项目的著作权设计策略：“宿主-载荷”与“双轨制”复合架构](my_docs/project_docs/1762636780_🚩🚩gromacs-2024.1_developer项目的著作权设计策略：“宿主-载荷”与“双轨制”复合架构.md)》。

## 开发协议（受保护区域）
- 默认沟通语言：简体中文。
- 区域保护：本区块受审查脚本保护，任何自动化脚本不得修改；仅允许人工编辑并经代码评审。
- 许可遵循：理论文档遵循 CC-BY-NC-ND 4.0；源代码遵循 GPL-3.0-only；源码头注需符合根级 AGENTS 的“源代码头注规范（MUST）”。
- 合规约束：严禁宣称“官方/official”身份；如需引用上游原文，须在 README 顶部保留非官方/派生声明进行对冲。
- 审查脚本：`python3 my_scripts/check_lbopb_readme_guard.py`（只读校验，不进行任何自动修改）。
- 自动化范围限制：`my_scripts/**` 与 `lbopb/scripts/**` 不得对本受保护区域进行写入类操作。

<!-- @guard-end: lbopb-legal-guard -->

# lbopb

lbopb（Local Binding & Operator Prototypes
Bundle）是“立体模拟人体”的生命总算子主纤维丛（LBOPB）的工程化实现：围绕多视角幺半群（PEM/PRM/TEM/PKTM/PGOM/PDEM/IEM）提供基本算子、算子包与幂集生成、跨切面的联络映射与路径积分，使
O3 理论在生物信息尺度落为可执行的计算引擎与 API。

## 对应实现

- 引擎与算子：`lbopb/src`（PEM/PRM/TEM/PKTM/PGOM/PDEM/IEM 多切面算子、幂集与跨切面联络映射）
- 示例与报告：`lbopb/lbopb_examples`（如 `hiv_therapy_case.py` 与生成的 Markdown 报告）
- 辅助脚本：`lbopb/scripts`（头注同步、规范化工具）

## GROMACS 集成与药效幺半群联动

本子项目在 `lbopb/src/pharmdesign` 中提供了面向药效幺半群（PDEM）的“化合物设计 + 分子模拟”接口层：

- 需求到设计（requirements/design）：
    - `PharmacodynamicRequirement`：药效/多维约束输入（目标/机制/ADMET/毒理/免疫）
    - `propose_small_molecule` / `propose_biologic`：生成小分子/大分子设计意图（药效团/母核/取代策略）
- 分子模拟（sim）：
    - `docking_degenerate_gromacs`：退化分子对接（基于 GROMACS rerun 能量评估流程的命令方案）
    - `md_classical_gromacs`：经典分子动力学（命令方案）
    - `md_qmmm_stub`：QM/MM 占位（对接 CP2K/ORCA 的命令草案）
- 路径积分与联络（pipeline）：
    - `pdem_path_integral`：PDEM 算子包的离散“点集拓扑路径积分”（Lagrangian 累加）
    - `map_pdem_sequence_to_fibers`：基于联络（`operator_crosswalk.json`）映射至各纤维丛（PRM/PEM/PKTM/PGOM/TEM/IEM）的离散拓扑序列

说明：API 默认仅返回命令方案与期望产物路径，不强制执行外部程序；调用方可根据环境选择实际运行。

示例：`lbopb/lbopb_examples/hiv_therapy_case.py` 展示了以病理为基底构造 HIV 治疗算子包、映射药效切面、并在六切面展开对齐序列，同时生成“立体序列”的详细
Markdown 报告（输出于 `lbopb/lbopb_examples/out/`）。

## 快速开始

从仓库根目录运行：

```
python -c "import sys,os; sys.path.insert(0, os.path.abspath('.')); import lbopb.lbopb_examples.hiv_therapy_case as m; m.run_case()"
```

或直接执行示例脚本：

```
python lbopb/lbopb_examples/hiv_therapy_case.py
```

规范说明：本子目录遵循仓库根级 `AGENTS.md`（最高规范）；并提供子目录专用规范见 `lbopb/AGENTS.md`。

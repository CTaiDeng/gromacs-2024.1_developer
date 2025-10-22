免责声明：本文件及其生成的报告仅用于方法学与技术演示，不构成医学建议或临床诊断/治疗方案；
亦不用于任何实际诊疗决策或药物使用指导。若需临床决策，请咨询专业医师并遵循监管要求。

# 案例：HIV_Therapy_Path

#### 说明：

以病理（PEM）为基底流形，构造 HIV 治疗算子包，经联络映射到药效切面（PDEM），并对六大切面给出对齐算子包。

#### 说明：

病理 repair_path（Inflammation→Apoptosis）与药效拮抗链（Bind→Antagonist）语义对齐：修复对应占有+抑制；其余切面选择与稳态回归/ADME/通路抑制/解毒与免疫分辨相协调的链。

## 立体序列总览

| 模块 | 序列 |
| :--- | :--- |
| pem | Inflammation → Apoptosis |
| pdem | Bind → Antagonist |
| pktm | Dose → Absorb → Distribute → Metabolize → Excrete |
| pgom | PathwayInhibition |
| tem | Detox → Repair |
| prm | Stimulus → Adaptation |
| iem | Activate → Differentiate → Memory |

## pem

**序列**: Inflammation → Apoptosis

**状态变化**:

| 指标 | s0 | s1 | Δ |
| :--- | :--- | :--- | :--- |
| B | 8 | 6.72 | -1.28 |
| P | 2 | 2.125 | +0.125 |
| N_comp | 3 | 4 | +1 |
| F | 0.6 | 0.67 | +0.07 |

**风险与代价（示意）**:

| 项 | 数值 | 备注 |
| :--- | :--- | :--- |
| Risk(s0) | 5 | 模块定义的风险函数 |
| Risk(s1) | 6.125 | 变化后风险 |
| ActionCost(seq; s0) | 0.615 | 仅示意权重 |

#### 说明：

病理侧期望通过凋亡/清除实现负荷与边界收敛、保真提升。

## pdem

**序列**: Bind → Antagonist

**状态变化**:

| 指标 | s0 | s1 | Δ |
| :--- | :--- | :--- | :--- |
| B | 1.5 | 1.44 | -0.06 |
| P | 0.8 | 0.756 | -0.044 |
| N_comp | 1 | 1 | +0 |
| F | 0.6 | 0.5632 | -0.0368 |

**风险与代价（示意）**:

| 项 | 数值 | 备注 |
| :--- | :--- | :--- |
| Risk(s0) | 1.9 | 模块定义的风险函数 |
| Risk(s1) | 1.877 | 变化后风险 |
| ActionCost(seq; s0) | 0.3288 | 仅示意权重 |

#### 说明：

药效侧通过结合+拮抗抑制关键效应链，期望 F 上行。

## pktm

**序列**: Dose → Absorb → Distribute → Metabolize → Excrete

**状态变化**:

| 指标 | s0 | s1 | Δ |
| :--- | :--- | :--- | :--- |
| B | 0.5 | 1.134 | +0.634 |
| P | 0.5 | 0.4516 | -0.04839 |
| N_comp | 1 | 2 | +1 |
| F | 0.95 | 1 | +0.05 |

**风险与代价（示意）**:

| 项 | 数值 | 备注 |
| :--- | :--- | :--- |
| Risk(s0) | 1.5 | 模块定义的风险函数 |
| Risk(s1) | 2.452 | 变化后风险 |
| ActionCost(seq; s0) | 1.673 | 仅示意权重 |

#### 说明：

ADME 链路用于保障暴露窗口与可达性，代谢/排泄控制全身风险。

## pgom

**序列**: PathwayInhibition

**状态变化**:

| 指标 | s0 | s1 | Δ |
| :--- | :--- | :--- | :--- |
| B | 3 | 2.7 | -0.3 |
| P | 1.5 | 1.425 | -0.075 |
| N_comp | 2 | 2 | +0 |
| F | 0.8 | 0.752 | -0.048 |

**风险与代价（示意）**:

| 项 | 数值 | 备注 |
| :--- | :--- | :--- |
| Risk(s0) | 3.5 | 模块定义的风险函数 |
| Risk(s1) | 3.425 | 变化后风险 |
| ActionCost(seq; s0) | 0.0336 | 仅示意权重 |

#### 说明：

通路抑制以避免不利转录级响应，辅以修复/表观调谐。

## tem

**序列**: Detox → Repair

**状态变化**:

| 指标 | s0 | s1 | Δ |
| :--- | :--- | :--- | :--- |
| B | 5 | 3.188 | -1.812 |
| P | 2 | 1.2 | -0.8 |
| N_comp | 1 | 1 | +0 |
| F | 0.9 | 1 | +0.1 |

**风险与代价（示意）**:

| 项 | 数值 | 备注 |
| :--- | :--- | :--- |
| Risk(s0) | 5.1 | 模块定义的风险函数 |
| Risk(s1) | 3.188 | 变化后风险 |
| ActionCost(seq; s0) | 0 | 仅示意权重 |

#### 说明：

解毒与修复以压低损伤负荷与炎症边界，控制毒理风险。

## prm

**序列**: Stimulus → Adaptation

**状态变化**:

| 指标 | s0 | s1 | Δ |
| :--- | :--- | :--- | :--- |
| B | 10 | 9.9 | -0.1 |
| P | 5 | 5.7 | +0.7 |
| N_comp | 1 | 1 | +0 |
| F | 0.8 | 0.84 | +0.04 |

**风险与代价（示意）**:

| 项 | 数值 | 备注 |
| :--- | :--- | :--- |
| Risk(s0) | 6 | 模块定义的风险函数 |
| Risk(s1) | 6.7 | 变化后风险 |
| ActionCost(seq; s0) | 1.22 | 仅示意权重 |

#### 说明：

刺激—适应表达稳态回归，B/P 收敛、F 提升。

## iem

**序列**: Activate → Differentiate → Memory

**状态变化**:

| 指标 | s0 | s1 | Δ |
| :--- | :--- | :--- | :--- |
| B | 2 | 2.16 | +0.16 |
| P | 1 | 0.9364 | -0.06364 |
| N_comp | 2 | 2 | +0 |
| F | 0.7 | 1 | +0.3 |

**风险与代价（示意）**:

| 项 | 数值 | 备注 |
| :--- | :--- | :--- |
| Risk(s0) | 2.3 | 模块定义的风险函数 |
| Risk(s1) | 2.16 | 变化后风险 |
| ActionCost(seq; s0) | 0.3606 | 仅示意权重 |

#### 说明：

免疫侧激活-分化-记忆，避免细胞因子过度释放。

## 分子设计与分子模拟计划

#### 说明：

基于药效切面（PDEM）的拮抗链，给出小分子设计意图与 GROMACS 退化对接/MD/QM-MM 的命令方案。

### 小分子设计意图

- 目标: HIV IN
- 机制: IN antagonist
- 药效团: tridentate_metal_chelation, aryl_hydrophobe, tertiary_amine_sidechain
- 母核: dihydroxy-aromatic + diketo-acid
- 取代策略: para/meta hydrophobe fitting, pKa tuned amine for solubility
- ADMET备注: target_solubility≥0.1 mg/mL, avoid_BBB, avoid_CYP:3A4
- 毒理备注: low_hERG

### 退化分子对接（命令方案）

```bash
# 生成随机姿势并打包为 TRR（伪指令，需对接构建工具）
python gen_poses.py --receptor protein.pdb --ligand ligand.sdf --out out/docking\poses.trr
# rerun 评估（示例命令）
gmx mdrun -s topol.tpr -rerun out/docking\poses.trr -g out/docking/rerun.log
python score_rerun.py --log out/docking/rerun.log --out out/docking\poses.scores.csv
```

### 经典分子动力学（命令方案）

```bash
gmx grompp -f md.mdp -c system.gro -p topol.top -o out/md/topol.tpr
gmx mdrun -deffnm out/md/md
```

### QM/MM 占位（命令草案）

```bash
# 准备 QM/MM 输入（片段）: qmmm.inp
# 示例：调用 CP2K/ORCA 进行 QM 区域能量/力评估并回填到 MD 步进
```

## 复现指引

```
python -c "import sys,os; sys.path.insert(0, os.path.abspath('.')); import lbopb_examples.hiv_therapy_case as m; m.run_case(pharm_cfg_path=None)"
```


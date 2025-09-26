最小“退化对接”雏形（基于 gmx CLI）

本示例以 GROMACS 命令行（subprocess）实现“候选位姿生成 → mdrun -rerun 能量评估 → 抽取 Protein–Ligand 能量分项 → 排序”。

先决条件
- 已安装 GROMACS（推荐 2021+，2024.1 测试通过），`gmx` 可在 PATH 中调用。
- 已准备好的 `topol.tpr`（必须包含 `energygrps = Protein Ligand`，建议在 MDP 中冻结 Protein 以实现刚性受体）。
- 与 `topol.tpr` 原子顺序一致的全体系坐标文件 `start.gro`（含 Protein+Ligand）。
- `index.ndx`，包含 `Protein` 与 `Ligand` 两个组（与 `energygrps` 一致）。

建议的最小 MDP 片段（若你需要制作 TPR）
```
integrator  = steep
nsteps      = 0            ; 仅用于生成 tpr（rerun 时不积分）
nstlist     = 20
rlist       = 1.2
coulombtype = Cut-off
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2
constraints = h-bonds
pbc         = xyz
energygrps  = Protein Ligand
freezegrps  = Protein
freezedim   = Y Y Y
```

快速开始
1) 准备 `topol.tpr / start.gro / index.ndx`。
2) 运行示例脚本，生成 N 个随机候选并评分：
```
python example/docking_prototype/dock_minimal.py \
  --tpr path/to/topol.tpr \
  --structure path/to/start.gro \
  --ndx path/to/index.ndx \
  --n-poses 50 \
  --trans 0.5 \
  --rot 20 \
  --workdir out --nt 1
```

参数说明（关键）
- `--tpr`：包含 `Protein` 与 `Ligand` 能量分组的 TPR。
- `--structure`：全体系起始坐标（与 TPR 原子顺序一致）。
- `--ndx`：`index.ndx`，须含 `[ Ligand ]` 组用于对配体刚体变换。
- `--n-poses`：随机候选数。
- `--trans`（nm）：每个轴向的平移最大幅度（均匀采样于 [-trans, trans]）。
- `--rot`（度）：绕随机轴的最大旋转角（均匀采样于 [-rot, rot]）。
- `--nt`：每个 mdrun 实例线程数（并发时建议 1）。
- `--jobs`：候选并发数（默认 1）。

输出
- 在 `--workdir` 下：`candidate_####.gro`、`pose_####.edr/.log`、`energy_####.xvg`、`scores.csv`（汇总评分）。
- 评分定义：`score = (Coul-SR:Protein-Ligand) + (LJ-SR:Protein-Ligand)`，取 rerun 单帧的数值（数值越低越优）。

注意事项
- 该脚本对配体做刚体变换后会做 `mdrun -rerun` 重算能量，不会积分或最小化；若初始严重重叠可导致能量极大，属于预期。
- 若你需要更稳健的评分，可将 `-rerun` 改为短 EM（将脚本中相应函数替换为 `grompp + mdrun`，本示例聚焦最小雏形故未展开）。
- 不处理复杂 PBC 拓扑与多分子配体拆分；若配体被拆分为多个拓扑分子，建议在 `index.ndx` 中将其组合为一个 `Ligand` 组并确认坐标连续性。


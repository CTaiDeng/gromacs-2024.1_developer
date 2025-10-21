# PEM_OPERATOR_PACKAGE

本文件介绍 `lbopb/src/pem` 的 Python 封装，依据：
`my_docs/project_docs/1761050237_病理演化幺半群 (PEM) 公理系统.md`。

- 目标：以可计算抽象复现 PEM 的基本要素（状态、可观测量、幺半群组合、典型算子与指标），用于快速原型与实验。
- 范畴：不绑定几何/测度细节；以统计量近似（B, N_comp, P, F）。几何级实现可在后续替换 `PEMState` 的表象。

## 结构（模块与对象）

- `PEMState`：状态表象（b, n_comp, perim, fidelity, meta）。
- `Observables`：可观测量族 Φ（默认包含 B, N_comp, P, F）。
- `PEMOperator`：算子基类；`Identity` 为恒等元。
- 典型算子：
  - `Metastasis` (O_meta)：增加组分数与边界复杂度，稀释负担密度、降低保真。
  - `Apoptosis` (O_apop)：降低负担/复杂度，提高保真。
  - `Inflammation` (O_inflam)：提升边界活性与负担、可能降低保真。
  - `Carcinogenesis` (O_carcin)：提升负担与边界复杂度、降低保真。
- 复合与幺半群：`compose(*ops)` 返回复合算子（结合律成立，恒等元可消去）。
- 指标：
  - `delta_phi(A,B,S)`：Δ_Φ(O_A,O_B;S)（先后次序差异）。
  - `non_commutativity_index(A,B,S)`：NC=Δ_Φ/(1+Σ_φ φ(S))。
  - `topo_risk(S, α1, α2)`：拓扑风险 α1 N_comp + α2 P。
  - `action_cost(Seq,S)`：作用量近似；上升的 B/P/N 与下降的 F 受罚。
  - `reach_probability(S,S*,{Seq})`：以 min 作用量的负指数给出可达性近似。

## 快速上手（示例）

```python
from lbopb.src.pem import (
    PEMState, Observables,
    Metastasis, Apoptosis, Inflammation, Carcinogenesis,
    compose, delta_phi, non_commutativity_index,
    topo_risk, action_cost, reach_probability,
)

# 初始状态（可视为 B=10, N=1, P=5, F=0.8）
s0 = PEMState(b=10.0, n_comp=1, perim=5.0, fidelity=0.8)

# 构造算子
O_meta = Metastasis(alpha_n=1.0, alpha_p=0.1, beta_b=0.0, beta_f=0.05)
O_apop = Apoptosis(gamma_b=0.2, gamma_n=0.1, gamma_p=0.15, delta_f=0.1)

# 复合
O = compose(O_meta, O_apop)  # O_apop∘O_meta
s1 = O(s0)

# 指标
phi = Observables.default()
print("ΔΦ:", delta_phi(O_meta, O_apop, s0, phi))
print("NC:", non_commutativity_index(O_meta, O_apop, s0, phi))
print("TopoRisk(s1):", topo_risk(s1, alpha1=1.0, alpha2=0.5))

# 作用量与可达性（候选路径）
seq1 = [O_meta, O_apop]
seq2 = [O_apop, O_meta]
print("A(seq1):", action_cost(seq1, s0))
print("Reach≈:", reach_probability(s0, s1, [seq1, seq2]))
```

## 与公理系统对应关系（摘记）

- 幺半群：`compose` 表示算子复合；`Identity` 为幺元；组合满足结合律。
- Φ：`Observables.default()` 提供 {B, N_comp, P, F}；可自定义扩展。
- Δ_Φ/NC：实现 `delta_phi` 与 `non_commutativity_index` 度量操作顺序效应。
- 典型算子：`Metastasis/Apoptosis/Inflammation/Carcinogenesis` 分别对应 O_meta/O_apop/O_inflam/O_carcin 的经验型近似作用。
- 运营指标：`topo_risk` 与 `action_cost`/`reach_probability` 反映 T2/T3 与 A11-A13 中的可观测与优化要素（近似化实现）。

## 设计与扩展建议

- 精细几何：可将 `PEMState.meta` 绑定几何句柄（网格/点集等），并重载可观测量求值。
- 学习映射：为 A12-A13，可在上层训练 `PEMOperator` 参数，使观测序列的预测误差最小（本包侧重前向近似与指标）。
- 度量拓展：根据具体任务增添新的 φ ∈ Φ 与对应风险/效用函数。

## 参考

- 规范文档：`my_docs/project_docs/1761050237_病理演化幺半群 (PEM) 公理系统.md`


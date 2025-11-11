# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 only.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# --- 著作权独立性声明 (Copyright Independence Declaration) ---
# 本文件（“载荷”）是作者 (GaoZheng) 的原创著作物，其知识产权
# 独立于其运行平台 GROMACS（“宿主”）。
# 本文件的授权遵循上述 SPDX 标识，不受“宿主”许可证的管辖。
# 详情参见项目文档 "my_docs/project_docs/1762636780_🚩🚩gromacs-2024.1_developer项目的著作权设计策略：“宿主-载荷”与“双轨制”复合架构.md"。
# ------------------------------------------------------------------

import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lbopb.src.pem import (
    PEMState,
    Observables,
    Metastasis,
    Apoptosis,
    Inflammation,
    Carcinogenesis,
    compose,
    delta_phi,
    non_commutativity_index,
    topo_risk,
    action_cost,
    reach_probability,
)


def main() -> None:
    s0 = PEMState(b=10.0, n_comp=1, perim=5.0, fidelity=0.8)
    O_meta = Metastasis(alpha_n=1.0, alpha_p=0.1, beta_b=0.0, beta_f=0.05)
    O_apop = Apoptosis(gamma_b=0.2, gamma_n=0.1, gamma_p=0.15, delta_f=0.1)

    O = compose(O_meta, O_apop)  # O_apop ∘ O_meta
    s1 = O(s0)

    phi = Observables.default()
    print("s0:", s0)
    print("s1:", s1)
    print("ΔΦ(meta, apop; s0):", delta_phi(O_meta, O_apop, s0, phi))
    print("NC(meta, apop; s0):", non_commutativity_index(O_meta, O_apop, s0, phi))
    print("TopoRisk(s1; α1=1, α2=0.5):", topo_risk(s1, alpha1=1.0, alpha2=0.5))

    seq1 = [O_meta, O_apop]
    seq2 = [O_apop, O_meta]
    print("Action(seq1):", action_cost(seq1, s0))
    print("Reach≈:", reach_probability(s0, s1, [seq1, seq2]))


if __name__ == "__main__":
    main()

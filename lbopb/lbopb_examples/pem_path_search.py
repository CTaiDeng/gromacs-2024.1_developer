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

import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lbopb.src.pem import (
    PEMState,
    Metastasis,
    Apoptosis,
    Inflammation,
    Carcinogenesis,
    action_cost,
    reach_probability,
)


def main() -> None:
    s0 = PEMState(b=12.0, n_comp=1, perim=4.0, fidelity=0.85)

    O_meta = Metastasis(alpha_n=1.0, alpha_p=0.1, beta_b=0.0, beta_f=0.05)
    O_apop = Apoptosis(gamma_b=0.25, gamma_n=0.2, gamma_p=0.2, delta_f=0.12)
    O_inflam = Inflammation(eta_b=0.05, eta_p=0.3, eta_f=0.06, dn=1)
    O_carcin = Carcinogenesis(k_b=0.2, k_p=0.18, k_f=0.1)

    # 候选路径：例如不同治疗/病理事件次序
    candidates = [
        [O_apop, O_apop],
        [O_meta, O_apop],
        [O_inflam, O_apop],
        [O_carcin, O_apop],
        [O_apop, O_meta, O_apop],
    ]

    # 目标状态可用某条路径的终点近似（演示用）
    s_star = candidates[0][-1](candidates[0][0](s0))

    costs = [action_cost(seq, s0) for seq in candidates]
    for i, c in enumerate(costs):
        print(f"Seq#{i + 1} cost = {c:.4f}")

    rp = reach_probability(s0, s_star, candidates)
    print("Reach probability ≈", rp)


if __name__ == "__main__":
    main()

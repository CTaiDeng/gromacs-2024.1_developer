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
    Inflammation,
    Carcinogenesis,
    delta_phi,
    non_commutativity_index,
)


def main() -> None:
    s0 = PEMState(b=8.0, n_comp=2, perim=6.0, fidelity=0.7)
    O_inflam = Inflammation(eta_b=0.05, eta_p=0.25, eta_f=0.08, dn=1)
    O_carcin = Carcinogenesis(k_b=0.25, k_p=0.15, k_f=0.1, dn=0)

    phi = Observables.default()
    dphi = delta_phi(O_inflam, O_carcin, s0, phi)
    nc = non_commutativity_index(O_inflam, O_carcin, s0, phi)

    print("s0:", s0)
    print("ΔΦ(inflam, carcin; s0):", dphi)
    print("NC(inflam, carcin; s0):", nc)


if __name__ == "__main__":
    main()

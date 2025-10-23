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

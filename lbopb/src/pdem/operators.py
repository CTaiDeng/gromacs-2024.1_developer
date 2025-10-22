# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .state import PDEMState


class PDEMOperator:
    """PDEM 调控算子基类：O: S -> S"""

    name: str = "PDEMOperator"
    params: Dict[str, float]

    def __init__(self, **params: float) -> None:
        self.params = dict(params)

    def __call__(self, s: PDEMState) -> PDEMState:
        return self.apply(s).clamp()

    def apply(self, s: PDEMState) -> PDEMState:  # pragma: no cover - abstract by convention
        return s

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"


class Identity(PDEMOperator):
    name = "Identity"

    def apply(self, s: PDEMState) -> PDEMState:
        return s


class Bind(PDEMOperator):
    """靶点结合：提升占有，适度提升边界与保真。

    - b' = b * (1 + alpha_b)
    - perim' = perim * (1 + alpha_p)
    - fidelity' = min(1, fidelity + delta_f)
    - n_comp' = n_comp + ceil(dn)
    """

    name = "Bind"

    def apply(self, s: PDEMState) -> PDEMState:
        from math import ceil

        alpha_b = float(self.params.get("alpha_b", 0.2))
        alpha_p = float(self.params.get("alpha_p", 0.05))
        delta_f = float(self.params.get("delta_f", 0.04))
        dn = int(self.params.get("dn", 0))

        b = s.b * (1.0 + max(0.0, alpha_b))
        p = s.perim * (1.0 + max(0.0, alpha_p))
        f = s.fidelity + max(0.0, delta_f)
        n = s.n_comp + max(0, ceil(dn))
        return PDEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Signal(PDEMOperator):
    """信号传导：提升效应强度与保真，边界上升。

    - b' = b * (1 + beta_b)
    - perim' = perim * (1 + beta_p)
    - fidelity' = min(1, fidelity + delta_f)
    """

    name = "Signal"

    def apply(self, s: PDEMState) -> PDEMState:
        beta_b = float(self.params.get("beta_b", 0.25))
        beta_p = float(self.params.get("beta_p", 0.1))
        delta_f = float(self.params.get("delta_f", 0.06))

        b = s.b * (1.0 + max(0.0, beta_b))
        p = s.perim * (1.0 + max(0.0, beta_p))
        f = s.fidelity + max(0.0, delta_f)
        return PDEMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


class Desensitization(PDEMOperator):
    """去敏/耐受：降低保真与效应，边界回落，位点可回落。

    - b' = b * (1 - gamma_b)
    - perim' = perim * (1 - gamma_p)
    - fidelity' = fidelity * (1 - gamma_f) 或 +(-)delta_f
    - n_comp' = max(1, round(n_comp * (1 - gamma_n)))
    """

    name = "Desensitization"

    def apply(self, s: PDEMState) -> PDEMState:
        gamma_b = float(self.params.get("gamma_b", 0.15))
        gamma_p = float(self.params.get("gamma_p", 0.1))
        gamma_n = float(self.params.get("gamma_n", 0.05))
        gamma_f = float(self.params.get("gamma_f", 0.08))

        b = s.b * (1.0 - max(0.0, min(gamma_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(gamma_p, 0.99)))
        f = s.fidelity * (1.0 - max(0.0, min(gamma_f, 0.95)))
        n = max(1, round(s.n_comp * (1.0 - max(0.0, min(gamma_n, 0.99)))))
        return PDEMState(b=b, n_comp=int(n), perim=p, fidelity=f, meta=s.meta)


class Antagonist(PDEMOperator):
    """拮抗：抑制效应链，降低保真与效应，边界下降。

    - b' = b * (1 - k_b)
    - perim' = perim * (1 - k_p)
    - fidelity' = fidelity * (1 - k_f)
    """

    name = "Antagonist"

    def apply(self, s: PDEMState) -> PDEMState:
        k_b = float(self.params.get("k_b", 0.2))
        k_p = float(self.params.get("k_p", 0.1))
        k_f = float(self.params.get("k_f", 0.12))

        b = s.b * (1.0 - max(0.0, min(k_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(k_p, 0.99)))
        f = s.fidelity * (1.0 - max(0.0, min(k_f, 0.95)))
        return PDEMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


class Potentiation(PDEMOperator):
    """增效（正变构/协同）：保真提升，边界略增，效应强化。

    - b' = b * (1 + xi_b)
    - perim' = perim * (1 + xi_p)
    - fidelity' = min(1, fidelity + delta_f)
    """

    name = "Potentiation"

    def apply(self, s: PDEMState) -> PDEMState:
        xi_b = float(self.params.get("xi_b", 0.05))
        xi_p = float(self.params.get("xi_p", 0.05))
        delta_f = float(self.params.get("delta_f", 0.1))

        b = s.b * (1.0 + max(0.0, xi_b))
        p = s.perim * (1.0 + max(0.0, xi_p))
        f = s.fidelity + max(0.0, delta_f)
        return PDEMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


class InverseAgonist(PDEMOperator):
    """反向激动：压低自发活性与效应，边界降低。

    - b' = b * (1 - rho_b)
    - perim' = perim * (1 - rho_p)
    - fidelity' = fidelity * (1 - rho_f)
    """

    name = "InverseAgonist"

    def apply(self, s: PDEMState) -> PDEMState:
        rho_b = float(self.params.get("rho_b", 0.1))
        rho_p = float(self.params.get("rho_p", 0.05))
        rho_f = float(self.params.get("rho_f", 0.08))

        b = s.b * (1.0 - max(0.0, min(rho_b, 0.99)))
        p = s.perim * (1.0 - max(0.0, min(rho_p, 0.99)))
        f = s.fidelity * (1.0 - max(0.0, min(rho_f, 0.95)))
        return PDEMState(b=b, n_comp=s.n_comp, perim=p, fidelity=f, meta=s.meta)


@dataclass(frozen=True)
class ComposedOperator(PDEMOperator):
    ops: Tuple[PDEMOperator, ...]
    name: str = "Composed"

    def __init__(self, *ops: PDEMOperator):  # type: ignore[override]
        object.__setattr__(self, "ops", tuple(ops))
        object.__setattr__(self, "params", {})

    def apply(self, s: PDEMState) -> PDEMState:
        out = s
        for op in self.ops:
            out = op(out)
        return out

    def __repr__(self) -> str:
        return "Composed(" + ", ".join(repr(o) for o in self.ops) + ")"


def compose(*ops: PDEMOperator) -> PDEMOperator:
    flat: list[PDEMOperator] = []
    for o in ops:
        if isinstance(o, Identity):
            continue
        if isinstance(o, ComposedOperator):
            flat.extend(o.ops)
        else:
            flat.append(o)
    if not flat:
        return Identity()
    if len(flat) == 1:
        return flat[0]
    return ComposedOperator(*flat)


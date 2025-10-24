# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from typing import Dict, List, Any
from pathlib import Path

import torch

from lbopb.src.rlsac.kernel.rlsac_connector.sampler import load_domain_packages, sample_random_connection
from lbopb.src.rlsac.kernel.rlsac_connector.oracle import ConnectorAxiomOracle, MODULES
from lbopb.src.rlsac.application.rlsac_nsclc.space import SimpleBoxInt32
from .observation import ObservationQuantizer

from lbopb.src.pem import PEMState, topo_risk as pem_risk
from lbopb.src.pdem import PDEMState, eff_risk as pdem_risk
from lbopb.src.pktm import PKTMState, topo_risk as pktm_risk
from lbopb.src.pgom import PGOMState, topo_risk as pgom_risk
from lbopb.src.tem import TEMState, tox_risk as tem_risk
from lbopb.src.prm import PRMState, topo_risk as prm_risk
from lbopb.src.iem import IEMState, imm_risk as iem_risk


def _default_init_states() -> Dict[str, Any]:
    return {
        "pem": PEMState(b=8.0, n_comp=3, perim=2.0, fidelity=0.6),
        "pdem": PDEMState(b=1.5, n_comp=1, perim=0.8, fidelity=0.6),
        "pktm": PKTMState(b=0.5, n_comp=1, perim=0.5, fidelity=0.95),
        "pgom": PGOMState(b=3.0, n_comp=2, perim=1.5, fidelity=0.8),
        "tem": TEMState(b=5.0, n_comp=1, perim=2.0, fidelity=0.9),
        "prm": PRMState(b=10.0, n_comp=1, perim=5.0, fidelity=0.8),
        "iem": IEMState(b=2.0, n_comp=2, perim=1.0, fidelity=0.7),
    }


def _risk(mod: str, st: Any) -> float:
    if mod == "pem": return float(pem_risk(st))
    if mod == "pdem": return float(pdem_risk(st))
    if mod == "pktm": return float(pktm_risk(st))
    if mod == "pgom": return float(pgom_risk(st))
    if mod == "tem": return float(tem_risk(st))
    if mod == "prm": return float(prm_risk(st))
    if mod == "iem": return float(iem_risk(st))
    return 0.0


class PemConnectorEnv:
    """以 PEM 为动作空间的联络打分环境（观测为整数索引）。"""

    def __init__(self, *, packages_dir: str | Path, observation_map: str | Path | None = None, cost_lambda: float = 0.2,
                 eps_change: float = 1e-3, use_llm_oracle: bool = False) -> None:
        self.packages_dir = Path(packages_dir)
        self.pkg_map = load_domain_packages(self.packages_dir)
        self.pem_pkgs = self.pkg_map.get("pem", []) or []
        self.states = _default_init_states()
        self.oracle = ConnectorAxiomOracle(cost_lambda=cost_lambda, eps_change=eps_change, use_llm=use_llm_oracle)
        map_path = observation_map or (Path(__file__).resolve().parent / "observation_map.json")
        self.quant = ObservationQuantizer(map_path)
        self.observation_space = SimpleBoxInt32(0, 1000, (35,))
        self.action_space = SimpleBoxInt32(0, max(1, len(self.pem_pkgs)), (1,))

    def _obs(self) -> torch.Tensor:
        risks = {m: _risk(m, self.states[m]) for m in MODULES}
        idxs = self.quant.quantize_full(self.states, risks)
        return torch.tensor(idxs, dtype=torch.int32)

    def reset(self) -> torch.Tensor:
        self.states = _default_init_states()
        return self._obs()

    def step(self, action: torch.Tensor | int):
        a = int(action if isinstance(action, int) else int(action.view(-1)[0].item()))
        a = max(0, min(a, max(0, len(self.pem_pkgs) - 1)))
        choice = sample_random_connection(self.pkg_map)
        if self.pem_pkgs:
            choice["pem"] = self.pem_pkgs[a]
        label, meta = self.oracle.judge({m: (choice[m].get("sequence", []) or []) for m in MODULES})
        reward = float(
            meta.get("delta_risk_sum", 0.0) + meta.get("consistency", 0.0) - float(meta.get("cost", 0.0)) * 1.0)
        obs = self._obs()
        done = True
        info = {"pem_pkg_id": choice.get("pem", {}).get("id", ""), "meta": meta, "label": int(label)}
        return obs, reward, done, info

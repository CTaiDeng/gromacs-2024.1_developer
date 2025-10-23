# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from typing import Dict, List
import json
from pathlib import Path

import torch

from .space import SimpleBoxInt32, SimpleBoxFloat32

MODULES: List[str] = ["pem", "pdem", "pktm", "pgom", "tem", "prm", "iem"]

from lbopb.src.pem import PEMState
from lbopb.src.pdem import PDEMState
from lbopb.src.pktm import PKTMState
from lbopb.src.pgom import PGOMState
from lbopb.src.tem import TEMState
from lbopb.src.prm import PRMState
from lbopb.src.iem import IEMState

from lbopb.src.pem import topo_risk as pem_risk, action_cost as pem_cost
from lbopb.src.pdem import eff_risk as pdem_risk, action_cost as pdem_cost
from lbopb.src.pktm import topo_risk as pktm_risk, action_cost as pktm_cost
from lbopb.src.pgom import topo_risk as pgom_risk, action_cost as pgom_cost
from lbopb.src.tem import tox_risk as tem_risk, action_cost as tem_cost
from lbopb.src.prm import topo_risk as prm_risk, action_cost as prm_cost
from lbopb.src.iem import imm_risk as iem_risk, action_cost as iem_cost

from lbopb.src.powerset import instantiate_ops


class NSCLCSequenceEnv:
    def __init__(self, case_json: str | Path, *, reward_lambda: float = 0.2) -> None:
        p = Path(case_json)
        data = json.loads(p.read_text(encoding="utf-8"))
        cp = data.get("case_packages", {})
        self.case_name = next(iter(cp.keys())) if cp else ""
        case = cp.get(self.case_name, {}) if self.case_name else {}
        seqs: Dict[str, List[str]] = case.get("sequences", {})
        ops: List[str] = []
        for m in MODULES:
            for op in seqs.get(m, []) or []:
                if op not in ops:
                    ops.append(op)
        self.ops = ops
        self.op2idx = {op: i for i, op in enumerate(self.ops)}
        self.seqs = {m: list(seqs.get(m, [])) for m in MODULES}

        self.mod_idx = 0
        self.ptr = 0
        self.reward_lambda = float(reward_lambda)

        self.states: Dict[str, object] = {
            "pem": PEMState(b=8.0, n_comp=3, perim=2.0, fidelity=0.6),
            "pdem": PDEMState(b=1.5, n_comp=1, perim=0.8, fidelity=0.6),
            "pktm": PKTMState(b=0.5, n_comp=1, perim=0.5, fidelity=0.95),
            "pgom": PGOMState(b=3.0, n_comp=2, perim=1.5, fidelity=0.8),
            "tem": TEMState(b=5.0, n_comp=1, perim=2.0, fidelity=0.9),
            "prm": PRMState(b=10.0, n_comp=1, perim=5.0, fidelity=0.8),
            "iem": IEMState(b=2.0, n_comp=2, perim=1.0, fidelity=0.7),
        }

        self.obs_dim = 7 + (5 * 7) + len(self.ops) + 1
        self.observation_space = SimpleBoxFloat32(0.0, 1.0, (self.obs_dim,))
        self.action_space = SimpleBoxInt32(0, len(self.ops), (1,))

    def _risk_of(self, mod: str, st: object) -> float:
        if mod == "pem":
            return float(pem_risk(st))
        if mod == "pdem":
            return float(pdem_risk(st))
        if mod == "pktm":
            return float(pktm_risk(st))
        if mod == "pgom":
            return float(pgom_risk(st))
        if mod == "tem":
            return float(tem_risk(st))
        if mod == "prm":
            return float(prm_risk(st))
        if mod == "iem":
            return float(iem_risk(st))
        return 0.0

    def _features_of_state(self, st: object) -> List[float]:
        try:
            b = float(getattr(st, "b")); p = float(getattr(st, "perim"))
            f = float(getattr(st, "fidelity")); n = float(getattr(st, "n_comp"))
        except Exception:
            b = p = f = n = 0.0
        return [b, p, f, n]

    def _vectorize(self) -> torch.Tensor:
        m_onehot = [0.0] * 7; m_onehot[self.mod_idx] = 1.0
        feats: List[float] = []
        for m in MODULES:
            st = self.states.get(m)
            feats += self._features_of_state(st)
            feats.append(self._risk_of(m, st))
        next_hot = [0.0] * len(self.ops)
        cur_mod = MODULES[self.mod_idx]
        seq = self.seqs.get(cur_mod, []) or []
        if self.ptr < len(seq):
            op = seq[self.ptr]; j = self.op2idx.get(op, -1)
            if j >= 0: next_hot[j] = 1.0
            pos = self.ptr / float(max(1, len(seq)))
        else:
            pos = 1.0
        return torch.tensor(m_onehot + feats + next_hot + [pos], dtype=torch.float32)

    def reset(self) -> torch.Tensor:
        self.mod_idx = 0; self.ptr = 0
        while self.mod_idx < len(MODULES) and len(self.seqs.get(MODULES[self.mod_idx], []) or []) == 0:
            self.mod_idx += 1; self.ptr = 0
        return self._vectorize()

    def _apply_op(self, mod: str, st: object, op_name: str) -> object:
        try:
            op = instantiate_ops(mod, [op_name])[0]
            return op(st)
        except Exception:
            return st

    def _single_cost(self, mod: str, st: object, op_name: str) -> float:
        try:
            op = instantiate_ops(mod, [op_name])[0]
        except Exception:
            return 0.0
        try:
            if mod == "pem": return float(pem_cost([op], st))
            if mod == "pdem": return float(pdem_cost([op], st))
            if mod == "pktm": return float(pktm_cost([op], st))
            if mod == "pgom": return float(pgom_cost([op], st))
            if mod == "tem": return float(tem_cost([op], st))
            if mod == "prm": return float(prm_cost([op], st))
            if mod == "iem": return float(iem_cost([op], st))
        except Exception:
            return 0.0
        return 0.0

    def step(self, action: torch.Tensor):
        a = int(action.view(-1)[0].item())
        cur_mod = MODULES[self.mod_idx]
        seq = self.seqs.get(cur_mod, []) or []
        reward = 0.0; done = False
        need_idx = self.op2idx.get(seq[self.ptr], -1) if self.ptr < len(seq) else -1

        st0 = self.states.get(cur_mod); risk0 = self._risk_of(cur_mod, st0)
        chosen_op = self.ops[a] if 0 <= a < len(self.ops) else None
        st1 = st0; risk1 = risk0; c = 0.0
        if chosen_op is not None and (chosen_op in seq):
            try:
                st1 = self._apply_op(cur_mod, st0, chosen_op)
                risk1 = self._risk_of(cur_mod, st1)
                c = self._single_cost(cur_mod, st0, chosen_op)
            except Exception:
                pass
        reward = (risk0 - risk1) - self.reward_lambda * c

        if need_idx >= 0 and a == need_idx:
            if chosen_op is not None and self.ops[a] == seq[self.ptr]:
                self.states[cur_mod] = st1
            else:
                self.states[cur_mod] = self._apply_op(cur_mod, st0, seq[self.ptr])
            self.ptr += 1

        if self.ptr >= len(seq):
            self.mod_idx += 1; self.ptr = 0
            while self.mod_idx < len(MODULES) and len(self.seqs.get(MODULES[self.mod_idx], []) or []) == 0:
                self.mod_idx += 1
            done = True

        return self._vectorize(), reward, done, {"module": cur_mod}


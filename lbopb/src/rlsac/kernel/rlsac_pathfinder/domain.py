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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type, TypeVar

TState = TypeVar("TState")


@dataclass
class DomainSpec:
    name: str
    state_cls: Type[Any]
    identity_cls: Type[Any]
    op_classes: Sequence[Type[Any]]  # 不包含 Identity，是否加入由 include_identity 控制
    dict_filename: str  # 辞海 JSON 文件名，如 "pem_operator_packages.json"


def _pem_spec() -> DomainSpec:
    from lbopb.src.pem.state import PEMState
    from lbopb.src.pem.operators import Identity, Apoptosis, Metastasis, Inflammation, Carcinogenesis
    return DomainSpec(
        name="pem",
        state_cls=PEMState,
        identity_cls=Identity,
        op_classes=(Apoptosis, Metastasis, Inflammation, Carcinogenesis),
        dict_filename="pem_operator_packages.json",
    )


def _pdem_spec() -> DomainSpec:
    from lbopb.src.pdem.state import PDEMState
    from lbopb.src.pdem.operators import Identity, Bind, Signal, Desensitization, Antagonist, Potentiation, \
        InverseAgonist
    return DomainSpec(
        name="pdem",
        state_cls=PDEMState,
        identity_cls=Identity,
        op_classes=(Bind, Signal, Desensitization, Antagonist, Potentiation, InverseAgonist),
        dict_filename="pdem_operator_packages.json",
    )


def _pktm_spec() -> DomainSpec:
    from lbopb.src.pktm.state import PKTMState
    from lbopb.src.pktm.operators import Identity, Dose, Absorb, Distribute, Metabolize, Excrete, Bind, Transport
    return DomainSpec(
        name="pktm",
        state_cls=PKTMState,
        identity_cls=Identity,
        op_classes=(Dose, Absorb, Distribute, Metabolize, Excrete, Bind, Transport),
        dict_filename="pktm_operator_packages.json",
    )


def _pgom_spec() -> DomainSpec:
    from lbopb.src.pgom.state import PGOMState
    from lbopb.src.pgom.operators import Identity, Activate, Repress, Mutation, RepairGenome, EpigeneticMod, \
        PathwayInduction, PathwayInhibition
    return DomainSpec(
        name="pgom",
        state_cls=PGOMState,
        identity_cls=Identity,
        op_classes=(Activate, Repress, Mutation, RepairGenome, EpigeneticMod, PathwayInduction, PathwayInhibition),
        dict_filename="pgom_operator_packages.json",
    )


def _tem_spec() -> DomainSpec:
    from lbopb.src.tem.state import TEMState
    from lbopb.src.tem.operators import Identity, Exposure, Absorption, Distribution, Lesion, Inflammation, Detox, \
        Repair
    return DomainSpec(
        name="tem",
        state_cls=TEMState,
        identity_cls=Identity,
        op_classes=(Exposure, Absorption, Distribution, Lesion, Inflammation, Detox, Repair),
        dict_filename="tem_operator_packages.json",
    )


def _prm_spec() -> DomainSpec:
    from lbopb.src.prm.state import PRMState
    from lbopb.src.prm.operators import Identity, Ingest, Exercise, Hormone, Proliferation, Adaptation, Stimulus
    return DomainSpec(
        name="prm",
        state_cls=PRMState,
        identity_cls=Identity,
        op_classes=(Ingest, Exercise, Hormone, Proliferation, Adaptation, Stimulus),
        dict_filename="prm_operator_packages.json",
    )


def _iem_spec() -> DomainSpec:
    from lbopb.src.iem.state import IEMState
    from lbopb.src.iem.operators import Identity, Activate, Suppress, Proliferate, Differentiate, CytokineRelease, \
        Memory
    return DomainSpec(
        name="iem",
        state_cls=IEMState,
        identity_cls=Identity,
        op_classes=(Activate, Suppress, Proliferate, Differentiate, CytokineRelease, Memory),
        dict_filename="iem_operator_packages.json",
    )


def get_domain_spec(name: str) -> DomainSpec:
    key = (name or "").strip().lower()
    if key == "pem":
        return _pem_spec()
    if key == "pdem":
        return _pdem_spec()
    if key == "pktm":
        return _pktm_spec()
    if key == "pgom":
        return _pgom_spec()
    if key == "tem":
        return _tem_spec()
    if key == "prm":
        return _prm_spec()
    if key == "iem":
        return _iem_spec()
    raise ValueError(f"Unknown domain: {name}")

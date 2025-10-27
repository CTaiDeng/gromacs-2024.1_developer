# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from typing import Any, Dict, List
from .pair_syntax_base import check_pair, check_conn as _check_conn


DOM_A = "tem"
DOM_B = "pdem"


def check(seq_a: List[str], seq_b: List[str]) -> Dict[str, Any]:
    return check_pair(DOM_A, DOM_B, seq_a, seq_b)


def check_conn(conn: Dict[str, List[str]]) -> Dict[str, Any]:
    return _check_conn(DOM_A, DOM_B, conn)

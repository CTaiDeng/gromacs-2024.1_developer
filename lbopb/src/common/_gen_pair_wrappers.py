# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from pathlib import Path

DOMAINS = ["pem", "prm", "tem", "pktm", "pgom", "pdem", "iem"]

TEMPLATE = """# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

from typing import Any, Dict, List
from .pair_syntax_base import check_pair, check_conn as _check_conn


DOM_A = "{A}"
DOM_B = "{B}"


def check(seq_a: List[str], seq_b: List[str]) -> Dict[str, Any]:
    return check_pair(DOM_A, DOM_B, seq_a, seq_b)


def check_conn(conn: Dict[str, List[str]]) -> Dict[str, Any]:
    return _check_conn(DOM_A, DOM_B, conn)
"""


def main() -> None:
    here = Path(__file__).resolve().parent
    for i, a in enumerate(DOMAINS):
        for b in DOMAINS[i + 1 :]:
            name = f"{a}_{b}_syntax_checker.py"
            code = TEMPLATE.format(A=a, B=b)
            p = (here / name)
            with p.open("w", encoding="utf-8", newline="\n") as f:
                f.write(code)


if __name__ == "__main__":
    main()

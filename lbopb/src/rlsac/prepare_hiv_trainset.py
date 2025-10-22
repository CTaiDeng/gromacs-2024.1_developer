# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    src = here.parent / "operator_crosswalk.json"
    dst = here / "operator_crosswalk_train.json"
    if not src.exists():
        raise FileNotFoundError(f"operator_crosswalk.json not found at: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    cp = data.get("case_packages", {})
    hiv = cp.get("HIV_Therapy_Path")
    if hiv is None:
        raise KeyError("HIV_Therapy_Path not found in case_packages")
    out = {"case_packages": {"HIV_Therapy_Path": hiv}}
    # Write UTF-8 (no BOM) + CRLF
    text = json.dumps(out, ensure_ascii=False, indent=2)
    text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    with open(dst, "wb") as f:
        f.write(text.encode("utf-8"))
    print(f"Wrote HIV case to {dst}")


if __name__ == "__main__":
    main()


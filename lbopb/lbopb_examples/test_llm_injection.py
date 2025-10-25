# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import argparse
import json
import os
import time as _t
from pathlib import Path
from typing import List


def _bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="测试：注入公理文档并发起 Gemini 请求，输出全过程调试信息"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="可选配置文件 JSON（默认尝试同名 test_llm_injection.json）",
    )
    parser.add_argument("--domain", default="pem", help="幺半群域，默认 pem")
    parser.add_argument(
        "--seq",
        default="Apoptosis,Inflammation",
        help="以英文逗号分隔的基本算子序列，例如: Apoptosis,Inflammation",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="请求超时（秒），若提供则覆盖 LBOPB_GEMINI_TIMEOUT_SEC",
    )
    parser.add_argument(
        "--debug",
        default="1",
        help="是否开启详细调试打印（0/1），默认 1 开启",
    )
    parser.add_argument(
        "--save",
        default="1",
        help="是否将提示词与结果落盘到 out/test_llm_injection/ 下（0/1），默认 1",
    )
    args = parser.parse_args()

    # 载入配置文件（优先使用 --config；否则尝试同目录下同名 JSON）
    cfg_path: Path | None = None
    if args.config:
        cfg_path = Path(args.config)
    else:
        _default_cfg = Path(__file__).with_suffix('.json')
        if _default_cfg.exists():
            cfg_path = _default_cfg

    def _load_cfg(p: Path) -> dict:
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}

    cfg = _load_cfg(cfg_path) if cfg_path else {}

    def _pick(cli_val, key: str, default_val):
        return cli_val if cli_val is not None else cfg.get(key, default_val)

    # 合并配置与 CLI（CLI 覆盖配置）
    _cfg_domain = _pick(args.domain, 'domain', 'pem')
    _cfg_seq = cfg.get('seq') if 'seq' in cfg else None
    if _cfg_seq is None:
        _cfg_seq = args.seq
    if isinstance(_cfg_seq, list):
        seq: List[str] = [str(x).strip() for x in _cfg_seq if str(x).strip()]
    else:
        seq: List[str] = [x.strip() for x in str(_cfg_seq).split(',') if x.strip()]
    domain = str(_cfg_domain).strip().lower()

    _cfg_timeout = _pick(args.timeout, 'timeout', None)
    _cfg_debug = _pick(args.debug, 'debug', '1')
    _cfg_save = _pick(args.save, 'save', '1')

    if _bool(_cfg_debug):
        os.environ.setdefault("LBOPB_GEMINI_DEBUG", "1")
    if _cfg_timeout is not None:
        os.environ["LBOPB_GEMINI_TIMEOUT_SEC"] = str(float(_cfg_timeout))

    # 延迟导入以便环境变量生效到客户端
    try:
        from lbopb.src.rlsac.kernel.common.llm_oracle import (
            build_pathfinder_prompt,
            call_llm,
        )
    except ImportError:
        # 兼容直接脚本执行：将仓库根加入 sys.path 后用绝对导入
        from pathlib import Path as _Path
        import sys as _sys
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
        from lbopb.src.rlsac.kernel.common.llm_oracle import (
            build_pathfinder_prompt,
            call_llm,
        )

    prompt = build_pathfinder_prompt(domain, seq)
    prompt_bytes = len(prompt.encode("utf-8"))
    print(
        f"[TEST] domain={domain} seq={seq} prompt_len={len(prompt)} bytes={prompt_bytes}"
    )
    pv = prompt[:600] + ("..." if len(prompt) > 600 else "")
    print(f"[TEST] prompt preview:\n{pv}")

    t0 = _t.time()
    res = call_llm(prompt)
    dt = _t.time() - t0

    err = isinstance(res, str) and (
            res.startswith("[Gemini Error]") or res.startswith("[Gemini HTTPError]")
    )
    used = not err and isinstance(res, str)
    print(
        f"[TEST] done: used={used} err={err} dt={dt:.2f}s len={len(res) if isinstance(res, str) else 0}"
    )
    rprev = res[:400] + ("..." if isinstance(res, str) and len(res) > 400 else "") if isinstance(res, str) else str(res)
    print(f"[TEST] result preview: {rprev}")

    if _bool(_cfg_save):
        try:
            repo_root = Path(__file__).resolve().parents[2]
            out_dir = repo_root / "out" / "test_llm_injection"
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"test_{int(_t.time())}.json"
            obj = {
                "domain": domain,
                "sequence": seq,
                "prompt_len": len(prompt),
                "prompt_bytes": prompt_bytes,
                "prompt": prompt,
                "result": res,
                "used": bool(used),
                "error": bool(err),
                "latency_sec": float(dt),
            }
            text = json.dumps(obj, ensure_ascii=False, indent=2)
            # 写入 CRLF 行尾
            text = text.replace("\r\n", "\n")
            fname.write_text(text, encoding="utf-8")
            print(f"[TEST] saved to {fname}")
        except Exception as e:
            print(f"[TEST] save error: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

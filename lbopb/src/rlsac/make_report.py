# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import sys
import re
from pathlib import Path
import time as _pytime


def gen_report_from_log(log_file: Path, out_md: Path) -> None:
    text = log_file.read_text(encoding='utf-8')
    pat = re.compile(r"^\[STEP\s+(?P<t>\d+)\]\s+a=(?P<a>-?\d+)\s+r=(?P<r>-?\d+\.?\d*)\s+d=(?P<d>True|False)\s+buf=(?P<buf>\d+)\s+upd=(?P<upd>\d+)\s+loss_q1=(?P<lq1>[-\.\d]+)\s+loss_q2=(?P<lq2>[-\.\d]+)\s+loss_pi=(?P<lpi>[-\.\d]+)\s+alpha=(?P<alpha>[-\.\d]+)$",
                     re.MULTILINE)
    rewards = []
    updates = 0
    last = {}
    for m in pat.finditer(text):
        r = float(m.group('r'))
        rewards.append(r)
        updates += int(m.group('upd'))
        last = {
            't': int(m.group('t')),
            'a': int(m.group('a')),
            'r': r,
            'd': m.group('d'),
            'buf': int(m.group('buf')),
            'upd': int(m.group('upd')),
            'lq1': m.group('lq1'),
            'lq2': m.group('lq2'),
            'lpi': m.group('lpi'),
            'alpha': m.group('alpha'),
        }
    mean_r = (sum(rewards) / len(rewards)) if rewards else 0.0
    last100 = (sum(rewards[-100:]) / len(rewards[-100:])) if len(rewards) >= 1 else 0.0
    lines = []
    lines.append(f"# 训练报告 (run: {log_file.parent.name})")
    lines.append("")
    lines.append(f"- 日期：{_pytime.strftime('%Y-%m-%d %H:%M:%S', _pytime.localtime())}")
    lines.append("")
    lines.append("## 概览")
    lines.append(f"- 总步数：{len(rewards)}")
    lines.append(f"- 总更新轮数：{updates}")
    lines.append(f"- 平均奖励：{mean_r:.6f}")
    lines.append(f"- 近100步平均奖励：{last100:.6f}")
    if last:
        lines.append(f"- 最后一次记录：t={last['t']} buf={last['buf']} upd={last['upd']} alpha={last['alpha']}")
        lines.append(f"  - loss_q1={last['lq1']} loss_q2={last['lq2']} loss_pi={last['lpi']}")
    # Ensure CRLF
    content = ("\n".join(lines)).replace("\r\n","\n").replace("\n","\r\n")
    out_md.write_text(content, encoding='utf-8')


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python -m lbopb.src.rlsac.make_report <run_dir_or_log>")
        return 1
    p = Path(argv[0])
    if p.is_dir():
        log = p / 'train.log'
        out = p / 'train_report.md'
    else:
        log = p
        out = log.with_suffix('.md')
    if not log.exists():
        print(f"Log not found: {log}")
        return 2
    gen_report_from_log(log, out)
    print(f"Report written: {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))



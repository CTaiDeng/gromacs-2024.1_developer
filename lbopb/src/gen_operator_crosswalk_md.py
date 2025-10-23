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

from __future__ import annotations

"""从 operator_crosswalk.json 生成 Markdown 预览表。

同步规范（规定）：
 - `operator_crosswalk.md` 为自动生成文件，请勿手工编辑；
 - 当 `operator_crosswalk.json` 发生变更后，需手动执行本脚本生成最新 Markdown：
     python -m lbopb.src.gen_operator_crosswalk_md [-o 输出路径]
 - 生成结果默认写入与 JSON 同目录的 `operator_crosswalk.md`。

数据来源：`lbopb/src/operator_crosswalk.json`（见其中 meta.docs 的知识库引用）。
"""

import argparse
import os
from typing import Any, Dict, Iterable, Mapping

from .op_crosswalk import (
    load_crosswalk,
    list_modules,
    basic_ops,
    crosswalk_for_tag,
    canonical_package,
)
from .powerset import get_powerset_config, generate_by_generator


def _h1(s: str) -> str:
    return f"# {s}\n\n"


def _h2(s: str) -> str:
    return f"## {s}\n\n"


def _table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    hs = list(headers)
    out = ["| " + " | ".join(hs) + " |"]
    out.append("| " + " | ".join([":---" for _ in hs]) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n\n"


def render_markdown(cw: Mapping[str, Any]) -> str:
    meta = cw.get("meta", {})
    modules = list_modules(cw)
    tags = cw.get("tags", {})

    lines: list[str] = []
    lines.append(_h1("LBOPB 幺半群算子联络预览（自动生成）"))
    lines.append("本文件由脚本自动生成（请勿手工编辑）。数据源：`lbopb/src/operator_crosswalk.json`。\n\n")

    # 概念与术语
    lines.append(_h2("概念与术语"))
    lines.append(
        "- 幺半群（Monoid）：带有结合律与单位元的代数结构；本文中各模块（PEM/PRM/TEM/PKTM/PGOM/PDEM/IEM）均是非交换幺半群。\n"
        "- 基本算子（Basic Operator）：各模块的最小过程单元（如 Dose/Absorb/Activate/Repair 等）。\n"
        "- 规范化算子包（Canonical Package）：仅由基本算子构成、能代表通用时序的序列（如 ADME 管线、损伤-修复链）。\n"
        "- 联络（Crosswalk）：以语义标签为桥，建立跨模块基本算子的类比映射与包的对齐规则。\n"
        "- 幂集（Powerset）：在约束下枚举仅基本算子构成的序列（自由幺半群），并可结合常用序列与生成器。\n\n"
    )

    # 对齐原则与使用建议
    guidelines = meta.get("guidelines", [])
    if guidelines:
        lines.append(_h2("对齐原则与使用建议"))
        for g in guidelines:
            lines.append(f"- {g}\n")
        lines.append("\n")
    # 基本信息
    lines.append(_h2("基本信息"))
    rows = [
        ("版本", str(meta.get("version", "-"))),
        ("模块", ", ".join(modules)),
    ]
    lines.append(_table(["键", "值"], rows))

    # 文档引用
    docs = meta.get("docs", [])
    if docs:
        lines.append(_h2("知识库文档引用"))
        for p in docs:
            lines.append(f"- `{p}`\n")
        lines.append("\n")

    # 语义标签
    if tags:
        lines.append(_h2("语义标签（统一字典）"))
        trows = [(k, str(v)) for k, v in tags.items()]
        trows.sort(key=lambda x: x[0])
        lines.append(_table(["标签", "说明"], trows))

    # 各模块基本算子与标签
    lines.append(_h2("各模块基本算子与标签"))
    for m in modules:
        ops = basic_ops(cw, m)
        if not ops:
            continue
        lines.append(f"### {m}\n\n")
        orows = []
        for op_name, tag_list in sorted(ops.items(), key=lambda kv: kv[0].lower()):
            orows.append((op_name, ", ".join(tag_list)))
        lines.append(_table(["基本算子", "语义标签"], orows))

    # 按标签的跨模块联络
    xw = cw.get("crosswalk_by_tag", {})
    if xw:
        lines.append(_h2("跨模块联络（按语义标签）"))
        for tag in sorted(xw.keys()):
            lines.append(f"### {tag}\n\n")
            mapm = crosswalk_for_tag(cw, tag)
            xrows = []
            for mod in modules:
                opl = mapm.get(mod, [])
                if opl:
                    xrows.append((mod, "、".join(opl)))
            if xrows:
                lines.append(_table(["模块", "基本算子"], xrows))

    # 规范化算子包（仅基本算子）
    pkgs = cw.get("canonical_packages", {})
    if pkgs:
        lines.append(_h2("规范化算子包（仅基本算子）"))
        desc_map = cw.get("canonical_packages_desc", {})
        for name in sorted(pkgs.keys()):
            lines.append(f"### {name}\n\n")
            d = desc_map.get(name)
            if d:
                lines.append("#### 说明：\n\n")
                lines.append(f"{d}\n\n")
            prow = []
            pkgmap = canonical_package(cw, name)
            for mod in modules:
                seq = pkgmap.get(mod)
                if seq:
                    prow.append((mod, " → ".join(seq)))
            if prow:
                lines.append(_table(["模块", "算子序列"], prow))

    # 幂集算法配置与常用幂集
    psets = cw.get("powersets", {})
    if psets:
        lines.append(_h2("幂集算法配置与常用幂集（仅基本算子）"))
        for mod in modules:
            if mod not in psets:
                continue
            cfg = get_powerset_config(cw, mod)
            lines.append(f"### {mod}\n\n")
            base = cfg.get("base", [])
            max_len = str(cfg.get("max_len", "-"))
            cons = cfg.get("constraints", {})
            cons_desc = ", ".join(
                [
                    ("禁止相邻重复" if cons.get("no_consecutive_duplicate", True) else "允许相邻重复"),
                    ("跳过Identity" if cons.get("skip_identity", True) else "包含Identity"),
                ]
            )
            lines.append(
                _table(["键", "值"], [("基本算子集", ", ".join(base)), ("最大长度", max_len), ("约束", cons_desc)]))
            if cfg.get("notes"):
                lines.append("#### 说明：\n\n")
                lines.append(f"{cfg.get('notes')}\n\n")

            fam = cfg.get("families", {})
            if fam:
                frows = []
                for fname, seqs in sorted(fam.items(), key=lambda kv: kv[0].lower()):
                    frows.append((fname, "；".join([" → ".join(seq) for seq in seqs])))
                lines.append(_table(["常用幂集族名", "算子序列组"], frows))
                fdesc = cfg.get("family_descriptions", {})
                if fdesc:
                    rows = [(k, str(v)) for k, v in sorted(fdesc.items())]
                    lines.append(_table(["族名", "说明"], rows))

            # 生成器（常用序列生成器）
            gens = cfg.get("generators", [])
            if gens:
                grows = []

                def render_step(st: Any) -> str:
                    if isinstance(st, str):
                        return st
                    if isinstance(st, dict) and "choice" in st:
                        return "(" + " | ".join(st["choice"]) + ")"
                    if isinstance(st, dict) and "repeat" in st:
                        spec = st["repeat"]
                        op = str(spec.get("op"))
                        mi = spec.get("min", 1)
                        ma = spec.get("max", mi)
                        return f"{op}{{{mi}..{ma}}}"
                    return "?"

                for g in gens:
                    name = str(g.get("name"))
                    chain = g.get("chain", [])
                    pattern = " → ".join([render_step(x) for x in chain])
                    grows.append((name, pattern))
                lines.append(_table(["生成器名", "链式模式"], grows))
                gdesc = cfg.get("generator_descriptions", {})
                if gdesc:
                    rows = [(k, str(v)) for k, v in sorted(gdesc.items())]
                    lines.append(_table(["生成器名", "说明"], rows))

                # 生成器示例（自动抽样）
                sample_rows: list[tuple[str, str]] = []
                SAMPLE_N = 5
                for g in gens:
                    name = str(g.get("name"))
                    try:
                        seqs = []
                        for i, seq in enumerate(generate_by_generator(mod, name)):
                            if i >= SAMPLE_N:
                                break
                            seqs.append(" → ".join(seq))
                        sample_rows.append((name, "；".join(seqs) if seqs else "(无生成)"))
                    except Exception as e:  # 容错输出
                        sample_rows.append((name, f"生成失败: {e}"))
                if sample_rows:
                    lines.append(_table(["生成器名", f"示例（前{SAMPLE_N}条）"], sample_rows))

    # 同步规范
    lines.append(_h2("同步规范（重要）"))
    lines.append(
        "- 本文件由 `python -m lbopb.src.gen_operator_crosswalk_md` 自动生成；源为 `operator_crosswalk.json`。\n"
    )
    lines.append("- 修改 JSON 后，请重新执行上述命令同步更新本文件。\n")
    lines.append("- 仓库 Git Hooks 不执行自动改写（遵循 AGENTS 规范）；需人工手动运行脚本。\n\n")

    # 案例包（可复现示例）
    cases = cw.get("case_packages", {})
    if cases:
        lines.append(_h2("案例包（可复现示例）"))
        for name in sorted(cases.keys()):
            case = cases[name]
            lines.append(f"### {name}\n\n")
            if case.get("description"):
                lines.append("#### 说明：\n\n")
                lines.append(f"{case['description']}\n\n")
            if case.get("notes"):
                lines.append("#### 说明：\n\n")
                lines.append(f"{case['notes']}\n\n")
            seqs = case.get("sequences", {})
            rows = []
            for mod in list_modules(cw):
                s = seqs.get(mod)
                if s:
                    rows.append((mod, " → ".join(s)))
            if rows:
                lines.append(_table(["模块", "算子序列"], rows))

            # 复现代码示例（基于 powerset.compose_sequence）
            lines.append("#### 复现示例（Python）：\n\n")
            code = []
            code.append("from lbopb.src.powerset import compose_sequence")
            code.append("from lbopb.src.op_crosswalk import load_crosswalk")
            code.append("cw = load_crosswalk()")
            code.append(f"case = cw['case_packages']['{name}']")
            code.append("seqs = case['sequences']")
            code.append("# 示例：药效（PDEM）复合并应用\nfrom lbopb.src.pdem import PDEMState")
            code.append("pdem_seq = seqs['pdem']")
            code.append("O = compose_sequence('pdem', pdem_seq)")
            code.append("s0 = PDEMState(b=1.5, n_comp=1, perim=0.8, fidelity=0.6)")
            code.append("s1 = O(s0)")
            code.append("print('PDEM seq:', pdem_seq)\nprint('s0→s1:', s0, '→', s1)")
            lines.append("```python\n" + "\n".join(code) + "\n```\n\n")

    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出 Markdown 文件路径；默认写入 JSON 同目录 operator_crosswalk.md",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="自定义 JSON 路径；默认使用包内 operator_crosswalk.json",
    )
    args = parser.parse_args()

    cw = load_crosswalk(args.json)
    md = render_markdown(cw)
    if args.output:
        out_path = args.output
    else:
        # 默认与 JSON 同目录
        json_path = args.json
        if json_path is None:
            # 包内默认 JSON
            json_path = os.path.join(os.path.dirname(__file__), "operator_crosswalk.json")
        out_path = os.path.join(os.path.dirname(os.path.abspath(json_path)), "operator_crosswalk.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote Markdown preview to: {out_path}")


if __name__ == "__main__":
    main()

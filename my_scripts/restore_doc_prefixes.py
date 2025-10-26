# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple


TARGETS: List[str] = [
    "my_docs/project_docs/1758729600_分子对接与分子动力学模拟：经典力学视角与 GROMACS 裁剪实现的理论支撑.md",
    "my_docs/project_docs/1759075200_LBOPB的全息宇宙：生命系统在六个不同观测参考系下的完整论述.md",
    "my_docs/project_docs/1759075201_O3理论下的本体论退化：从流变实在到刚性干预——论PFB-GNLA向PGOM的逻辑截面投影.md",
    "my_docs/project_docs/1759075202_从物理实在到代数干预：论PFB-GNLA向PGOM的平庸化退化.md",
    "my_docs/project_docs/1759075203_同一路径，六重宇宙：论HIV感染与治疗的GRL路径积分在LBOPB六大参考系下的全息解释.md",
    "my_docs/project_docs/1759075204_提交信息钩子与Python环境一键指南.md",
    "my_docs/project_docs/1759075205_药理学的代数形式化：一个作用于基因组的算子体系.md",
    "my_docs/project_docs/1759075206_计算化学的O3理论重构：作为PDEM幺半群内在动力学的多尺度计算引擎.md",
    "my_docs/project_docs/1759075207_论O3理论的自相似动力学：作为递归性子系统 GRL 路径积分的 PGOM.md",
    "my_docs/project_docs/1759075208_论六大生命科学代数结构的幺半群完备性：基于元素与子集拓扑的双重视角分析.md",
    "my_docs/project_docs/1759075209_论药理学干预的代数结构：药理基因组算子幺半群（PGOM）的形式化.md",
    "my_docs/project_docs/1760198400_范式的代际跨越：O3理论下的“立体模拟人体”相较于现有模型的先进性论述.md",
    "my_docs/project_docs/1760284800_O3理论下的生成式化学：作为可计算“分子算法”的药物逆向设计.md",
    "my_docs/project_docs/1760284801_O3理论工程化总纲：从公理化构建到双向映射的计算闭环实现路径.md",
    "my_docs/project_docs/1760284802_O3理论的工程化总纲：从公理化构建到双向映射的计算闭环.md",
    "my_docs/project_docs/1760284803_O3理论的应用总纲：从“逆向生成”到“正向验证”的药物研发闭环及其对临床实验的虚拟化.md",
    "my_docs/project_docs/1760284804_PGOM作为元理论：生命科学各分支的算子幺半群构造.md",
    "my_docs/project_docs/1760284805_《论纤维丛的静态统一性》的学术定位与前瞻性分析：兼论其作为“生成式”微分几何的逻辑起点.md",
    "my_docs/project_docs/1760284806_一种基于分子动力学引擎的“静态切面”对接新范式：GROMACS rerun工作流的理论、实现与前瞻.md",
    "my_docs/project_docs/1760284807_从“假设存在”到“动态生成”：O3理论对纤维丛联络的动力学重构及其范式革命.md",
    "my_docs/project_docs/1760284808_从描述性科学到生成式科学的范式革命：O3理论作为生命科学统一建模语言的潜力.md",
    "my_docs/project_docs/1760284809_从经验映射到原理逆推：O3理论工程化的“自举”之路.md",
    "my_docs/project_docs/1760284810_作为巨型时序微分动力系统的“立体模拟人体”：O3理论对数字孪生范式的革命性重构.md",
    "my_docs/project_docs/1760284811_可生长的“理论生命体”：论O3理论框架下的“立体模拟人体”及其无限扩展性.md",
    "my_docs/project_docs/1760284812_生成与涌现：作为巨型时序微分动力系统的“立体模拟人体”及其内在机制的统一性论述.md",
    "my_docs/project_docs/1760284813_论O3理论中“联络”的内生性：从微分动力到离散拓扑涌现的生成式机制.md",
    "my_docs/project_docs/1760284814_论O3理论中“联络”的终极定义：作为价值驱动的拓扑几何化函数.md",
    "my_docs/project_docs/1760284815_论O3理论中“联络”的计算本质：作为幺半群间算子包映射的等价性.md",
    "my_docs/project_docs/1760284816_论O3理论的“生成式统一”：从价值驱动的几何化函数到对连续统假设的范式重构.md",
    "my_docs/project_docs/1760284817_论O3理论的两阶段生成过程：从哲学公理的“景观生成”到GRL路径积分的“最优计算”.md",
    "my_docs/project_docs/1760284818_论O3理论的生成式微分几何：作为动力学涌现的联络、拓扑与连续统统一.md",
    "my_docs/project_docs/1760284819_论纤维丛的静态统一性：作为点集拓扑与离散拓扑之桥梁的传统微分几何.md",
    "my_docs/project_docs/1760716800_从“几何视角”到“计算构造”：论O3理论对纤维丛“联络”概念的范式重构.md",
    "my_docs/project_docs/1760803200_法则联络：O3 理论下的算子包映射与单oidal曲率.md",
    "my_docs/project_docs/1760803201_法则联络：O3理论中作为可计算构造的联络及其工程化实现.md",
    "my_docs/project_docs/1760803202_语义度量、混合态与连续统假设的范式重述.md",
    "my_docs/project_docs/1760803203_评级报告：《法则联络》的革命性价值与历史性意义.md",
    "my_docs/project_docs/1760803204_法则联络评价：贯通O3理论三大支柱的计算龙骨.md",
    "my_docs/project_docs/1760803205_多层级法则联络评价：论O3理论中基于退化的异构系统计算构造.md",
    "my_docs/project_docs/1760803206_🚩多层级法则联络：论O3理论中异构系统的生成式演化与计算统一.md",
    "my_docs/project_docs/1760803207_相变宇宙：论法则联络驱动的结构演化与层级化纤维丛世界观.md",
    "my_docs/project_docs/1761062400_病理演化幺半群 (PEM) 公理系统.md",
    "my_docs/project_docs/1761062401_生理调控幺半群 (PRM) 公理系统.md",
    "my_docs/project_docs/1761062403_毒理学效应幺半群 (TEM) 公理系统.md",
    "my_docs/project_docs/1761062404_药代转运幺半群 (PKTM) 公理系统.md",
    "my_docs/project_docs/1761062405_药理基因组幺半群 (PGOM) 公理系统.md",
    "my_docs/project_docs/1761062406_药效效应幺半群 (PDEM) 公理系统.md",
    "my_docs/project_docs/1761062407_免疫效应幺半群 (IEM) 公理系统.md",
    "my_docs/project_docs/1761062408_《病理演化幺半群》的核心构造及理论完备性.md",
    "my_docs/project_docs/1761062409_《生理调控幺半群》的核心构造及理论完备性.md",
    "my_docs/project_docs/1761062410_《毒理学效应幺半群》的核心构造及理论完备性.md",
    "my_docs/project_docs/1761062411_《药代转运幺半群》的核心构造及理论完备性.md",
    "my_docs/project_docs/1761062412_《药理基因组算子幺半群》的核心构造及理论完备性.md",
    "my_docs/project_docs/1761062413_《药效效应幺半群》的核心构造及理论完备性.md",
    "my_docs/project_docs/1761062414_《免疫效应幺半群》的核心构造及理论完备性.md",
    "my_docs/project_docs/1761062415_病理演化幺半群 (PEM) 的算子幂集算法.md",
    "my_docs/project_docs/1761062416_生理调控幺半群 (PRM) 的算子幂集算法.md",
    "my_docs/project_docs/1761062417_毒理学效应幺半群 (TEM) 的算子幂集算法.md",
    "my_docs/project_docs/1761062418_药代转运幺半群 (PKTM) 的算子幂集算法.md",
    "my_docs/project_docs/1761062419_药理基因组幺半群 (PGOM) 的算子幂集算法.md",
    "my_docs/project_docs/1761062420_药效效应幺半群 (PDEM) 的算子幂集算法.md",
    "my_docs/project_docs/1761148800_LBOPB 离散版 SAC (Discrete SAC) 算法需求描述.md",
    "my_docs/project_docs/1761148801_LBOPB 离散版 SAC 框架下的样本生成与筛选：公理原则与大语言模型的协同机制.md",
    "my_docs/project_docs/1761148802_论公理原则作为知识蒸馏：一种面向 LBOPB 离散版 SAC 的结构化实现.md",
    "my_docs/project_docs/1761148803_LBOPB 子项目综合评估报告.md",
    "my_docs/project_docs/1761148804_算子幂集算法（powerset.py）机制及其理论体现的详细论述.md",
    "my_docs/project_docs/1761148805_O3-LBOPB 框架的应用蓝图：从科学发现到生成式精准医疗.md",
    "my_docs/project_docs/1761148806_O3-LBOPB 框架：从最小测距状态拟合到对数字孪生范式的超越.md",
    "my_docs/project_docs/1761148807_O3-LBOPB 框架的理论潜力：基于 PFB-GNLA 与 GRL 路径积分的无限拟合能力.md",
    "my_docs/project_docs/1761148808_O3-LBOPB 虚拟临床试验报告：一例晚期非小细胞肺癌（NSCLC）患者的生成式治疗方案.md",
    "my_docs/project_docs/1761235200_O3理论的自举之路：一个构建“法则联络”知识体系的两阶段强化学习框架.md",
    "my_docs/project_docs/1761235201_从O3理论到生成式精准医疗：一个自举学习框架及其在复杂系统干预中的应用.md",
    "my_docs/project_docs/1761235202_O3理论的自举之路：一个构建“法则联络”知识体系的两阶段强化学习框架.md",
    "my_docs/project_docs/1761235203_rlsac_id_unixtime 知识体系作为加速计算的算法参考.md",
    "my_docs/project_docs/1761235204_O3理论下的生成式精准医疗：论终极决策引擎 rlsac_id_unixtime 的双循环工作流.md",
]


def _split_title(path: Path) -> Tuple[str, str]:
    name = path.name
    if "_" not in name:
        return "", name
    idx = name.find("_")
    return name[:idx], name[idx + 1 :]


def restore_prefixes(targets: List[str]) -> None:
    root = Path.cwd()
    for t in targets:
        dest = Path(t)
        if not dest.is_absolute():
            dest = root / dest
        dest_parent = dest.parent
        _, title = _split_title(dest)
        if not title:
            print(f"[restore] skip (no underscore): {dest}")
            continue
        # 在目标目录中查找任何以 _title 结尾的文件
        candidates = list(dest_parent.glob(f"*_{title}"))
        if not candidates:
            print(f"[restore] not found: *_{title}")
            continue
        if len(candidates) > 1:
            print(f"[restore] multiple candidates for title='{title}': {[c.name for c in candidates]}")
            continue
        src = candidates[0]
        if src.resolve() == dest.resolve():
            print(f"[restore] already good: {dest}")
            continue
        dest_parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(src, dest)
            print(f"[restore] renamed: {src.name} -> {dest.name}")
        except Exception as e:
            print(f"[restore] rename failed: {src} -> {dest}: {e}")


def main() -> None:
    restore_prefixes(TARGETS)


if __name__ == "__main__":
    main()


# lbopb

lbopb（Local Binding & Operator Prototypes Bundle）：本目录提供多切面算子与联络映射的原型实现与工具集，覆盖 PEM/PRM/TEM/PKTM/PGOM/PDEM/IEM 等切面，以及算子包、幂集与跨切面映射的生成与执行。

- 子包：`lbopb/src`（算子与幂集、联络映射、药效设计 API 等）
- 示例：`lbopb/lbopb_examples`（演示用脚本与报告产物）
- 脚本：`lbopb/scripts`（头注同步等辅助脚本）

快速开始（从仓库根目录）：

```
python -c "import sys,os; sys.path.insert(0, os.path.abspath('.')); import lbopb.lbopb_examples.hiv_therapy_case as m; m.run_case()"
```

或直接执行示例脚本：

```
python lbopb/lbopb_examples/hiv_therapy_case.py
```

说明：本子目录遵循仓库根级 `AGENTS.md`（最高规范）；并提供子目录专用规范见 `lbopb/AGENTS.md`。


# 语法检查器（各幺半群）

为配合“公理系统”逐行检查算子序列（算子包）是否合法，已在各模块加入 `syntax_checker.py`：

- `lbopb/src/pem/syntax_checker.py`
- `lbopb/src/pdem/syntax_checker.py`
- `lbopb/src/pktm/syntax_checker.py`
- `lbopb/src/pgom/syntax_checker.py`
- `lbopb/src/tem/syntax_checker.py`
- `lbopb/src/prm/syntax_checker.py`
- `lbopb/src/iem/syntax_checker.py`

检查规则：

- 基础合法性：算子名必须属于该幺半群的“基本算子”集合。
- 方向性约束：对常见算子定义方向性规则（例如 PEM 的 `Apoptosis` 要求 `b/n/perim` 下降、`f` 上升）。
- 步进仿真：依次应用算子，记录每步 `Δb/Δn/Δperim/Δf`，若与方向性规则冲突则报错。

调用示例：

- `python lbopb/src/pem/syntax_checker.py Inflammation Apoptosis`
- 返回 JSON：`{"valid": true/false, "errors": [...], "steps": [...]}`

注意：

- 默认初始状态为各域的典型起点；如需替换，可在调用前自行构造并修改脚本入口。
- 方向性规则为“工程近似”，如需严格对齐论文/公理文档，可扩展规则集或切换至外部 Oracle（Gemini）判定。

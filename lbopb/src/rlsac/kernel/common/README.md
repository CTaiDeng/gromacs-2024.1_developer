# LLM（Gemini）判定注入模板与使用说明

本说明给出当 `use_llm_oracle=true` 时，与外部 LLM（如 Gemini）配合所需的“定义模板（prompt）”，以及接口与返回约束。模板已在 `llm_oracle.py` 中实现：

- 单域（Pathfinder）模板函数：`build_pathfinder_prompt(domain, sequence)`
- 跨域（Connector）模板函数：`build_connector_prompt(conn)`

接口入口：`call_llm(prompt, provider='gemini')`（在 `my_scripts/gemini_client.py` 对接实际 API）

## 1) 判定约束（强制）

- LLM 必须只返回单个字符：`'1'` 或 `'0'`
  - `'1'` 表示“符合相应公理系统/联络要求”
  - `'0'` 表示“不符合”
- 任何其它字符、文字、解释均视为非合规输出（调用方可做兜底处理）
- 当 `syntax_checker` 给出“显著错误（errors 非空）”时，LLM 不会被调用（直接判 0）
- 当仅有“警告（warnings 非空）”时，才启用 LLM 辅助；若 `warnings` 为空，默认仅使用内置启发式/一致性判定

## 2) 单域（Pathfinder）模板

函数：`build_pathfinder_prompt(domain: str, sequence: List[str]) -> str`

模板示例（变量注入后）：

```
你是一名严格的形式系统审查器。
幺半群域: pem
公理文档: my_docs/project_docs/1761062400_病理演化幺半群 (PEM) 公理系统.md
任务: 判断以下基本算子序列(算子包)是否严格符合该域的公理系统(所有必要约束: 方向性/不可交换/序次/阈值/停机等)。
算子序列: ["Inflammation", "Apoptosis"]
要求: 只返回单个字符 '1' 或 '0'。1 表示符合公理系统, 0 表示不符合。不得输出其他字符或解释。
```

说明：
- `domain`: pem/pdem/pktm/pgom/tem/prm/iem
- `公理文档`: 由 `DOC_MAP` 查表得到
- `序列`: 由训练/评估时的候选“算子包”注入

## 3) 跨域（Connector）模板（两两整合）

函数：`build_connector_prompt(conn: Dict[str, List[str]]) -> str`

- 其中 `conn = {"pem": [...], "pdem": [...], ..., "iem": [...]}` 表示七域各自的算子包序列
- 模板对每个“对偶域”构造一段说明，并要求 LLM 最终对“整体联络是否成立”返回 `1/0`

模板示例（片段）：

```
你是一名严格的形式系统审查器, 要判断跨七域的联络候选体是否成立。
请基于各域公理系统, 对以下对偶域进行逐一一致性判定(因果/量纲/时序/阈值), 最后给出整体判定: 

对偶域 pdem<->pktm
文档A: my_docs/project_docs/1761062406_药效效应幺半群 (PDEM) 公理系统.md
文档B: my_docs/project_docs/1761062404_药代转运幺半群 (PKTM) 公理系统.md
序列A: ["Bind", "Signal"]
序列B: ["Absorb", "Distribute"]

...（其余对偶域类似展开）...

请严格依公理判定整体联络是否成立: 若所有对偶域均一致且不违背任何域公理, 返回 '1'; 否则返回 '0'。
只返回单个字符 '1' 或 '0'。不得输出其他文本。
```

说明：
- 对偶域对在 `PAIRWISE` 中定义：`[(pdem,pktm), (pgom,pem), (tem,pktm), (prm,pem), (iem,pem)]`
- 模板会注入每个域的公理文档路径与该域的算子包序列
- LLM 必须只返回 `1/0`

## 4) my_scripts/gemini_client.py 对接建议

- 推荐在 `my_scripts/gemini_client.py` 暴露一个纯文本接口函数：

```
# 任意一个函数名，call_llm 会按顺序探测：ask/generate/chat/gemini_text/query/run

def ask(prompt: str) -> str:
    # 这里调用实际 Gemini API，返回 '1' 或 '0'
    ...
```

- 返回值应严格为 `'1'` 或 `'0'`；若返回其它内容，调用方会尝试解析（不保证可靠）
- 为提升输出稳定性，建议：
  - 设置较低 `temperature` 与 `top_p`（使输出更确定）
  - 使用系统/用户指令强约束：“只返回单个字符 '1' 或 '0'，不得输出其它文本”

## 5) 调用路径与开关

- Pathfinder（单域）
  - `lbopb/src/rlsac/kernel/rlsac_pathfinder/oracle.py` 中 `AxiomOracle`
  - `use_llm_oracle=true` 且 syntax_checker 仅出现“警告”时，调用 `build_pathfinder_prompt()` + `call_llm()`
- Connector（跨域）
  - `lbopb/src/rlsac/kernel/rlsac_connector/oracle.py` 中 `ConnectorAxiomOracle`
  - `use_llm_oracle=true` 且任一域仅出现“警告”时，调用 `build_connector_prompt()` + `call_llm()`

## 6) 与 syntax_checker 的关系（双重判定）

- 显著错误（errors 非空）
  - 直接判定为 0，**不会**调用 LLM
- 仅有警告（warnings 非空）
  - 训练/评估时，按配置启用 LLM 辅助判定：LLM 返回 1 才与启发式/一致性判定共同决策
- 无警告
  - 默认仅使用内置启发式/一致性判定（不调用 LLM）

## 7) 典型集成步骤

1. 在 `my_scripts/gemini_client.py` 实现 `ask(prompt: str) -> str`（或 generate/chat 等之一），返回 `'1'/'0'`
2. 在 pathfinder/connector 的配置中开启 `use_llm_oracle=true`
3. 运行训练：
   - 单域：`python -m lbopb.src.rlsac.kernel.rlsac_pathfinder.train`
   - 跨域：`python -m lbopb.src.rlsac.kernel.rlsac_connector.train`
4. 确认日志中仅在出现“警告”的样本上调用 LLM；显著错误均本地拦截

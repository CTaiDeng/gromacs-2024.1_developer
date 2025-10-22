# lbopb_examples

示例展示如何使用 `lbopb/src` 的各子包与工具。

- `pem_basic_usage.py`：状态、算子复合与指标。
- `pem_noncommutativity_demo.py`：非交换性的影响（NC 指数）。
- `pem_path_search.py`：路径代价与可达性。
- `hiv_therapy_case.py`：HIV 治疗案例——以病理为基底，经联络映射药效，并在六切面展开算子包（参考 `lbopb/src/operator_crosswalk.json` 的 `case_packages`）。

运行（从仓库根目录）：

```
python -c "import sys,os; sys.path.insert(0, os.path.abspath('.')); import lbopb.lbopb_examples.hiv_therapy_case as m; m.run_case()"
```

或直接：

```
python -c "import sys,os; sys.path.insert(0, os.path.abspath('.')); exec(open('lbopb/lbopb_examples/hiv_therapy_case.py', 'r', encoding='utf-8').read())"
```


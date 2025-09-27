自定义脚本与数据清单（my_scripts）

概览
- 该目录收纳本项目新增的“非原生脚本”和数据，便于集中维护与备份，不污染上游 GROMACS 的原生 scripts/ 内容。

分类与说明
- 环境安装/配置（WSL/Ubuntu）
  - install_gromacs_wsl.sh：一键安装 GROMACS（apt 或源码编译，默认 2024.1）。
  - install_ambertools_acpype_wsl.sh：一键安装 AmberTools + ACPYPE（基于 conda-forge/micromamba）。
  - install_openbabel.ps1（Windows）：在 Conda 环境中安装 Open Babel 并验证。

- 参数化与转换（小分子/力场）
  - run_cgenff_local.ps1（Windows）：离线调用本地 cgenff 生成小分子 .str；配合 CHARMM36 使用。
  - fetch_paramchem_str.py（跨平台）：半自动（Selenium）从 ParamChem 获取 .str，支持人工介入。
  - data/：第三方数据（如 charmm36-jul2021.ff 力场目录等）。

- 文档对齐/知识库维护
  - align_my_documents.py：
    - 将 my_docs/** 下以“<时间戳>_”命名的文件，重命名为 Git 首次入库时间戳（秒）。
    - 在 Markdown 首个标题下方插入“日期：YYYY年MM月DD日”。
  - align_my_documents.ps1：Windows 包装器，调用同目录的 Python 脚本。

使用要点
- Windows / PowerShell：
  - 文档对齐：`pwsh -File my_scripts/align_my_documents.ps1`
  - Open Babel 安装：`pwsh -File my_scripts/install_openbabel.ps1`
  - 本地 CGenFF：`pwsh -File my_scripts/run_cgenff_local.ps1 -CGenFFBin C:\path\to\cgenff.exe -Mol2 path\to\ligand.mol2 -OutDir out -CharmmFFDir my_scripts\data\charmm36-jul2021.ff`

- WSL / Ubuntu：
  - 安装 GROMACS：`bash my_scripts/install_gromacs_wsl.sh --method source --version 2024.1 -j 8`
  - 安装 AmberTools+ACPYPE：`bash my_scripts/install_ambertools_acpype_wsl.sh`

维护约定
- my_scripts 下均为本项目新增或改造的脚本与资源；若需提交上游或分享，请将共性逻辑抽离并撰写 README。
- 若脚本之间存在调用关系，请优先使用相对路径（如 `$PSScriptRoot`）以保证可移动性。

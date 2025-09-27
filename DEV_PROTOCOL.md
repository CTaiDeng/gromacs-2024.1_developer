# my_docs 文档与知识库对齐协议

当触发“文档对齐指令”时，执行以下两项整理以保持知识库一致性与可检索性：

1) 文件名时间戳标准化
- 目标：将 `my_docs/**` 下文件名以 `<数字>_` 开头的文档，其时间戳前缀改为该文档的 Git 首次入库时间（Unix 秒 `%at`）。
- 做法：使用 `git mv` 重命名，以保留历史。

2) 标题下方日期补齐
- 目标：对 Markdown 文档，在首个 `# ` 标题下插入/规范日期行为 `日期：YYYY年MM月DD日`（来源为文件名中的时间戳前缀）。
- 若已有“日期”行，统一为上述格式；若无标题，则以文件名（去前缀与扩展名）生成一级标题并写入日期。

执行指令（文档对齐指令）
- PowerShell（Windows / PowerShell 7）：
  - `pwsh -File my_scripts/align_my_documents.ps1`
- 或 Python 直接调用：
  - `python3 my_scripts/align_my_documents.py`

脚本说明
- 路径：`my_scripts/align_my_documents.py`
- 包装器：`my_scripts/align_my_documents.ps1`
- 幂等：多次执行无副作用；仅在需要时重命名/写入。
- 重命名仅作用于以 `<数字>_` 命名的文件；未跟踪文件不会被重命名。

注意事项
- 请在干净工作区（`git status` 无未保存改动）下运行，便于审阅本次对齐的差异与提交。
- 若文件此前从未提交过，无法从 Git 获得时间戳，脚本将跳过该文件。

目录约定
- `my_docs/dev_docs`：项目开发文档（开发说明、环境、依赖、流程、安装脚本说明等）。
- `my_docs/project_docs`：项目知识库（论文笔记、方案、研究记录等，一般以“时间戳_标题.md”命名）。
 
�����ٷ����ű�Լ��
- ���м��η�/�����Զ���ű�ͳһ���� `my_scripts/` Ŀ¼��֧�� bash/ps1/py �ȡ�
- �µ����ű�����д�룬��ֹ�洢�� `scripts/` ��Ŀ¼��
- ʾ�������� `my_scripts/install_cmake_wsl.sh`��

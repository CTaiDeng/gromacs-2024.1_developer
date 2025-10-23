# rlsac1（DummyEnv 版）

- 独立的离散 SAC 实现，使用 `DummyEnv` 进行连通性与基线训练。
- 运行：
  - `python lbopb/src/rlsac1/train.py`（配置见同目录 `config.json`）
- 产物：`out/train_*/` 下生成 `policy.pt` 等权重与日志。

说明：本包不依赖 `lbopb/src/rlsac`，内部自洽（env/models/utils/replay_buffer/train）。

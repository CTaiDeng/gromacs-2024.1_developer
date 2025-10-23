# rlsac2（SequenceEnv 版）

- 独立的离散 SAC 实现，使用 `LBOPBSequenceEnv` 将 `operator_crosswalk_train.json` 的“模块→算子序列”转为训练样本。
- 运行：
  - `python lbopb/src/rlsac2/train.py`（配置见同目录 `config.json`）
- 产物：`out/train_*/` 下生成 `policy.pt` 等权重与日志；自动导出 `op_index.json`（op→id 对照表）。

说明：本包不依赖 `lbopb/src/rlsac`，内部自洽（sequence_env/models/utils/replay_buffer/train）。

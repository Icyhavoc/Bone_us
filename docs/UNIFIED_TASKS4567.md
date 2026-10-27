# 统一数据集任务 4–7（2026-10-26）

运行 `python -X utf8 -u run_unified_tasks4567.py`，正式输出
`experiments/unified_tasks4567_20261026/`。任务 1–3 的原结果只读保留。

baseline 完整参数 B 从任务1正式结果读取，为 random_0131：patch=64、dim=96、depth=4、heads=8、mlp_dim=384、lr=1e-4、dropout=0、weight_decay=0、batch=24、warmup=5。不通过 auto preset 重新选择参数。

|任务|输入与模型|超参|
|---|---|---|
|4|max1 + dyn_envelope + channel4 + ViT reward off|B|
|5|同任务4|共同52套候选中验证选优|
|6|max1 + dyn_envelope + channel3 + 四分支cross-vote|B|
|7|同任务6|共同52套候选中验证选优|

先完成 4、6 的固定参数对照，再运行 5、7 的搜索。每种方法搜索时复用已训练的 B，共 104 次训练。channel4 保留双Top10交集及旧相似度回退，channel3 为独立双Beam max1。cross-vote 保留逐分支分类、平均概率NLL训练和概率投票argmax，不迁移参考包按Beam融合的两头结构。

沿用任务1–3的模型、前处理、数据及完整52套候选；启动时校验数据、参考配置和训练源码哈希。训练上限40轮、patience12、seed2023、现行预热余弦与5e-6下限保持一致。每套按val loss选epoch，超参按验证accuracy降序、val loss升序、ID升序选优；搜索不读取test信号，只对固定参数和选出的最优模型评估test。

此前任务1的label1错分率9/26=34.62%，已通过用户的40%停止条件。本轮不新增其他停止条件。

`manifest.json`、`sources/` 固定协议和源码，逐套保存权重、配置、审计、历史和概率。`progress.json` 每轮更新，Windows占用时有限重试并写备用 `progress_fallback.json`。监控选择 updated_at 较新的有效状态，每15分钟报告。

新增结果保存于 `results.json`；`results_all7.json`、`REPORT.md`、`confusion_matrices_all7.png` 汇总此前1–3和本轮4–7，包含完整参数。短流程诊断位于 `_unified_tasks4567_20261026/`，不作为正式性能。

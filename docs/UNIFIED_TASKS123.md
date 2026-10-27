# 统一数据集任务 1–3（2026-10-25）

运行 `python -X utf8 -u run_unified_tasks123.py`。正式输出为
`experiments/unified_tasks123_20261025/`，同协议可复用已完成配置；候选、源码或数据哈希变化时拒绝混写。

- 任务 1：max1 + dyn_envelope + channel3 + ViT，reward off。
- 任务 2：max1 + dyn_envelope_full + channel3 + ViT，固定任务 1 最优参数。
- 任务 3：同任务 2 输入，搜索本方法最优参数。

五份参考 Top15 按完整参数去重，共 52 套。各次最多 40 轮、patience=12、seed2023，使用现行 AdamW、预热余弦、三次样条与 Tukey。channel3 的两个 Beam 独立取各自主体窗 max1。阈值固定 0.5。

搜索只训练和评估 train/val；每套按 val loss 选 epoch，配置按 val accuracy 降序、val loss 升序、ID 升序选优。只对选定配置评估 test。复用同方法已完成 checkpoint 时不再训练；任务 2 的配置可直接供任务 3 复用。

任务 1 完成后检查所选模型的测试 label=1 错分率 `FN/(FN+TP)`。严格大于 40% 则状态为 `stopped_by_label1_guard`，不会运行任务 2、3；等于 40% 不停止。完整测试有 26 个 label=1，错 11 个即触发。

`progress.json` 每轮原子更新；`manifest.json` 固定协议、源码与数据哈希，`sources/` 保存源码；每套保存配置、审计、历史、权重和概率。最终 `REPORT.md`、`results.json`、`gate.json`、`confusion_matrices.png` 汇总结果；被选模型目录另存测试预测 CSV。

训练 worker 独立运行，stdout/stderr 位于 `_unified_tasks123_20261025/`，同目录 `process.json` 和 `monitor.json` 记录 PID、分钟进度心跳。终态停止心跳。短流程验证位于该目录下的 smoke_runner*，不属于正式结果。

递归删除旧结果被自动审批拒绝；旧统一数据集实验和图像已从活动结果目录移入 `bin/backup/unified_cleanup_20261025/`，核对清单位于 `_unified_tasks123_20261025/archived_targets.json`。原始与派生数据集均保留。

2026-10-25 首次训练在完成 13 套后因 Windows 拒绝替换 progress.json（WinError 5）中断。进度 I/O 现使用独立临时文件，遇 Windows 5/32/33 最多尝试 12 次；进度主文件持续被占用时写 progress_fallback.json，进度异常不再中断训练。监控读取两个文件中 updated_at 较新的有效状态。

本次修复只涉及 atomic_json、Batch.update 和 tempfile 导入，AST 核对确认其余训练、搜索、选优及停止逻辑相同。原始 manifest、源码和失败日志保存在 `maintenance/io_retry_repair_20261025/`；该目录 repair_audit.json 记录修复前后源码哈希、13 套已完成产物哈希及真实 Windows 文件占用重试验证。原 13 套结果复用，从第 14 套续跑；原超参数、数据和停止条件不变。

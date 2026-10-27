# 参考对齐实验归档

侧分支 `reference-alignment-full-20261027` 保存两个正式实验的完整可核验产物：

- `reference_alignment_full_20261027/`：保留项目三次样条的五折实验，测试 55/67。
- `reference_alignment_linear_exact_20261027/`：与参考缓存逐值相同的线性输入及精确参考执行协议，测试 56/67。

每个目录均包含：

- 五折 `best.pt`、`history.json`、`complete.json` 和验证概率；
- 预处理输入、样本记录、逐样本输入审计和运行时源码快照；
- OOF 概率、逐折 Beam/集成测试概率、预测 CSV、manifest、progress、result 和报告。

权重使用 Git LFS；克隆后须执行 `git lfs pull` 才能取得实际 `.pt` 内容。中间线性运行 `reference_alignment_linear_20261027` 与 smoke 运行未归档，因为它们被最终精确线性运行取代。

最终线性目录内的 `reference_native_verification/` 保存直接运行 `code_reference` 冻结原生代码得到的源码快照、训练历史、验证/测试概率、指标和 checkpoint 哈希。参考原生五折模型张量与最终线性目录的五个 `best.pt` 逐元素相同，因此不重复存储第二套权重。

核验：

```powershell
python experiments/reference_alignment_linear_exact_20261027/reference_native_verification/verify_artifacts.py
```

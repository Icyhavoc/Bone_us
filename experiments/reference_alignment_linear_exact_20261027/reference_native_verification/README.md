# 参考原生代码本机复核

本目录来自 `code_reference/run.py` 的冻结单候选 worker，配置为 `six_channel_beam_vote/random_0086`。五折训练完成后，测试由参考包自身的 `test_implementation.infer` 与 `export_case` 执行。

本机结果为 56/67、accuracy 83.58%、AUC 0.912758、混淆矩阵 `[[16,10],[1,40]]`。五折验证概率、逐折测试概率和五折集成概率与上级精确线性实验逐元素相同。

参考原生 checkpoint 没有在这里重复保存。归档前已核对五折全部模型张量：

- `branches.*` 对应上级权重的 `encoders.*`；
- `beam_heads.*` 对应上级权重的 `heads.*`；
- `*.pos` 对应上级权重的 `*.position`；
- 映射后五折所有张量均逐元素相同。

原生 checkpoint 的 SHA256 保存在 `frozen_selection.json` 和每折 `complete.json`；实际可加载权重为上级 `fold_1/best.pt` 至 `fold_5/best.pt`。`code_snapshot/`、`test_implementation.py`、历史、概率和预测文件均为参考运行原样复制。

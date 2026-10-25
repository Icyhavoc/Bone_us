# ViT 超参数记录（2026-10-25）

## 修改前快照

同步前 HEAD 为 `b80f724`（`info:gui fix`），远端 `origin/main` 同样为该提交。
未提交的探索改动已备份为 Git stash：`backup before dataset+hyperparameter update 20261025`（含未跟踪文件，未自动恢复）。工作区外另有 `../ViT_hyperparameters_before_20261025.md` 原始记录。

| 参数 | 修改前 ViT / cross-vote 默认 |
| --- | --- |
| patch_size | 8 |
| dim | 256 |
| depth | 3 |
| heads | 2 |
| mlp_dim | 512 |
| dropout | 0.1 |
| optimizer | AdamW，betas=(0.9,0.999)，eps=1e-8 |
| learning_rate | 0.0003 |
| weight_decay | 0.05 |
| batch_size | 8 |
| epochs / patience | 40 / 12 |
| lr_schedule | warmup_cosine |
| warmup_epochs / min_learning_rate | 5 / 0.000005 |
| seed / gradient_clip_max_norm | 2023 / 1.0 |
| label_smoothing | 0 |
| checkpoint selection | validation loss 最低 |
| target_length / tukey_alpha | 512 / 0.3 |
| form / region_set / channel_mode | max1 / dyn_envelope / 3 |
| 标准二分类判定 | 默认 val 搜阈值，positive_reward=2；--no-reward 固定 0.5 |
| cross-vote 损失及判定 | vote_nll；概率平均后 argmax |
| 数据集 | CLI raw_data（1.0 mm）；GUI cls2_thr1.3（重划分） |

来源：修改前 `vit_model.py`、`run_vit_experiments.py`、`learning_rate_schedule.py`、`gui_app.py`。
备份中的探索选项默认值为 `branch_normalization=none`、`amplitude_mode=none`、`window_reference=locator`、`training_recipe=standard`。

## 更新后参数

来源：`ViT_reference/top15_hyperparameters_test_accuracy_20261024`。各方法按测试准确率降序、同分按 ID 升序取第一组，按本项目对应的输入与分类头选择，不额外搜索。

| 参数 | 普通拼接 ViT | 头/尾/全段拼接 ViT | ViT+cross-vote |
| --- | --- | --- | --- |
| preset | baseline_top1 | six_channel_top1 | six_channel_beam_vote_top1 |
| 参考文件 | 01_baseline_top15.json | 03_six_channel_top15.json | 05_six_channel_beam_vote_top15.json |
| 参考 ID | random_0047 | random_0048 | random_0086 |
| patch_size | 16 | 8 | 8 |
| dim | 64 | 256 | 256 |
| depth | 4 | 4 | 2 |
| heads | 8 | 4 | 4 |
| mlp_dim | 256 | 1024 | 1024 |
| dropout | 0.1 | 0.1 | 0.4 |
| learning_rate | 0.0003 | 0.001 | 0.0003 |
| weight_decay | 0.05 | 0.02 | 0 |
| batch_size | 4 | 24 | 4 |
| warmup_epochs | 5 | 5 | 5 |
| label_smoothing | 0 | 0 | 0 |

`--hyperparameter-preset auto` 为 CLI/GUI 默认：拼接且 `--region-set dyn_envelope_full` 选 `six_channel_top1`，其他拼接选 `baseline_top1`；cross-vote 在 `dyn_envelope_full` 选 `six_channel_beam_vote_top1`，在新增支持的 `dyn_envelope` 选 `beam_vote_top1`（来自 04_beam_vote_top15.json 的 random_0096：patch=8、dim=256、depth=4、heads=4、mlp_dim=512、lr=0.001、dropout=0.25、weight_decay=0.01、batch=4、warmup=5）。cross-vote 未传区域时 CLI 仍默认 full，GUI 可选择两种动态组合。结构对比需要显式传完整参数或 preset 以锁定超参。`--region-set all` 对拼接遍历原四个区域，不含 dyn_envelope_full，统一用 baseline_top1。

参数集中在 `vit_hyperparameters.py`；直接构造 `ViTConfig` 默认与 baseline_top1 一致。显式 CLI 参数优先，包括 0 值；`--hyperparameter-preset legacy` 恢复修改前参数。`config.json` 保存 preset、参考 ID 和实际模型/训练参数，实验目录与 summary 增加 `__hp_<preset>` 以区分历史结果。

```powershell
python run_vit_experiments.py --data-dir raw_data_relabeled/统一_cls2_thr1.3
python run_vit_experiments.py --data-dir raw_data_relabeled/统一_cls2_thr1.3 --region-set dyn_envelope_full
python run_vit_experiments.py --data-dir raw_data_relabeled/统一_cls2_thr1.3 --model-variant cross_vote --channel-mode 4
python run_vit_experiments.py --hyperparameter-preset legacy --batch-size 8
```

三次样条、主体窗选帧、val loss 早停、reward/投票规则和 `5e-6` 学习率下限继续沿用本项目；上限 40 轮、patience=12、seed=2023、AdamW、梯度裁剪 1 不变。迁移范围为参数与外层 train/val/test，不合并开发集做五折训练。参考排名已查看测试集，属于探索性选择，不能把参考准确率作为本项目实测性能。

## 统一数据集

GUI 名称：**统一 · 二分类 · 阈值 1.3 mm · cls2_thr1.3**。目录 `raw_data_relabeled/统一_cls2_thr1.3`，独立 tag `cls2_thr1.3_unified`。旧 `raw_data`、`cls2_thr1.3`、三分类数据全部保留。GUI 默认选统一数据集，CLI 原默认 `raw_data` 保留。

| split | 总数 | 0=薄骨（<1.3 mm） | 1=厚骨（≥1.3 mm） |
| --- | --- | --- | --- |
| train | 161 | 100 | 61 |
| val | 40 | 25 | 15 |
| test | 67 | 41 | 26 |

逐 ID、逐行沿用 `ViT_reference/dataset/{train,validation,test}/arrays/point_ids.npy`，validation 映射为 val；与 top15 五种方法的开发/测试名单核对一致。参考训练缓存标签为 0=厚、1=薄，reference/dataset 原始标签则与本项目同向；新数据按本项目 depth_value 贴标。

`dataset_protocols/cls2_thr1.3_unified.json` 固定 ID 次序、类别计数及参考文件 SHA-256。重建只依赖现有 raw_data 和仓库内清单：

```powershell
python create_unified_dataset.py
# 输出必须不存在；需要副本时指定新目录
python create_unified_dataset.py --output-dir raw_data_relabeled/统一_cls2_thr1.3_copy
```

构建器按 sample_id 匹配，不假定两套数据池行号一致；拒绝重复、遗漏、额外样本、类别计数不符、覆盖已有数据与源目录重叠。完整生成后才发布目标目录。X.npy 逐值复制本项目原始数据，读取时 ADC 归零后与参考 source_signal 逐值一致。

遵循 `.gitignore` 的现有策略：信号数组和生成产物留在本地，Git 保存构建脚本、固定清单、参数及文档，可据此重建。

## 本次验证

- `python -X utf8 -m unittest discover -s tests -v`：15 项通过，包括 split 匹配、阈值边界、拒绝覆盖、参数覆盖及原有通道/投票/学习率检查。
- 三套 preset 各在 CUDA 上运行 1 轮，每个 split 取前 8 点；训练、阈值/投票、权重、概率和指标保存均通过。CPU 重载权重后的验证概率与 CUDA 保存结果最大误差为 `3.04e-6`，在验证容差内；逐路概率平均/求和与 cross-vote 产物一致。
- 全部 268 点逐条核对 ID 归属与行顺序，三个 split 的标签与参考 dataset 一致；X 与本项目原信号逐值一致，ADC 归零后与参考完整信号逐值一致。
- 修改前已有的 50 个数据文件 SHA-256 均未变；GUI 实际启动后默认选择统一数据集，旧数据集仍可选择，三种模型训练命令均指向选中目录；EMD 入口可导入。
- Python 编译与 `git diff --check` 通过。检查日志、原文件哈希及短训练产物保存在忽略目录 `_dataset_hyperparameter_update_20261025/`。本次仅验证链路，未运行完整 40 轮性能评估。

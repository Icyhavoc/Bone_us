# `experiments/` 产物与当前代码的一致性

更新时间：2026-09-17

> **当前状态**：本文件中提到的全部旧实验已归档到
> `backup/5_experiments_2026-09-17/`（27 个实验目录 + 1 个 `summary.json`）。
> `experiments/` 目录现在只保留本说明，等待用当前代码重新生成结果。
> 归档保留的原因是下面列出的两类问题，**旧指标不要再引用**。

## 结论速览

| 实验目录（均已归档） | 数据源 | 状态 |
|---|---|---|
| `form_top3_mean__regions_dyn_envelope__channels_1` | `raw_data` | ✅ 已用移植后算法 + ADC 直流归零重跑 |
| `form_max1__regions_full__channels_1` | `raw_data` | ⚠️ 2026-08-12 生成，**早于 ADC 直流归零** |
| `form_max1__regions_bone_plus_post__channels_1` | `raw_data` | ⚠️ 2026-08-12 生成，**早于 ADC 直流归零** |
| `form_top3_mean__regions_full__channels_1` | `raw_data` | ⚠️ 2026-08-12 生成，**早于 ADC 直流归零** |
| `form_top3_mean__regions_bone_plus_post__channels_1` | `raw_data` | ⚠️ 2026-08-12 生成，**早于 ADC 直流归零** |
| `form_top3_mean__regions_bone__channels_1` | `raw_data` | ⚠️ 无 `metrics.json`，是一次未完成的运行 |
| `**/regions_after_split_pair__*`、`**/regions_main__*`、`**/regions_tail__*` | `after_split_data` | ❌ **不可复现的历史产物**（见下） |

## 为什么 `raw_data` 的实验会变

`emd_pipeline.load_split()` 现在统一调用 `replace_adc_dc_level()`：源 ADC 的 127 与 128
两个码值都表示零电平，归一化后分别是 $\pm 0.5/127.5 \approx \pm 0.003922$。
把这 $\pm 0.0039$ 抖动抹平后，**所有**帧处理方式的特征都会变化（不只 `dyn_envelope`）。
在 `test` 前 20 个样本上实测的特征变化量：

| form | region | 特征最大偏差 | 特征平均绝对偏差 | 选帧是否改变 |
|---|---|---:|---:|---|
| `mean_std` | `bone` | 1.06e-01 | 6.41e-03 | 否 |
| `mean_std` | `full` | 1.63e-01 | 1.55e-02 | 否 |
| `max1` | `bone` | 8.25e-02 | 1.02e-02 | 否 |
| `max1` | `full` | 1.99e-01 | 2.34e-02 | 否 |
| `top3_mean` | `bone` | 1.04e-01 | 1.00e-02 | 否 |
| `top3_mean` | `full` | 1.61e-01 | 1.88e-02 | 否 |
| `raw50` | `bone` | 7.59e-02 | 1.10e-02 | 否 |

选帧结果未改变，因此差异来自特征数值本身，而不是帧选择跳变。
影响量级为 $10^{-2}$，对指标的影响需重跑才能确定。

## 为什么 `after_split_data` 的实验不可复现

这批目录的 `config.json` 里有当前代码已不存在的字段：

- `region_name = after_split_pair`，但 `run_emd_experiments.REGION_PRESETS` 现在只有
  `full` / `bone` / `bone_plus_post` / `dyn_envelope`
- `selection_region = after_split_covered`，`dataset = after_split`、
  `combined_branches = [main, tail]` 均为旧设计
- `--data-dir after_split_data` 也无法运行：`load_split()` 按 `train` / `val` / `test`
  拼路径，而 `after_split_data/` 下的切分目录叫 `validation/`

因此这批产物只能作为历史记录保留，不代表当前算法的输出。

## 建议

1. 需要引用指标时，用当前代码重新运行 `run_emd_experiments.py`，不要沿用归档目录里的旧数字。
2. 归档位置：`backup/5_experiments_2026-09-17/`（其中 `README.md` 记录了归档原因与清单）。
   需要时可以直接把它们拷回来做对照，但不应当作当前算法的基线。
3. 重跑全矩阵基线的命令（4 种帧处理 × 4 种区域 × 双通道独立）：

   ```powershell
   D:\Miniconda3\python.exe run_emd_experiments.py --forms all --region-set all --channel-mode 3 --output-dir experiments
   ```

4. `form_top3_mean__regions_dyn_envelope__channels_1` 在算法移植前后的对比
   （同一配置、同一数据源，旧结果已归档）：

   | 版本 | val AUC | test AUC | test accuracy | test sensitivity |
   |---|---:|---:|---:|---:|
   | 移植前（`backup/4_stale_dyn_preport/`） | 0.7803 | 0.7016 | 0.6269 | 0.5556 |
   | 移植后（`backup/5_experiments_2026-09-17/`） | 0.8813 | 0.8297 | 0.7313 | 0.8056 |

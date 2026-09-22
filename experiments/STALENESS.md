# `experiments/` 产物与当前代码的一致性

更新时间：2026-09-17（最近补充：2026-09-22，S1/S3a/S3b/S4 落地后的重放结论）

> **当前状态**：本文件中提到的全部旧实验已归档到
> `bin/backup/5_experiments_2026-09-17/`（27 个实验目录 + 1 个 `summary.json`）。
> 归档保留的原因是下面列出的两类问题，**旧指标不要再引用**。

## 现有目录（全部由当前代码重新生成）

| 目录 | 说明 |
|---|---|
| `form_top3_mean__regions_dyn_envelope__channels_1` | 移植后算法 + ADC 直流归零 |
| `form_top3_mean__regions_dyn_envelope__channels_2` | 移植后算法 + ADC 直流归零 |
| `form_raw50__regions_dyn_envelope__channels_1` | 移植后算法 + ADC 直流归零 |
| `form_max1__regions_full__channels_1` | 移植后算法 + ADC 直流归零 |

> ⚠️ **不含任何 `channels_3` 结果。** 通道 3 的语义在双塔 MLP 改写后已变化：
> 默认 `--channel-aggregation per_channel` 让两个通道各成一组特征、各有一座 MLP 塔，
> 特征宽度从 8 变成 16，网络拓扑也随之改变。改写前生成的通道 3 指标对应的是
> `pooled`（通道平均）行为，不能与当前默认结果混用。需要通道 3 基线时请重跑：

```powershell
D:\Miniconda3\python.exe run_emd_experiments.py --forms all --region-set all --channel-mode 3
```

> 目录命名：默认 `per_channel` 沿用 `form_<form>__regions_<region>__channels_<channel>`；
> `--channel-aggregation pooled` 会写成同名 + `__pooled` 后缀，两者不会互相覆盖。

## 结论速览（已归档的旧产物）

| 实验目录（均已归档） | 数据源 | 状态 |
|---|---|---|
| `form_top3_mean__regions_dyn_envelope__channels_1` | `raw_data` | ✅ 已用移植后算法 + ADC 直流归零重跑 |
| `form_max1__regions_full__channels_1` | `raw_data` | ⚠️ 2026-08-12 生成，**早于 ADC 直流归零** |
| `form_max1__regions_bone_plus_post__channels_1` | `raw_data` | ⚠️ 2026-08-12 生成，**早于 ADC 直流归零** |
| `form_top3_mean__regions_full__channels_1` | `raw_data` | ⚠️ 2026-08-12 生成，**早于 ADC 直流归零** |
| `form_top3_mean__regions_bone_plus_post__channels_1` | `raw_data` | ⚠️ 2026-08-12 生成，**早于 ADC 直流归零** |
| `form_top3_mean__regions_bone__channels_1` | `raw_data` | ⚠️ 无 `metrics.json`，是一次未完成的运行 |
| `**/regions_after_split_pair__*`、`**/regions_main__*`、`**/regions_tail__*` | `bin/after_split_data` | ❌ **不可复现的历史产物**（见下） |
| 全部 `*__channels_3` | `raw_data` | ❌ 改写前生成，对应 `pooled` 行为，当前默认已改为 `per_channel` |


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

## 为什么基于 `bin/after_split_data` 的实验不可复现

这批目录的 `config.json` 里有当前代码已不存在的字段：

- `region_name = after_split_pair`，但 `run_emd_experiments.REGION_PRESETS` 现在只有
  `full` / `bone` / `bone_plus_post` / `dyn_envelope`
- `selection_region = after_split_covered`，`dataset = after_split`、
  `combined_branches = [main, tail]` 均为旧设计
- `--data-dir after_split_data` 也无法运行：`load_split()` 按 `train` / `val` / `test`
  拼路径，而 `bin/after_split_data/` 下的切分目录叫 `validation/`

> 该数据集本身已于 2026-09-21 随 `reference_code/` 一起归档进 `bin/`；
> 它不再参与训练，**仅**被 `verify_dyn_envelope.py` 当作算法参照读取。

因此这批产物只能作为历史记录保留，不代表当前算法的输出。

## 建议

1. 需要引用指标时，用当前代码重新运行 `run_emd_experiments.py`，不要沿用归档目录里的旧数字。
2. 归档位置：`bin/backup/5_experiments_2026-09-17/`（其中 `README.md` 记录了归档原因与清单）。
   需要时可以直接把它们拷回来做对照，但不应当作当前算法的基线。
3. 重跑全矩阵基线的命令（4 种帧处理 × 4 种区域 × 双通道独立）：

   ```powershell
   D:\Miniconda3\python.exe run_emd_experiments.py --forms all --region-set all --channel-mode 3 --output-dir experiments
   ```

4. `form_top3_mean__regions_dyn_envelope__channels_1` 在算法移植前后的对比
   （同一配置、同一数据源，旧结果已归档）：

   | 版本 | val AUC | test AUC | test accuracy | test sensitivity |
   |---|---:|---:|---:|---:|
   | 移植前（`bin/backup/4_stale_dyn_preport/`） | 0.7803 | 0.7016 | 0.6269 | 0.5556 |
   | 移植后（`bin/backup/5_experiments_2026-09-17/`） | 0.8813 | 0.8297 | 0.7313 | 0.8056 |

## S1 / S3a / S3b / S4 落地后的重放结论（2026-09-22）

本轮新增了配对评价协议（`compare_label_schemes.py`）与三个可选特征开关
（`--feature-set shape/spectral_ext/envelope/...`、`--branch-ratio`、`--channel-contrast`）。
它们的**默认值就等于历史行为**，因此上表四个目录仍然代表当前代码的输出：

- 三个开关默认分别是 `compact`、`none`、`none`，不会改变任何已有列或网络拓扑；
- 用 `experiments/form_top3_mean__regions_dyn_envelope__channels_1` 自己的 `config.json`
  重放，`features_*.npy` 与原产物的 **16 列中有 15 列逐位相同**；
- 唯一差异是 `dynamic_tail_window.spectral_centroid`，源自 S3a 去掉了谱质心功率和里的
  一个 `+1e-12`，实测偏移为**相对 2.51e-07 = 2.1 个 float32 ulp**（纯舍入）。
  该列已成为 `verify_branch_ratio.py` / `verify_channel_contrast.py` 里唯一被容忍的例外，
  上限 8 ulp（超一个数量级即判为真实改动）。

> S3b / S4 的 A/B 特征提取与配对报告都写在 `bin/_ab13_*`、`bin/_s3a_rich`、`bin/_stack_dim`，
> 故意**不**落进 `experiments/`，以免与正式基线混在一起。

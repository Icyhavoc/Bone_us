# `experiments/` 产物与当前代码的一致性

更新时间：2026-09-17（最近补充：2026-09-22，S1/S3a/S3b/S4 落地后的重放结论；
同日补：数据集 tag 冲突的修法与 `bin/_ab13_keep` 重新对齐；
2026-09-23 补：严格 LOSO 测评、标本级划分与围栏对照落地后的引用规则）

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

## 旧产物缺少围栏与阈值口径（2026-09-23）

`experiments/` 下现存目录都生成于围栏对照与 `--threshold-source` 落地**之前**，
所以它们的 `metrics.json` **没有 `controls` 和 `decision_threshold` 两块**。
这不影响其中任何一个已存数字：`metrics.metrics[split]` 仍然按 0.5 计算，
新代码在这块上**行为未变**（`--threshold-source` 只往 `decision_threshold` 里写）。

需要引用时必须区分：

- 要「本模型 vs 多数类 / 标本指纹」这条围栏，或要看三种阈值口径的敏感度
  → 用 `run_emd_experiments.py` 重跑对应命令（§11.8），旧目录里的数字无法补出对照线；
- 要**绝对可分性**结论 → `raw_data` 的划分是按**点**分层的
  （train ∩ test 共享 43 块骨头），这些目录的 test 指标含乐观偏差，
  应改用 `run_loso_evaluation.py`（§11.7）；
- 要「A 特征是否优于 B」→ 用 `compare_label_schemes.py` 的配对报告（§11.1），
  与目录里的单次 test 指标无关。

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

## 数据集 tag 冲突与覆盖保护（2026-09-22 补）

`label_scheme.json` 的 `tag` 只由**类别数**与**阈值**决定，不反映划分方式，因此
`raw_data_relabeled/cls2_thr1.3`（`stratified_random_split`，seed 42）与
`raw_data_relabeled/cls2_thr1.3_keepsplit`（`keep_source_split`，沿用 `raw_data` 划分）
拿到的是**同一个 tag** `cls2_thr1.3`。两者阈值相同 → 标签相同，268 样本池相同 →
`X` 逐位相同，但 train/val/test 归属不同（重叠 train 92/161、val 4/40、test 14/67，
即 **268 个样本里有 158 个换了 split**）。

而实验目录名是 `form_...__regions_...__channels_...__<tag>`，两轮实验会落到**同一个目录**：
`run_emd_experiments.py` 原先是 `mkdir(exist_ok=True)` 后直接覆写，后跑的赢，
`config.json` 的 `data_dir` 只指向最后那一次；GUI 的 `visualizations/<kind>/<tag>/` 同样冲突。

同一命令只换 `--data-dir` 的实测差异（`top3_mean` + `dyn_envelope` + 通道 1）：

| 数据源 | test accuracy |
|---|---:|
| `cls2_thr1.3`（stratified） | 0.8358 |
| `cls2_thr1.3_keepsplit` | 0.7164 |

> 这一步也解释了之前「两个 1.3 mm 数据集跑出来结果一样」的现象：
> `compare_label_schemes.py` 的配对协议把 268 个样本合并后按 `depth_value` 分位数**重新**折分、
> 根本不读已存 split，所以对它而言两份数据集逐位等价（报告 SHA256 相同）；
> 而真正使用已存 split 的 `run_emd_experiments.py` 得到的是上表那两个不同的数。

已实施的修法分两层：

1. **结构性**：`raw_data/relabel_by_thickness.py` 新增 `dataset_tag(scheme, method)`，
   在 `--keep-split` 时把 tag 追加成 `cls2_thr1.3_keepsplit`，目录名天然不同。
   `label_scheme.json` / `split_metadata.json` / `README.txt` 都记录这个 tag，
   下游的目录名、文件名、GUI 分桶全自动跟随。
2. **兜底**：`run_emd_experiments._guard_existing_experiment()` 在写入任何文件**之前**
   读目标目录的 `config.json`，若 `data_dir` / `regions` / `num_classes` / `label_scheme`
   与本次请求不一致就 `SystemExit` 并打印 recorded vs requested；非致命项（如 `epochs`）
   只打印 `[guard] …这些设置已变化`。确需覆盖时显式加 `--allow-config-mismatch`。
   被拦下时**一个文件都不写**。

已核实的安全性：用新代码重建 `cls2_thr1.3` 与 `cls2_thr1.3_keepsplit`，与磁盘上的旧版本比对
**12 个 payload 文件（train/val/test × `X.npy`/`y.npy`/`indices.npy`/`samples.json`）SHA256 全部相同**，
所以改 tag 不会破坏「`--thresholds 1.0 --keep-split` 逐位复现 `raw_data`」这一性质。

### `bin/_ab13_keep` 的数据源对齐

A/B 三件套里 `bin/_ab13_keep` 原先指向 `cls2_thr1.3_keepsplit`，而 `bin/_ab13_ratio` /
`bin/_ab13_atten` 指向 `cls2_thr1.3`（三者目录名后缀当时都是 `__cls2_thr1.3`）。
现已把 `_ab13_keep` 用 `--data-dir raw_data_relabeled/cls2_thr1.3 --allow-config-mismatch`
重跑（新守护确实拦下了第一次尝试，拦截信息与 §2.3 描述一致），三件套现在同源。

因为 §11.1 的 `bin/_s1_thresholds` 用的是**不读已存 split** 的配对协议，
重跑后 `comparison_report.txt` 与 `comparison_results.json` 与归档件 **SHA256 完全相同**
（`DD67C66C…` / `AC79F184…`），§11.1 的所有数字不受影响。新增 `test_accuracy` 为 0.8358。

> 提醒：`bin/_ab13_*` / `bin/_s1_thresholds` 里的数字都由当时的数据集与代码产生，
> 目录名里的 `__cls2_thr1.3` 是 **tag**，不代表目录内 `config.json` 的 `data_dir`。
> 引用前先看 `config.json`。

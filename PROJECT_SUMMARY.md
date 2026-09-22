# 当前项目工作总结

本文档用于发布到 Git 时说明当前版本已经完成的工作、实验约定、运行方式和已知注意事项。

## 1. 项目目标

本项目针对 BMU 骨分层数据，完成以下实验流程：

```text
raw_data
  -> 帧预处理
  -> 深度区域 branch 划分
  -> branch 重采样到 512 点
  -> EMD 分解
  -> EMD 特征提取
  -> NumPy MLP 分类（默认二分类，支持任意 k 类）
```

当前代码位于 `经验小波分解/`，与 `raw_data/` 并列。

## 2. 数据集约定

- 每个样本形状：`[50, 2, 896]`。
- 50：帧数。
- 2：两个物理信号通道。
- 896：每帧采样点数。
- 默认 896 个采样点对应 0–5 mm。
- 标签约定：`label=0` 为即将穿透，`label=1` 为安全（这是 `raw_data/` 的**默认**阈值，
  其他阈值方案见 §2.2）。
- 当前实验不考虑同一骨头不同采样点之间的相关性和数据泄露问题。
- 数据划分由 `raw_data/train`、`raw_data/val`、`raw_data/test` 提供。

### 2.1 直流电平归零（`--data-dir raw_data` 的硬性前提）

源 ADC 同时用码值 127 和 128 表示“零”，两者归一化后分别是
$\pm 0.5/127.5 \approx \pm 0.003922$，即同一个物理电平被拆到了 127.5 中点的两侧。
``raw_data/`` 保留了这一 $\pm 0.0039$ 抖动，而参考数据集（现位于 `bin/after_split_data/`）
是从抖动已被抹平的信号生成的；
这点量化噪声足以让 `dyn_envelope` 的过阈点偏移几十个采样点
（实测 `N35_P2_01` 通道 1 会从 324 漂到 267，从而触发人工修正守卫报错）。

因此 `emd_pipeline.load_split()` 现在统一调用 `replace_adc_dc_level()`
（`ADC_DC_MAGNITUDE = 0.5 / 127.5`，`ADC_DC_ATOL = 1e-6`）把这一对码值塌陷到 0.0。
由于入口唯一，训练与四个可视化脚本都自动获得同一份直流归零后的帧；
对本身已归零的 `bin/after_split_data/` 该操作是幂等的。实验目录的 `config.json`
会用 `adc_dc_replacement` 字段记录该常数。

深度映射由 `DepthMapper` 封装，默认规则为：

```text
index = round(depth_mm / 5.0 * 896)
```

例如 1 mm 对应约 179 号采样点。后续如需改变映射，只需修改 `DepthMapper` 接口。

### 2.2 骨头厚度阈值与通用 k 分类

`raw_data/` 的标签是 `depth_value >= 1.0` 的结果，但骨层厚度本身是连续量，阈值应当是可调的。
`raw_data/relabel_by_thickness.py` 负责按新阈值重建数据集：

```powershell
python raw_data/relabel_by_thickness.py --thresholds 1.3
python raw_data/relabel_by_thickness.py --thresholds 1.3 --output-dir raw_data_relabeled/cls2_thr1.3
python raw_data/relabel_by_thickness.py --thresholds 0.8,1.2 --output-dir raw_data_relabeled/cls3_thr0.8_1.2
```

- 阈值语义是 `numpy.digitize(..., right=False)` 的**闭区间下界**：`--thresholds 1.0` 等价于
  `depth >= 1.0 → 1`，与现有 `y.npy` 的构造规则**完全一致**；k 个阈值 → k+1 个类别。
- `raw_data/{train,val,test}` 只是一份 268 样本池的划分（`indices.npy` 记录池内行号，三个划分的
  下标集合恰好铺满 `range(268)`），所以脚本能从现有划分反推出整个池，再重新分层划分；
  `--keep-split` 则只重贴标签、沿用原划分。上游原始数据集已不在磁盘上，这是唯一可行的重建方式。
- `_check_output_dir` 会在**做任何事之前**拒绝写入 `raw_data/` 或任何与源目录重叠的路径
  （`--overwrite` 也不能绕过），避免又一次性事故。
- 输出目录多一个 `label_scheme.json`（`thresholds_mm` / `num_classes` / `class_names` / `tag` /
  `label_rule`），`tag` 形如 `cls2_thr1.3`、`cls3_thr0.8-1.2`。

下游全链路自动跟随数据集切换类别数：

- `run_emd_experiments.py --data-dir <新目录>`：类别数默认从标签推断，`--num-classes` 可覆盖
  （给得比数据里的小会直接报错）；`label_scheme` 与 `num_classes` 写入 `config.json`/`metrics.json`，
  目录名追加 `__<tag>`，因此三分类结果不会覆盖已有的二分类目录。
- `emd_pipeline.classification_metrics(y, p, num_classes)`：两类时逐字复现原有二分类指标（含
  `TN`/`FP`/`FN`/`TP` 键）；多于两类时改用 argmax，输出逐类 `precision/recall/f1/specificity`
  与一站式 AUC、`k×k` 混淆矩阵（行 = 真实，列 = 预测）、宏平均和 `balanced_accuracy`。
  宏平均同时挂回历史键名，所以 `history.json`、训练曲线与 `val_auc` 早停逻辑无需改动。
- `run_baseline_models.py`：类别数从 `label_scheme`（先读实验目录、再读 `data_dir`）或标签推断，
  多于两类时把每个基模型包成 `shallow_models.OneVsRestClassifier`（`prior` 除外）。
- 四个可视化脚本：`--class-labels` 选类（多类时文件名改用 `class_label{k}`），错分厚度图的参考线
  优先读 `label_scheme`（多阈逐线），混淆矩阵按 `metrics.json` 的类别数自动切换 k×k 画法。

产物约定：二分类（默认 `raw_data/`，无 `label_scheme.json`）的所有历史行为与图纸、指标**逐位不变**。

## 3. 帧预处理方式

当前实现了 PDF 中要求的四种帧处理方式：

| 名称 | 处理方式 |
|---|---|
| `mean_std` | 对 50 帧逐点计算均值和标准差，得到两条信号流 |
| `raw50` | 保留全部 50 帧 |
| `max1` | 按指定区域的综合 RMS 选择能量最大的一帧 |
| `top3_mean` | 按综合 RMS 选择能量最高的 3 帧并求均值 |

`max1` 和 `top3_mean` 默认在完整 0–5 mm 区间选帧，选择区域可由 `--selection-region-json` 修改。

## 4. 通道模式

`--channel-mode` 支持：

- `1`：只使用物理通道 1；
- `2`：只使用物理通道 2；
- `3`：同时保留通道 1 和 2。

通道维度和深度 branch 是两个**相互独立**的维度，物理通道不会被误当成深度 branch。

### 4.1 通道 3 的处理流程（`per_channel`，默认）

两个通道**各自独立**切片、重采样、做 EMD，各得一组特征；两组特征**拼接**后进入
MLP 分类头，并且**每组特征各有一座独立的塔**，塔的输出再拼接起来交给共享头。
分组方式由 `EMDConfig.channel_aggregation` 控制，CLI 对应 `--channel-aggregation`：

| 取值 | 含义 |
|---|---|
| `per_channel`（默认） | 每个选中通道形成一组特征，组间拼接，每组一座 MLP 塔 |
| `pooled` | 通道轴与分量轴一起平均成一组，复现历史行为（见 §4.3） |

以某个 branch 为例：

```text
channel_mode=1 / 2                     channel_mode=3
──────────────────                     ──────────────
[50,1,896] 选中 1 个物理通道           [50,2,896] 选中通道 1+2
     ↓ 帧处理                               ↓ 帧处理
[1,1,896]                              [1,2,896]
     ↓ 切片 + 重采样 + Tukey                ↓ 切片 + 重采样 + Tukey
[1,512]  1 条流                        [2,512]  2 条流
     ↓ EMD 跑 1 次                          ↓ EMD 跑 2 次（每通道各 1 次）
 8 维特征（1 组）                        8 + 8 = 16 维特征（2 组）
     ↓                                    ↓
 1 座塔 8→64                            2 座塔 8→64 → 拼接成 128
     ↓                                    ↓
 共享头 64→32→2                          共享头 128→32→2
     ↓                                    ↓
 2 类输出                                2 类输出
```

关键点：

1. **两个通道各自独立做预处理和 EMD**，不共享也不串扰；
2. 合并发生在**分类头之前**，是**拼接**而不是平均：
   通道 3 的每个 branch 宽 **16 维**（`8 × 2`），通道 1 / 2 仍是 8 维；
3. 特征按 **group-major** 排列，即 `[组0 的所有 branch | 组1 的所有 branch]`，
   这样分类头可以按 `feature_info.json` 里的 `feature_group_dims` 直接切片，
   把每组特征喂给对应的塔。
4. 单通道时只有一组，`per_channel` 与 `pooled` **完全等价**，
   已做逐位回归验证（见 §15）。

### 4.2 `per_channel` 与「信号相加」的区别

通道 3 的两组特征来自**两次独立 EMD**，不是把两个通道加起来的 EMD。
EMD 是**非线性**分解，"先相加再分解"与"先分别分解"结果完全不同：

| 做法 | 与通道 3 实际结果的关系 |
|---|---|
| 两个通道分别分解、各自成组 | ✅ 就是现在的做法 |
| 信号级相加后再分解 $f(x_1+x_2)$ | ❌ 与分别分解的结果差异达 $10^{-2}$ 量级 |

### 4.3 历史行为：`--channel-aggregation pooled`

历史版本的通道 3 会把通道轴和分量轴**一起求平均**，得到单组 8 维特征：

```text
[2,6,8]  2 通道 × 6 分量
     ↓ np.mean(component_matrix, axis=(0, 1))
  8 维特征  ← 维度不翻倍，两个通道的个性被平均掉
```

实测同一帧、同一 branch、同一选帧下，该平均关系满足（误差 `7.5e-9`，float32 精度下限）：

$$f_{\text{ch3}}^{\text{pooled}} = \tfrac{1}{2}\left(f_{\text{ch1}} + f_{\text{ch2}}\right)$$

| 做法 | 与 `pooled` 通道 3 实际结果的偏差 |
|---|---:|
| 特征级平均 $\tfrac{1}{2}(f_1+f_2)$ | $7.5\times10^{-9}$ ✅ 就是它 |
| 信号级相加后再分解 $f(x_1+x_2)$ | $1.7\times10^{-2}$ ❌ 差 6 个数量级 |

> 现默认已改为 `per_channel`，`pooled` 仅用于复现历史结果与消融对照。

### 4.4 选帧会随通道模式变化（重要）

`max1` / `top3_mean` 靠 RMS 排名选帧，而 `_rms_scores()` 对**通道轴和采样轴一起**求均方根：

```python
np.sqrt(np.mean(np.square(segment, dtype=np.float64), axis=(1, 2)))
```

因此同一帧在不同通道集下得分不同，排名可能不同。实测同一样本：

| `--channel-mode` | `selected_frames` |
|---|---|
| 1 | `[13 11 12]` |
| 2 | `[28 27 26]` |
| 3 | `[26 27 24]` |

结果是：**通道 1 单独运行时所选的那一帧，与通道 3 里第一个通道所用的帧并不是同一帧**。
所以 §4.3 的"严格平均"只在**帧选择相同时**成立：

| form | 是否选帧 | $\max\left|f_{\text{ch3}}^{\text{pooled}} - \tfrac{1}{2}(f_{\text{ch1}}+f_{\text{ch2}})\right|$ |
|---|---|---|
| `mean_std` | 否 | $1.5\times10^{-8}$ → 严格平均 |
| `raw50` | 否 | $7.5\times10^{-8}$ → 严格平均 |
| `max1` | 是 | $1.2\times10^{-2}$ → 偏离 |
| `top3_mean` | 是 | $6.6\times10^{-3}$ → 偏离 |

即：不选帧的 `mean_std` / `raw50` 严格满足平均关系；`max1` / `top3_mean` 因选帧差异而偏离。

> 因此横向比较 `channels_1` 与 `channels_3` 的实验时，差异既包含"多了通道 2"的贡献，
> 也包含"选帧变了"的贡献，两者无法从结果中分离。需要干净的消融对照时，
> 应固定 `--selection-region-json`，或改用不使用选帧的 `mean_std` / `raw50`。

## 5. 区域组合

固定区域组合：

- `full`：`[[0, 5]]`；
- `bone`：`[[1, 3]]`；
- `bone_plus_post`：`[[0.2, 1.5], [1.5, 5.0]]`；
- `custom`：通过 `--regions-json` 自定义，可使用重叠、嵌套或不连续区域。

新增动态区域组合：

- `dyn_envelope`：逐通道定位被检测信号的起振波包，并据此划分两个 branch。

固定区域不要求覆盖完整 0–5 mm，也不要求 branch 等长。所有 branch 都分别处理，最后再拼接特征。

## 6. 动态包络区域 `dyn_envelope`

`dyn_envelope` 是 `bin/reference_code/pipeline.py` 中 `segment()` 的**完整移植**，与
`bin/after_split_data/` 的产物逐位一致（见 `verify_dyn_envelope.py`）。它不改变信号，只用包络结果确定 branch 边界。

> ⚠️ 本区域对输入的直流电平敏感：必须先执行 §2.1 的 ADC 127/128 归零，
> 否则 $\pm 0.0039$ 的量化抖动会把越阈点推偏几十个采样点，并触发人工修正守卫
> `recorded manual correction no longer applies`。`load_split()` 已统一处理。

### 定位信号

定位信号与帧处理方式**无关**：每个通道各自按 `max(abs(x))` 选出幅值最大的 3 帧（并列取较早帧），在 `float64` 下求平均后回写 `float32`，再取 Hilbert 包络。因此切换 `mean_std` / `raw50` / `max1` / `top3_mean` 不会移动分割边界。

### 检测流程

1. 包络前 71 点置零，其余用 `mode='nearest'` 的长度 5 移动平均平滑。
2. 参考点：第 71 点起首个严格 `> 0.015` 的采样点；单通道缺失时借用另一通道，全部缺失时改用 RNorm 融合包络（0.5×最大值）回退。
3. 局部峰：`[参考点-50, 参考点+150)` 窗口内包络的最大值。
4. 比例阈值：`0.05 × 局部峰值`。
5. 前置波包合并：以含该峰的高于阈值连续段为主波包，向前合并满足以下全部条件的段：间隔 ≤ 8、宽度 ≥ 8、面积 ≥ 主波包面积的 5%、相对主波包起点的累计提前量 ≤ 96。
6. 阈值点 `t` = 合并后的最早越阈点。
7. 主窗：`s = max(t - 20, 70)`，`[s, s + 170)`，**不补零、不平移**（越界直接报错）。
8. 尾部：固定 `[251, 896)`，与主窗**允许重叠或留间隔**。

### 逐通道边界

两个通道**各自独立**定位，因此同一采样点的主窗起点、与尾部的重叠长度都可能不同。分支在切片后各自重采样到 512 点并加 Tukey 窗，再按 `stream × channel` 展平。

### 默认参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `locator_top_k` | `3` | 每通道用于定位的帧数 |
| `noise_start` | `71` | 包络噪声头长度 |
| `weak_threshold` | `0.015` | 参考点越阈水平 |
| `smooth_window` | `5` | 包络平滑窗口 |
| `peak_window_back` / `peak_window_forward` | `50` / `150` | 局部峰搜索窗 |
| `peak_ratio` | `0.05` | 比例阈值系数 |
| `gap_max` / `min_run_width` | `8` / `8` | 波包合并间隔与最小宽度 |
| `min_run_area_ratio` | `0.05` | 合并最小面积比 |
| `merge_max_lead` | `96` | 最大累计提前量 |
| `lead_back` | `20` | 由阈值点回退到主窗起点 |
| `main_start_min` | `70` | 主窗起点下限 |
| `main_length` | `170` | 主窗固定长度 |
| `tail_start` | `251` | 尾部固定起点 |
| `apply_manual_corrections` | `True` | 是否套用参考实现的人工复核修正表 |

每个样本逐通道的阈值点、局部峰、参考点、回退规则、主窗/尾部边界、重叠与间隔长度都会写入实验目录的 `feature_info.json`；参数写入 `config.json`。

调用：

```powershell
python run_emd_experiments.py `
  --forms top3_mean `
  --region-set dyn_envelope `
  --channel-mode 3
```

`--dyn-*` 参数由 `dyn_cli.py` 统一定义，默认值全部取自 `DynamicEnvelopeConfig`；GUI 训练与四个可视化脚本共用同一套默认值，不再各自硬编码。

一致性校验：

```powershell
python verify_dyn_envelope.py
```

该脚本对 268 个点位 × 2 通道同时校验「边界逐位相同」与「分支内容是无补零的纯切片」。

## 7. 重采样、Tukey 窗和可视化顺序

实际训练和 EMD 路径：

```text
branch 切片
  -> 线性重采样到 512 点
  -> Tukey 窗
  -> EMD
```

当前默认 Tukey 参数为 `alpha=0.3`，可通过 `--tukey-alpha` 修改。

预处理可视化路径特意不加 Tukey 窗：

```text
branch 切片
  -> 线性重采样到 512 点
  -> 绘图
```

这样可视化图展示的是预处理/重采样后的原始信号形态，而不会把窗函数边缘衰减误认为原始信号特征。EMD 分解可视化仍展示实际进入 EMD 的加窗信号。

## 8. EMD 和特征维度

EMD 使用当前 `emd_pipeline.py` 中的 NumPy 实现，默认最多提取 5 个 IMF，并保留 residue。

### 8.1 每个 EMD 分量的统计特征

候选统计量共 8 个，实际启用哪些由 `EMDConfig.feature_names`（CLI：`--feature-set`）决定：

| 名称 | 含义 |
|---|---|
| `mean` | 分量均值 |
| `std` | 分量标准差 |
| `rms` | 均方根 |
| `energy` | 能量 |
| `abs_mean` | 绝对均值 |
| `peak_abs` | 绝对峰值 |
| `zero_crossing_rate` | 过零率 |
| `spectral_centroid` | 谱质心 |

三个预设（`EMD_FEATURE_PRESETS`）：

| `--feature-set` | 包含 | 说明 |
|---|---|---|
| `legacy` | 全部 8 个 | 历史行为，用于复现旧结果 |
| `compact`（默认） | 7 个，**去掉 `mean`** | 见 §8.3 |
| `lean` | 6 个，再去掉 `energy` | 备选 |

`spectral_centroid` 使用**物理单位**：`sample_spacing_mm` 由该分支的物理深度跨度除以重采样点数
（`target_length`）得到，谱质心因此以 **cycles/mm** 表示，主窗口与尾窗口可以跨分支比较
（两者重采样到同样点数，但物理跨度差约 3.8 倍，见 §6）。

默认 `stream_aggregation=pooled`：

- 同一 branch 内，对 EMD 分量和信号流做均值池化；
- 每个 branch 的每组特征因此只占 `len(feature_names)` 维；
- 多个 branch 之间不求均值，直接拼接。

通道方向由 `channel_aggregation` 决定：

- `per_channel`（默认）：每个通道各一组，组间拼接；
- `pooled`：通道轴并入流轴一起池化，只剩一组。

### 8.2 定位特征（`--locator-features`）

`dyn_envelope` 的主窗口是**从信号里切出来再重采样**的：窗口内的所有回波都被重新对齐到窗口起点，
**绝对到达深度被丢弃了**。branch 内任何统计量（std/rms/energy…）都无法把这一信息找回来。
因此额外提供一组**定位特征**，直接描述检测器把主窗口放在了深度轴的哪个位置
（同样属于 `LOCATOR_BRANCH_NAME = "dynamic_main_window"` 这一个 branch）：

| `--locator-features` | 包含 | 说明 |
|---|---|---|
| `core`（默认） | `onset_mm`、`peak_mm` | 合并后的过阈起点深度、包络峰深度 |
| `full` | `core` 再加 `weak_onset_mm`、`rise_mm`、`merge_extension_mm`、`peak_amplitude` | 实测**更差**：多出的列与 `onset_mm` 高度共线，主要贡献方差 |
| `none` | 空 | 复现 change #1 之前的行为 |

### 8.3 维度表（默认 `--feature-set compact --locator-features core`）

每组宽度 = `len(feature_names) × branch 数`，`dyn_envelope` 的主 branch 再追加定位特征：

| 区域组合 | `--channel-mode 1` / `2` | `--channel-mode 3`（`per_channel`） | `--channel-mode 3`（`pooled`） |
|---|---:|---:|---:|
| `full` / `bone` | 7 | 14 | 7 |
| `bone_plus_post` | 14 | 28 | 14 |
| `dyn_envelope` | 16 | **32**（实测 `feature_dim=32`） | 16 |

`--feature-set legacy --locator-features none` 可还原历史宽度（`full`/`bone` 8 维、
`bone_plus_post` 16 维、`dyn_envelope` 16 维 / 32 维）。

### 8.4 `compact` + `core` 的实测效果（change #1 / #2）

同一划分、5 个训练种子（42–46）、`form_top3_mean__regions_dyn_envelope__channels_1`：

| 配置 | test accuracy | test AUC |
|---|---:|---:|
| `legacy` + `none`（历史） | 0.6896 ± 0.0324 | 0.7785 ± 0.0387 |
| **`compact` + `core`（默认）** | **0.7373 ± 0.0133** | **0.8206 ± 0.0090** |

即 **+4.8 pt 准确率，同时方差减半**。去掉 `mean` 的依据：EMD 的筛分过程本身把每个 IMF 的均值
压向 0，该列在本数据上几乎只剩 float32 舍入误差（branch 1 实测 `std = 3.6e-5`，
而 `peak_abs = 8.4e-2`）——标准化后看起来无害，但它对第一层全连接始终是一个**死输入**。

维度进一步压缩还有收益：把 16 列用前向选择砍到 5 列后 test accuracy 反而**升高**
（0.7761 → 0.8060），说明**有效维度远小于 16**。相关列分组（`|r| ≥ 0.98`）显示
`{std, rms, abs_mean, peak_abs}` 在**每个 branch 内**彼此高度共线（8 列约等于 2 个独立方向）。

`flatten` 和 `stats` 模式仍保留用于对照实验。

## 9. MLP 分类头

当前模型为 NumPy 实现的轻量 MLP，不使用 CNN。

**单组特征**（`--channel-mode 1` / `2`，或 `--channel-mode 3 --channel-aggregation pooled`）
时，网络就是历史结构：

```text
Linear(input_dim, 64)
  -> ReLU + Dropout(0.1)
  -> Linear(64, 32)
  -> ReLU + Dropout(0.1)
  -> Linear(32, 2)
```

**多组特征**（`--channel-mode 3`，默认 `per_channel`）时，改为「每组建一座塔 + 共享头」：

```text
组0 (8 维) -> Linear(8, 64)  -> ReLU + Dropout(0.1) -┐
组1 (8 维) -> Linear(8, 64)  -> ReLU + Dropout(0.1) -┤ concat -> 128 维
                                                     ↓
                                        Linear(128, 32) -> ReLU + Dropout(0.1)
                                                     ↓
                                                  Linear(32, 2)
```

说明：

- 塔的宽度取 `hidden_dims[0]`，共享头取 `hidden_dims[1:]` 再接 2 类输出（k 分类时最后一层是 `Linear(32, k)`）；
- 因此 `--channel-mode 3` 的双通道不再在 MLP 之前被平均掉，两个通道的特征
  在分类头里**各自经过一层非线性**后才融合；
- 只有一组时塔会塌缩成第一层全连接，网络与历史结构**完全等价**
  （已用逐位回归验证，见 §15）；
- `mlp_model.npz` 会额外记录 `group_dims` 与 `tower_count`，`feature_info.json`
  会记录 `feature_groups` 与 `feature_group_dims`。

当前使用 k 个输出节点的 softmax 交叉熵（k 默认 2，见 §2.2）。对于二分类，它与单输出 sigmoid BCE 在数学上等价。

训练规则：

- 标准化参数只在训练集上拟合；
- validation loss/AUC 用于早停和最佳 epoch 选择；
- test 集不参与模型选择；
- 每个 epoch 的 test loss 仅用于曲线观察；
- 默认最多训练 200 个 epoch，默认 patience 为 40。

## 10. 线性/浅层基线对照（change #3）

### 10.1 目的

原现象的观测是 **train / test 错误率都在 25% 左右**——既不是"train 很低、test 很高"的过拟合，
也不是"两边都退到多数类"的欠拟合。要判断瓶颈在**特征**还是在 **MLP 容量**，最直接的办法是：
在同一份特征、同一份划分上换成**没有隐藏层**的模型。若线性模型就能达到 MLP 的水平，
说明再加网络容量不会有用，瓶颈在特征或数据。

为此新增四个文件（见 §16.1），**不修改任何已有训练流程**：

| 文件 | 作用 |
|---|---|
| `shallow_models.py` | 纯 NumPy 浅层模型：`L2LogisticRegression`、`L1LogisticRegression`、`LinearDiscriminantAnalysis`、`RBFLSSVM`、`NearestCentroid` |
| `feature_selection.py` | 单变量 AUC 排序、相关性并查集分组、前向/后向贪心选择、`ColumnSelector`；多于两类时单列得分改成**成对平均 AUC**（Hand & Till），而不是一站式宏平均——后者对完全单调的三类特征得 0.5（0.0/0.5/1.0 的平均），会把最好的列排到最后 |
| `baseline_evaluation.py` | 分层 CV、`PriorClassifier`、`holdout_evaluate`；**每个 fold 内部重新拟合 `StandardScaler`** |
| `run_baseline_models.py` | CLI 驱动，输出 `baseline_report.txt` / `baseline_results.json` |

### 10.2 协议

- 直接读取 `experiments/.../features_{split}.npy`。这些文件是**原始未标准化**的
  （标准化在训练时由 `run_emd_experiments.py` 内部完成），因此基线**在每个 fold 内自行拟合 scaler**，
  与 §9 的训练规则一致；
- test 集**只用于最后报告**，超参在 train 上的 CV 里选题；网格内并列时取**正则最强**的一档；
- 5 折 × 5 重复分层 CV，seed 0；
- 多数类地板：train 0.5466 / val 0.5500 / test 0.5373。test 共 67 个样本，
  因此 **1 个样本 = 1.49 pt**，`0.7164` 与 `0.7612` 之间只差 3 个样本。

运行方式：

```powershell
python run_baseline_models.py `
  --experiments bin/_ab2/compact_core_s42/form_top3_mean__regions_dyn_envelope__channels_1 `
  --output-dir bin/_baseline_full
```

`--feature-selection forward|backward` 可同时跑特征选择（`--max-features`、`--min-gain`）。

### 10.3 结果（`form_top3_mean__regions_dyn_envelope__channels_1`，16 列）

| 模型 | 选中超参 | CV AUC | test acc | test AUC |
|---|---|---:|---:|---:|
| prior（地板） | — | 0.5000 | 0.5373 | 0.5000 |
| nearest_centroid | — | 0.7640 | 0.7164 | 0.7652 |
| **MLP（§9，同划分）** | — | — | **0.7164** | **0.8082** |
| rbf_lssvm | `gamma=1, sigma=4` | 0.7987 | 0.7313 | 0.8118 |
| lda | `shrinkage=0.3` | 0.8040 | 0.7612 | 0.8342 |
| l2_logistic | `l2=0.1` | 0.8003 | 0.7612 | 0.8262 |
| l1_logistic | `l1=0.03` | 0.7962 | 0.7612 | 0.8378 |
| **lda + 前向选择 5/16 列** | — | — | **0.8060** | **0.8504** |

（MLP 行是 seed 42 单次；5 个种子的均值为 **0.7373 ± 0.0133 / AUC 0.8206 ± 0.0090**。
同一份特征、同一划分、不同种子得到的特征矩阵**逐位相同**，所以这 0.0133 完全是训练噪声。）

### 10.4 结论

1. **线性模型 ≥ MLP。** `lda` / `l2_logistic` / `l1_logistic` 三者并列 0.7612，
   比 MLP 的 seed 42 值高 3 个 test 样本；对比 5 种子均值只高约 1.6 个样本。
   差距不显著，但**可以确定的结论是：MLP 相对线性模型没有带来任何增益**，
   即多层非线性没有提取到线性模型看不到的结构。
2. **瓶颈在特征，不在容量。** 加大 MLP 宽度/深度、加正则、换优化器都不会突破这个水平；
   §10.3 最后一行反而说明**减少**特征更有效。
3. **有效维度远小于 16。** 前向选择把 16 列砍到 5 列，test accuracy 从 0.7761 升到 0.8060、
   AUC 从 0.8441 升到 0.8504。选中的 5 列是
   `dynamic_main_window.spectral_centroid`、`dynamic_main_window.peak_mm`、
   `dynamic_tail_window.energy`、`dynamic_tail_window.abs_mean`、
   `dynamic_tail_window.zero_crossing_rate`。
4. **列内高度共线。** `|r| ≥ 0.98` 的分组显示 `{std, rms, abs_mean, peak_abs}` 在**每个 branch 内**
   互相共线（8 列 ≈ 2 个独立方向）。单列区分度排序为：
   `tail.zero_crossing_rate` 0.2935 > `tail.spectral_centroid` 0.2737 >
   `main.energy` 0.2611 > `main.std` = `main.rms` 0.2554。
5. **`main.spectral_centroid` 在主窗口上是死列**（|AUC−0.5| = 0.0064，全表最低），
   但它在尾窗口上是第二强的列（0.2737）。这与 §6 的物理事实一致：两个 branch 重采样到同样
   512 点，但主窗口只覆盖 ~0.95 mm、尾窗口 ~3.60 mm，逐点物理间距差 3.8 倍，
   只有尾窗口有足够带宽让谱质心表达出差异。前向选择最后一步把它加入只换来
   +0.0077 的 CV AUC，**大概率是噪声**，不应据此认为它有价值。
6. **两个跨配置稳健的列**：前向选择在 `compact+core` 与 `legacy+none` 两种特征集上都把
   `dynamic_tail_window.zero_crossing_rate` → `dynamic_tail_window.energy` 选在前面，
   这是目前最有把握保留的特征对。

> 下一步若要把错误率压到 25% 以下，方向是**换数据划分的物理意义**（例如 1.3 mm 阈值，
> 见 §2.2）或**引入新的信号表征**，而不是继续调 MLP。
>
> §2.2 已经把 1.3 mm 与 0.8/1.2 mm 两种方案实现成数据集并用同一套基线跑过一遍：
> 三分类（0.8/1.2）下 `lda`/`l1_logistic` 的 test accuracy 0.5821、宏平均 AUC 0.75–0.76，
> 高于多数类地板 0.4179，与同数据的 MLP（test AUC 0.7662）基本持平——与二分类的结论
> 一致：线性模型与 MLP 无差距，瓶颈在特征。

## 11. GUI 功能

`gui_app.py` 提供 Tkinter 图形化实验查看和训练工具：

- 选择数据集（标签方案）：`raw_data/`（历史 1.0 mm 二分类）或 `raw_data_relabeled/` 下任意带 `label_scheme.json` 的目录，整个界面跟随该选择；
- 选择帧预处理方式、区域组合和通道模式；
- 查看 summary、特征维度、Best Epoch 和训练指标（多分类为宏平均）；
- 在 GUI 中启动当前选择的训练，训练脚本会收到 `--data-dir` 与 `--num-classes`；
- 查看四种帧预处理信号图；
- 查看 validation/test loss-epoch 曲线；
- 查看 test 集混淆矩阵（二分类 2×2，多分类 k×k）；
- 查看每个 branch 的 EMD 分量和 residue；
- 预处理页将当前组合和目标组合按类别并排展示（`2k` 列，二分类即历史的 label=0/1 四列）；
- 所有列图像共用横向滚动和纵向滚动，支持多 branch 图像查看；
- 目标组合选择不受顶部当前区域组合限制；
- 训练完成后可自动生成预处理图、EMD 图、训练曲线图、混淆矩阵图和错分厚度分布图。

带标签的数据集隔离存放，互不覆盖：

- 实验目录名带 `label_scheme` 标签后缀，图库写入 `visualizations/<kind>/<tag>/`；
- 历史默认数据集（无 `label_scheme.json`）保持原有目录名、图片文件名与字节内容完全不变；
- 查看结果时按 `label_scheme` 标签过滤，切换数据集不会串台。

运行：

```powershell
python gui_app.py
```

运行环境必须包含完整 Tcl/Tk。若基础 Python 环境缺少 `init.tcl`，需要使用带 Tcl/Tk 的 Python/conda 环境运行 GUI。

## 12. 可视化脚本

- `visualize_preprocessing.py`：每个类别随机选择 10 个样本，绘制四种预处理结果；深度横轴显示到小数点后两位。
- `visualize_emd_components.py`：每个类别随机选择 1 个样本，绘制每个 branch 的 IMF 和 residue。
- `visualize_training_curves.py`：绘制 validation loss/test loss 随 epoch 的变化。
- `visualize_confusion_matrix.py`：绘制 test 集混淆矩阵。
- `visualize_error_by_thickness.py`：按骨头厚度（`samples_*.json` 的 `depth_value`，即原始表的“总厚度”列，不是 0-5 mm 采集深度轴）分桶统计 train/val/test 的错分样本数并绘制堆叠柱状图；浅色宽柱为该桶全部样本，深色窄柱为各 split 的错分样本，1 mm 处有 label 分界参考线。默认桶宽 0.05 mm、判定阈值 0.5，只读取训练已保存的 `probabilities_{split}.npy`/`labels_{split}.npy`，不重跑推理。

默认类别和随机设置：

- label 0：即将穿透；
- label 1：安全；
- 随机种子：42；
- 样本池：train + val + test。

## 13. 输出目录

训练默认输出：

```text
experiments/
  form_<form>__regions_<region>__channels_<channel>/          # per_channel（默认）
  form_<form>__regions_<region>__channels_<channel>__pooled/  # 历史通道平均行为
    config.json
    features_train.npy
    features_val.npy
    features_test.npy
    labels_train.npy
    labels_val.npy
    labels_test.npy
    standard_scaler.npz
    mlp_model.npz
    history.json
    metrics.json
    feature_info.json
    probabilities_train.npy
    probabilities_val.npy
    probabilities_test.npy
  summary.json
```

> `--channel-aggregation pooled` 会写入带 `__pooled` 后缀的目录，因为它是消融对照，
> 不能覆盖默认的 `per_channel` 结果。单个通道（`--channel-mode 1` / `2`）两种设置等价，
> 但目录名仍按传入值区分，便于对照。
>
> 用 `--data-dir` 指向带 `label_scheme.json` 的数据集时（§2.2），目录名再追加 `__<tag>`，
> 例如 `form_top3_mean__regions_dyn_envelope__channels_1__cls3_thr0.8-1.2`，
> 因此多分类实验与二分类实验平行存在、互不覆盖。

可视化默认输出到：

```text
visualizations/preprocessing/
visualizations/emd/
visualizations/training_curves/
visualizations/confusion_matrices/
```

浅层基线输出（`run_baseline_models.py --output-dir ...`）：

```text
bin/_baseline_full/
  baseline_report.txt   # 可直接阅读的对照表
  baseline_results.json # 含每个 fold 的指标与选中列
```

> `bin/_*` 都被 `.gitignore` 的 `_*/` 匹配，属于**本地产物**，不入库；
> 需要时用 §10.2 的命令重新生成即可。

## 14. 推荐运行命令

运行全部帧处理方式、全部区域组合和指定通道：

```powershell
python run_emd_experiments.py `
  --forms all `
  --region-set all `
  --channel-mode 3
```

> `--channel-mode 3` 默认使用 `--channel-aggregation per_channel`，
> 即两个通道各成一组特征并各有一座 MLP 塔。需要复现历史（通道平均）结果时加
> `--channel-aggregation pooled`。

运行单组动态包络实验：

```powershell
python run_emd_experiments.py `
  --forms top3_mean `
  --region-set dyn_envelope `
  --channel-mode 1 `
  --dyn-lead-back 20 `
  --dyn-main-length 170
```

校验 `dyn_envelope` 与参考实现逐位一致：

```powershell
python verify_dyn_envelope.py
```

换厚度阈值跑多分类（阈值可任意指定，见 §2.2）：

```powershell
python raw_data/relabel_by_thickness.py --thresholds 0.8,1.2 `
  --output-dir raw_data_relabeled/cls3_thr0.8_1.2

python run_emd_experiments.py `
  --data-dir raw_data_relabeled/cls3_thr0.8_1.2 `
  --forms top3_mean `
  --region-set dyn_envelope `
  --channel-mode 1

python run_baseline_models.py `
  --experiments experiments/form_top3_mean__regions_dyn_envelope__channels_1__cls3_thr0.8-1.2 `
  --output-dir bin/_baseline_cls3 `
  --feature-selection forward
```

> `experiments/` 中部分目录早于本次算法移植或早于 §2.1 的直流归零，
> 引用其中的指标前请先读 `experiments/STALENESS.md`。

快速验证代码：

```powershell
python run_emd_experiments.py `
  --forms top3_mean `
  --region-set dyn_envelope `
  --channel-mode 1 `
  --max-samples 2 `
  --max-imfs 1 `
  --max-sift-iterations 2 `
  --epochs 2 `
  --patience 2 `
  --output-dir smoke_test
```

跑浅层基线对照（用法与结果见 §10）：

```powershell
python run_baseline_models.py `
  --experiments bin/_ab2/compact_core_s42/form_top3_mean__regions_dyn_envelope__channels_1 `
  --output-dir bin/_baseline_full `
  --feature-selection forward
```

只用已有实验目录的特征（`experiments/.../features_{split}.npy`），
**不重跑** EMD、也不写回任何训练结果。

## 15. 当前验证情况

已完成：

- 所有核心 Python 文件语法检查；
- 固定区域训练流程验证；
- `dyn_envelope` 训练流程验证；
- `dyn_envelope` 与参考实现 268 点位 × 2 通道逐位一致性验证（`verify_dyn_envelope.py`，边界与分支内容全部通过）；
- 动态 branch 两个分支均重采样到 512 点的验证；
- Tukey 窗仍保留在训练/EMD路径的验证；
- 预处理可视化不加 Tukey 窗的验证；
- EMD 分量可视化验证；
- GUI 导入和训练命令构建验证；
- 混淆矩阵和训练曲线可视化验证；
- 双塔 MLP 的数值梯度检验（`verify_mlp_gradients.py`）：6 种拓扑（1 / 2 / 3 塔，
  `hidden_dims` 为 `(64,32)`、`(32,16)`、`(16,12,6)`、`(64,)`、`(128,64)`）的
  方向导数相对误差中位数 ≤ $2.2\times10^{-3}$、最差单方向 $1.8\times10^{-2}$
  （逐方向取中位数是为了排除踩到 ReLU 折点的方向，这是损失面性质而非实现错误）；
  逐参数中心差分 6 种拓扑中 5 种 100% 通过，1 种 57/58 通过（差值为 float32 舍入地板）；
- 单通道逐位回归（`verify_channel_aggregation.py`）：`--channel-mode 1` / `2` 下
  `--channel-aggregation per_channel` 与 `pooled` 的特征、标签、概率和全部指标
  **逐位相同**；
- 通道 3 双塔端到端验证（同一脚本）：`per_channel` 得到 `feature_group_dims=[8, 8]`、
  `tower_count=2`、共享头首层 `fan_in=128`；`pooled` 得到 `[8]`、`tower_count=1`、
  `fan_in=64`，即历史结构；两者精度不同（例：test AUC 0.7333 vs 0.7176，32 样本冒烟运行）；
  （该次验证跑在 §8.4 的特征改动之前，所以 `feature_group_dims` 是当时的 8 维；
  换成当前的 `compact+core` 后同样会是 `[16, 16]` / `[16]`，拓扑结论不变）；
- 特征改动（§8.4）的 A/B 对照：`legacy+none` 与 `compact+core` 各跑 5 个种子（42–46），
  同一划分、同一份 `raw_data`，结果见 §8.4；同一配置不同种子得到的特征矩阵**逐位相同**，
  因此种子间的差异是训练噪声而非特征差异；
- 特征列名回归（`feature_dimension_names`）：`compact+core` 在
  `channel-mode 1` 下输出 16 列、`channel-mode 3` 下输出 32 列，
  且 `onset_mm` / `peak_mm` 只挂在 `dynamic_main_window` 这个 branch 上（实测 `feature_dim=32`）；
- 浅层基线工具链自检（§10）：对一个**纯噪声**特征集，in-sample 准确率 0.6087 而 CV 只有 0.4856，
  证明 CV 确实扣掉了乐观偏差；3 个信号列 + 3 个噪声列的合成集上 CV accuracy 0.8714 / AUC 0.9378；
  常量列的 AUC 严格等于 0.5，`ColumnSelector` 不泄露 test 集；
- 两个可视化脚本在 `dyn_envelope` 上的端到端重跑（`visualize_emd_components.py`、
  `visualize_preprocessing.py`）：均正常出图；新的 branch 标题由配置和检测结果**推导**
  （`Branch 1: main | 170 pt = 0.95 mm | 0.00185 mm/pt`，
  `Branch 2: tail | 645 pt = 3.60 mm | 0.00703 mm/pt`），
  不再硬编码 `170 samples` / `[251, 896)`，因此在传入非默认
  `--dyn-main-length` / `--dyn-tail-start` 时也不会说谎；标题过长时会自动逐段截短。
- **通用 k 分类改造的全量回归**（§2.2）：
  - 二分类逐位不变：`--channel-mode 1` + `dyn_envelope` 的冒烟跑与完整基线（6 模型 × 25 folds
    + 前向选择）重跑后，`baseline_results.json` 除新增 `num_classes` 键外**深度相等**，
    `baseline_report.txt` **逐字节相同**（4932 字节）；
  - 三分类（`cls3_thr0.8-1.2`）端到端跑通：MLP 训练、混淆矩阵、错分厚度图、浅层基线
    （`OneVsRestClassifier` + 成对平均 AUC 排序）全部正常，且指标优于多数类地板；
  - 单列得分的合成数据校验：完全单调的三分类特征得 0.5（满分）、纯噪声 0.11、
    只有中间类可分的特征 0.015（确实无法用单一阈值刻画中间带）；
    二分类分支与旧 `_roc_auc` 公式数值一致。

三个自检脚本都不依赖 `experiments/` 里的历史产物，可用当前代码直接重跑：

```powershell
python verify_dyn_envelope.py        # 动态包络切分与参考实现逐位一致
python verify_mlp_gradients.py       # 双塔 MLP 解析梯度
python verify_channel_aggregation.py # 通道聚合语义与网络拓扑
```

## 16. 项目结构与归档约定

### 16.1 根目录（活跃代码与文档）

| 类别 | 文件 |
|---|---|
| 核心流水线 | `emd_pipeline.py`、`run_emd_experiments.py`、`dyn_cli.py` |
| 浅层基线（change #3，见 §10） | `shallow_models.py`、`feature_selection.py`、`baseline_evaluation.py`、`run_baseline_models.py` |
| GUI | `gui_app.py` |
| 可视化（GUI 调用 + 文档记录） | `visualize_preprocessing.py`、`visualize_emd_components.py`、`visualize_training_curves.py`、`visualize_confusion_matrix.py`、`visualize_error_by_thickness.py` |
| 自检回归 | `verify_dyn_envelope.py`、`verify_mlp_gradients.py`、`verify_channel_aggregation.py` |
| 文档 | `README.md`、`PROJECT_SUMMARY.md` |
| 数据/输出（本地，多数被忽略） | `raw_data/`（唯一的训练数据源）、`experiments/`、`visualizations/` |
| 数据重建 | `raw_data/relabel_by_thickness.py`（按指定厚度阈值重建数据集，见 §2.2） |

### 16.2 `bin/`（归档区）

根目录只保留仍然在用的脚本；一次性、已被取代或与项目无关的文件统一移入 `bin/`，
具体清单与原因见 `bin/README.md`。当前 `bin/` 内有：

- `bin/after_split_data/`（`reference_code/pipeline.py` 的输出，**只有** `verify_dyn_envelope.py` 读它作参照；训练一律走 `raw_data/`）
- `bin/reference_code/`（参考实现的原始目录，是算法出处；没有任何代码 import 它）
- `bin/_recon.ps1`（2026-09 环境迁移的一次性侦查脚本）
- `bin/visualize_dyn_envelope_cases.py`（`dyn_envelope` 阈值回退的专项诊断出图，未被 GUI/文档引用）
- `bin/backup/`（历史代码与实验快照：`0_no_gui` ~ `5_experiments_2026-09-17`）
- `bin/_recon_check/`、`bin/_smoke_all/`、`bin/_verify_fix/`（一次性冒烟/校验输出）
- `bin/_ab2/`、`bin/_ab_legacy/`（2026-09-22 特征集 A/B 扫描的 10 组输出，见 §8.4）
- `bin/_baseline_full/`、`bin/_baseline_smoke/`（浅层基线的全量与冒烟输出，见 §10）
- `bin/_dimcheck/`、`bin/_vis3f/`（维度自检与可视化回归的临时输出）
- `bin/_ab_run.ps1`（驱动 A/B 扫描的 PowerShell 脚本；**是文件不是目录**，仍被 git 跟踪）
- `bin/_gui_dataset/`、`bin/_kclass_check/`、`bin/_relabel_check/`（2026-09-22 多数据集选择器与
  k 分类重构的无头冒烟/校验脚本，外加按厚度重标注的校验数据集 `_relabel_check/cls3/`；
  搬家后脚本内的 `parents[N]` 已改为 `parents[2]`，可直接运行）
- `bin/__pycache__/`（含已删除模块 `dynamic_split`、`_grad_check` 的陈旧 pyc，可随时删除；
  根目录下由 Python 重新生成的同名目录属正常现象）

> `bin/` 下的目录仍被 `.gitignore` 的 `after_split_data/`、`reference_code/`、`backup/`、`_*/`、
> `__pycache__/` 规则忽略（这些规则都不带前导斜杠，可在任意层级匹配），所以归档内容不会进入版本库；
> 直接放在 `bin/` 下的三个脚本文件（`_recon.ps1`、`_ab_run.ps1`、`visualize_dyn_envelope_cases.py`）
> 则继续保持被跟踪。
> 依赖方向是“根目录 → `bin/`”，`bin/` 内的东西从不反向依赖根目录；唯一例外是
> `verify_dyn_envelope.py` 通过 `REFERENCE_DIR` 读取 `bin/after_split_data/`（可用环境变量
> `DYN_REFERENCE_DIR` 重定向）。

## 17. Git 发布说明

本目录已经是 Git 仓库根的子目录：远端为 `https://github.com/Icyhavoc/Bone_us.git`（原名
`bone_us.git`，已改名），当前分支 `custom-dyn_envelop` 跟踪 `origin/custom-dyn_envelop`。
`.gitignore` 已排除 `raw_data/`、`raw_data_relabeled/`（重标注数据集，由
`raw_data/relabel_by_thickness.py` 重新生成；该脚本本身用 `git add -f` 强制入库）、
`after_split_data/`、`reference_code/`（后两者已归档进 `bin/`，
规则仍按任意深度生效）、`experiments/*`（仅保留 `experiments/*.md`）、`visualizations/`、`backup/`、
`__pycache__/` 和 `_*/` 临时输出目录（这些规则不带前导斜杠，因此同样覆盖 `bin/` 下的归档副本）。

因此提交时只需：

```powershell
git add -A
git commit -m "..."
```

`_*/` 规则只匹配目录，所以根目录下已不再有临时目录；被归档的一次性脚本
`bin/_recon.ps1`、`bin/_ab_run.ps1`、`bin/visualize_dyn_envelope_cases.py` 是文件而非目录，仍在版本控制中；
新增的 `verify_*.py` 也会正常被跟踪。

如需发布可复现实验结果，可以额外选择性提交对应实验目录的 `config.json`、
`metrics.json`、`history.json` 和 `summary.json`（需临时放开 `experiments/*` 的忽略规则）。

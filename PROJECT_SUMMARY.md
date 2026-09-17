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
  -> NumPy MLP 二分类
```

当前代码位于 `经验小波分解/`，与 `raw_data/` 并列。

## 2. 数据集约定

- 每个样本形状：`[50, 2, 896]`。
- 50：帧数。
- 2：两个物理信号通道。
- 896：每帧采样点数。
- 默认 896 个采样点对应 0–5 mm。
- 标签约定：`label=0` 为即将穿透，`label=1` 为安全。
- 当前实验不考虑同一骨头不同采样点之间的相关性和数据泄露问题。
- 数据划分由 `raw_data/train`、`raw_data/val`、`raw_data/test` 提供。

### 2.1 直流电平归零（`--data-dir raw_data` 的硬性前提）

源 ADC 同时用码值 127 和 128 表示“零”，两者归一化后分别是
$\pm 0.5/127.5 \approx \pm 0.003922$，即同一个物理电平被拆到了 127.5 中点的两侧。
`raw_data/` 保留了这一 $\pm 0.0039$ 抖动，而 `after_split_data/` 是从抖动已被抹平的信号生成的；
这点量化噪声足以让 `dyn_envelope` 的过阈点偏移几十个采样点
（实测 `N35_P2_01` 通道 1 会从 324 漂到 267，从而触发人工修正守卫报错）。

因此 `emd_pipeline.load_split()` 现在统一调用 `replace_adc_dc_level()`
（`ADC_DC_MAGNITUDE = 0.5 / 127.5`，`ADC_DC_ATOL = 1e-6`）把这一对码值塌陷到 0.0。
由于入口唯一，训练与四个可视化脚本都自动获得同一份直流归零后的帧；
对本身已归零的 `after_split_data/` 该操作是幂等的。实验目录的 `config.json`
会用 `adc_dc_replacement` 字段记录该常数。

深度映射由 `DepthMapper` 封装，默认规则为：

```text
index = round(depth_mm / 5.0 * 896)
```

例如 1 mm 对应约 179 号采样点。后续如需改变映射，只需修改 `DepthMapper` 接口。

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
   已做逐位回归验证（见 §14）。

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

`dyn_envelope` 是 `reference_code/pipeline.py` 中 `segment()` 的**完整移植**，与 `after_split_data/` 的产物逐位一致（见 `verify_dyn_envelope.py`）。它不改变信号，只用包络结果确定 branch 边界。

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

每个 IMF 或 residue 提取 8 个统计特征：

- Mean；
- Std；
- RMS；
- Energy；
- Mean absolute value；
- Peak absolute value；
- Zero-crossing rate；
- Spectral centroid。

默认 `stream_aggregation=pooled`：

- 同一 branch 内，对 EMD 分量和信号流做均值池化；
- 每个 branch 的每组特征得到 8 维；
- 多个 branch 之间不求均值，直接拼接。

通道方向由 `channel_aggregation` 决定：

- `per_channel`（默认）：每个通道各一组，组间拼接；
- `pooled`：通道轴并入流轴一起池化，只剩一组。

因此每个 branch 的宽度是 `8 × 通道组数 × branch 数`：

| 区域组合 | `--channel-mode 1` / `2` | `--channel-mode 3`（`per_channel`） | `--channel-mode 3`（`pooled`） |
|---|---:|---:|---:|
| `full` / `bone` | 8 | 16 | 8 |
| `bone_plus_post` | 16 | 32 | 16 |
| `dyn_envelope` | 16 | 32 | 16 |

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

- 塔的宽度取 `hidden_dims[0]`，共享头取 `hidden_dims[1:]` 再接 2 类输出；
- 因此 `--channel-mode 3` 的双通道不再在 MLP 之前被平均掉，两个通道的特征
  在分类头里**各自经过一层非线性**后才融合；
- 只有一组时塔会塌缩成第一层全连接，网络与历史结构**完全等价**
  （已用逐位回归验证，见 §14）；
- `mlp_model.npz` 会额外记录 `group_dims` 与 `tower_count`，`feature_info.json`
  会记录 `feature_groups` 与 `feature_group_dims`。

当前使用两个输出节点的 softmax 交叉熵。对于二分类，它与单输出 sigmoid BCE 在数学上等价。

训练规则：

- 标准化参数只在训练集上拟合；
- validation loss/AUC 用于早停和最佳 epoch 选择；
- test 集不参与模型选择；
- 每个 epoch 的 test loss 仅用于曲线观察；
- 默认最多训练 200 个 epoch，默认 patience 为 40。

## 10. GUI 功能

`gui_app.py` 提供 Tkinter 图形化实验查看和训练工具：

- 选择帧预处理方式、区域组合和通道模式；
- 查看 summary、特征维度、Best Epoch 和训练指标；
- 在 GUI 中启动当前选择的训练；
- 查看四种帧预处理信号图；
- 查看 validation/test loss-epoch 曲线；
- 查看 test 集混淆矩阵；
- 查看每个 branch 的 EMD 分量和 residue；
- 预处理页将当前组合和目标组合分别按 label=0/1 展示；
- 四列图像共用横向滚动和纵向滚动，支持多 branch 图像查看；
- 目标组合选择不受顶部当前区域组合限制；
- 训练完成后可自动生成预处理图、EMD 图、训练曲线图和混淆矩阵图。

运行：

```powershell
python gui_app.py
```

运行环境必须包含完整 Tcl/Tk。若基础 Python 环境缺少 `init.tcl`，需要使用带 Tcl/Tk 的 Python/conda 环境运行 GUI。

## 11. 可视化脚本

- `visualize_preprocessing.py`：每个类别随机选择 10 个样本，绘制四种预处理结果；深度横轴显示到小数点后两位。
- `visualize_emd_components.py`：每个类别随机选择 1 个样本，绘制每个 branch 的 IMF 和 residue。
- `visualize_training_curves.py`：绘制 validation loss/test loss 随 epoch 的变化。
- `visualize_confusion_matrix.py`：绘制 test 集混淆矩阵。

默认类别和随机设置：

- label 0：即将穿透；
- label 1：安全；
- 随机种子：42；
- 样本池：train + val + test。

## 12. 输出目录

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

可视化默认输出到：

```text
visualizations/preprocessing/
visualizations/emd/
visualizations/training_curves/
visualizations/confusion_matrices/
```

## 13. 推荐运行命令

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

## 14. 当前验证情况

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
  `fan_in=64`，即历史结构；两者精度不同（例：test AUC 0.7333 vs 0.7176，32 样本冒烟运行）。

三个自检脚本都不依赖 `experiments/` 里的历史产物，可用当前代码直接重跑：

```powershell
python verify_dyn_envelope.py        # 动态包络切分与参考实现逐位一致
python verify_mlp_gradients.py       # 双塔 MLP 解析梯度
python verify_channel_aggregation.py # 通道聚合语义与网络拓扑
```

## 15. Git 发布说明

本目录已经是 Git 仓库根的子目录：远端为 `https://github.com/Icyhavoc/bone_us.git`，
当前分支 `custom-dyn_envelop` 跟踪 `origin/custom-dyn_envelop`。
`.gitignore` 已排除 `raw_data/`、`after_split_data/`、`reference_code/`、
`experiments/*`（仅保留 `experiments/*.md`）、`visualizations/`、`backup/`、
`__pycache__/` 和 `_*/` 临时输出目录。

因此提交时只需：

```powershell
git add -A
git commit -m "..."
```

`_*/` 规则只匹配目录，所以 `_recon.ps1` 仍在版本控制中；新增的
`verify_*.py` 是脚本文件而不是目录，会正常被跟踪。

如需发布可复现实验结果，可以额外选择性提交对应实验目录的 `config.json`、
`metrics.json`、`history.json` 和 `summary.json`（需临时放开 `experiments/*` 的忽略规则）。

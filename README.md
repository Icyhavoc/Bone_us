# EMD 特征提取 + MLP 分类

本目录实现当前阶段的完整实验管线：

```text
raw_data
  -> 四种帧处理方式
  -> 任意深度区间 branch
  -> 每个 branch 重采样到 512 点
  -> 每个 branch 独立进行 EMD
  -> branch 特征拼接
  -> NumPy MLP 分类头
```

代码文件与 `raw_data` 同级：

- `emd_pipeline.py`：数据加载、帧处理、深度映射、重采样、EMD、特征和 MLP 实现。
- `run_emd_experiments.py`：命令行入口，可运行单组或全部 16 组标准实验。
- `dyn_cli.py`：`--dyn-*` 参数的统一定义，供三个入口脚本和 GUI 共用。
- `gui_app.py`：Tkinter 图形界面。
- `visualize_*.py`：五张图的可视化脚本（预处理、EMD 分量、训练曲线、混淆矩阵、错分厚度分布）。
- `verify_*.py`：三份可重跑的回归自检。
- `bin/`：**归档区**（一次性脚本、历史快照、参考实现与参考数据），清单见 `bin/README.md`。
- `README.md`：使用说明。

> 根目录只保留仍在用的脚本：每次整理时把“已经用不到”的东西移入 `bin/`，
> 而不是删除。`bin/` 里的内容被 `.gitignore` 忽略，但两个脚本文件（`bin/_recon.ps1`、
> `bin/visualize_dyn_envelope_cases.py`）仍在版本控制中。

## 数据格式

默认从 `raw_data` 读取数据。每个 `train`、`val`、`test` 子目录包含：

- `X.npy`：`[N, 50, 2, 896]`，依次为样本、帧、物理通道、采样点。
- `y.npy`：二分类标签。
- `samples.json`：样本编号、深度值和标签信息。

当前数据集的标签保持不变：`label=0` 对应 `depth < 1 mm`，`label=1` 对应 `depth >= 1 mm`。

## 四种帧处理方式

命令行中的 `--forms` 支持：

- `mean_std`：对 50 帧逐点计算 Mean 和 Std。
- `raw50`：保留全部 50 帧。
- `max1`：依据整段 0–5 mm 信号的综合 RMS 选择 1 帧。
- `top3_mean`：依据综合 RMS 选择 3 帧并求平均。

Max 1 Frame 和 Top-3 Mean 默认在完整 0–5 mm 区间上选择帧，然后把相同的帧处理结果提供给各个深度 branch。选择范围可通过 `--selection-region-json` 修改。

### 直流电平归零

源 ADC 用 127 和 128 两个码值表示同一个零电平，归一化后分别是 $\pm 0.5/127.5 \approx \pm 0.003922$。`raw_data/` 保留了这一抖动，而参考数据集（现位于 `bin/after_split_data/`）来自抖动已抹平的信号；该抖动足以让 `dyn_envelope` 的过阈点偏移几十个采样点。`emd_pipeline.load_split()` 因此统一调用 `replace_adc_dc_level()` 把这对码值塌陷到 `0.0`，训练与四个可视化脚本共用同一入口；对已归零的 `bin/after_split_data/` 该操作幂等。实验 `config.json` 用 `adc_dc_replacement` 记录该常数。

## 深度 branch

标准区域组合为：

- `full`：`[[0, 5]]`
- `bone`：`[[1, 3]]`
- `bone_plus_post`：`[[1, 3], [3, 5]]`
- `dyn_envelope`：逐通道定位起振波包。每个通道按 `max(abs(x))` 取幅值最大的 3 帧求平均得到定位信号，再按包络检测起振点；第 1 个 branch 为 `[max(t-20,70), +170)`，第 2 个 branch 为固定尾部 `[251, 896)`。

branch 不要求覆盖完整 0–5 mm，也不要求互斥、连续或等长。可以使用任意嵌套或重叠区间，例如：

```powershell
python run_emd_experiments.py --forms max1 --regions-json "[[0,5],[1,3]]"
```

每个 branch 独立切片、重采样到 512 点、进行 EMD，并在最后拼接特征。

重采样完成后，每个 branch 的每条信号都会乘以长度为 512、`alpha=0.3` 的 Tukey 窗，再进入 EMD。该参数可通过 `--tukey-alpha` 修改。

### 深度到采样点映射

默认映射为：

```text
index = round(depth_mm / 5.0 * 896)
```

因此 1 mm 默认对应 `round(896 * 0.2) = 179`。映射由 `DepthMapper` 封装，后续可以替换 `depth_to_index()` 和 `index_to_depth()`，无需修改 branch 或预处理逻辑。

边界采用左闭右开区间 `[start, end)`，避免相邻区域重复使用边界采样点。

### 动态包络 branch

`dyn_envelope` 不改变原始信号，只用包络分析结果确定 branch 边界。它是 `bin/reference_code/pipeline.py` 中 `segment()` 的完整移植，与 `bin/after_split_data/` 的产物逐位一致。

定位信号与帧处理方式无关：每个通道按 `max(abs(x))` 选幅值最大的 `locator_top_k=3` 帧（并列取较早帧）求平均，再取 Hilbert 包络。检测流程为：包络前 71 点置零后用 `mode='nearest'` 的长度 5 移动平均平滑；在第 71 点起找首个 `> 0.015` 的参考点；在 `[参考点-50, 参考点+150)` 内取局部峰；以 `0.05 × 峰值` 为比例阈值提取连续段；从含该峰的主波包向前合并间隔 ≤ 8、宽度 ≥ 8、面积 ≥ 主波包 5%、累计提前 ≤ 96 的段；阈值点 `t` 取合并后最早越阈点；主窗为 `[max(t-20,70), +170)`，尾部为固定 `[251, 896)`。

两个通道**各自独立**定位，因此主窗起点与尾部重叠长度可能不同；主窗不补零、不平移，尾部允许与主窗重叠或留间隔。切片后各分支重采样到 512 点并加 Tukey 窗，再进入 EMD。

全部参数由 `DynamicEnvelopeConfig` 提供默认值，`--dyn-*` 选项由 `dyn_cli.py` 统一定义：

```powershell
python run_emd_experiments.py `
  --forms top3_mean `
  --region-set dyn_envelope `
  --channel-mode 3 `
  --dyn-smooth-window 5 `
  --dyn-weak-threshold 0.015 `
  --dyn-main-length 170 `
  --dyn-tail-start 251
```

每个样本逐通道的阈值点、局部峰、参考点、回退规则和两个 branch 的采样点边界会写入该实验的 `feature_info.json`；参数会写入 `config.json`。

用 `python verify_dyn_envelope.py` 可对 268 个点位 × 2 通道重新校验边界与分支内容是否与参考实现完全一致。

## 通道选择

`--channel-mode` 支持：

- `1`：只使用物理通道 1。
- `2`：只使用物理通道 2。
- `3`：同时使用通道 1 和 2。

通道选择与深度 branch 是两个独立维度；物理通道不会被误当成深度 branch。

### 通道 3 如何处理两个通道

两个通道**各自独立**切片、重采样、做 EMD，各得一组特征；两组特征**拼接**后进入分类头，
而且**每组特征各有一座独立的 MLP 塔**，塔的输出再拼接起来交给共享头。
分组方式由 `--channel-aggregation` 控制：

| 取值 | 含义 |
|---|---|
| `per_channel`（默认） | 每个通道一组特征，组间拼接，每组一座 MLP 塔 |
| `pooled` | 通道轴与分量轴一起平均成一组，复现历史行为 |

```text
channel_mode=1 / 2                    channel_mode=3 (per_channel)
[1,512]  1 条流                       [2,512]  2 条流
   ↓ EMD 跑 1 次                         ↓ EMD 跑 2 次（每通道各 1 次）
 8 维特征（1 组）                       8 + 8 = 16 维特征（2 组）
   ↓                                     ↓
 1 座塔 8→64                           2 座塔 8→64 -> 拼成 128 -> 共享头 128→32→2
```

所以通道 3 每个 branch 的特征宽度是 **16 维**（`8 ⅹ 2`），通道 1 / 2 仍是 8 维。
两个通道不再在分类头之前被平均掉，而是各自经过一层非线性后才融合。

**不是相加。** 两组特征来自**两次独立的 EMD**。EMD 是非线性分解，
"两通道先相加再做 EMD"的结果与分别分解相差 $10^{-2}$ 量级，完全不是一回事。

### 历史行为：`--channel-aggregation pooled`

历史版本的通道 3 会把「通道轴 × 分量轴」一起取平均，得到单组 8 维特征：

```text
[2,6,8]  2 通道 × 6 分量
   ↓ np.mean(component_matrix, axis=(0, 1))
 8 维特征  ← 维度不翻倍，两个通道的个性被平均掉
```

实测同帧、同选帧下满足 $f_{\text{ch3}}^{\text{pooled}} = \tfrac{1}{2}(f_{\text{ch1}} + f_{\text{ch2}})$
（误差 $7.5\times10^{-9}$）。该选项现在仅用于复现历史结果和消融对照。

### 选帧随通道模式变化

`max1` / `top3_mean` 用 RMS 排名选帧，而 RMS 是对**通道轴和采样轴一起**求的，
所以通道集不同 → 得分不同 → 可能选中不同帧。实测同一样本：

| `--channel-mode` | `selected_frames` |
|---|---|
| 1 | `[13 11 12]` |
| 2 | `[28 27 26]` |
| 3 | `[26 27 24]` |

即通道 1 单独跑时选的帧，和通道 3 里第一个通道用的帧**不是同一帧**。因此上面那个
严格平均关系只在帧选择相同时成立：

| form | 是否选帧 | 与平均关系的偏差 |
|---|---|---|
| `mean_std` | 否 | $1.5\times10^{-8}$（严格成立） |
| `raw50` | 否 | $7.5\times10^{-8}$（严格成立） |
| `max1` | 是 | $1.2\times10^{-2}$（偏离） |
| `top3_mean` | 是 | $6.6\times10^{-3}$（偏离） |

> 对比 `channels_1` 与 `channels_3` 的实验时，"多了通道 2"和"选帧变了"两个因素混在一起
> 无法分离。需要干净的消融对照时，固定 `--selection-region-json`，或改用不选帧的
> `mean_std` / `raw50`。

## EMD 特征

当前实现包含一个仅依赖 NumPy 的确定性 EMD 实现。默认最多提取 5 个 IMF，并保留残余项。每个 IMF/残余项提取：

- Mean
- Std
- RMS
- Energy
- Mean absolute value
- Peak absolute value
- Zero-crossing rate
- Spectral centroid

默认使用 `--stream-aggregation pooled`：对每个 branch 内的所有 EMD 分量和信号流做均值池化，每个 branch 的每组特征得到 8 个统计量；多个 branch 之间不做均值，而是直接拼接。

通道方向由 `--channel-aggregation` 决定：`per_channel`（默认）让每个通道各成一组，组间拼接；`pooled` 把通道轴并入流轴一起池化，只剩一组。

因此每个 branch 的宽度是 `8 × 通道组数 × branch 数`：

| 区域组合 | `--channel-mode 1` / `2` | 通道 3（`per_channel`） | 通道 3（`pooled`） |
|---|---:|---:|---:|
| `full` / `bone` | 8 | 16 | 8 |
| `bone_plus_post` | 16 | 32 | 16 |
| `dyn_envelope` | 16 | 32 | 16 |

这里的"信号流"既包括 `mean_std` 的两条流（均值/标准差）或 `raw50` 的 50 帧，也包括通道 3 展开出的两个通道——在 `pooled` 下它们都被同一个均值池化掉。

`flatten` 和 `stats` 仍保留作对照实验，不再设置最终特征维数上限。

## MLP 分类头

NumPy 实现的轻量 MLP，不使用 CNN。

**单组特征**（通道 1 / 2，或通道 3 加 `--channel-aggregation pooled`）时就是历史结构：

```text
Linear(input_dim, 64)
-> ReLU + Dropout(0.1)
-> Linear(64, 32)
-> ReLU + Dropout(0.1)
-> Linear(32, 2)
```

**多组特征**（通道 3，默认 `per_channel`）时改为「每组建一座塔 + 共享头」：

```text
组0 (8 维) -> Linear(8, 64) -> ReLU + Dropout(0.1) -┐
组1 (8 维) -> Linear(8, 64) -> ReLU + Dropout(0.1) -┤ concat -> 128 维
                                                    ↓
                                       Linear(128, 32) -> ReLU + Dropout(0.1)
                                                    ↓
                                                 Linear(32, 2)
```

塔的宽度取 `hidden_dims[0]`，共享头取 `hidden_dims[1:]` 再接 2 类输出。只有一组时塔会塌缩成第一层
全连接，网络与历史结构**完全等价**（已用逐位回归验证）。

MLP 只负责 branch 特征融合和二分类，不使用 CNN。标准化参数只在训练集上拟合，验证集用于早停；测试集不参与训练或模型选择，每个 epoch 的 test loss 仅用于曲线观察。

当前 MLP 使用两个输出节点的 softmax 交叉熵。对于二分类，它与单输出节点 sigmoid BCE 数学等价，因此不需要仅因为类别数为 2 而更换 loss；测试集 loss 只记录用于训练曲线，不参与模型选择。

## 运行方式

在本目录下执行。

### 运行全部 16 组标准实验

```powershell
python run_emd_experiments.py --forms all --region-set all --channel-mode 3
```

### 运行一组实验

```powershell
python run_emd_experiments.py `
  --forms max1 `
  --region-set bone_plus_post `
  --channel-mode 3
```

### 自定义 N 个 branch

```powershell
python run_emd_experiments.py `
  --forms top3_mean `
  --regions-json "[[0,5],[1,3],[3,4.5]]" `
  --channel-mode 3
```

### 快速 smoke test

只处理每个 split 的前 4 个样本并训练 2 个 epoch：

```powershell
python run_emd_experiments.py `
  --forms max1 `
  --region-set full `
  --channel-mode 3 `
  --max-samples 4 `
  --epochs 2 `
  --patience 2 `
  --output-dir smoke_test
```

### 自检脚本

三个脚本不依赖 `experiments/` 里的历史产物，可用当前代码直接重跑：

```powershell
python verify_dyn_envelope.py         # 动态包络切分与参考实现逐位一致
python verify_mlp_gradients.py        # 双塔 MLP 的解析梯度（数值梯度对照）
python verify_channel_aggregation.py  # 通道聚合语义与网络拓扑
```

`verify_channel_aggregation.py` 会真实调用 CLI 跑 6 次小规模训练，验证：单通道下
`per_channel` 与 `pooled` 逐位相同；通道 3 下 `per_channel` 的特征宽度是 `pooled` 的两倍、
`tower_count=2`、共享头首层输入翻倍。加 `--keep` 可保留临时目录 `_agg_regress/`。

## 输出文件

默认输出到 `experiments/`，每组实验一个目录：

- `--channel-aggregation per_channel`（默认）：`form_<form>__regions_<region>__channels_<channel>/`
- `--channel-aggregation pooled`（历史行为/消融对照）：同名 + `__pooled` 后缀

单个通道下两种设置等价，但仍分开存放，便于对照。

```text
experiments/
  form_max1__regions_full__channels_3/
    config.json
    features_train.npy
    features_val.npy
    features_test.npy
    labels_train.npy
    labels_val.npy
    labels_test.npy
    standard_scaler.npz
    mlp_model.npz
    history.json                 # 每个 epoch 的 train/validation/test loss
    metrics.json
    feature_info.json
    probabilities_train.npy
    probabilities_val.npy
    probabilities_test.npy
  summary.json
```

`metrics.json` 保存 `best_epoch`、`trained_epochs`、最佳验证集指标，以及 Accuracy、Precision、F1-score、Sensitivity、Specificity、AUC 和混淆矩阵计数。

## 图形化实验查看器

`gui_app.py` 提供本地 Tkinter 图形界面，读取已有的 `experiments/` 和 `visualizations/` 结果，并支持在界面中启动当前选项的训练。运行 GUI 的 Python 环境需要包含完整的 Tcl/Tk 运行库。

启动方式：

```powershell
python gui_app.py
```

界面功能：

- 选择 `全部预处理方式` 时，显示当前已生成实验的 summary 对比表。
- Summary 显示特征维数、Best Epoch、实际训练 Epoch 数和 Train/Val/Test 指标；Best Epoch 遵循训练检查点规则（默认 `min_delta=1e-4`），优先选择验证集 AUC 改善的 epoch，AUC 相同时选择 validation loss 改善的 epoch。
- 选择单一预处理方式时，显示对应的预处理信号图和 EMD 分量图。
- “Training Curve” 页位于预处理可视化和 EMD 分解之间，显示当前筛选实验的 validation loss/test loss-epoch 曲线，并标出 Best Epoch。
- “Confusion Matrix” 页显示当前顶部筛选对应的 test 集混淆矩阵，并可在右侧选择栏中独立选择其他已生成实验进行对比。
- “错分厚度分布” 页显示当前顶部筛选对应的“骨头厚度—错分数”柱状分布（浅色宽柱为该厚度桶的全部样本，深色窄柱为 train/val/test 各自的错分样本，1 mm 处有 label 分界参考线），并可在右侧选择栏中独立选择其他已生成实验进行对比。
- 预处理页不再提供全局类别下拉框；四个等宽列分别显示当前组合 label=0/1 和目标组合 label=0/1。顶部筛选条件只控制当前组合，目标组合选择栏可独立选择所有已生成的 form/region/channel 结果。四列共用横向滑动条，按图片序号同步切换，多 branch 图像可横向查看。
- 支持区域组合和通道模式筛选；EMD 页默认同时显示两个类别。
- 点击“开始训练”后，会按当前预处理方式、区域组合和通道模式调用现有训练脚本；训练在后台执行，日志会显示在 Summary 页底部。
- 默认训练完成后自动生成对应的预处理图、EMD 分量图、训练曲线图、test 集混淆矩阵图和错分厚度分布图；可以取消“训练后生成图像”。
- 旧版不含通道号的图片文件也可以读取；新生成的图片文件名会包含通道模式，避免不同通道结果互相覆盖。

训练时需要选择具体通道 1、2 或 3，不能选择“全部通道”。预处理方式和区域组合可以选择“全部”，这会按现有命令行脚本运行多组实验。

如果某个筛选组合尚未运行实验或生成图像，界面会提示缺少结果；需要先使用 `run_emd_experiments.py`、`visualize_preprocessing.py`、`visualize_emd_components.py`、`visualize_training_curves.py`、`visualize_confusion_matrix.py` 或 `visualize_error_by_thickness.py` 生成对应文件。

## 数据可视化

新增五个可视化脚本：

- `visualize_preprocessing.py`：对四种帧处理方式的输出进行可视化，每个类别随机选择 10 个样本。
- `visualize_emd_components.py`：对 EMD 得到的 IMF 和 Residue 进行可视化，每个类别随机选择 1 个样本。
- `visualize_training_curves.py`：读取 `history.json`，绘制 validation loss 和 test loss 随 epoch 的变化；test loss 不参与 early stopping。
- `visualize_confusion_matrix.py`：读取 `metrics.json`，绘制 test 集混淆矩阵。
- `visualize_error_by_thickness.py`：读取 `samples_{split}.json` 与 `probabilities_{split}.npy`/`labels_{split}.npy`，按骨头厚度分桶统计 train/val/test 的错分样本个数并绘制堆叠柱状图；默认桶宽 0.05 mm、判定阈值 0.5。

两个脚本默认都使用：

- `label=0` 作为“即将穿透”；
- `label=1` 作为“安全”；
- 固定随机种子 42；
- `train + val + test` 的全部样本池；
- 四种帧处理方式和四种区域组合（`full`、`bone`、`bone_plus_post`、`dyn_envelope`）；
- 通道模式 3，即同时保留物理通道 1 和 2。

运行预处理结果可视化：

```powershell
python visualize_preprocessing.py `
  --forms all `
  --region-set all `
  --channel-mode 3
```

运行 EMD 分量可视化：

```powershell
python visualize_emd_components.py `
  --forms all `
  --region-set all `
  --channel-mode 3
```

运行训练曲线可视化（需先完成训练）：

```powershell
python visualize_training_curves.py `
  --experiments-dir experiments `
  --output-dir visualizations/training_curves
```

输出分别位于：

```text
visualizations/preprocessing/
visualizations/emd/
visualizations/training_curves/
visualizations/confusion_matrices/
```

如果只想检查当前实验配置，可以运行：

```powershell
python visualize_preprocessing.py `
  --forms top3_mean `
  --region-set bone_plus_post `
  --channel-mode 1

python visualize_emd_components.py `
  --forms top3_mean `
  --region-set bone_plus_post `
  --channel-mode 1
```

预处理图中的横轴为 `Depth / mm`，纵轴为归一化 `RF Amplitude`，纵轴默认固定为 `[-1, 1]`。EMD 图中每一行对应一个 IMF 或残余项。Raw 50 Frames 会对每个帧/通道信号流分别做 EMD，并在同一分量图中叠加显示各信号流及其均值。

当前代码默认使用 NumPy，不依赖 PyTorch、SciPy、scikit-learn 或 PyEMD。若后续改用经过验证的 EMD 库，只需替换 `emd_pipeline.py` 中的 `emd_decompose()`，其余 branch、特征和 MLP 接口保持不变。

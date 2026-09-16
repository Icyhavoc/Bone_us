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
- `3`：同时保留通道 1 和 2，两个通道分别处理，不做数值相加。

通道维度和深度 branch 是相互独立的概念。

## 5. 区域组合

固定区域组合：

- `full`：`[[0, 5]]`；
- `bone`：`[[1, 3]]`；
- `bone_plus_post`：`[[0.2, 1.5], [1.5, 5.0]]`；
- `custom`：通过 `--regions-json` 自定义，可使用重叠、嵌套或不连续区域。

新增动态区域组合：

- `dyn_envelope`：对每个样本独立检测第一个显著包络峰，并据此划分两个 branch。

固定区域不要求覆盖完整 0–5 mm，也不要求 branch 等长。所有 branch 都分别处理，最后再拼接特征。

## 6. 动态包络区域 `dyn_envelope`

动态区域不会替换原始信号，只使用包络分析结果确定 branch 边界。

当前检测流程：

1. 对帧预处理和通道选择后的每条信号计算 Hilbert 包络。
2. 对信号首尾进行反射填充，降低 FFT/Hilbert 变换产生的边界伪峰。
3. 对全部信号流和物理通道的包络逐点取平均。
4. 用长度为 21 的移动平均平滑融合包络。
5. 用包络中位数估计背景基线，用 MAD 估计噪声尺度。
6. 构造自适应显著性阈值：同时考虑 `2σ` 噪声阈值和全局峰值相对基线高度的 20%。
7. 从采样点起点开始寻找第一个超过阈值且足够宽的局部峰。
8. 从峰顶向左回溯到峰值相对基线高度的 50%，将该位置定义为峰起始点。
9. 从峰起始点开始截取长度为 `a` 的第一个 branch，之后的全部信号作为第二个 branch。

当前默认参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `branch_length_mm` | `0.75 mm` | 动态 branch 1 的长度 `a` |
| `smooth_window` | `21` | 包络平滑窗口 |
| `prominence_sigma` | `2.0` | 噪声尺度倍数 |
| `min_relative_height` | `0.2` | 全局峰值相对高度比例 |
| `min_peak_width` | `12` | 候选峰最小宽度 |

当前映射下，`0.75 mm` 约等于 134 个原始采样点。可以直接通过命令行调整：

```powershell
python run_emd_experiments.py `
  --forms top3_mean `
  --region-set dyn_envelope `
  --channel-mode 1 `
  --dyn-a 0.75
```

GUI 启动训练时会从 `DynamicEnvelopeConfig().branch_length_mm` 读取默认 `a`，因此修改 `emd_pipeline.py` 中的默认值后，GUI 训练和自动可视化会同步使用新值。

每个样本实际检测到的起点、峰位置、阈值和 branch 边界会写入实验目录的 `feature_info.json`；实验参数写入 `config.json`。

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
- 每个 branch 得到 8 维特征；
- 多个 branch 之间不求均值，直接拼接。

因此：

```text
full / bone：8 维
bone_plus_post：16 维
dyn_envelope：16 维
```

`flatten` 和 `stats` 模式仍保留用于对照实验。

## 9. MLP 分类头

当前模型为 NumPy 实现的轻量 MLP，不使用 CNN：

```text
Linear(input_dim, 64)
  -> ReLU + Dropout(0.1)
  -> Linear(64, 32)
  -> ReLU + Dropout(0.1)
  -> Linear(32, 2)
```

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
  form_<form>__regions_<region>__channels_<channel>/
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

运行单组动态包络实验：

```powershell
python run_emd_experiments.py `
  --forms top3_mean `
  --region-set dyn_envelope `
  --channel-mode 1 `
  --dyn-a 0.75
```

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
- 动态 branch 两个分支均重采样到 512 点的验证；
- Tukey 窗仍保留在训练/EMD路径的验证；
- 预处理可视化不加 Tukey 窗的验证；
- EMD 分量可视化验证；
- GUI 导入和训练命令构建验证；
- 混淆矩阵和训练曲线可视化验证。

## 15. Git 发布建议

当前工作目录本身未检测到 `.git` 仓库，因此本文档是 Git 发布用的项目说明，不代表已经完成 Git commit。

发布前建议：

1. 将 `经验小波分解/` 作为代码目录纳入仓库。
2. 根据数据共享要求决定是否提交 `raw_data/`；原始数据通常不建议直接公开。
3. 根据结果文件大小决定是否提交 `experiments/` 和 `visualizations/`；它们属于可再生成产物。
4. 不提交 `__pycache__/` 和临时 smoke test 输出。
5. 在真正的 Git 根目录执行：

```powershell
git add 经验小波分解/README.md 经验小波分解/PROJECT_SUMMARY.md 经验小波分解/*.py
git commit -m "Implement EMD preprocessing experiments and GUI"
```

如果需要发布可复现实验结果，可以额外选择性提交对应实验目录的 `config.json`、`metrics.json`、`history.json` 和 `summary.json`。

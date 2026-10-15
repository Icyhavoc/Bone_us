# BMU 骨分层：ViT 分类与 EMD+MLP 对照

本目录的现行 ViT 主流程：

```text
X.npy [样本, 50 帧, 2 通道, 896 点]
  → ADC 零电平归一化
  → 按需定位双通道动态主体窗，再选择通道与帧处理方式
  → 固定深度区间或逐样本动态分区
  → 每个通道、每个分区独立三次样条重采样和 Tukey 加窗
  → 每个“分区 × 物理通道”进入独立 1D ViT 编码器
  → 拼接编码向量，由一个 k 类分类头输出
```

原有 EMD 统计特征、训练集标准化和 NumPy MLP 作为独立对照路径继续可运行。

## 数据与标签

`raw_data/train`、`val`、`test` 各含 `X.npy`、`y.npy`、`samples.json`。原始 `X.npy` 形状为 `[N,50,2,896]`，采集深度默认 0–5 mm。默认标签来自 `samples.json` 的 `depth_value`（骨头总厚度，**不是**采集深度）：厚度小于 1.0 mm 为 0，达到 1.0 mm 为 1。现有数据按采样点划分，同一标本的点可能分布在多个 split 中；因此现有 test 指标不能当作未见标本的泛化指标。

`emd_pipeline.load_split` 在内存中把代表同一 ADC 零电平的 `±0.5/127.5` 归零，不改写 `X.npy`。训练和可视化共用这个读取入口。

按其他厚度阈值重贴标签：

```powershell
python raw_data/relabel_by_thickness.py --thresholds 0.8,1.2
python raw_data/relabel_by_thickness.py --thresholds 0.8,1.2 --output-dir raw_data_relabeled/cls3_thr0.8_1.2
```

第一条只预览分布。一个阈值给二分类，两个阈值给三分类；阈值属于其右侧类别，例如 0.8 mm 属于第 1 类。默认按新标签分层重新划分 train/val/test；`--keep-split` 只重贴标签并保留原归属，其数据集 tag 追加 `_keepsplit`。输出包含 `label_scheme.json`、分组统计和样本清单。请为不同阈值或划分使用独立数据目录；源 `raw_data` 不会被覆盖。

当前保留的 1.3 mm 二分类数据集为 `raw_data_relabeled/cls2_thr1.3`（按新标签分层重划分）。GUI 启动时默认选择它；原始 `raw_data` 的 1.0 mm 标签及 CLI 默认数据目录仍保持不变。先前生成的 `cls2_thr1.3_keepsplit` 与它标签和样本池相同，但有 158/268 条样本的 split 归属不同；该副本已删除，`--keep-split` 接口仍可在需要时重新生成该划分。

## 信号与模型

帧方式：`mean_std`、`raw50`、`max1`、`top3_mean`。后两种先用完整双通道 50 帧定位各自的动态主体窗（默认 170 点），即使最终使用固定分区也按主体窗选帧。单通道按本通道主体窗内的最大绝对幅值排序；双通道先取通道 1 主体窗峰值最高的 10 帧，再按两通道相似度排序，`max1` 取相似度最高的 1 帧，`top3_mean` 取最高 3 帧并逐通道平均。相似度为同一绝对位置窗口内、允许 ±5 点位移的最大绝对去均值相关系数，乘以两通道 RMS 的 `min/max` 能量比。通道 1 峰值只用于筛选候选和同分排序，不决定最终首帧。

固定区域预置：`full` = [0,5] mm、`bone` = [1,3] mm、`bone_plus_post` = [0.2,1.5] 与 [1.5,5] mm。可用 `--regions-json` 输入自定义区间，允许重叠或不连续，少于 4 个原始采样点的区间会报错。`dyn_envelope` 按每个通道定位主体窗，尾窗紧接该通道主体窗末尾取 600 点，超过原信号末尾则右侧补零。包络只决定边界，进入模型的仍是对应有符号波形。每个分区使用 SciPy `CubicSpline` 三次样条重采样到默认 512 点，再乘 Tukey 窗。

`--channel-mode 1`、`2` 取单个物理通道；`3` 保留两个通道。通道 1 和通道 2 各自切片，**不相加或平均**。

ViT 输入为 `[样本, 分区×物理通道, 流, 512]`。`mean_std` 有 2 条流，`raw50` 有 50 条流，`max1` / `top3_mean` 各有 1 条流；这些流属于同一个样本。每路独立进行 1D patch embedding、位置编码、Transformer 和池化，拼接后由一个 k 类分类头输出。默认模型参数参考 `ViT_reference`，当前训练使用现有 train/val/test，按 val loss 早停，冻结权重后才处理 test 信号；不加载参考权重，也不翻转标签。

ViT 二分类保留普通交叉熵训练和 val loss 选模型；模型确定后，默认只在 **val** 上选择使 `TN/(TN+FP) + w×TP/(TP+FN)` 最大的概率阈值（`w=2`，同分取较高阈值），再固定用于 train/val/test 指标和混淆矩阵。两类识别率分别归一化，因此验证集类别数量不会直接决定权重；val 必须包含两类。`--positive-reward` 可修改 label=1 召回率权重（至少 1），GUI 可勾选启用并调整。关闭 GUI 开关或传入 `--no-reward` 时固定使用原始 0.5 阈值，不进行阈值搜索。普通 accuracy 仍原样报告，另报告加权双类召回率 `[TN/(TN+FP) + w×TP/(TP+FN)]/(1+w)`、正类召回率及所选阈值；关闭时该指标取 `w=1`，缺少某一类的划分显示 `n/a`。该规则只改变判定，**不改变 ViT 训练损失**。三分类及以上仍使用 argmax。旧的固定阈值实验及先前按原始正确数加权的 `__w1_...` 实验保持原样；新实验用 `__norm_w1_2p0` 或 `__fixed0p5` 区分，互不覆盖。

EMD+MLP 路径仍对每条信号最多提取 5 个 IMF 和余项，对每个分量计算 Mean、Std、RMS、Energy、绝对值均值、绝对峰值、过零率和谱质心。同一通道、同一分区内对帧流及分量求均值，得到 8 维。单通道单分区为 8 维；双通道双分区为 32 维。MLP 默认是 `输入 → 64 → 32 → k 类 softmax`，标准化器只在 train 上拟合。

## 运行

从本目录执行：

```powershell
python run_vit_experiments.py --data-dir raw_data_relabeled/cls2_thr1.3 --forms max1 --region-set dyn_envelope --channel-mode 3 --positive-reward 2
python run_vit_experiments.py --data-dir raw_data_relabeled/cls2_thr1.3 --forms max1 --region-set dyn_envelope --channel-mode 3 --no-reward
python run_emd_experiments.py --forms max1 --region-set full --channel-mode 3
python run_emd_experiments.py --data-dir raw_data_relabeled/cls3_thr0.8_1.2 --forms top3_mean --region-set dyn_envelope --channel-mode 3
python gui_app.py
```

ViT 默认运行 `max1 + dyn_envelope + 通道 3`；显式指定 `--forms all --region-set all` 可运行全部 16 个方式/区域组合。ViT 二分类实验写入 `experiments/form_<方式>__regions_<区域>__channels_<通道>__vit__headsim_cubic[__数据集tag]__norm_w1_2p0/`；关闭 reward 时后缀为 `__fixed0p5`。两者均保存配置、分支次序、逐样本分段审计、权重、判定阈值、训练历史、概率和指标；多分类目录不带判定后缀。EMD+MLP 保持原有 `__core__headsim_cubic` 目录及产物。已有目录配置不一致时拒绝混写，换设置请指定新的 `--output-dir`；小样本 `--max-samples` 仅用于链路验证，指标不可引用为模型性能。

主要入口：

| 文件 | 用途 |
| --- | --- |
| `emd_pipeline.py` | 数据加载、ADC 归零、分区、EMD、特征、标准化和 MLP |
| `vit_pipeline.py`、`vit_model.py` | 共用前处理后的 ViT 输入布局与独立分支模型 |
| `run_vit_experiments.py` | ViT 训练与保存结果（需要 PyTorch） |
| `experiment_setup.py` | 两条训练路径共用的帧、区域与输出保护配置 |
| `run_emd_experiments.py` | 训练与保存结果 |
| `raw_data/relabel_by_thickness.py` | 可变阈值重标与数据集生成 |
| `dyn_cli.py` | 动态分区参数 |
| `gui_app.py` | 选择数据集、方式、区域和通道；启动训练及查看结果 |
| `visualize_preprocessing.py`、`visualize_emd_components.py` | 信号和分量图 |
| `visualize_training_curves.py`、`visualize_confusion_matrix.py` | 当前主流程的训练曲线和混淆矩阵 |
| `verify_dyn_envelope.py` | 动态分区校验 |

GUI 默认选择 ViT 和 1.3 mm 数据集，可切换到 EMD+MLP。ViT 自动生成预处理、训练曲线和混淆矩阵图；EMD+MLP 另生成 EMD 分量图。通道 `all` 会依次运行模式 1、2、3。历史实验、原始数据和 `bin/` 归档保留在原处；旧探索脚本不属于现行入口。

项目目标与实现约束见工作区的 `../.codex/skills/bmu-emd-project/SKILL.md`，当前代码范围及历史产物说明见 `PROJECT_SUMMARY.md`。

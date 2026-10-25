# BMU 骨分层：ViT、ViT+cross-vote 与 EMD+MLP 对照

本目录的现行 ViT 主流程：

```text
X.npy [样本, 50 帧, 2 通道, 896 点]
  → ADC 零电平归一化
  → 按需定位双通道动态主体窗，再选择通道与帧处理方式
  → 固定深度区间或逐样本动态分区
  → 每个通道、每个分区独立三次样条重采样和 Tukey 加窗
  → 每个“分区 × 物理通道”进入独立 1D ViT 编码器
  → 标准 ViT 拼接编码向量后分类，或 cross-vote 各路独立分类后逐类投票
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

GUI 默认数据集为 **统一 · 二分类 · 阈值 1.3 mm · cls2_thr1.3**，目录 `raw_data_relabeled/统一_cls2_thr1.3`，独立 tag 为 `cls2_thr1.3_unified`。逐样本、逐行沿用 `ViT_reference/dataset` 的 train/validation/test（validation 映射为 val），共 161/40/67 点；薄骨/厚骨分别为 100/61、25/15、41/26，仍为 0=薄骨（<1.3 mm）、1=厚骨（≥1.3 mm）。冻结清单已与 top15 五种方法的开发/测试名单核对。

```powershell
python create_unified_dataset.py
```

构建器使用现有 `raw_data` 与仓库内 `dataset_protocols/cls2_thr1.3_unified.json`，按 sample_id 对齐，不重新随机划分，已有输出拒绝覆盖。旧 `raw_data_relabeled/cls2_thr1.3`（按新标签分层重划分）、三分类及 `raw_data` 全部保留；CLI 默认目录仍为 `raw_data`。信号逐值复制，ADC 归零在读取时进行。数组继续留在本地，Git 保存可重建脚本和固定清单。详细参数快照及数据协议见 [ViT_HYPERPARAMETERS.md](docs/ViT_HYPERPARAMETERS.md)。

## 信号与模型

帧方式：`mean_std`、`raw50`、`max1`、`top3_mean`。后两种先用完整双通道 50 帧定位各自动态主体窗（默认 170 点），固定分区也使用主体窗选帧。channel 1/2 在所选通道主体窗内按最大绝对幅值排序；**channel 3 为 Beam1 max1 + Beam2 max1，两个 Beam 独立选帧，帧号可以不同**。`top3_mean` 对两个 Beam 各自选前三帧并分别平均；`mean_std/raw50` 保留原逻辑。同幅度按原帧号升序。选中帧在各自 Beam 的主体、尾窗及可选全段共用，不跨 Beam 相加。审计中的 `channel3_selection.selected_frames_by_channel` 和 `selected_frames` 均按 `[物理通道, 选中帧]` 记录，例如 `[[3],[17]]`。新训练目录增加 `__sel_independent_beam_peak_v1`，与旧 channel3 结果隔离。

`--channel-mode 4` 保持双 Top10 交集规则：在各自主体窗内分别按最大绝对幅值选两个 Beam 的前 10 帧，再取帧号交集。`max1` 有交集则取其中 Beam1 幅值最大的同一帧；`top3_mean` 交集至少 3 帧则取 Beam1 幅值最大的三帧逐通道平均。数量不足时保留旧双通道相似度回退：在 Beam1 Top10 中，按同一绝对主体窗内、允许 ±5 点位移的最大绝对去均值相关系数乘 RMS 能量比排序；不调用新的独立选帧 channel3。审计中的历史回退名 `channel3_similarity_ranking` 指该旧算法。channel4 数值行为保持不变。

固定区域预置：`full` = [0,5] mm、`bone` = [1,3] mm、`bone_plus_post` = [0.2,1.5] 与 [1.5,5] mm。可用 `--regions-json` 输入自定义区间，允许重叠或不连续，少于 4 个原始采样点的区间会报错。`dyn_envelope` 按每个通道定位主体窗，尾窗紧接该通道主体窗末尾取 600 点，超过原信号末尾则右侧补零。包络只决定边界，进入模型的仍是对应有符号波形。每个分区使用 SciPy `CubicSpline` 三次样条重采样到默认 512 点，再乘 Tukey 窗。

`dyn_envelope_full` 在动态主体窗和动态尾窗后追加 [0,5] mm 全段分区。双通道得到“头部、尾部、全段 × 两物理通道”的 6 路输入；单通道得到 3 路。`--region-set all` 仍只遍历原有四种区域配置，不自动加入此对照区域。

`--channel-mode 1`、`2` 取单个物理通道；`3`、`4` 都保留两个通道，区别仅在选帧规则。通道 1 和通道 2 各自切片，**不相加或平均**。

ViT 输入为 `[样本, 分区×物理通道, 流, 512]`。`mean_std` 有 2 条流，`raw50` 有 50 条流，`max1` / `top3_mean` 各有 1 条流；这些流属于同一个样本。每路独立进行 1D patch embedding、位置编码、Transformer 和池化，拼接后由一个 k 类分类头输出。训练使用所选数据集的 train/val/test，按 val loss 早停，冻结权重后才处理 test 信号。

ViT 默认 `--hyperparameter-preset auto`，从参考 `top15_hyperparameters_test_accuracy_20261024` 迁移第一组参数：普通拼接用 `baseline/random_0047`；三分区拼接用 `six_channel/random_0048`；cross-vote 在 `dyn_envelope` 下用 `beam_vote/random_0096`，在 `dyn_envelope_full` 下用 `six_channel_beam_vote/random_0086`。显式 CLI 参数优先，`--hyperparameter-preset legacy` 恢复旧参数；比较不同结构使用同一套超参时必须显式传 preset 或完整参数，避免 auto 自动切换。配置保存来源及实际取值，实验名增加 `__hp_<preset>`。参考排名来自已查看的测试集，只用于参数迁移；当前逐分支 cross-vote 与参考按 Beam 融合的软投票结构和损失不同，参数相同不表示复现参考成绩。完整对照见 [ViT_HYPERPARAMETERS.md](docs/ViT_HYPERPARAMETERS.md)。

`ViT+cross-vote` 复用同一输入和独立编码器，但每路有自己的 k 类分类头；对每路 softmax 概率逐类求和，以最大和对应的标签作为结果。训练用各路**平均概率**对正确类的负对数作为损失，平均与求和的 argmax 相同；仍按 val loss 早停。`probabilities_*.npy` 保存归一化的平均概率，`branch_probabilities_*.npy` 保存逐路概率，`vote_scores_*.npy` 保存逐类求和得分。二分类也直接 argmax，不使用 reward 或验证集调阈值。

ViT 二分类保留普通交叉熵训练和 val loss 选模型；模型确定后，默认只在 **val** 上选择使 `TN/(TN+FP) + w×TP/(TP+FN)` 最大的概率阈值（`w=2`，同分取较高阈值），再固定用于 train/val/test 指标和混淆矩阵。两类识别率分别归一化，因此验证集类别数量不会直接决定权重；val 必须包含两类。`--positive-reward` 可修改 label=1 召回率权重（至少 1），GUI 可勾选启用并调整。关闭 GUI 开关或传入 `--no-reward` 时固定使用原始 0.5 阈值，不进行阈值搜索。普通 accuracy 仍原样报告，另报告加权双类召回率 `[TN/(TN+FP) + w×TP/(TP+FN)]/(1+w)`、正类召回率及所选阈值；关闭时该指标取 `w=1`，缺少某一类的划分显示 `n/a`。该规则只改变判定，**不改变 ViT 训练损失**。三分类及以上仍使用 argmax。旧的固定阈值实验及先前按原始正确数加权的 `__w1_...` 实验保持原样；新实验用 `__norm_w1_2p0` 或 `__fixed0p5` 区分，互不覆盖。

EMD+MLP 路径仍对每条信号最多提取 5 个 IMF 和余项，对每个分量计算 Mean、Std、RMS、Energy、绝对值均值、绝对峰值、过零率和谱质心。同一通道、同一分区内对帧流及分量求均值，得到 8 维。单通道单分区为 8 维；双通道双分区为 32 维。MLP 默认是 `输入 → 64 → 32 → k 类 softmax`，标准化器只在 train 上拟合。

## 运行

从本目录执行：

```powershell
python run_vit_experiments.py --data-dir raw_data_relabeled/统一_cls2_thr1.3 --forms max1 --region-set dyn_envelope --channel-mode 3 --positive-reward 2
python run_vit_experiments.py --data-dir raw_data_relabeled/统一_cls2_thr1.3 --forms max1 --region-set dyn_envelope --channel-mode 3 --no-reward
python run_vit_experiments.py --data-dir raw_data_relabeled/统一_cls2_thr1.3 --model-variant cross_vote --forms max1 --region-set dyn_envelope_full --channel-mode 4
python run_emd_experiments.py --forms max1 --region-set full --channel-mode 3
python run_emd_experiments.py --data-dir raw_data_relabeled/cls3_thr0.8_1.2 --forms top3_mean --region-set dyn_envelope --channel-mode 3
python gui_app.py
```

ViT 默认运行 `max1 + dyn_envelope + 通道 3`；`--forms all --region-set all` 可运行原有 16 个方式/区域组合。ViT 二分类开启 reward 的产物带 `__norm_w1_2p0`，关闭时带 `__fixed0p5`。cross-vote 目录用独立的 `__vit_cross_vote__headsim_cubic`，支持 `--region-set dyn_envelope`（主体＋尾窗，双 Beam 共四分支）和 `dyn_envelope_full`（追加全段，共六分支）；未显式传区域时保留原 CLI 默认 dyn_envelope_full。其余区域、自定义区域会明确报错。各分支独立分类，平均概率的负对数用于训练，逐类投票 argmax 用于推理，不启用 reward。两种 ViT 保存配置、分支次序、样本审计、权重、历史、概率及指标；channel3 增加选择版本后缀。EMD+MLP 保持独立 core 产物。已有目录配置不一致时拒绝混写，短样本检查指标仅用于链路验证。

三种 GUI 模型（ViT、ViT+cross-vote、EMD+MLP）共用逐轮学习率规则：默认预热 5 轮，再按余弦曲线降到 `5e-6`；ViT 峰值由 preset 决定（普通拼接和 cross-vote 为 `3e-4`，三分区拼接为 `1e-3`），EMD+MLP 峰值为 `1e-3`。两个入口支持 `--learning-rate`、`--min-learning-rate`、`--warmup-epochs`、`--lr-schedule {warmup_cosine,constant}`。实际学习率写入 `history.json`，配置写入 `config.json`，实验目录带学习率及 ViT 参数 preset 后缀，保留历史结果。按验证集早停时，实际训练可能尚未降到下限。

主要入口：

| 文件 | 用途 |
| --- | --- |
| `emd_pipeline.py` | 数据加载、ADC 归零、分区、EMD、特征、标准化和 MLP |
| `vit_pipeline.py`、`vit_model.py` | 共用前处理后的 ViT 输入布局、拼接分类及 cross-vote 独立分类模型 |
| `run_vit_experiments.py` | ViT 训练与保存结果（需要 PyTorch） |
| `experiment_setup.py` | 两条训练路径共用的帧、区域与输出保护配置 |
| `learning_rate_schedule.py` | 三种模型共用的预热、余弦衰减和实验标签 |
| `vit_hyperparameters.py` | ViT 参考 preset、旧参数与自动选择 |
| `create_unified_dataset.py`、`dataset_protocols/` | 按冻结参考 ID 顺序重建统一数据集 |
| `run_emd_experiments.py` | 训练与保存结果 |
| `raw_data/relabel_by_thickness.py` | 可变阈值重标与数据集生成 |
| `dyn_cli.py` | 动态分区参数 |
| `gui_app.py` | 选择数据集、方式、区域和通道；启动训练及查看结果 |
| `visualize_preprocessing.py`、`visualize_emd_components.py` | 信号和分量图 |
| `visualize_training_curves.py`、`visualize_confusion_matrix.py` | 当前主流程的训练曲线和混淆矩阵 |
| `verify_dyn_envelope.py` | 动态分区校验 |

GUI 默认选择 ViT 和统一数据集。cross-vote 的区域下拉框可选择 `dyn_envelope` 或 `dyn_envelope_full`，切换模型时保留有效选项，否则初始选择 dyn_envelope；reward 控件不可用。Summary 表格支持按列排序。ViT 自动生成预处理、训练曲线和混淆矩阵，EMD+MLP 另生成分量图；曲线只显示 Train/Validation loss。混淆矩阵左侧按当前模型筛选，右侧允许跨模型比较，只显示当前 experiments 中仍有完整配置、指标和图像的实验；图名包含完整实验目录标识。通道 all 依次运行 1、2、3、4，历史产物继续保留。

项目目标与实现约束见工作区的 `../.codex/skills/bmu-emd-project/SKILL.md`，当前代码范围及历史产物说明见 `PROJECT_SUMMARY.md`。

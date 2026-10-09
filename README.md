# BMU 骨分层：EMD 特征与 MLP 分类

本目录的现行主流程：

```text
X.npy [样本, 50 帧, 2 通道, 896 点]
  → ADC 零电平归一化
  → 选择通道与帧处理方式
  → 固定深度区间或逐样本动态分区
  → 每个通道、每个分区独立重采样和 EMD
  → 统计特征按通道与分区拼接
  → 仅用训练集拟合标准化器
  → 单个 NumPy MLP 分类
```

## 数据与标签

`raw_data/train`、`val`、`test` 各含 `X.npy`、`y.npy`、`samples.json`。原始 `X.npy` 形状为 `[N,50,2,896]`，采集深度默认 0–5 mm。默认标签来自 `samples.json` 的 `depth_value`（骨头总厚度，**不是**采集深度）：厚度小于 1.0 mm 为 0，达到 1.0 mm 为 1。现有数据按采样点划分，同一标本的点可能分布在多个 split 中；因此现有 test 指标不能当作未见标本的泛化指标。

`emd_pipeline.load_split` 在内存中把代表同一 ADC 零电平的 `±0.5/127.5` 归零，不改写 `X.npy`。训练和可视化共用这个读取入口。

按其他厚度阈值重贴标签：

```powershell
python raw_data/relabel_by_thickness.py --thresholds 0.8,1.2
python raw_data/relabel_by_thickness.py --thresholds 0.8,1.2 --output-dir raw_data_relabeled/cls3_thr0.8-1.2
```

第一条只预览分布。一个阈值给二分类，两个阈值给三分类；阈值属于其右侧类别，例如 0.8 mm 属于第 1 类。默认按新标签分层重新划分 train/val/test；`--keep-split` 只重贴标签并保留原归属，其数据集 tag 追加 `_keepsplit`。输出包含 `label_scheme.json`、分组统计和样本清单。请为不同阈值或划分使用独立数据目录；源 `raw_data` 不会被覆盖。

## 信号、特征与模型

帧方式：`mean_std`、`raw50`、`max1`、`top3_mean`。`max1` 和 `top3_mean` 按选定深度区间的综合 RMS 选帧。

固定区域预置：`full` = [0,5] mm、`bone` = [1,3] mm、`bone_plus_post` = [0.2,1.5] 与 [1.5,5] mm。可用 `--regions-json` 输入自定义区间，允许重叠或不连续。`dyn_envelope` 根据每个样本、每个通道的包络定位主窗和尾窗；包络只决定边界，进入 EMD 的仍是对应原始处理信号。每个分区默认线性重采样到 512 点，再乘 Tukey 窗。

`--channel-mode 1`、`2` 取单个物理通道；`3` 保留两个通道。通道 1 和通道 2 各自切片、EMD 和提特征，**不相加或平均**；最后将两组特征拼接，送入同一个 MLP。

每条信号默认最多提取 5 个 IMF 和余项，对每个分量计算 Mean、Std、RMS、Energy、绝对值均值、绝对峰值、过零率和谱质心。同一通道、同一分区内对帧流及分量求均值，得到 8 维。单通道单分区为 8 维；双通道双分区为 32 维。分类器默认是 `输入 → 64 → 32 → k 类 softmax`，`k` 从三个 split 的标签推断，也可显式指定 `--num-classes`。标准化器只在 train 上拟合，val 用于早停，test 用于结果与曲线展示。

## 运行

从本目录执行：

```powershell
python run_emd_experiments.py --forms max1 --region-set full --channel-mode 3
python run_emd_experiments.py --data-dir raw_data_relabeled/cls3_thr0.8-1.2 --forms top3_mean --region-set dyn_envelope --channel-mode 3
python gui_app.py
```

`--forms all --region-set all` 运行四种帧方式与四个区域方案，共 16 组；默认也是这个组合。先用一组确认参数与耗时。实验写入 `experiments/form_<方式>__regions_<区域>__channels_<通道>__core[__数据集tag]/`，包含配置、特征、模型、预测概率、指标、训练历史和分区审计。`__core` 用于区分旧探索结果。已有目录的配置不一致时会拒绝混写；要运行不同 seed 或其他参数，请指定不同 `--output-dir`。

主要入口：

| 文件 | 用途 |
| --- | --- |
| `emd_pipeline.py` | 数据加载、ADC 归零、分区、EMD、特征、标准化和 MLP |
| `run_emd_experiments.py` | 训练与保存结果 |
| `raw_data/relabel_by_thickness.py` | 可变阈值重标与数据集生成 |
| `dyn_cli.py` | 动态分区参数 |
| `gui_app.py` | 选择数据集、方式、区域和通道；启动训练及查看结果 |
| `visualize_preprocessing.py`、`visualize_emd_components.py` | 信号和分量图 |
| `visualize_training_curves.py`、`visualize_confusion_matrix.py` | 当前主流程的训练曲线和混淆矩阵 |
| `verify_dyn_envelope.py` | 动态分区校验 |

GUI 可自动运行以上四类可视化。历史实验、原始数据和 `bin/` 归档保留在原处；旧探索脚本不属于现行入口。

项目目标与实现约束见工作区的 `../.codex/skills/bmu-emd-project/SKILL.md`，当前代码范围及历史产物说明见 `PROJECT_SUMMARY.md`。

# 项目现状与范围

## 目标

用 BMU 超声 A 扫信号构建骨厚度分类主流程：ADC 零电平归一化、帧处理、深度分区和独立分支 1D ViT。ViT+cross-vote 是独立概率投票对照，EMD 统计特征、训练集标准化和 NumPy MLP 保留为独立对照。厚度分类阈值可变；原始 `raw_data` 与 CLI 默认目录采用 1.0 mm 二分类，GUI 初始选择 `raw_data_relabeled/统一_cls2_thr1.3`。训练、指标、图像和 GUI 保留 k 分类接口。

## 数据约定

- 单样本 `[50,2,896]`，分别是帧、物理通道、采集深度点。896 点默认对应 0–5 mm。
- 默认 `raw_data` 有 train/val/test。厚度取样本元数据的 `depth_value`，它与采集深度轴不同。
- `raw_data/relabel_by_thickness.py` 按一个或多个阈值生成独立数据集。默认按新标签分层重划分；`--keep-split` 保留原 split 并使用不同 tag。
- 新建“统一 · 二分类 · 阈值 1.3 mm · cls2_thr1.3”，独立目录 `统一_cls2_thr1.3`、tag `cls2_thr1.3_unified`。逐样本、逐行沿用 `ViT_reference/dataset` 的 train/validation/test，映射为 train/val/test 共 161/40/67 点；薄/厚分别 100/61、25/15、41/26。`create_unified_dataset.py` 使用版本化 ID 清单重建，拒绝覆盖已有目录；原 `raw_data`、`cls2_thr1.3` 及三分类数据均保留。信号逐值复制，归零后与参考完整信号一致。
- `load_split` 在读取时将 ADC 127/128 对应的 `±0.5/127.5` 统一归零；存储的数组不变。
- 现有划分粒度是点，同一标本可能跨 split。已有指标只说明该划分上的表现。

## 现行实现

1. 选择 `mean_std`、`raw50`、`max1` 或 `top3_mean`。后两者先在完整双通道 50 帧定位各自动态主体窗；channel3 的两个 Beam 分别按各自主体窗最大绝对幅值独立选 max1 或前三帧平均，帧号可以不同。选中帧在该 Beam 的所有分区共用。`channel3_selection` 与二维 `selected_frames` 保存逐 Beam 帧号；训练配置记录 `independent_beam_peak_v1`，目录追加同名选择版本，隔离旧结果。
   channel4 继续按各自主体窗分别取双 Beam Top10 交集，max1 取 Beam1 幅值最大的交集帧，top3_mean 要求至少三帧交集。交集不足时保留旧双通道相似度排序作为回退，不调用新 channel3；两种不选帧的形式、主体定位、尾窗补零及重采样保持原逻辑。
2. 按固定区间或 `dyn_envelope` 形成深度分区。动态主体窗定位器每样本只运行一次；尾窗从每通道主体窗末尾连续取 600 点，越过 896 点则右侧补零。包络只负责定位，不替代有符号波形输入。
3. 每个分区按通道独立切片、用 SciPy 三次样条重采样到默认 512 点、加 Tukey 窗。通道模式 3 不做通道相加。
4. ViT 输入保留 `[分区×通道, 流, 长度]`：每个分区和物理通道用独立的 patch/Transformer 编码器，拼接后由一个 k 类头分类。训练按 val loss 早停；二分类仅用 val 按 `TN/(TN+FP) + w×TP/(TP+FN)` 选择正类概率阈值（默认 w=2），也可关闭 reward 固定使用 0.5；多分类仍为 argmax。普通 accuracy 与加权双类召回率、正类召回率和阈值一起保存。`input_audit_*.json` 保存选中帧、动态边界及尾窗补零。
5. EMD+MLP 对每通道每分区提取 8 个统计量，训练集拟合标准化器，再由单个 `64 → 32 → k` MLP 分类；原入口保持可运行。
6. `dyn_envelope_full` 在动态主体和尾窗之后追加 [0,5] mm 全段，双通道共六路。ViT+cross-vote 的每路编码向量经独立分类头得到概率，逐类求和后 argmax；训练对平均概率使用负对数损失，按 val loss 早停。此模型二分类也不用 reward 或阈值搜索，结果另存 `__vit_cross_vote__` 目录，并保存每路概率、平均概率及求和得分。GUI 左侧混淆矩阵按当前模型筛选，右侧只列仍在当前 `experiments/` 中的实验，并允许跨 ViT、ViT+cross-vote、EMD+MLP 选图比较；旧 PNG 不再产生孤立选项。
7. 三条训练路径共用逐轮学习率策略：默认预热 5 轮后余弦衰减，下限 `5e-6`。ViT 的 auto preset 按对应结构使用 top15 第一组：普通拼接 `baseline/random_0047`、三分区拼接 `six_channel/random_0048`、cross-vote `six_channel_beam_vote/random_0086`；峰值分别 `3e-4/1e-3/3e-4`，MLP 仍为 `1e-3`。参数集中在 `vit_hyperparameters.py`，CLI 显式值优先，legacy preset 可恢复原值；每次配置保存来源和实际取值，实验及 summary 增加 `__hp_<preset>`。原参数与完整对照见 `docs/ViT_HYPERPARAMETERS.md`。参考排名已查看测试集，只用于探索性参数迁移；本项目前处理、损失、验证及判定规则继续沿用。

代码入口为 `emd_pipeline.py`、`vit_pipeline.py`、`vit_model.py`、`run_vit_experiments.py`、`run_emd_experiments.py` 和 `gui_app.py`。ViT 二分类新规则产物带 `__vit__headsim_cubic...__norm_w1_2p0`，关闭 reward 的固定阈值产物带 `__fixed0p5`；旧 ViT 产物保留原名，先前按计数加权的产物仍带 `__w1_...`。cross-vote 产物带 `__vit_cross_vote__headsim_cubic`，EMD+MLP 产物带 `__core__headsim_cubic`。GUI 按模型和数据集筛选，标准 ViT 可启停二分类 reward 并调整权重。预处理、训练曲线和混淆矩阵用于三条路径，EMD 分量图只用于 EMD+MLP。

## 本轮收敛

先前移除了为探索网络结构或提升单次指标而添加的 LOSO 入口、浅层模型/特征搜索、通道聚合与对比、跨分区比值、新特征族及错分厚度图。当前主线接入 ViT，但保留可变阈值、多分类、ADC 归零和独立物理通道。没有引入 EMD 融合、参考包五折协议或参考权重。历史指标与当前 ViT 输入及模型不可直接比较。

运行与参数见 `README.md`；长期约束见工作区 `../.codex/skills/bmu-emd-project/SKILL.md`。

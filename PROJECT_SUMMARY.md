# 项目现状与范围

## 目标

用 BMU 超声 A 扫信号构建骨厚度分类主流程：ADC 零电平归一化、帧处理、深度分区和独立分支 1D ViT。ViT+cross-vote 是独立概率投票对照，EMD 统计特征、训练集标准化和 NumPy MLP 保留为独立对照。厚度分类阈值可变；原始 `raw_data` 与 CLI 默认目录采用 1.0 mm 二分类，GUI 初始选择 `raw_data_relabeled/cls2_thr1.3`。训练、指标、图像和 GUI 保留 k 分类接口。

## 数据约定

- 单样本 `[50,2,896]`，分别是帧、物理通道、采集深度点。896 点默认对应 0–5 mm。
- 默认 `raw_data` 有 train/val/test。厚度取样本元数据的 `depth_value`，它与采集深度轴不同。
- `raw_data/relabel_by_thickness.py` 按一个或多个阈值生成独立数据集。默认按新标签分层重划分；`--keep-split` 保留原 split 并使用不同 tag。
- 当前保留 `cls2_thr1.3` 作为 1.3 mm 二分类数据集；同标签但沿用原 split 的 `cls2_thr1.3_keepsplit` 副本已删除。两者标签相同、样本池相同，但 train/val/test 归属不同；保留重建接口以便以后做划分对照。
- `load_split` 在读取时将 ADC 127/128 对应的 `±0.5/127.5` 统一归零；存储的数组不变。
- 现有划分粒度是点，同一标本可能跨 split。已有指标只说明该划分上的表现。

## 现行实现

1. 选择 `mean_std`、`raw50`、`max1` 或 `top3_mean`。后两者先在双通道完整 50 帧上定位各自动态主体窗：单通道按自身主体窗峰值选帧；双通道先取通道 1 峰值前 10 帧，再按同一绝对窗口内的波形相关性与 RMS 能量比的乘积排序，选最高 1 帧或最高 3 帧。最终帧索引共用于两个通道，三帧均值仍逐通道计算。
   新增 channel 4 作为双通道对照：两个通道分别在**各自主体窗**按最大绝对幅值取前 10 帧，比较帧号交集。`max1` 有交集取 beam 1 幅值最大的交集帧；`top3_mean` 交集至少 3 帧时取 beam 1 幅值最大的 3 帧均值。各自不满足数量条件时完整回退到原 channel 3 排序。`mean_std`、`raw50` 与 channel 3 一致；训练审计记录两个前 10、交集、是否回退和最终帧号。
2. 按固定区间或 `dyn_envelope` 形成深度分区。动态主体窗定位器每样本只运行一次；尾窗从每通道主体窗末尾连续取 600 点，越过 896 点则右侧补零。包络只负责定位，不替代有符号波形输入。
3. 每个分区按通道独立切片、用 SciPy 三次样条重采样到默认 512 点、加 Tukey 窗。通道模式 3 不做通道相加。
4. ViT 输入保留 `[分区×通道, 流, 长度]`：每个分区和物理通道用独立的 patch/Transformer 编码器，拼接后由一个 k 类头分类。训练按 val loss 早停；二分类仅用 val 按 `TN/(TN+FP) + w×TP/(TP+FN)` 选择正类概率阈值（默认 w=2），也可关闭 reward 固定使用 0.5；多分类仍为 argmax。普通 accuracy 与加权双类召回率、正类召回率和阈值一起保存。`input_audit_*.json` 保存选中帧、动态边界及尾窗补零。
5. EMD+MLP 对每通道每分区提取 8 个统计量，训练集拟合标准化器，再由单个 `64 → 32 → k` MLP 分类；原入口保持可运行。
6. `dyn_envelope_full` 在动态主体和尾窗之后追加 [0,5] mm 全段，双通道共六路。ViT+cross-vote 的每路编码向量经独立分类头得到概率，逐类求和后 argmax；训练对平均概率使用负对数损失，按 val loss 早停。此模型二分类也不用 reward 或阈值搜索，结果另存 `__vit_cross_vote__` 目录，并保存每路概率、平均概率及求和得分。GUI、曲线和混淆矩阵按模型类型独立筛选。

代码入口为 `emd_pipeline.py`、`vit_pipeline.py`、`vit_model.py`、`run_vit_experiments.py`、`run_emd_experiments.py` 和 `gui_app.py`。ViT 二分类新规则产物带 `__vit__headsim_cubic...__norm_w1_2p0`，关闭 reward 的固定阈值产物带 `__fixed0p5`；旧 ViT 产物保留原名，先前按计数加权的产物仍带 `__w1_...`。cross-vote 产物带 `__vit_cross_vote__headsim_cubic`，EMD+MLP 产物带 `__core__headsim_cubic`。GUI 按模型和数据集筛选，标准 ViT 可启停二分类 reward 并调整权重。预处理、训练曲线和混淆矩阵用于三条路径，EMD 分量图只用于 EMD+MLP。

## 本轮收敛

先前移除了为探索网络结构或提升单次指标而添加的 LOSO 入口、浅层模型/特征搜索、通道聚合与对比、跨分区比值、新特征族及错分厚度图。当前主线接入 ViT，但保留可变阈值、多分类、ADC 归零和独立物理通道。没有引入 EMD 融合、参考包五折协议或参考权重。历史指标与当前 ViT 输入及模型不可直接比较。

运行与参数见 `README.md`；长期约束见工作区 `../.codex/skills/bmu-emd-project/SKILL.md`。

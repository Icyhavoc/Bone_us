# -*- coding: utf-8 -*-
"""四种输入各有独立配置；修改 INPUT_INDEX 即可切换整个训练流程。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ===== 数据集选择：平时只改这一行；分界点自动读取 source_metadata.json =====
DATASET_INDEX = 1             # 当前本地数据：1.3 mm、seed42、新170/600分割
# 仅映射1.0/1.3 mm二分类；dataset中的三分类目录不接入当前研究。
DATASET_SOURCES = {
    0: Path("dataset/手动阈值_1_split_60_15_25_seed42_stratified"),
    1: Path("dataset/手动阈值_1.3_split_60_15_25_seed42_stratified"),
    2: Path("dataset/手动阈值_1.3_split_60_15_25_seed50_stratified"),
}


def get_dataset_root(index=None, project_root=None):
    index = DATASET_INDEX if index is None else index
    if index not in DATASET_SOURCES:
        raise ValueError(f"DATASET_INDEX={index} 不在 DATASET_SOURCES 中")
    path = Path(DATASET_SOURCES[index]).expanduser()
    base = ROOT if project_root is None else Path(project_root)
    return (path if path.is_absolute() else base / path).resolve()


DATASET_ROOT = get_dataset_root()
FIXED_SEQ_LEN = 512           # 所有方案固定512点，不参与搜索
TUKEY_ALPHA = 0.3             # 重采样后沿采样轴加对称Tukey窗

# 0 = 50帧 mean/std；1 = 原始50帧矩阵；2 = 原信号幅值最大1帧；3 = 原信号幅值最大3帧平均
INPUT_INDEX = 2


class CFGMeanStd:
    split_beams = True         # 每路输入同一Beam的mean、std两个通道
    input_index = 0
    input_name = "mean_std"
    dataset_root = DATASET_ROOT
    seq_len = FIXED_SEQ_LEN
    patch_size = 8
    channels = 4               # mean Beam1/2 + std Beam1/2
    frames = 1
    num_classes = 2
    dim = 128
    depth = 3
    heads = 16
    mlp_dim = 256
    dropout = 0.25
    epochs = 120
    patience = 20
    batch_size = 24
    lr = 1e-4
    weight_decay = 0.02
    label_smoothing = 0.03
    warmup = 5
    seeds = (2023, 2024, 2025)
    rank_metric = "peak"


class CFGRaw50:
    split_beams = True         # 每路保留同一Beam的全部50帧
    input_index = 1
    input_name = "raw_50"
    dataset_root = DATASET_ROOT
    seq_len = FIXED_SEQ_LEN     # 50帧逐帧重采样，保留帧与Beam两个轴
    patch_size = 8
    channels = 2
    frames = 50
    num_classes = 2
    dim = 128
    depth = 3
    heads = 16
    mlp_dim = 256
    dropout = 0.25
    epochs = 120
    patience = 20
    batch_size = 24
    lr = 1e-4
    weight_decay = 0.02
    label_smoothing = 0.03
    warmup = 5
    seeds = (2023, 2024, 2025)
    rank_metric = "peak"


class CFGMaxFrame:
    # 当前修改中的四分支结构；历史双分支实验结果不代表此结构的性能。
    split_beams = True         # 主体/尾部 × Beam1/2：四个独立ViT
    input_index = 2
    input_name = "max_frame"
    dataset_root = DATASET_ROOT
    seq_len = FIXED_SEQ_LEN
    patch_size = 8
    channels = 2
    frames = 1
    num_classes = 2
    dim = 128
    depth = 3
    heads = 16
    mlp_dim = 256
    dropout = 0.25
    epochs = 120
    patience = 20
    batch_size = 24
    lr = 0.0001
    weight_decay = 0.02
    label_smoothing = 0.03
    warmup = 5
    seeds = (2023, 2024, 2025, 2026, 2027)  # 与复核实验的五种子集成口径一致
    rank_metric = "peak"         # 完整原信号 max(abs(x))，主体/尾部共享帧索引


class CFGTop3Mean:
    split_beams = True         # 每路输入同一Beam的Top3平均波形
    # 1.3mm 默认采用已复核的 baseline 参数；此类输入仍为幅值最大三帧平均。
    # 参数来源：common_3h_2workers_20260927_145405；稳定性复核针对最大单帧。
    input_index = 3
    input_name = "top3_mean"
    dataset_root = DATASET_ROOT
    seq_len = FIXED_SEQ_LEN
    patch_size = 8
    channels = 2
    frames = 1
    num_classes = 2
    dim = 128
    depth = 3
    heads = 16
    mlp_dim = 256
    dropout = 0.25
    epochs = 120
    patience = 20
    batch_size = 24
    lr = 1e-4
    weight_decay = 0.02
    label_smoothing = 0.03
    warmup = 5
    seeds = (2023, 2024, 2025, 2026, 2027)  # 与原实验的五种子集成口径一致
    rank_metric = "peak"


# 四个类独立定义全部参数，改一个类不会修改其他方案。
CFG_LIST = (CFGMeanStd, CFGRaw50, CFGMaxFrame, CFGTop3Mean)


def get_cfg(index):
    if index not in range(len(CFG_LIST)):
        raise ValueError("INPUT_INDEX 只能为 0、1、2、3")
    return CFG_LIST[index]


CFG = get_cfg(INPUT_INDEX)

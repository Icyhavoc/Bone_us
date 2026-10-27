# -*- coding: utf-8 -*-
"""四种输入均按主体/尾部与Beam拆为四个独立Transformer，拼接特征分类。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. 单个区域的 Transformer 编码器
# =========================================================
class Branch(nn.Module):
    def __init__(self, seq_len, patch_size, channels, dim, depth, heads,
                 mlp_dim, dropout, frames=1):
        super().__init__()
        if seq_len <= 0 or patch_size <= 0 or frames < 1:
            raise ValueError("seq_len、patch_size、frames 必须为正数")
        if dim <= 0 or heads <= 0 or dim % 2 or dim % heads or depth < 1:
            raise ValueError("dim 须为正偶数且能被 heads 整除，depth 须为正数")

        self.seq_len = seq_len
        self.channels = channels
        self.frames = frames
        self.padding = (-seq_len) % patch_size
        self.patch = nn.Conv1d(channels * frames, dim, kernel_size=patch_size,
                               stride=patch_size)

        # 固定正弦/余弦位置编码，与当前 Compact ViT 一致。
        num_patches = (seq_len + self.padding) // patch_size
        position = torch.arange(num_patches)[:, None]
        frequency = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.) / dim))
        pe = torch.zeros(num_patches, dim)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("pos", pe[None])

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, signal):
        expected = ((self.frames, self.channels, self.seq_len) if self.frames > 1
                    else (self.channels, self.seq_len))
        if tuple(signal.shape[1:]) != expected:
            raise ValueError(f"输入应为 [batch, {expected}]，实际为 {tuple(signal.shape)}")
        # 按固定帧顺序展开；四分支时每个Beam保留50帧，不变成独立样本。
        if self.frames > 1:
            signal = signal.reshape(signal.shape[0], self.frames * self.channels, self.seq_len)
        # 仅在模型内补齐末尾 patch；原始矩阵不重采样、不裁剪。
        signal = F.pad(signal, (0, self.padding))
        x = self.patch(signal).transpose(1, 2)
        x = self.transformer(x + self.pos)
        return self.norm(x.mean(dim=1))


# =========================================================
# 2. 主体 + 尾部联合分类
# =========================================================
class ViT(nn.Module):
    def __init__(self, *, seq_len=512, patch_size=8, num_classes=2,
                 dim=128, depth=3, heads=4, mlp_dim=256, channels=4,
                 stats_dim=0, dropout=0.25, main_seq_len=None,
                 tail_seq_len=None, frames=1, split_beams=False):
        super().__init__()
        lengths = (seq_len if main_seq_len is None else main_seq_len,
                   seq_len if tail_seq_len is None else tail_seq_len)
        self.split_beams = split_beams
        self.channels = channels
        self.frames = frames
        if split_beams and (channels, frames) not in ((4, 1), (2, 1), (2, 50)):
            raise ValueError("四分支模型支持mean/std、单帧或原始50帧的两个Beam输入")
        self.branch_names = (("main_beam1", "main_beam2", "tail_beam1", "tail_beam2")
                             if split_beams else ("main", "tail"))
        branch_lengths = (lengths[0], lengths[0], lengths[1], lengths[1]) if split_beams else lengths
        self.branches = nn.ModuleList([
            Branch(length, patch_size, channels // 2 if split_beams else channels,
                   dim, depth, heads, mlp_dim, dropout, frames)
            for length in branch_lengths
        ])
        # 当前四方案只输入对应波形；stats_dim>0 仅保留历史模型结构兼容能力。
        self.stats = (nn.Sequential(nn.Linear(stats_dim, dim), nn.GELU(), nn.Dropout(dropout))
                      if stats_dim else None)
        self.head = nn.Sequential(
            nn.Linear(dim * (len(self.branches) + int(bool(stats_dim))), dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, main_signal, tail_signal, signal_stats=None):
        if self.split_beams:
            for signal in (main_signal, tail_signal):
                expected_ndim = 4 if self.frames > 1 else 3
                if (signal.ndim != expected_ndim or signal.shape[-2] != self.channels
                        or (self.frames > 1 and signal.shape[1] != self.frames)):
                    raise ValueError("四分支输入的帧轴或通道轴与配置不符")
            # mean/std顺序为mean B1、mean B2、std B1、std B2；按奇偶通道拆Beam。
            # raw50保留帧轴，只拆最后第二维Beam轴。
            signals = (main_signal[..., 0::2, :], main_signal[..., 1::2, :],
                       tail_signal[..., 0::2, :], tail_signal[..., 1::2, :])
            features = [branch(signal) for branch, signal in zip(self.branches, signals)]
        else:
            features = [self.branches[0](main_signal), self.branches[1](tail_signal)]
        if self.stats is not None:
            if signal_stats is None:
                raise ValueError("历史统计特征模型需要 signal_stats")
            features.append(self.stats(signal_stats))
        elif signal_stats is not None:
            raise ValueError("当前四方案模型不接收额外统计特征")
        return self.head(torch.cat(features, dim=1))

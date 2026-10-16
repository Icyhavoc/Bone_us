"""Independent 1D Transformer encoders for each region and physical channel."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ViTConfig:
    branches: int
    streams: int
    length: int
    num_classes: int
    patch_size: int = 8
    dim: int = 256
    depth: int = 3
    heads: int = 2
    mlp_dim: int = 512
    dropout: float = 0.1

    def validate(self) -> None:
        if min(self.branches, self.streams, self.length, self.num_classes,
               self.patch_size, self.dim, self.depth, self.heads, self.mlp_dim) < 1:
            raise ValueError("ViT dimensions must be positive")
        if self.num_classes < 2 or self.dim % 2 or self.dim % self.heads:
            raise ValueError("ViT needs at least two classes and an even dim divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class WaveformEncoder(nn.Module):
    """Patch embedding, fixed positional encoding, Transformer and mean pool."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.streams = config.streams
        self.length = config.length
        self.padding = (-config.length) % config.patch_size
        patches = (config.length + self.padding) // config.patch_size
        self.patch = nn.Conv1d(config.streams, config.dim, config.patch_size,
                               stride=config.patch_size)
        positions = torch.arange(patches, dtype=torch.float32)[:, None]
        frequency = torch.exp(
            torch.arange(0, config.dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / config.dim)
        )
        encoding = torch.zeros(1, patches, config.dim)
        encoding[0, :, 0::2] = torch.sin(positions * frequency)
        encoding[0, :, 1::2] = torch.cos(positions * frequency)
        self.register_buffer("position", encoding)
        layer = nn.TransformerEncoderLayer(
            d_model=config.dim, nhead=config.heads, dim_feedforward=config.mlp_dim,
            dropout=config.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.depth, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or tuple(signal.shape[1:]) != (self.streams, self.length):
            raise ValueError(
                f"expected [batch, {self.streams}, {self.length}], got {tuple(signal.shape)}"
            )
        tokens = self.patch(F.pad(signal, (0, self.padding))).transpose(1, 2)
        return self.norm(self.transformer(tokens + self.position).mean(dim=1))


class RegionChannelViT(nn.Module):
    """One encoder per (region, physical channel), followed by one k-class head.

    Input is [batch, branches, streams, length].  Streams are mean/std, the
    original 50 frames, or one selected/averaged frame of the *same* sample.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.encoders = nn.ModuleList(
            WaveformEncoder(config) for _ in range(config.branches)
        )
        self.head = nn.Sequential(
            nn.Linear(config.branches * config.dim, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, config.num_classes),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        expected = (self.config.branches, self.config.streams, self.config.length)
        if signals.ndim != 4 or tuple(signals.shape[1:]) != expected:
            raise ValueError(f"expected [batch, {expected}], got {tuple(signals.shape)}")
        encoded = [encoder(signals[:, index]) for index, encoder in enumerate(self.encoders)]
        return self.head(torch.cat(encoded, dim=1))


class RegionChannelCrossVoteViT(nn.Module):
    """Independent branch classifiers, combined by summed class probabilities."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.encoders = nn.ModuleList(
            WaveformEncoder(config) for _ in range(config.branches)
        )
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.dim, config.dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dim, config.num_classes),
            )
            for _ in range(config.branches)
        )

    def branch_probabilities(self, signals: torch.Tensor) -> torch.Tensor:
        expected = (self.config.branches, self.config.streams, self.config.length)
        if signals.ndim != 4 or tuple(signals.shape[1:]) != expected:
            raise ValueError(f"expected [batch, {expected}], got {tuple(signals.shape)}")
        logits = torch.stack([
            head(encoder(signals[:, index]))
            for index, (encoder, head) in enumerate(zip(self.encoders, self.heads))
        ], dim=1)
        return logits.softmax(dim=-1)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        # NLLLoss on the mean probability trains the same decision used at
        # inference.  Summing instead of averaging gives the same argmax.
        probabilities = self.branch_probabilities(signals).mean(dim=1)
        return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()

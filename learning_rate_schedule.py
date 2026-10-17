"""Shared epoch-level learning-rate policy for ViT and EMD+MLP training."""

from __future__ import annotations

import math


DEFAULT_LR_SCHEDULE = "warmup_cosine"
DEFAULT_MIN_LEARNING_RATE = 5e-6
DEFAULT_WARMUP_EPOCHS = 5
LR_SCHEDULES = ("warmup_cosine", "constant")


def validate_learning_rate_settings(
    peak_lr: float, min_lr: float, warmup_epochs: int, schedule: str,
) -> None:
    if (not math.isfinite(peak_lr) or not math.isfinite(min_lr)
            or not 0 < min_lr <= peak_lr):
        raise ValueError("require 0 < min learning rate <= peak learning rate")
    if warmup_epochs < 0:
        raise ValueError("warmup epochs must be nonnegative")
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown learning-rate schedule: {schedule}")


def learning_rate_for_epoch(
    epoch: int, epochs: int, peak_lr: float, min_lr: float,
    warmup_epochs: int, schedule: str,
) -> float:
    """Linearly warm up, then decay by cosine to the configured floor."""

    if not 1 <= epoch <= epochs:
        raise ValueError("epoch must be between 1 and epochs")
    validate_learning_rate_settings(peak_lr, min_lr, warmup_epochs, schedule)
    if schedule == "constant":
        return peak_lr
    warmup = min(warmup_epochs, max(epochs - 2, 0))
    if epoch <= warmup:
        return min_lr + (peak_lr - min_lr) * epoch / warmup
    decay_intervals = epochs - warmup - 1
    if decay_intervals == 0:
        return peak_lr
    progress = (epoch - warmup - 1) / decay_intervals
    return min_lr + (peak_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


def learning_rate_tag(
    peak_lr: float, min_lr: float, warmup_epochs: int, schedule: str,
) -> str:
    """Keep new runs separate from historical fixed-rate experiment outputs."""

    validate_learning_rate_settings(peak_lr, min_lr, warmup_epochs, schedule)

    def token(value: float) -> str:
        mantissa, exponent = f"{value:.11e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
        power = int(exponent)
        return f"{mantissa}e{'m' if power < 0 else 'p'}{abs(power)}"

    policy = f"warmcos{warmup_epochs}" if schedule == "warmup_cosine" else "constant"
    return f"lr_{policy}_peak{token(peak_lr)}_min{token(min_lr)}"

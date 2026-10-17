"""Verify the shared rate policy is used by every current classifier."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the spline pipeline before PyTorch on the Windows Conda runtime.
from run_vit_experiments import _fit
from emd_pipeline import MLPConfig, NumpyMLPClassifier
from learning_rate_schedule import learning_rate_for_epoch, learning_rate_tag
from vit_model import RegionChannelCrossVoteViT, RegionChannelViT, ViTConfig

import torch


class LearningRateScheduleTests(unittest.TestCase):
    def test_schedule_boundaries_and_run_tags(self) -> None:
        rates = [
            learning_rate_for_epoch(epoch, 40, 3e-4, 5e-6, 5, "warmup_cosine")
            for epoch in range(1, 41)
        ]
        self.assertAlmostEqual(rates[4], 3e-4)
        self.assertAlmostEqual(rates[-1], 5e-6)
        self.assertTrue(all(5e-6 <= rate <= 3e-4 for rate in rates))
        self.assertTrue(all(a >= b for a, b in zip(rates[4:-1], rates[5:])))
        self.assertAlmostEqual(
            learning_rate_for_epoch(2, 3, 1e-3, 5e-6, 0, "constant"), 1e-3,
        )
        self.assertEqual(
            learning_rate_tag(3e-4, 5e-6, 5, "warmup_cosine"),
            "lr_warmcos5_peak3em4_min5em6",
        )
        self.assertNotEqual(
            learning_rate_tag(3e-4, 5e-6, 5, "warmup_cosine"),
            learning_rate_tag(3e-4, 5e-6, 5, "constant"),
        )

    def test_mlp_history_records_applied_rates(self) -> None:
        x = np.random.default_rng(7).normal(size=(12, 4)).astype(np.float32)
        y = np.asarray([0, 1] * 6, dtype=np.int64)
        config = MLPConfig(
            hidden_dims=(8,), epochs=3, patience=3, batch_size=4,
            dropout=0.0, warmup_epochs=0,
        )
        history = NumpyMLPClassifier(4, config).fit(x, y, x, y)
        self.assertEqual(len(history), 3)
        self.assertAlmostEqual(history[0]["learning_rate"], 1e-3)
        self.assertAlmostEqual(history[-1]["learning_rate"], 5e-6)

    def test_both_vit_variants_record_applied_rates(self) -> None:
        signals = np.random.default_rng(3).normal(size=(4, 2, 1, 16)).astype(np.float32)
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        config = ViTConfig(
            branches=2, streams=1, length=16, num_classes=2,
            patch_size=4, dim=8, depth=1, heads=2, mlp_dim=16, dropout=0.0,
        )
        args = argparse.Namespace(
            learning_rate=3e-4, min_learning_rate=5e-6,
            warmup_epochs=0, lr_schedule="warmup_cosine",
            weight_decay=0.0, batch_size=2, seed=7,
            epochs=3, patience=3,
        )
        torch.set_num_threads(1)
        for model_type in (RegionChannelViT, RegionChannelCrossVoteViT):
            with self.subTest(model_type=model_type.__name__):
                model = model_type(config)
                history, _, _ = _fit(
                    model, signals, labels, signals, labels, args, torch.device("cpu"),
                )
                self.assertEqual(len(history), 3)
                self.assertAlmostEqual(history[0]["learning_rate"], 3e-4)
                self.assertAlmostEqual(history[-1]["learning_rate"], 5e-6)


if __name__ == "__main__":
    unittest.main()

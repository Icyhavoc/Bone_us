"""Protect channel routing, fixed baseline parameters, and vote decisions."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_vit_experiments as trainer
from emd_pipeline import DynamicEnvelopeConfig
from run_unified_tasks123 import Batch


class RemainingTasksTests(unittest.TestCase):
    def test_channel_preprocessing_caches_do_not_mix(self):
        batch = Batch.__new__(Batch)
        batch.cache, batch.test_cache = {}, {}
        batch.base = argparse.Namespace()
        batch.options = argparse.Namespace(max_samples=None)
        batch.data_dir = Path("unused")
        batch.channel_mode = 4
        with patch("run_unified_tasks123.dyn_config_from_args", return_value=DynamicEnvelopeConfig()), \
             patch.object(trainer, "_load_prepared_split", side_effect=[object(), object()]) as load:
            channel4 = batch.prepared("dyn_envelope", "train")
            batch.channel_mode = 3
            channel3 = batch.prepared("dyn_envelope", "train")
            batch.channel_mode = 4
            self.assertIs(batch.prepared("dyn_envelope", "train"), channel4)
            self.assertIsNot(channel4, channel3)
            self.assertEqual([call.kwargs["channel_mode"] for call in load.call_args_list], [4, 3])

    def test_cross_vote_fit_uses_complete_fixed_baseline_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Batch.__new__(Batch)
            batch.root = Path(temporary)
            batch.data_dir = batch.root / "data"
            batch.scheme = {"tag": "cls2_thr1.3_unified"}
            batch.device = trainer._device("cpu")
            batch.channel_mode, batch.model_variant = 3, "cross_vote"
            with patch.object(sys, "argv", ["run_vit_experiments.py", "--no-reward", "--defer-test",
                                           "--device", "cpu", "--hyperparameter-preset", "legacy"]):
                batch.base = trainer._parse_args()
            batch.prepared = lambda region, split: object()
            candidate = {"id": "baseline", "parameters": {"patch_size": 64, "dim": 96,
                "depth": 4, "heads": 8, "mlp_dim": 384, "lr": 0.0001,
                "dropout": 0, "weight_decay": 0, "batch_size": 24,
                "label_smoothing": 0.0, "warmup": 5}}
            metrics = {"metrics": {"val": {"accuracy": 0.85}}, "best_val_loss": 0.4,
                       "best_epoch": 16, "trained_epochs": 28}
            with patch.object(trainer, "run_one", return_value=metrics) as run:
                row = batch.fit("task6", "dyn_envelope", candidate, batch.root / "trial")
            args = run.call_args.args[5]
            self.assertEqual((args.model_variant, args.channel_mode, args.region_set),
                             ("cross_vote", 3, "dyn_envelope"))
            self.assertEqual((args.patch_size, args.dim, args.learning_rate, args.batch_size,
                              args.dropout, args.weight_decay), (64, 96, 0.0001, 24, 0, 0))
            self.assertTrue(args.no_reward and args.defer_test)
            self.assertIsNone(run.call_args.args[8])
            self.assertEqual(row["model_variant"], "cross_vote")

    def test_saved_cross_vote_tie_remains_argmax(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Batch.__new__(Batch)
            batch.root = Path(temporary)
            batch.rows = []
            batch.update = lambda **fields: None
            trial = batch.root / "trial"
            trial.mkdir()
            (trial / "metrics.json").write_text(json.dumps({"metrics": {"test": {"accuracy": 1.0}}}), "utf-8")
            np.save(trial / "labels_test.npy", np.array([0], dtype=np.int64))
            np.save(trial / "probabilities_test.npy", np.array([[0.5, 0.5]], dtype=np.float32))
            row = {"directory": str(trial), "model_variant": "cross_vote", "candidate": {"id": "tie"}}
            result = batch.evaluate(6, row)
            self.assertEqual(result["test"]["confusion_matrix"], [[1, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()

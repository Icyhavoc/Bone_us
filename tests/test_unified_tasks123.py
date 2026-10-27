"""Keep the label-1 stop gate exact and the search isolated from test data."""
from __future__ import annotations

import sys
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_vit_experiments as trainer
from emd_pipeline import RegionSpec
from run_unified_tasks123 import Batch, atomic_json, stop_for_label1


class ProgressWriteTests(unittest.TestCase):
    def windows_denial(self):
        error = PermissionError("simulated Windows sharing denial")
        error.winerror = 5
        return error

    def test_atomic_update_retries_transient_windows_denial(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "progress.json"
            target.write_text('{"epoch": 1}', "utf-8")
            replace = os.replace
            calls = []

            def sometimes_busy(source, destination):
                calls.append(source)
                if len(calls) < 3:
                    self.assertEqual(json.loads(target.read_text("utf-8")), {"epoch": 1})
                    raise self.windows_denial()
                replace(source, destination)

            with patch("run_unified_tasks123.os.replace", side_effect=sometimes_busy), \
                 patch("run_unified_tasks123.time.sleep"):
                atomic_json(target, {"epoch": 2})
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads(target.read_text("utf-8")), {"epoch": 2})
            self.assertEqual(list(Path(temporary).iterdir()), [target])

    def test_nontransient_failure_preserves_previous_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "progress.json"
            target.write_text('{"epoch": 1}', "utf-8")
            with patch("run_unified_tasks123.os.replace", side_effect=PermissionError("permanent denial")):
                with self.assertRaises(PermissionError):
                    atomic_json(target, {"epoch": 2})
            self.assertEqual(json.loads(target.read_text("utf-8")), {"epoch": 1})
            self.assertEqual(list(Path(temporary).iterdir()), [target])

    def test_occupied_progress_uses_fallback_without_aborting_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Batch.__new__(Batch)
            batch.root = Path(temporary)
            batch.progress = {"status": "running"}
            with patch("run_unified_tasks123.atomic_json", side_effect=[self.windows_denial(), None]) as write:
                batch.update(epoch=7)
            self.assertEqual(write.call_count, 2)
            self.assertEqual(write.call_args_list[1].args[0], batch.root / "progress_fallback.json")
            self.assertEqual(batch.progress["epoch"], 7)


class StopGateTests(unittest.TestCase):
    def test_actual_test_count_boundary(self):
        self.assertFalse(stop_for_label1({"fn": 10, "tp": 16})["stop"])
        self.assertTrue(stop_for_label1({"fn": 11, "tp": 15})["stop"])
        self.assertFalse(stop_for_label1({"fn": 2, "tp": 3})["stop"])

    def test_missing_positive_class_is_not_treated_as_success(self):
        with self.assertRaises(ValueError):
            stop_for_label1({"fn": 0, "tp": 0})

    def test_deferred_search_does_not_read_or_save_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(sys, "argv", ["run_vit_experiments.py", "--defer-test", "--no-reward", "--device", "cpu"]):
                args = trainer._parse_args()
            args.channel_mode = 1
            args.target_length = 16
            args.patch_size, args.dim, args.depth = 4, 8, 1
            args.heads, args.mlp_dim, args.dropout = 2, 16, 0
            args.epochs, args.patience, args.batch_size = 1, 1, 4
            epochs = []
            args.epoch_callback = lambda row, best: epochs.append((row["epoch"], best))
            x = np.random.default_rng(7).normal(size=(4, 1, 1, 16)).astype(np.float32)
            y = np.asarray([0, 1, 0, 1], dtype=np.int64)
            block = (x, y, [{"sample_id": str(i)} for i in range(4)],
                     [{"branch_names": ["full_ch1"]} for _ in range(4)])
            with patch.object(trainer, "_load_prepared_split", side_effect=AssertionError("unexpected source read")):
                result = trainer.run_one(root / "data", root / "output", "max1", "full",
                    [RegionSpec(0, 5)], args, 2, None, "fixed0p5",
                    prepared_splits={"train": block, "val": block})
            self.assertEqual(set(result["metrics"]), {"train", "val"})
            self.assertFalse((root / "output" / "probabilities_test.npy").exists())
            self.assertEqual(epochs, [(1, 1)])


if __name__ == "__main__":
    unittest.main()

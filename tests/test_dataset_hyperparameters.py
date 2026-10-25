"""Check exact ID alignment, source preservation, and CLI preset precedence."""

from __future__ import annotations

import hashlib
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_unified_dataset import create_dataset, reference_assignment
from run_vit_experiments import _parse_args
from gui_app import DatasetProfile, GUI_DEFAULT_DATASET_TAG
from vit_model import ViTConfig


class UnifiedDatasetTests(unittest.TestCase):
    def test_exact_order_threshold_boundary_and_existing_data_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            depth = [1.3, 0.4, 2.0, 0.7, 1.7, 1.1]
            frames = np.random.default_rng(7).normal(size=(6, 50, 2, 896)).astype(np.float32)
            for split, rows in {"train": [3, 0], "val": [5, 1], "test": [2, 4]}.items():
                directory = source / split
                directory.mkdir(parents=True)
                np.save(directory / "X.npy", frames[rows])
                np.save(directory / "y.npy", [int(depth[i] >= 1.0) for i in rows])
                np.save(directory / "indices.npy", rows)
                (directory / "samples.json").write_text(json.dumps([
                    {"sample_id": f"s{i}", "depth_value": depth[i], "label": int(depth[i] >= 1.0)}
                    for i in rows
                ]), "utf-8")
            source_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
            splits = {"train": ["s1", "s4"], "val": ["s0", "s5"], "test": ["s3", "s2"]}
            manifest = {
                "schema_version": 1, "thresholds_mm": [1.3], "num_classes": 2,
                "tag": "cls2_thr1.3_unified", "display_name": "统一 · 二分类 · 阈值 1.3 mm · cls2_thr1.3",
                "ratios": [0.6, 0.15, 0.25], "seed": 42,
                "source": "test reference", "source_files_sha256": {}, "splits": splits,
                "class_distribution": {split: {"0": 1, "1": 1} for split in splits},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), "utf-8")
            output = root / "unified"
            create_dataset(source, output, manifest_path)
            for split, ids in splits.items():
                records = json.loads((output / split / "samples.json").read_text("utf-8"))
                self.assertEqual([r["sample_id"] for r in records], ids)
                rows = [int(s[1:]) for s in ids]
                np.testing.assert_array_equal(np.load(output / split / "X.npy"), frames[rows])
                np.testing.assert_array_equal(np.load(output / split / "indices.npy"), rows)
                np.testing.assert_array_equal(np.load(output / split / "y.npy"), [int(depth[i] >= 1.3) for i in rows])
            # s0 is exactly 1.3 mm and must remain class 1.
            self.assertEqual(int(np.load(output / "val" / "y.npy")[0]), 1)
            profile = DatasetProfile(output, 2, (1.3,), GUI_DEFAULT_DATASET_TAG)
            self.assertEqual(profile.display, manifest["display_name"])
            self.assertEqual(json.loads((output / "label_scheme.json").read_text("utf-8"))["tag"], GUI_DEFAULT_DATASET_TAG)
            with self.assertRaises(FileExistsError):
                create_dataset(source, output, manifest_path)
            with self.assertRaises(ValueError):
                create_dataset(source, source / "child", manifest_path)
            for path, digest in source_hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_reject_duplicate_missing_and_unknown_ids(self) -> None:
        valid = {"train": ["a"], "val": ["b"], "test": ["c"]}
        for ids, splits in [
            (["a", "a", "c"], valid),
            (["a", "b", "c"], {**valid, "test": ["b"]}),
            (["a", "b", "c"], {**valid, "test": ["z"]}),
            (["a", "b", "c", "extra"], valid),
            (["a", "b", "c"], {**valid, "val": []}),
        ]:
            with self.subTest(ids=ids, splits=splits), self.assertRaises(ValueError):
                reference_assignment(ids, {"splits": splits})


class HyperparameterTests(unittest.TestCase):
    def parse(self, *options: str):
        with patch.object(sys, "argv", ["run_vit_experiments.py", *options]):
            return _parse_args()

    def test_auto_maps_each_architecture(self) -> None:
        standard = self.parse()
        self.assertEqual((standard.hyperparameter_preset, standard.patch_size, standard.dim,
                          standard.depth, standard.heads, standard.mlp_dim, standard.batch_size),
                         ("baseline_top1", 16, 64, 4, 8, 256, 4))
        full = self.parse("--region-set", "dyn_envelope_full")
        self.assertEqual((full.hyperparameter_preset, full.learning_rate, full.batch_size),
                         ("six_channel_top1", 1e-3, 24))
        vote = self.parse("--model-variant", "cross_vote")
        self.assertEqual((vote.hyperparameter_preset, vote.depth, vote.heads, vote.mlp_dim,
                          vote.dropout, vote.weight_decay, vote.batch_size),
                         ("six_channel_beam_vote_top1", 2, 4, 1024, 0.4, 0.0, 4))
        config = ViTConfig(branches=4, streams=1, length=512, num_classes=2)
        self.assertEqual((config.patch_size, config.dim, config.heads),
                         (standard.patch_size, standard.dim, standard.heads))

    def test_cross_vote_respects_two_region_selection(self) -> None:
        two = self.parse("--model-variant", "cross_vote", "--region-set", "dyn_envelope")
        self.assertEqual(two.region_set, "dyn_envelope")
        self.assertTrue(two.no_reward)
        self.assertEqual((two.hyperparameter_preset, two.depth, two.dropout, two.learning_rate),
                         ("beam_vote_top1", 4, 0.25, 1e-3))
        three = self.parse("--model-variant", "cross_vote", "--region-set", "dyn_envelope_full")
        self.assertEqual(three.region_set, "dyn_envelope_full")
        self.assertEqual(three.hyperparameter_preset, "six_channel_beam_vote_top1")

    def test_cross_vote_rejects_unsupported_and_custom_regions(self) -> None:
        for options in [("--region-set", "full"), ("--region-set", "all"),
                        ("--regions-json", "[[0,5]]")]:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse("--model-variant", "cross_vote", *options)

    def test_legacy_and_explicit_overrides_including_zero(self) -> None:
        legacy = self.parse("--hyperparameter-preset", "legacy")
        self.assertEqual((legacy.patch_size, legacy.dim, legacy.depth, legacy.heads,
                          legacy.mlp_dim, legacy.batch_size), (8, 256, 3, 2, 512, 8))
        args = self.parse("--model-variant", "cross_vote", "--batch-size", "7",
                          "--dropout", "0", "--weight-decay", "0", "--warmup-epochs", "0")
        self.assertEqual((args.batch_size, args.dropout, args.weight_decay, args.warmup_epochs),
                         (7, 0.0, 0.0, 0))


if __name__ == "__main__":
    unittest.main()

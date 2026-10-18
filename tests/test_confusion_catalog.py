"""Confusion selectors must reflect current experiments across model families."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emd_pipeline import PREPROCESSING_TAG
from experiment_setup import confusion_image_name
from gui_app import DatasetProfile, ExperimentCatalog


class ConfusionCatalogTests(unittest.TestCase):
    def test_stale_images_are_hidden_and_models_are_named(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            root = Path(temporary)
            experiments = root / "experiments"
            images = root / "visualizations" / "confusion_matrices" / "cls2_thr1.3"
            experiments.mkdir()
            images.mkdir(parents=True)
            profile = DatasetProfile(root / "dataset", 2, (1.3,), "cls2_thr1.3")
            catalog = ExperimentCatalog(experiments, root / "visualizations", profile)

            def add_experiment(model: str, suffix: str = "") -> tuple[Path, Path]:
                token = {"emd": "core", "vit": "vit", "vit_cross_vote": "vit_cross_vote"}[model]
                name = (
                    f"form_max1__regions_dyn_envelope_full__channels_4"
                    f"__{token}__{PREPROCESSING_TAG}__cls2_thr1.3{suffix}"
                )
                directory = experiments / name
                directory.mkdir()
                (directory / "config.json").write_text(json.dumps({
                    "feature_layout": PREPROCESSING_TAG, "model_family": model,
                    "label_scheme": {"tag": "cls2_thr1.3"},
                    "num_classes": 2, "form": "max1",
                    "region_name": "dyn_envelope_full", "channel_mode": 4,
                }), encoding="utf-8")
                (directory / "metrics.json").write_text(json.dumps({
                    "feature_dim": 12, "best_epoch": 2, "trained_epochs": 3,
                    "metrics": {"test": {"accuracy": 0.5}},
                }), encoding="utf-8")
                image = images / confusion_image_name(name, "test")
                image.touch()
                return directory, image

            vit_dir, vit_image = add_experiment("vit")
            _, vote_image = add_experiment("vit_cross_vote", "__lr_warmcos5_peak3em4_min5em6")
            _, emd_image = add_experiment("emd")
            orphan = images / "form_top3_mean__regions_full__channels_1__split_test__vit.png"
            orphan.touch()

            catalog.reload()
            options = catalog.confusion_matrix_options()
            self.assertEqual(set(options.values()), {vit_image, vote_image, emd_image})
            self.assertTrue(any(key.startswith("model=ViT |") for key in options))
            self.assertTrue(any(key.startswith("model=ViT+cross-vote |") for key in options))
            self.assertTrue(any(key.startswith("model=EMD+MLP |") for key in options))
            self.assertTrue(any("run=lr_warmcos5_peak3em4_min5em6" in key for key in options))
            self.assertEqual(
                catalog.confusion_matrix_files(
                    "max1", "dyn_envelope_full", "4", model_family="vit_cross_vote",
                ),
                [vote_image],
            )

            archive = root / "archive"
            archive.mkdir()
            vit_dir.rename(archive / vit_dir.name)
            catalog.reload()
            self.assertNotIn(vit_image, catalog.confusion_matrix_options().values())


if __name__ == "__main__":
    unittest.main()

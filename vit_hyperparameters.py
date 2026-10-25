"""ViT defaults from the 2026-10-24 reference shortlist; no runtime dependency."""

from __future__ import annotations

from typing import Any


REFERENCE_PACKAGE = "ViT_reference/top15_hyperparameters_test_accuracy_20261024"
LEGACY_PARAMETERS = {
    "patch_size": 8, "dim": 256, "depth": 3, "heads": 2,
    "mlp_dim": 512, "dropout": 0.1, "learning_rate": 3e-4,
    "weight_decay": 0.05, "batch_size": 8, "warmup_epochs": 5,
}
PRESETS: dict[str, dict[str, Any]] = {
    "legacy": {"parameters": LEGACY_PARAMETERS, "reference": None},
    "baseline_top1": {
        "parameters": {**LEGACY_PARAMETERS, "patch_size": 16, "dim": 64,
                       "depth": 4, "heads": 8, "mlp_dim": 256, "batch_size": 4},
        "reference": {"file": "01_baseline_top15.json", "method": "baseline",
                      "id": "random_0047", "position": 1},
    },
    "six_channel_top1": {
        "parameters": {**LEGACY_PARAMETERS, "depth": 4, "heads": 4,
                       "mlp_dim": 1024, "learning_rate": 1e-3,
                       "weight_decay": 0.02, "batch_size": 24},
        "reference": {"file": "03_six_channel_top15.json", "method": "six_channel",
                      "id": "random_0048", "position": 1},
    },
    "beam_vote_top1": {
        "parameters": {**LEGACY_PARAMETERS, "depth": 4, "heads": 4,
                       "learning_rate": 1e-3, "dropout": 0.25,
                       "weight_decay": 0.01, "batch_size": 4},
        "reference": {"file": "04_beam_vote_top15.json", "method": "beam_vote",
                      "id": "random_0096", "position": 1},
    },
    "six_channel_beam_vote_top1": {
        "parameters": {**LEGACY_PARAMETERS, "depth": 2, "heads": 4,
                       "mlp_dim": 1024, "dropout": 0.4,
                       "weight_decay": 0.0, "batch_size": 4},
        "reference": {"file": "05_six_channel_beam_vote_top15.json",
                      "method": "six_channel_beam_vote", "id": "random_0086",
                      "position": 1},
    },
}
DEFAULT_MODEL_PARAMETERS = PRESETS["baseline_top1"]["parameters"]


def resolve_hyperparameters(args: Any) -> None:
    """Fill unspecified CLI options; explicit options always take precedence."""
    preset = args.hyperparameter_preset
    if preset == "auto":
        if args.model_variant == "cross_vote":
            preset = ("six_channel_beam_vote_top1" if args.region_set == "dyn_envelope_full"
                      else "beam_vote_top1")
        else:
            preset = "six_channel_top1" if args.region_set == "dyn_envelope_full" else "baseline_top1"
    args.hyperparameter_preset = preset
    for key, value in PRESETS[preset]["parameters"].items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def hyperparameter_reference(preset: str) -> dict[str, Any] | None:
    reference = PRESETS[preset]["reference"]
    return None if reference is None else {"package": REFERENCE_PACKAGE, **reference}

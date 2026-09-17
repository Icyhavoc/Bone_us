"""Run EMD + MLP experiments for the prepared BMU dataset.

Examples from the ``经验小波分解`` directory::

    python run_emd_experiments.py --forms all --region-set all --channel-mode 3
    python run_emd_experiments.py --forms max1 --region-set full --max-samples 4 --epochs 2
    python run_emd_experiments.py --forms raw50 --regions-json "[[0,5],[1,3]]"

The default ``--region-set all`` runs the 4 frame forms against the three
fixed region configurations and the dynamic envelope configuration, for 16
experiments in total.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from emd_pipeline import (
    DepthMapper,
    EMDConfig,
    FORM_ORDER,
    MLPConfig,
    RegionSpec,
    StandardScaler,
    ADC_DC_MAGNITUDE,
    binary_metrics,
    build_feature_matrix,
    canonical_form,
    json_ready,
    load_split,
    parse_regions,
    write_json,
    NumpyMLPClassifier,
)
from dyn_cli import add_dyn_arguments, dyn_config_from_args


REGION_PRESETS: dict[str, list[list[float]]] = {
    "full": [[0.0, 5.0]],
    "bone": [[1.0, 3.0]],
    "bone_plus_post": [[0.2, 1.5], [1.5, 5.0]],
}
DYNAMIC_REGION_NAME = "dyn_envelope"
REGION_CHOICES = ["all", *REGION_PRESETS.keys(), DYNAMIC_REGION_NAME]
def _parse_forms(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(FORM_ORDER)
    forms = [canonical_form(item) for item in value.split(",") if item.strip()]
    if not forms:
        raise ValueError("At least one frame form is required")
    # Keep the canonical order and remove duplicates.
    return [form for form in FORM_ORDER if form in forms]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EMD feature extraction + MLP classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--forms", default="all", help="all or comma-separated forms")
    parser.add_argument(
        "--region-set",
        choices=REGION_CHOICES,
        default="all",
        help="standard region presets; ignored when --regions-json is supplied",
    )
    parser.add_argument(
        "--regions-json",
        default=None,
        help='custom depth regions, e.g. "[[0,5],[1,3]]"; regions may overlap or nest',
    )
    parser.add_argument("--channel-mode", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--tukey-alpha", type=float, default=0.3)
    # dyn_envelope mirrors reference_code/pipeline.py; see dyn_cli for the flags.
    add_dyn_arguments(parser)
    parser.add_argument("--max-depth-mm", type=float, default=5.0)
    parser.add_argument("--signal-length", type=int, default=896)
    parser.add_argument("--rounding", choices=["round", "floor", "ceil"], default="round")
    parser.add_argument("--selection-region-json", default="[[0,5]]")
    parser.add_argument("--max-imfs", type=int, default=5)
    parser.add_argument("--max-sift-iterations", type=int, default=30)
    parser.add_argument("--sift-sd-threshold", type=float, default=0.2)
    parser.add_argument(
        "--stream-aggregation",
        choices=["pooled", "flatten", "stats"],
        default="pooled",
        help="pooled averages EMD components/streams within each branch, then concatenates branches",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--hidden-dims", default="64,32")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="limit every split for a quick smoke test; omit for the full dataset",
    )
    return parser.parse_args()


def _region_experiments(args: argparse.Namespace) -> list[tuple[str, list[RegionSpec]]]:
    if args.regions_json is not None:
        return [("custom", parse_regions(args.regions_json))]
    if args.region_set == "all":
        names = [*REGION_PRESETS.keys(), DYNAMIC_REGION_NAME]
    else:
        names = [args.region_set]
    return [
        (name, [] if name == DYNAMIC_REGION_NAME else parse_regions(REGION_PRESETS[name]))
        for name in names
    ]


def _experiment_name(form: str, region_name: str, channel_mode: int) -> str:
    return f"form_{form}__regions_{region_name}__channels_{channel_mode}"


def _best_history_record(
    history: list[dict[str, object]], min_delta: float = 1e-4
) -> dict[str, object] | None:
    """Select the checkpoint criterion used by the MLP."""

    if not history:
        return None
    best: dict[str, object] | None = None
    best_auc = float("-inf")
    best_loss = float("inf")
    for record in history:
        auc_value = record.get("val_auc")
        loss_value = float(record.get("val_loss", float("inf")))
        auc = float("-inf") if auc_value is None else float(auc_value)
        improved = auc > best_auc + min_delta
        if not improved and auc == best_auc and loss_value < best_loss - min_delta:
            improved = True
        if improved:
            best = record
            best_auc = auc
            best_loss = loss_value
    return best


def run_one(
    data_dir: Path,
    output_dir: Path,
    form: str,
    region_name: str,
    regions: Sequence[RegionSpec],
    args: argparse.Namespace,
) -> dict[str, object]:
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    mapper = DepthMapper(
        max_depth_mm=args.max_depth_mm,
        signal_length=args.signal_length,
        rounding=args.rounding,
    )
    selection_region = parse_regions(args.selection_region_json)[0]
    emd_config = EMDConfig(
        max_imfs=args.max_imfs,
        max_sift_iterations=args.max_sift_iterations,
        sift_sd_threshold=args.sift_sd_threshold,
        stream_aggregation=args.stream_aggregation,
    )
    dynamic_envelope_config = (
        dyn_config_from_args(args) if region_name == DYNAMIC_REGION_NAME else None
    )
    hidden_dims = tuple(int(item) for item in args.hidden_dims.split(",") if item.strip())
    mlp_config = MLPConfig(
        hidden_dims=hidden_dims,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    experiment_config = {
        "form": form,
        "region_name": region_name,
        "regions": regions,
        "channel_mode": args.channel_mode,
        "target_length": args.target_length,
        "tukey_alpha": args.tukey_alpha,
        "dynamic_envelope": dynamic_envelope_config,
        "depth_mapper": mapper,
        "selection_region": selection_region,
        "emd": emd_config,
        "mlp": mlp_config,
        "data_dir": data_dir,
        "adc_dc_replacement": ADC_DC_MAGNITUDE,
        "max_samples": args.max_samples,
    }
    write_json(output_dir / "config.json", experiment_config)

    split_data: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, object]]]] = {}
    feature_data: dict[str, np.ndarray] = {}
    feature_info: dict[str, object] = {}
    for split in ("train", "val", "test"):
        x, y, samples = load_split(data_dir, split)
        if args.max_samples is not None:
            y = y[: args.max_samples]
            samples = samples[: args.max_samples]
        split_data[split] = (x, y, samples)
        print(f"[{form}/{region_name}] extracting {split}: {x.shape[0]} samples")
        features, info = build_feature_matrix(
            x,
            form=form,
            regions=regions,
            channel_mode=args.channel_mode,
            mapper=mapper,
            target_length=args.target_length,
            tukey_alpha=args.tukey_alpha,
            dynamic_envelope_config=dynamic_envelope_config,
            emd_config=emd_config,
            score_region=selection_region,
            limit=args.max_samples,
            sample_ids=[str(record.get("sample_id", "")) for record in samples],
        )
        feature_data[split] = features
        feature_info[split] = info
        np.save(output_dir / f"features_{split}.npy", features)
        np.save(output_dir / f"labels_{split}.npy", y)
        write_json(output_dir / f"samples_{split}.json", samples)

    train_features = feature_data["train"]
    scaler = StandardScaler().fit(train_features)
    scaled = {split: scaler.transform(features) for split, features in feature_data.items()}
    scaler.save(output_dir / "standard_scaler.npz")
    model = NumpyMLPClassifier(input_dim=train_features.shape[1], config=mlp_config)
    history = model.fit(
        scaled["train"],
        split_data["train"][1],
        scaled["val"],
        split_data["val"][1],
        scaled["test"],
        split_data["test"][1],
    )
    model.save(output_dir / "mlp_model.npz")
    write_json(output_dir / "history.json", history)
    best_record = _best_history_record(history, min_delta=mlp_config.min_delta)

    metrics: dict[str, object] = {
        "feature_dim": int(train_features.shape[1]),
        "best_epoch": None if best_record is None else best_record.get("epoch"),
        "trained_epochs": 0 if not history else history[-1].get("epoch"),
        "best_val_auc": None if best_record is None else best_record.get("val_auc"),
        "best_val_loss": None if best_record is None else best_record.get("val_loss"),
        "split_counts": {split: int(split_data[split][1].size) for split in split_data},
        "class_counts": {
            split: {
                "0": int(np.sum(split_data[split][1] == 0)),
                "1": int(np.sum(split_data[split][1] == 1)),
            }
            for split in split_data
        },
        "metrics": {},
    }
    for split in ("train", "val", "test"):
        probabilities = model.predict_proba(scaled[split])
        metrics["metrics"][split] = binary_metrics(split_data[split][1], probabilities)
        np.save(output_dir / f"probabilities_{split}.npy", probabilities)
    write_json(output_dir / "feature_info.json", feature_info)
    write_json(output_dir / "metrics.json", metrics)
    elapsed = time.time() - start_time
    print(
        f"[{form}/{region_name}] done: feature_dim={train_features.shape[1]}, "
        f"test_auc={metrics['metrics']['test']['auc']}, time={elapsed:.1f}s"
    )
    return metrics


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    forms = _parse_forms(args.forms)
    region_experiments = _region_experiments(args)
    all_results: dict[str, object] = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "forms": forms,
        "regions": {name: regions for name, regions in region_experiments},
        "channel_mode": args.channel_mode,
        "results": {},
    }
    for form in forms:
        for region_name, regions in region_experiments:
            experiment_name = _experiment_name(form, region_name, args.channel_mode)
            experiment_dir = output_dir / experiment_name
            result = run_one(
                data_dir=data_dir,
                output_dir=experiment_dir,
                form=form,
                region_name=region_name,
                regions=regions,
                args=args,
            )
            all_results["results"][experiment_name] = result
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", all_results)
    print(f"All experiments completed. Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

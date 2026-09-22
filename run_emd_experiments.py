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
    BRANCH_RATIO_PRESETS,
    CHANNEL_CONTRAST_KINDS,
    DepthMapper,
    EMD_FEATURE_PRESETS,
    EMDConfig,
    FORM_ORDER,
    LOCATOR_FEATURE_PRESETS,
    MLPConfig,
    RegionSpec,
    StandardScaler,
    ADC_DC_MAGNITUDE,
    build_feature_matrix,
    canonical_form,
    classification_metrics,
    feature_dimension_names,
    json_ready,
    label_scheme_tag,
    load_split,
    parse_regions,
    read_label_scheme,
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
    # dyn_envelope mirrors bin/reference_code/pipeline.py; see dyn_cli for the flags.
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
    parser.add_argument(
        "--channel-aggregation",
        choices=["per_channel", "pooled"],
        default="per_channel",
        help=(
            "per_channel keeps one feature group per selected channel and gives each "
            "group its own MLP tower; pooled averages the channels into a single group "
            "(the historical channel-3 behaviour)"
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=sorted(EMD_FEATURE_PRESETS),
        default="compact",
        help=(
            "per-IMF statistic set: legacy keeps the original 8 (including the "
            "mathematically zero 'mean' of an IMF), compact drops 'mean', lean also "
            "drops 'energy'"
        ),
    )
    parser.add_argument(
        "--locator-features",
        choices=sorted(LOCATOR_FEATURE_PRESETS),
        default="core",
        help=(
            "depth-locator block appended to the envelope main window: core adds the "
            "merged onset and the envelope peak in mm, full adds the weak threshold "
            "crossing, rise time, merge extension and peak amplitude, none disables it"
        ),
    )
    parser.add_argument(
        "--channel-contrast",
        choices=list(CHANNEL_CONTRAST_KINDS),
        default="none",
        help=(
            "append one extra feature group contrasting the first two channels "
            "elementwise (S4): 'normalized' is (a-b)/(|a|+|b|), invariant to a common "
            "gain, 'log_ratio' is the signed log of the attenuation ratio, "
            "'difference' keeps the raw units; 'none' leaves the layout untouched"
        ),
    )
    parser.add_argument(
        "--branch-ratio",
        choices=sorted(BRANCH_RATIO_PRESETS),
        default="none",
        help=(
            "append cross-branch contrast columns inside every channel group (S3b): "
            "'ratio' divides the deep window's statistics by the shallow window's, "
            "'attenuation' takes the same contrasts in log form; both add one "
            "separation-normalised attenuation coefficient in 1/mm. Requires a "
            "region plan with two or more windows; 'none' leaves the layout untouched"
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--hidden-dims", default="64,32")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=(
            "number of classes in the target; omit to infer it from the labels "
            "of every split (2 for the default depth>=1.0mm labels)"
        ),
    )
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


def _experiment_name(
    form: str,
    region_name: str,
    channel_mode: int,
    channel_aggregation: str = "per_channel",
    label_tag: str | None = None,
    channel_contrast: str = "none",
    branch_ratio: str = "none",
) -> str:
    """Directory name for one experiment.

    The default ``per_channel`` aggregation keeps the historical name so that
    existing directories still line up.  ``pooled`` (the legacy channel-3
    behaviour, now used as an ablation baseline) gets a suffix, otherwise it
    would silently overwrite the ``per_channel`` run in the same directory.

    A dataset that ships a ``label_scheme.json`` (i.e. one produced by
    ``raw_data/relabel_by_thickness.py``) appends its tag for the same reason:
    a 3-class run must not land on top of the binary results it is meant to be
    compared with.

    A cross-channel contrast is likewise its own experiment -- it adds a feature
    group and therefore a classifier tower -- so it gets a tag too.  Without one,
    a ``channels_3`` contrast run would overwrite the plain ``channels_3``
    baseline it exists to be compared against.

    A cross-branch ratio adds columns but no tower, so it would not change the
    model layout; it still gets a tag, because its feature dimension differs and a
    silent mix-up would be impossible to spot from the metrics alone.
    """
    name = f"form_{form}__regions_{region_name}__channels_{channel_mode}"
    if channel_aggregation != "per_channel":
        name = f"{name}__{channel_aggregation}"
    if channel_contrast != "none":
        name = f"{name}__contrast_{channel_contrast}"
    if branch_ratio != "none":
        name = f"{name}__branchratio_{branch_ratio}"
    if label_tag:
        name = f"{name}__{label_tag}"
    return name


def _dataset_tag(data_dir: Path) -> str | None:
    """Label-scheme tag of a relabelled dataset, sanitised for a directory name."""

    return label_scheme_tag(_read_label_scheme(data_dir))


def _feature_group_dims(info: object) -> tuple[int, ...]:
    """Read the per-group feature widths recorded by the extractor.

    The widths tell the classifier where one channel's features end and the
    next one's begin.  A single entry means the channels were pooled and the
    network stays a plain MLP.
    """

    if not isinstance(info, list) or not info:
        raise ValueError("feature info is empty; cannot determine the feature groups")
    first = info[0]
    if not isinstance(first, dict):
        raise ValueError("feature info entries must be mappings")
    dims = first.get("feature_group_dims")
    if not dims:
        raise ValueError("feature info is missing 'feature_group_dims'")
    return tuple(int(width) for width in dims)


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


def _infer_num_classes(data_dir: Path, max_samples: int | None = None) -> int:
    """Smallest class count that covers every label in every split.

    Only ``y.npy`` is read here, so this costs microseconds compared with the
    EMD feature extraction that follows and lets the MLP head be sized before
    any of that work is done.
    """

    highest = -1
    for split in ("train", "val", "test"):
        path = data_dir / split / "y.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing label file: {path}")
        labels = np.load(path)
        if max_samples is not None:
            labels = labels[:max_samples]
        if labels.size:
            highest = max(highest, int(labels.max()))
    if highest < 0:
        raise ValueError(f"no labels found under {data_dir}")
    return highest + 1


def _read_label_scheme(data_dir: Path) -> dict[str, object] | None:
    """The threshold scheme recorded by ``raw_data/relabel_by_thickness.py``.

    Thin wrapper around :func:`emd_pipeline.read_label_scheme` so that the
    visualisation scripts and the training script agree on both the file name
    and the "absent file means the historical 1.0 mm binary split" rule.
    """

    return read_label_scheme(data_dir)


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
        channel_aggregation=args.channel_aggregation,
        feature_names=EMD_FEATURE_PRESETS[args.feature_set],
        locator_features=args.locator_features,
        channel_contrast=args.channel_contrast,
        branch_ratio=args.branch_ratio,
    )
    dynamic_envelope_config = (
        dyn_config_from_args(args) if region_name == DYNAMIC_REGION_NAME else None
    )
    hidden_dims = tuple(int(item) for item in args.hidden_dims.split(",") if item.strip())
    inferred_classes = _infer_num_classes(data_dir, args.max_samples)
    if args.num_classes is not None and args.num_classes < inferred_classes:
        raise ValueError(
            f"--num-classes {args.num_classes} cannot hold the labels in {data_dir}, "
            f"which need at least {inferred_classes} classes"
        )
    num_classes = inferred_classes if args.num_classes is None else int(args.num_classes)
    mlp_config = MLPConfig(
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )
    label_scheme = _read_label_scheme(data_dir)
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
        "num_classes": num_classes,
        "label_scheme": label_scheme,
        "adc_dc_replacement": ADC_DC_MAGNITUDE,
        "max_samples": args.max_samples,
    }
    write_json(output_dir / "config.json", experiment_config)
    print(
        f"[{form}/{region_name}] data_dir={data_dir.name}, num_classes={num_classes}"
        + (
            ""
            if label_scheme is None
            else f", thresholds_mm={label_scheme.get('thresholds_mm')}"
        )
    )

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
    # A degenerate column carries no variation, so the classifier sees a constant.
    # Name them: an unexpected entry here means the feature set is not what it
    # was thought to be.
    dimension_names = feature_dimension_names(feature_info["train"][0], emd_config)
    degenerate_indices = scaler.degenerate_indices
    if degenerate_indices:
        named = ", ".join(
            dimension_names[index] if index < len(dimension_names) else f"dim{index}"
            for index in degenerate_indices
        )
        print(f"[{form}/{region_name}] constant feature columns (mapped to 0): {named}")
    feature_group_dims = _feature_group_dims(feature_info["train"])
    model = NumpyMLPClassifier(
        input_dim=train_features.shape[1],
        config=mlp_config,
        group_dims=feature_group_dims,
    )
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
        "feature_group_dims": [int(width) for width in feature_group_dims],
        "feature_dimension_names": dimension_names,
        "degenerate_feature_indices": [int(index) for index in degenerate_indices],
        "degenerate_feature_names": [
            dimension_names[index] if index < len(dimension_names) else f"dim{index}"
            for index in degenerate_indices
        ],
        "model_layout": model.parameter_layout(),
        "best_epoch": None if best_record is None else best_record.get("epoch"),
        "trained_epochs": 0 if not history else history[-1].get("epoch"),
        "best_val_auc": None if best_record is None else best_record.get("val_auc"),
        "best_val_loss": None if best_record is None else best_record.get("val_loss"),
        "split_counts": {split: int(split_data[split][1].size) for split in split_data},
        "num_classes": num_classes,
        "label_scheme": label_scheme,
        "class_counts": {
            split: {
                str(label): int(np.sum(split_data[split][1] == label))
                for label in range(num_classes)
            }
            for split in split_data
        },
        "metrics": {},
    }
    for split in ("train", "val", "test"):
        probabilities = model.predict_proba(scaled[split])
        metrics["metrics"][split] = classification_metrics(
            split_data[split][1], probabilities, num_classes
        )
        np.save(output_dir / f"probabilities_{split}.npy", probabilities)
    write_json(output_dir / "feature_info.json", feature_info)
    write_json(output_dir / "metrics.json", metrics)
    elapsed = time.time() - start_time
    print(
        f"[{form}/{region_name}] done: feature_dim={train_features.shape[1]}, "
        f"num_classes={num_classes}, test_accuracy={metrics['metrics']['test']['accuracy']}, "
        f"test_auc={metrics['metrics']['test']['auc']}, time={elapsed:.1f}s"
    )
    return metrics


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    forms = _parse_forms(args.forms)
    region_experiments = _region_experiments(args)
    dataset_tag = _dataset_tag(data_dir)
    all_results: dict[str, object] = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "forms": forms,
        "regions": {name: regions for name, regions in region_experiments},
        "channel_mode": args.channel_mode,
        "dataset_tag": dataset_tag,
        "label_scheme": _read_label_scheme(data_dir),
        "results": {},
    }
    for form in forms:
        for region_name, regions in region_experiments:
            experiment_name = _experiment_name(
                form,
                region_name,
                args.channel_mode,
                args.channel_aggregation,
                dataset_tag,
                args.channel_contrast,
                args.branch_ratio,
            )
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

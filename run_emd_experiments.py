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
import json
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
    threshold_scan,
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
# The threshold convention is a reporting choice, not a modelling one: it is
# stored for every supported variant so a reader can see how far the headline
# moves when it changes.  "fixed" is the historical 0.5 rule and is the default
# only for the stored artefacts' sake -- "val" is the defensible live choice.
THRESHOLD_SOURCES = ("val", "train", "fixed")
DEFAULT_THRESHOLD_SOURCE = "val"
def _parse_forms(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(FORM_ORDER)
    forms = [canonical_form(item) for item in value.split(",") if item.strip()]
    if not forms:
        raise ValueError("At least one frame form is required")
    # Keep the canonical order and remove duplicates.
    return [form for form in FORM_ORDER if form in forms]


def _parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def build_parser() -> argparse.ArgumentParser:
    """The full experiment parser, exposed so other entry points can inherit it.

    ``run_loso_evaluation`` builds its parser with this one as a ``parent`` and
    ``conflict_handler="resolve"``.  That is the only way to guarantee the two
    scripts agree on every feature-related default: a copied list of flags would
    drift the moment either file gained an option, and a drifted default is
    invisible in the results -- it just silently describes a different feature
    vector.

    Nothing here parses anything, so importing this module stays side-effect
    free.
    """

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
        "--threshold-source",
        choices=THRESHOLD_SOURCES,
        default=DEFAULT_THRESHOLD_SOURCE,
        help=(
            "which split chooses the binary decision threshold: 'val' (default, the "
            "textbook choice -- fitted on data the model did not train on, still not "
            "the scored split), 'train' (in-sample, can overshoot), or 'fixed' for the "
            "plain 0.5 rule. Every variant is stored under 'decision_threshold' "
            "regardless of this flag, so the sensitivity of the headline to the "
            "threshold convention is always visible"
        ),
    )
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
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help=(
            "overwrite an existing experiment directory even when its recorded "
            "data_dir/regions/num_classes/label_scheme differ from this run. "
            "Without the flag such a collision aborts: the directory name does not "
            "encode the dataset variant, so the previous results would be lost "
            "without a trace"
        ),
    )
    return parser


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


SPECIMEN_FIELD_INDEX = 0


def specimen_ids(samples: Sequence[dict[str, object]]) -> np.ndarray:
    """Specimen (bone) identifier of every sample.

    Read from ``point_id``, which is ``<specimen>_<point>[_<channel>]``.  It is
    deliberately *not* ``sample_id``: that one is unique per sample, so grouping
    on it would quietly degenerate back into the point-level split this is meant
    to expose.
    """

    ids: list[str] = []
    for record in samples:
        raw = record.get("point_id")
        if not raw:
            raise ValueError("sample record has no 'point_id'; cannot group by specimen")
        ids.append(str(raw).split("_")[SPECIMEN_FIELD_INDEX])
    return np.asarray(ids, dtype=object)


def _label_counts(
    labels: np.ndarray, specimens: np.ndarray, num_classes: int
) -> dict[str, np.ndarray]:
    """Per-specimen label histogram, keyed by specimen id."""

    labels = np.asarray(labels, dtype=np.int64).ravel()
    table: dict[str, np.ndarray] = {}
    for specimen in np.unique(specimens):
        counts = np.bincount(labels[specimens == specimen], minlength=int(num_classes))
        table[str(specimen)] = counts.astype(np.float64)
    return table


def _prior_rows(
    specimens: np.ndarray,
    table: dict[str, np.ndarray],
    fallback: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Per-sample class prior taken from a specimen's known label histogram.

    Specimens absent from ``table`` fall back to the global prior, which makes
    them coincide with the majority-class predictor.  A small floor keeps an
    all-negative specimen from producing an exact 0/1 row, so the tie convention
    of the metrics decides the label instead of a floating-point edge case.
    """

    rows = np.tile(np.asarray(fallback, dtype=np.float64).reshape(1, -1), (specimens.size, 1))
    for index, specimen in enumerate(specimens):
        counts = table.get(str(specimen))
        if counts is not None:
            rows[index] = counts
    rows = np.clip(rows, 1e-9, None)
    return rows / rows.sum(axis=1, keepdims=True)


def control_baselines(
    split_data: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, object]]]],
    num_classes: int,
) -> dict[str, object]:
    """Reference lines that separate "learned the signal" from "remembered the bone".

    Every reported accuracy has to be read against these.  The splits here are
    made of *points*, not bones, and 43 of 95 bones appear in both train and
    test, so a predictor that carries no signal at all can still score well by
    recognising which specimen a point came from:

    ``majority_class``
        The train prior, constant for every sample.  The floor: it cannot be
        beaten without information.
    ``specimen_fingerprint``
        The specimen's label histogram **as seen in train**, replayed onto its
        samples in the scored split.  This predictor never looks at a waveform.
        It is the bar a real method must clear -- if the model sits near it, the
        split is leaking specimens and the number means nothing.
    ``specimen_oracle``
        The specimen's label histogram **in the split being scored**.  This
        cheats, and exists only as an upper bound: it isolates how much of the
        task is "which bone is this" rather than "how thick is this bone".

    ``specimen_overlap`` counts the bones shared by each split pair.  It is the
    single number that says how much a point-level split flatters the model.
    """

    split_names = ("train", "val", "test")
    specimens = {split: specimen_ids(split_data[split][2]) for split in split_names}
    overlap = {
        f"{split_names[i]}|{split_names[j]}": int(
            len(set(np.unique(specimens[split_names[i]])) & set(np.unique(specimens[split_names[j]])))
        )
        for i in range(len(split_names))
        for j in range(i + 1, len(split_names))
    }
    train_labels = np.asarray(split_data["train"][1], dtype=np.int64)
    train_table = _label_counts(train_labels, specimens["train"], num_classes)
    counts = np.bincount(train_labels, minlength=int(num_classes)).astype(np.float64)
    global_prior = counts / max(1.0, counts.sum())

    controls: dict[str, object] = {
        "note": (
            "Control predictors, stored so that accuracy/AUC can be attributed. "
            "'specimen_fingerprint' uses no waveform information at all and is the "
            "bar a real method must clear; if it is close to the model, the split "
            "leaks specimens. 'specimen_oracle' cheats and is only an upper bound."
        ),
        "specimen_overlap": overlap,
        "specimen_overlap_max": max(overlap.values()) if overlap else 0,
        "specimen_counts": {split: int(np.unique(specimens[split]).size) for split in split_names},
        "specimen_total": int(np.unique(np.concatenate([specimens[s] for s in split_names])).size),
        "train_prior": [float(value) for value in global_prior],
    }
    for split in split_names:
        labels = np.asarray(split_data[split][1], dtype=np.int64)
        per_split = specimens[split]
        rows = {
            "majority_class": np.tile(global_prior.reshape(1, -1), (labels.size, 1)),
            "specimen_fingerprint": _prior_rows(per_split, train_table, global_prior, num_classes),
            "specimen_oracle": _prior_rows(
                per_split,
                _label_counts(labels, per_split, num_classes),
                global_prior,
                num_classes,
            ),
        }
        controls[split] = {
            name: classification_metrics(labels, value, num_classes) for name, value in rows.items()
        }
    return controls


def _fence_lines(metrics: dict[str, object], num_classes: int) -> list[str]:
    """Console lines that place the model between the control predictors."""

    controls = metrics.get("controls")
    if not isinstance(controls, dict):
        return []
    test = controls.get("test")
    if not isinstance(test, dict):
        return []
    model = metrics.get("metrics", {}).get("test", {})  # type: ignore[union-attr]
    lines = [f"  [fence] 标本重叠最大 {controls['specimen_overlap_max']} 块  详情 {controls['specimen_overlap']}"]
    for name, label in (
        ("majority_class", "多数类"),
        ("specimen_fingerprint", "标本指纹(无信号)"),
        ("specimen_oracle", "标本oracle(作弊上界)"),
    ):
        entry = test.get(name)
        if not isinstance(entry, dict):
            continue
        auc = entry.get("auc")
        auc_text = "n/a" if auc is None else f"{float(auc):.3f}"
        lines.append(f"  [fence] {label}: acc={float(entry['accuracy']):.3f} auc={auc_text}")
    if num_classes == 2 and isinstance(model, dict) and "accuracy" in model:
        auc = model.get("auc")
        auc_text = "n/a" if auc is None else f"{float(auc):.3f}"
        lines.append(f"  [fence] 本模型: acc={float(model['accuracy']):.3f} auc={auc_text}")
    decision = metrics.get("decision_threshold")
    if isinstance(decision, dict):
        chosen = decision["metrics"]["test"]
        auc = chosen.get("auc")
        auc_text = "n/a" if auc is None else f"{float(auc):.3f}"
        lines.append(
            f"  [fence] 阈值口径 {decision['source_flag']} t={float(decision['value']):.3f}: "
            f"test acc={float(chosen['accuracy']):.3f} auc={auc_text}"
        )
        parts = "  ".join(
            f"{name}=acc {float(entry['test_accuracy']):.3f}/t {float(entry['threshold']):.2f}"
            for name, entry in decision["sensitivity"].items()
        )
        lines.append(f"  [fence] 阈值敏感度 {parts}")
    return lines


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


# Keys whose change means the directory would hold a different experiment.
# ``_experiment_name`` encodes the form, the region set, the channel mode, the
# aggregations and the label-scheme tag, but none of these: a changed dataset or
# a changed class count lands on the same path.
_GUARD_FATAL_KEYS = ("data_dir", "regions", "num_classes", "label_scheme")


def _short(value: object, limit: int = 140) -> str:
    text = repr(json_ready(value))
    return text if len(text) <= limit else f"{text[:limit]}..."


def _same_path(left: object, right: object) -> bool:
    """Case-insensitive, resolved comparison so that ``D:\\x`` == ``d:/x``."""

    if left is None or right is None:
        return left is right
    return str(Path(str(left)).resolve()).casefold() == str(Path(str(right)).resolve()).casefold()


def _guard_existing_experiment(
    output_dir: Path,
    experiment_config: dict[str, object],
    allow_mismatch: bool,
) -> None:
    """Refuse to silently overwrite a directory that holds a different run.

    ``run_one`` opens with ``mkdir(parents=True, exist_ok=True)``, so a name
    collision costs the earlier run without any message.  That is not
    hypothetical: ``cls2_thr1.3`` and the keep-split variant of the same
    thresholds shared a tag and therefore a directory name, and 158 of the 268
    samples sat in a different split -- so the second run overwrote the first and
    ``config.json`` ended up describing only whichever ran last.

    Re-running an identical configuration stays silent.  A changed input
    (dataset, region plan, class count, threshold scheme) aborts unless
    ``--allow-config-mismatch`` is given.  Any other difference -- epochs, seed,
    feature set, sample limit -- is printed and then allowed through, because
    those are deliberate ablations and the directory name is still correct.
    """

    config_path = output_dir / "config.json"
    if not config_path.is_file():
        return
    try:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[guard] cannot read {config_path} ({exc}); overwriting in place")
        return
    if not isinstance(previous, dict):
        print(f"[guard] {config_path} is not a config mapping; overwriting in place")
        return

    fatal: list[str] = []
    changed: list[str] = []
    for key, new_value in experiment_config.items():
        if key not in previous:
            changed.append(key)
            continue
        old_value = previous[key]
        if key == "data_dir":
            same = _same_path(old_value, new_value)
        else:
            same = json_ready(old_value) == json_ready(new_value)
        if not same:
            (fatal if key in _GUARD_FATAL_KEYS else changed).append(key)
    for key in previous:
        if key not in experiment_config:
            changed.append(key)

    if changed:
        print(
            f"[guard] {output_dir.name}: reusing an existing directory; "
            f"these settings change ({', '.join(sorted(changed))})"
        )

    if not fatal:
        return
    details = "\n".join(
        f"    {key}\n"
        f"      recorded : {_short(previous.get(key))}\n"
        f"      requested: {_short(experiment_config.get(key))}"
        for key in fatal
    )
    if allow_mismatch:
        print(
            f"[guard] {output_dir.name}: overwriting a different experiment "
            f"because --allow-config-mismatch was given:\n{details}"
        )
        return
    raise SystemExit(
        f"refusing to overwrite {config_path}:\n{details}\n"
        "  The directory name does not encode the dataset variant, so this run "
        "would replace the recorded results.\n"
        "  Use a different --output-dir, give the dataset a distinct "
        "label_scheme.json tag, or pass --allow-config-mismatch to overwrite "
        "on purpose."
    )


def run_one(
    data_dir: Path,
    output_dir: Path,
    form: str,
    region_name: str,
    regions: Sequence[RegionSpec],
    args: argparse.Namespace,
) -> dict[str, object]:
    start_time = time.time()
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
        "threshold_source": args.threshold_source,
    }
    _guard_existing_experiment(output_dir, experiment_config, args.allow_config_mismatch)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    split_probabilities: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        split_probabilities[split] = model.predict_proba(scaled[split])
        metrics["metrics"][split] = classification_metrics(
            split_data[split][1], split_probabilities[split], num_classes
        )
        np.save(output_dir / f"probabilities_{split}.npy", split_probabilities[split])
    # The threshold is a fitted quantity here, not a constant: the claim being
    # tested is that the classes separate *at some* threshold.  Whichever split
    # selects it, the result is stored, and every other supported convention is
    # stored beside it -- on a 161-point train split the train-selected threshold
    # can cost several points of test accuracy, so a single number would hide a
    # reporting choice behind what looks like a result.  The 0.5 numbers in
    # "metrics" above are left untouched so every stored artefact and the
    # training curves stay comparable with earlier runs.
    if num_classes == 2:
        candidates: dict[str, float] = {
            "fixed": 0.5,
            "train": threshold_scan(split_data["train"][1], split_probabilities["train"])["threshold"],
            "val": threshold_scan(split_data["val"][1], split_probabilities["val"])["threshold"],
        }
        chosen_source = args.threshold_source
        chosen_threshold = candidates[chosen_source]
        sensitivity = {}
        for name, value in candidates.items():
            at_value = classification_metrics(
                split_data["test"][1], split_probabilities["test"], num_classes, value
            )
            sensitivity[name] = {
                "threshold": value,
                "test_accuracy": at_value["accuracy"],
                "test_sensitivity": at_value["sensitivity"],
                "test_specificity": at_value["specificity"],
            }
        metrics["decision_threshold"] = {
            "selected_on": "fixed 0.5 rule" if chosen_source == "fixed" else chosen_source,
            "value": chosen_threshold,
            "source_flag": chosen_source,
            "reason": (
                "Goal is separation at some threshold, so the threshold is fitted and "
                "reported instead of hardcoded at 0.5. 'val' is the live default because "
                "it is off the training data yet is not the scored split; 'train' is "
                "in-sample and can overshoot on the 161-point train split. AUC is "
                "threshold-free and is the primary separability number."
            ),
            "metrics": {
                split: classification_metrics(
                    split_data[split][1],
                    split_probabilities[split],
                    num_classes,
                    chosen_threshold,
                )
                for split in ("train", "val", "test")
            },
            "sensitivity": sensitivity,
        }
    else:
        metrics["decision_threshold"] = None
    # Stored next to the model metrics on purpose: a test number that cannot be
    # read against the majority-class and specimen-fingerprint lines is not
    # attributable, and on a point-level split the fingerprint line is the one
    # that decides whether the result means anything.
    metrics["controls"] = control_baselines(split_data, num_classes)
    write_json(output_dir / "feature_info.json", feature_info)
    write_json(output_dir / "metrics.json", metrics)
    elapsed = time.time() - start_time
    for line in _fence_lines(metrics, num_classes):
        print(f"[{form}/{region_name}]{line}")
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

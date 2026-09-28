"""Leave-one-specimen-out evaluation of the frozen bone-thickness dataset.

Why this script exists
----------------------
``run_emd_experiments.py`` scores a single train/val/test holdout.  That is the
right shape for model selection, and the wrong shape for the claim this project
actually needs to make.  The claim is *separability*: "the remaining vertical
bone thickness is recoverable from this 896-point record under some decision
threshold".  A single holdout answers that with one number, and every practical
problem with a single holdout -- a lucky split, a threshold tuned on the scored
data, samples from the same bone appearing on both sides -- shows up as the same
symptom: the number is a little too high and nothing in the output says why.

The protocol here removes all three by construction:

1. **Leave one specimen out.**  Every fold holds out one whole bone.  A bone is a
   ``point_id`` prefix (``point_id.split("_")[0]``), *not* a ``sample_id``: the
   latter is unique per sample, so grouping by it would silently collapse back to
   a point-level split and reproduce exactly the leak this exists to prevent.
   With every bone held out once, the specimen-fingerprint channel -- memorising
   "this trace came from bone 17, and bone 17 is thick" -- has nothing to memorise
   on the held-out side, so its accuracy collapses to the majority class.  That
   collapse is checked and reported rather than assumed.

2. **In-fold standardisation.**  Mean and standard deviation come from the
   training specimens of that fold only.  Fitting the scaler once on all data and
   then cross-validating standardises every held-out point using a mean that
   includes itself, which is a leak whether or not it moves the number today.
   With the probe used here it currently does *not* move the number: replacing the
   in-fold scaler with the pooled one changed the point AUC by 0.0000 at
   ``--l2 1`` and by +0.0001 at ``--l2 400`` (it does change the fitted
   coefficients -- the sampled probabilities move by 2e-5 and 2e-3 respectively).
   The invariance is a property of an almost unpenalised linear probe, not a
   reason to skip the rule: the same mistake stops being invisible the moment the
   probe is materially penalised, widened, or replaced by the MLP.

3. **In-fold threshold.**  The decision threshold is fitted on the fold's training
   specimens and applied to the held-out bone.  The threshold is reported
   alongside the accuracy it produced, because "separates at some threshold" makes
   the threshold a fitted quantity: quoting an accuracy at a hardcoded 0.5 answers
   a question nobody asked.

Threshold-free AUC is the primary number for the same reason.  Accuracy is
reported at three thresholds -- 0.5, the in-fold fit, and the best achievable on
the held-out fold -- so the reader can see how much of the headline is the
classifier and how much is the threshold choice.

The labels are *derived* from ``depth_value`` at ``--threshold-mm`` instead of
being read from ``y.npy``.  The stored labels are a snapshot of one labelling
decision; deriving them makes the threshold an explicit argument, lets one pooled
dataset answer both the 1.0 mm and the 1.3 mm question, and the run verifies its
own convention against ``y.npy`` on the way through.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from emd_pipeline import (
    DepthMapper,
    EMDConfig,
    EMD_FEATURE_PRESETS,
    StandardScaler,
    binary_metrics,
    build_feature_matrix,
    feature_dimension_names,
    json_ready,
    load_split,
    parse_regions,
    threshold_scan,
    write_json,
)
from run_emd_experiments import (
    DYNAMIC_REGION_NAME,
    REGION_PRESETS,
    build_parser,
    canonical_form,
    dyn_config_from_args,
)

# --------------------------------------------------------------------------- #
# Dataset plumbing
# --------------------------------------------------------------------------- #

# ``point_id`` looks like ``<specimen>_<something>``.  The specimen (the physical
# bone that was drilled) is field 0.  ``sample_id`` is deliberately NOT used for
# grouping: it is unique per sample, so grouping by it would hand every sample its
# own fold -- no training data at all -- or, worse, look like a working group
# split while providing zero isolation.
SPECIMEN_FIELD_INDEX = 0

SPLITS = ("train", "val", "test")

# Two label conventions differ only in what happens exactly on the boundary.
# Which one the stored ``y.npy`` used is measured at run time instead of assumed.
LABEL_CONVENTIONS = ("ge", "gt")

# The physical feature source.  ``emd`` is the model of record; ``bands`` is the
# deliberately minimal baseline -- one log-envelope amplitude per depth window per
# channel, each with a stated physical meaning -- used to show that the separator
# is present before any EMD machinery is applied to it.
FEATURE_SOURCES = ("emd", "bands")

BAND_WINDOWS: tuple[tuple[int, int], ...] = ((96, 200), (200, 270), (270, 500))

# Hilbert envelope of a real signal: keep the positive frequencies, double the
# interior bins, then take the magnitude.  896 is even, so the one-sided trick is
# exact (there is no Nyquist bin to argue about).
ENVELOPE_EPS = 1e-6

DEFAULT_THRESHOLD_MM = 1.0
DEFAULT_L2 = 1.0


def specimen_ids(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """The physical bone each record came from.

    Raises instead of guessing.  A silent fallback here would produce a split
    that looks grouped and is not, which is the failure this whole script is
    built to rule out.
    """

    ids: list[str] = []
    for index, record in enumerate(records):
        raw = record.get("point_id")
        if raw is None or str(raw).strip() == "":
            raise ValueError(
                f"record {index} has no usable 'point_id'; the specimen cannot be "
                "determined and a point-level split would be reported as a grouped one"
            )
        ids.append(str(raw).strip().split("_")[SPECIMEN_FIELD_INDEX])
    array = np.asarray(ids, dtype=object)
    if array.size == 0:
        raise ValueError("no records were loaded; cannot group by specimen")
    unique = np.unique(array)
    if unique.size < 2:
        raise ValueError(
            f"only {unique.size} distinct specimen(s) found, so leave-one-specimen-out "
            "has nothing to hold out"
        )
    if unique.size == array.size:
        raise ValueError(
            "every record has its own specimen id; the field used for grouping is "
            "unique per sample (a 'sample_id', not a bone id)"
        )
    return array


def depth_values(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Measured remaining vertical thickness in mm, one per record."""

    values: list[float] = []
    for index, record in enumerate(records):
        raw = record.get("depth_value")
        if raw is None:
            raise ValueError(f"record {index} has no 'depth_value'")
        values.append(float(raw))
    return np.asarray(values, dtype=np.float64)


def labels_from_depth(depth: np.ndarray, threshold_mm: float, convention: str) -> np.ndarray:
    """Binary label: 1 when the remaining thickness is at least ``threshold_mm``."""

    if convention == "ge":
        positive = depth >= threshold_mm
    elif convention == "gt":
        positive = depth > threshold_mm
    else:
        raise ValueError(f"unknown label convention {convention!r}")
    return positive.astype(np.int64)


def load_pool(data_dir: Path) -> dict[str, Any]:
    """Pool every split of a dataset directory into one array set.

    Leave-one-specimen-out does not need the shipped train/val/test partition at
    all -- it re-partitions by bone.  Pooling first is therefore not a loss of
    information: it makes the split's point-level leakage irrelevant by ignoring
    the split.  The per-split counts are kept for the report so the pooled total
    can be checked against the split sizes.
    """

    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    per_split: dict[str, int] = {}
    for split in SPLITS:
        x, y, samples = load_split(data_dir, split)
        chunks.append(np.asarray(x, dtype=np.float32))
        labels.append(np.asarray(y, dtype=np.int64).ravel())
        records.extend(dict(record) for record in samples)
        per_split[split] = int(np.asarray(y).size)
    x_pool = np.concatenate(chunks, axis=0)
    y_pool = np.concatenate(labels, axis=0)
    if x_pool.shape[0] != len(records):
        raise RuntimeError(
            f"pooled {x_pool.shape[0]} feature rows but {len(records)} metadata records"
        )
    return {
        "x": x_pool,
        "labels_stored": y_pool,
        "records": records,
        "per_split": per_split,
        "specimens": specimen_ids(records),
        "depth": depth_values(records),
    }


def detect_convention(depth: np.ndarray, stored: np.ndarray) -> dict[str, Any]:
    """Find which boundary rule the stored labels used, and how well they match.

    Reported rather than fixed silently: the project's own README describes the
    threshold as an "exclusive lower bound" while the code applies an inclusive
    one, so at least one of the two statements is wrong.  Measuring it here is
    cheaper than arguing about it.
    """

    report: dict[str, Any] = {}
    for convention in LABEL_CONVENTIONS:
        # ``--threshold-mm`` is not known at this point, so the comparison is done
        # against the cutoff that actually reproduces ``y.npy``: every distinct
        # depth is a candidate boundary.
        candidates = np.unique(depth)
        for boundary in candidates:
            prediction = labels_from_depth(depth, float(boundary), convention)
            if np.array_equal(prediction, stored):
                report[convention] = float(boundary)
                break
        else:
            report[convention] = None
    return report


def envelope(x: np.ndarray) -> np.ndarray:
    """Analytic-signal magnitude along the last axis, in float64."""

    length = x.shape[-1]
    spectrum = np.fft.fft(x, axis=-1)
    kernel = np.zeros(length)
    kernel[0] = 1.0
    kernel[length // 2] = 1.0
    kernel[1 : length // 2] = 2.0
    out = np.empty(x.shape, dtype=np.float64)
    for start in range(0, x.shape[0], 1024):
        stop = min(start + 1024, x.shape[0])
        out[start:stop] = np.abs(np.fft.ifft(spectrum[start:stop] * kernel, axis=-1))
    return out


def band_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Six physical features: one log amplitude per channel per depth window.

    Amplitude of a log-envelope average, so the numbers keep units of a
    logarithmic amplitude and the columns can be named.  Frames are averaged
    first: the 50 frames are repeated acquisitions of the same point, so their
    mean is the measurement and their spread is noise.
    """

    n, frames, channels, length = x.shape
    amplitude = envelope(x.reshape(-1, length).astype(np.float64))
    amplitude = amplitude.reshape(n, frames, channels, length).mean(axis=1)
    log_amplitude = np.log(amplitude + ENVELOPE_EPS)
    columns = [
        log_amplitude[:, channel, start:stop].mean(axis=1)
        for start, stop in BAND_WINDOWS
        for channel in range(channels)
    ]
    names = [
        f"logband_{start}_{stop}_ch{channel}"
        for start, stop in BAND_WINDOWS
        for channel in range(channels)
    ]
    return np.stack(columns, axis=1).astype(np.float32), names


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def build_parser_loso() -> argparse.ArgumentParser:
    """The LOSO parser: every experiment feature flag, plus the protocol flags.

    The experiment parser is used as a ``parent`` and conflicts are resolved in
    favour of this script.  Copying the flags instead would leave the two files
    free to disagree about a default, and a drifted default is invisible in the
    output -- it just quietly evaluates a different feature vector.
    """

    parser = argparse.ArgumentParser(
        parents=[build_parser()],
        conflict_handler="resolve",
        description="Leave-one-specimen-out separability evaluation",
        epilog=(
            "examples:\n"
            "  python run_loso_evaluation.py --threshold-mm 1.0\n"
            "  python run_loso_evaluation.py --threshold-mm 1.3 --feature-source bands\n"
            "  python run_loso_evaluation.py --forms max1 --region-set full --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--forms",
        default="max1",
        help=(
            "exactly one frame form: the protocol reasons about one feature vector at a "
            "time, so a sweep belongs in the experiment script (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--region-set",
        choices=[*REGION_PRESETS.keys(), DYNAMIC_REGION_NAME],
        default="full",
        help=(
            "one depth-region preset; 'all' is not offered because it would report "
            "five protocols as one (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--threshold-mm",
        type=float,
        default=DEFAULT_THRESHOLD_MM,
        help=(
            "remaining-thickness cutoff in mm that defines the positive class; the "
            "labels are derived from depth_value so one pooled dataset can answer "
            "both the 1.0 mm and the 1.3 mm question (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--feature-source",
        choices=FEATURE_SOURCES,
        default="emd",
        help=(
            "'emd' runs the pipeline's real feature extractor, 'bands' uses six "
            "log-envelope amplitudes over three depth windows (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--probe",
        choices=("logistic", "quadratic"),
        default="logistic",
        help=(
            "'logistic' is a ridge-penalised logistic regression solved by Newton "
            "iterations -- deterministic, no learning rate, and the honest probe for "
            "'is the information there'; 'quadratic' is the same solver on "
            "[x, x^2, x_i x_j], which asks whether a curved boundary is worth "
            "anything.  An MLP is deliberately not offered: epochs, learning rate "
            "and dropout would each have to be selected inside the fold to keep the "
            "protocol clean, and a probe whose own hyperparameters were picked by "
            "hand is one more place a result can be tuned into looking better than "
            "it is (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=DEFAULT_L2,
        help=(
            "L2 penalty for all probes, divided by the fold's training count so the "
            "value means the same thing at every fold size.  Because of that "
            "division a small value is effectively no penalty at all: on this "
            "dataset ~255 training rows means --l2 1 contributes 0.004 per "
            "coefficient against a curvature of order 255.  Sweep it (1, 10, 100, "
            "400, ...) before reading a penalised or wide run as evidence about the "
            "data (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=5,
        help=(
            "how many specimen groups the training part of each fold is split into "
            "when fitting the decision threshold; the threshold must not be chosen "
            "on the bone it is then scored on (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation"),
        help="where the per-run result directory is written (default: %(default)s)",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="name of the result directory; default encodes threshold and feature source",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the pooled dataset and the fold composition, then stop",
    )
    return parser


def _resolve_form(args: argparse.Namespace) -> str:
    """The single frame form under test, or a placeholder for the bands source.

    ``--forms`` inherits its default from the experiment script, which accepts the
    word ``all``.  Running every form here would produce five protocols reported as
    one, so ``all`` is refused rather than expanded.
    """

    if args.feature_source == "bands":
        return "n/a"
    value = str(args.forms).strip()
    if value.lower() == "all":
        raise SystemExit(
            "--forms was set to 'all'; a LOSO run is per-configuration, so pass "
            "exactly one form, e.g. --forms max1"
        )
    forms = [canonical_form(item) for item in value.split(",") if item.strip()]
    if len(forms) != 1:
        raise SystemExit(
            f"--forms must name exactly one form for a LOSO run, got {args.forms!r}; "
            "the protocol is per-configuration and a sweep would bury it"
        )
    return forms[0]


def _resolve_regions(args: argparse.Namespace) -> tuple[str, list[Any]]:
    """The depth windows under test, or a placeholder for the bands source."""

    if args.feature_source == "bands":
        return "n/a", []
    if args.regions_json is not None:
        return "custom", parse_regions(args.regions_json)
    if args.region_set == "all":
        raise SystemExit(
            "--region-set all would run five protocols at once; pass one of "
            f"{', '.join(REGION_PRESETS)} or {DYNAMIC_REGION_NAME}"
        )
    if args.region_set == DYNAMIC_REGION_NAME:
        return DYNAMIC_REGION_NAME, []
    return args.region_set, parse_regions(REGION_PRESETS[args.region_set])


def _default_run_tag(args: argparse.Namespace, region_name: str, form: str) -> str:
    """Directory name that encodes the three inputs a reader must know.

    The threshold, the feature source and the probe each change what the numbers
    mean, so each is in the name.  A name that omits them would let a 1.3 mm run
    overwrite a 1.0 mm one and leave no trace of which is which.
    """

    source = args.feature_source if args.feature_source == "bands" else f"{form}_{region_name}"
    threshold = f"{args.threshold_mm:g}".replace(".", "p")
    return f"loso_{source}_thr{threshold}mm_{args.probe}"


def build_features(
    args: argparse.Namespace, form: str, region_name: str, regions: Sequence[Any], pool: Mapping[str, Any]
) -> tuple[np.ndarray, list[str]]:
    """The feature matrix under test, plus the name of every column."""

    if args.feature_source == "bands":
        matrix, names = band_features(np.asarray(pool["x"], dtype=np.float32))
        return matrix, names
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
    matrix, info = build_feature_matrix(
        np.asarray(pool["x"], dtype=np.float32),
        form=form,
        regions=regions,
        channel_mode=args.channel_mode,
        mapper=mapper,
        target_length=args.target_length,
        tukey_alpha=args.tukey_alpha,
        dynamic_envelope_config=dynamic_envelope_config,
        emd_config=emd_config,
        score_region=selection_region,
        sample_ids=[str(record.get("sample_id", "")) for record in pool["records"]],
    )
    return matrix, feature_dimension_names(info[0], emd_config)


def guard_output_dir(output_dir: Path, config: Mapping[str, Any], allow_mismatch: bool) -> None:
    """Refuse to overwrite a directory that holds a different protocol.

    ``run_emd_experiments`` grew the same guard for the same reason: a directory
    name that does not encode every input silently keeps whichever run finished
    last.  The run tag here does encode threshold, feature source and probe, so
    only an unrelated protocol should ever collide.
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
        return
    differences = [
        key
        for key, value in config.items()
        if json.dumps(json_ready(previous.get(key)), sort_keys=True)
        != json.dumps(json_ready(value), sort_keys=True)
    ]
    if not differences:
        return
    detail = "\n".join(f"    {key}: {previous.get(key)!r} -> {config[key]!r}" for key in differences)
    if allow_mismatch:
        print(f"[guard] overwriting {output_dir} on request:\n{detail}")
        return
    raise SystemExit(
        f"refusing to overwrite {config_path}:\n{detail}\n"
        "  Pass --allow-config-mismatch to overwrite on purpose, or --run-tag to "
        "write somewhere else."
    )


def write_scores(
    path: Path,
    pool: Mapping[str, Any],
    labels: np.ndarray,
    result: Mapping[str, Any],
) -> None:
    """One row per sample: what it was, what the protocol said, which cutoff applied."""

    probability = np.asarray(result["probability"], dtype=np.float64)
    threshold = np.asarray(result["threshold"], dtype=np.float64)
    lines = ["specimen,depth_mm,label,probability,fold_threshold,decision"]
    for index in range(labels.size):
        decision = "" if np.isnan(probability[index]) else str(
            int(probability[index] >= threshold[index])
        )
        lines.append(
            ",".join(
                [
                    f'"{pool["specimens"][index]}"',
                    f"{float(pool['depth'][index]):.4f}",
                    str(int(labels[index])),
                    "" if np.isnan(probability[index]) else f"{float(probability[index]):.6f}",
                    "" if np.isnan(threshold[index]) else f"{float(threshold[index]):.6f}",
                    decision,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def design_diagnostics(
    args: argparse.Namespace,
    feature_dimension: int,
    pooled_rows: int,
    specimens: np.ndarray,
) -> dict[str, Any]:
    """How wide the fitted design is next to the smallest fold's training set.

    One function rather than two inline computations, because the printed caveat
    and the stored caveat have to be the same caveat.  If the report said "wide"
    while the artefact said nothing, the number would eventually be quoted on its
    own -- and an under-penalised wide fit fails *downwards*, so it reads as
    "nonlinearity does not help" when it actually means "the penalty was wrong".
    """

    if args.probe != "quadratic":
        return {
            "probe": args.probe,
            "columns": int(feature_dimension),
            "wide": False,
        }
    expanded = 2 * feature_dimension + feature_dimension * (feature_dimension - 1) // 2
    _, bone_sizes = np.unique(specimens, return_counts=True)
    smallest = int(pooled_rows - int(bone_sizes.max()))
    return {
        "probe": args.probe,
        "linear_columns": int(feature_dimension),
        "columns": int(expanded),
        "smallest_fold_training_points": smallest,
        "wide": bool(expanded > smallest // 4),
        "note": (
            "the quadratic probe is only comparable to the linear one while the "
            "expansion stays small relative to the fold; when 'wide' is true the "
            "shared --l2 is spread over too many coefficients, so a drop in AUC "
            "measures the penalty, not the data.  Before reading a quadratic run as "
            "'nonlinearity does not help', sweep --l2 until the AUC plateaus and "
            "compare the plateau against the same-feature linear probe; an "
            "under-penalised wide fit fails downwards and mimics 'no signal'."
        ),
    }


def print_report(
    args: argparse.Namespace,
    form: str,
    region_name: str,
    pool: Mapping[str, Any],
    labels: np.ndarray,
    scores: Mapping[str, Any],
    fences: Mapping[str, Any],
    feature_dimension: int,
    elapsed: float,
) -> None:
    points = scores["points"]
    specimen_metrics = scores["specimens"]
    print()
    print("=" * 88)
    print("leave-one-specimen-out evaluation")
    print("=" * 88)
    print(
        f"  data_dir={args.data_dir}  pooled={pool['x'].shape[0]} rows  "
        f"specimens={int(np.unique(pool['specimens']).size)}  "
        f"points/bone={pool['x'].shape[0] / float(np.unique(pool['specimens']).size):.2f}"
    )
    print(
        f"  label rule: depth >= {args.threshold_mm:g} mm  ->  "
        f"{int(labels.sum())} positive / {labels.size} ({float(labels.mean()):.3f})"
    )
    print(
        f"  features  : source={args.feature_source} form={form} region={region_name} "
        f"dim={feature_dimension}"
    )
    if args.probe == "quadratic":
        diagnostics = design_diagnostics(
            args, feature_dimension, int(pool["x"].shape[0]), pool["specimens"]
        )
        print(
            f"  design    : {diagnostics['linear_columns']} -> {diagnostics['columns']} "
            f"columns after the expansion (smallest fold trains on "
            f"{diagnostics['smallest_fold_training_points']} rows)"
        )
        if diagnostics["wide"]:
            print(
                "  [!]       design is wide relative to the fold: l2 is spread over "
                f"{diagnostics['columns']} coefficients instead of "
                f"{diagnostics['linear_columns']}, so this run is a penalty-limited fit, "
                "NOT evidence about the data.  Sweep --l2 until the AUC plateaus, then "
                "compare that plateau against the same-feature linear probe: an "
                "under-penalised wide fit fails downwards and mimics 'no signal'."
            )
    print(
        f"  probe     : {args.probe}  l2={args.l2:g}  inner threshold folds={args.inner_folds}"
    )
    print(f"  folds     : {scores['per_fold']['n']} scored, {scores['skipped_folds']} skipped")
    print()
    print("  point level (every sample scored by a model that never saw its bone)")
    print(f"    AUC (threshold-free)        : {float(points['auc']):.3f}")
    print(
        f"    accuracy @0.5               : {float(points['accuracy_at_0.5']):.3f}"
    )
    print(
        f"    accuracy @ fold threshold   : {float(points['accuracy_at_fold_threshold']):.3f}"
        f"   (balanced {float(points['balanced_accuracy_at_fold_threshold']):.3f})"
    )
    print(
        f"    accuracy @ oracle threshold : {float(points['accuracy_at_oracle_threshold']):.3f}"
        f"   (t={float(points['oracle_threshold']):.3f}, not selectable in advance)"
    )
    print(
        f"    fold thresholds             : mean {float(points['threshold_mean']):.3f} "
        f"range {float(points['threshold_min']):.3f}..{float(points['threshold_max']):.3f}"
    )
    print(
        f"    per-bone accuracy           : mean "
        f"{float(scores['per_fold']['mean_accuracy']):.3f} "
        f"sd {float(scores['per_fold']['std_accuracy']):.3f} "
        f"over {int(scores['per_fold']['n'])} bones "
        f"({int(scores['per_fold']['single_point_folds'])} of them hold a single point)"
    )
    print()
    print("  specimen level (one decision per bone)")
    print(f"    AUC                         : {float(specimen_metrics['auc']):.3f}")
    print(f"    accuracy @0.5               : {float(specimen_metrics['accuracy_at_0.5']):.3f}")
    print(
        f"    accuracy @ oracle threshold : "
        f"{float(specimen_metrics['accuracy_at_oracle_threshold']):.3f}"
    )
    print()
    print("  fence lines (all four are needed to read the numbers above)")
    for line in format_fences(fences):
        print(line)
    print()
    print(f"  [{time.time() - elapsed:.1f}s total]")



# --------------------------------------------------------------------------- #
# The probe
# --------------------------------------------------------------------------- #


def fit_ridge_logistic(
    design: np.ndarray, labels: np.ndarray, penalty: np.ndarray, max_iterations: int = 100
) -> tuple[np.ndarray, int, bool]:
    """Ridge-penalised logistic regression by Newton / IRLS iterations.

    Chosen over gradient descent so the protocol has no learning rate and no
    iteration budget to tune: with a 14-column design the Newton step converges in
    single-digit iterations, and a probe whose own hyperparameters are fitted by
    hand is one more place a result can be tuned into looking better than it is.

    ``penalty`` is supplied per coefficient (0 for the intercept) and is already
    divided by the fold's count, so the same ``--l2`` means the same thing whether
    a fold trains on 250 rows or 260.  The tiny diagonal added before solving is
    what keeps a fold with a constant column from raising instead of returning a
    finite coefficient.
    """

    weights = np.zeros(design.shape[1], dtype=np.float64)
    converged = False
    iterations = 0
    ridge = 1e-9 * np.eye(design.shape[1])
    for iterations in range(1, max_iterations + 1):
        scores = design @ weights
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -35.0, 35.0)))
        curvature = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
        gradient = design.T @ (probabilities - labels) + penalty * weights
        hessian = (design.T * curvature) @ design + np.diag(penalty) + ridge
        step = np.linalg.solve(hessian, gradient)
        weights = weights - step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    return weights, iterations, converged


def predict_ridge_logistic(design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = design @ weights
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -35.0, 35.0)))


def _design(matrix: np.ndarray) -> np.ndarray:
    """Feature matrix with the intercept column in front."""

    return np.hstack([np.ones((matrix.shape[0], 1)), matrix])


PROBES = ("logistic", "quadratic")


def _standardise(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise with the training fold's own statistics only.

    The scaler is the quietest leak in a cross-validation loop: fit it once on all
    data and every held-out point has been standardised using a mean that includes
    itself.

    Measured, so nobody has to guess the size of it: swapping this for a scaler
    fitted on all 268 pooled rows leaves the point AUC at 0.7760 (``--l2 1``) or
    moves it by +0.0001 (``--l2 400``), because the penalty is divided by the fold
    size and an unpenalised logistic fit is invariant under a rescaled design.
    The fitted coefficients *do* change, so this is a reparameterisation that
    happens to be harmless for this probe -- not a licence to fit the scaler on
    the scored rows, which is what the rule actually forbids.
    """

    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(other)


def expand_quadratic(matrix: np.ndarray) -> np.ndarray:
    """``[x, x^2, x_i x_j]`` -- a nonlinearity check with no second trainer.

    Applied *after* standardisation, so the squares and products are on the same
    scale as the linear terms and the shared ``--l2`` penalises them fairly.  The
    expansion is a fixed function with no fitted parameters, so it cannot leak;
    putting it inside the fold is about the penalty being comparable, not about
    honesty.
    """

    columns = [matrix, matrix * matrix]
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            columns.append((matrix[:, left] * matrix[:, right])[:, None])
    return np.hstack(columns)


def _prepare(
    train: np.ndarray, other: np.ndarray, probe: str
) -> tuple[np.ndarray, np.ndarray]:
    """Everything that turns raw features into the two design matrices of a fold.

    One function because the outer fold and the inner threshold fold must build
    their design matrices the same way: if they disagreed, the threshold would be
    fitted on scores from a different model class than the one being scored.
    """

    train_std, other_std = _standardise(train, other)
    if probe == "quadratic":
        train_std = expand_quadratic(train_std)
        other_std = expand_quadratic(other_std)
    return _design(train_std), _design(other_std)


def inner_threshold_groups(specimens: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Assign the training specimens of one fold to ``folds`` inner groups.

    Whole specimens, for the same reason the outer loop holds out whole
    specimens: the threshold is a fitted quantity and it must not be fitted on
    points from the bone it is about to be scored on.
    """

    unique = np.unique(specimens)
    if folds <= 1 or unique.size < 2:
        return np.zeros(specimens.size, dtype=np.int64)
    order = np.random.default_rng(seed).permutation(unique.size)
    lookup = {str(unique[index]): int(rank % folds) for rank, index in enumerate(order)}
    return np.asarray([lookup[str(value)] for value in specimens], dtype=np.int64)


# --------------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------------- #


def leave_one_specimen_out(
    features: np.ndarray,
    labels: np.ndarray,
    specimens: np.ndarray,
    l2: float,
    inner_folds: int,
    seed: int,
    probe: str = "logistic",
) -> dict[str, Any]:
    """Score every sample from a bone the model never trained on.

    Returns the out-of-fold probability of every sample, the threshold that fold
    fitted, and the fold bookkeeping.  Nothing is averaged away here: an averaged
    cross-validation score hides the folds where the protocol failed, and those
    are the ones worth looking at.
    """

    if probe not in PROBES:
        raise SystemExit(f"unknown probe {probe!r}; expected one of {PROBES}")
    probability = np.full(labels.size, np.nan, dtype=np.float64)
    threshold = np.full(labels.size, np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    unique = np.unique(specimens)
    for index, bone in enumerate(unique):
        held_out = specimens == bone
        training = ~held_out
        if np.unique(labels[training]).size < 2:
            folds.append(
                {
                    "specimen": str(bone),
                    "points": int(held_out.sum()),
                    "skipped": "the training specimens carry a single class",
                }
            )
            continue
        design_train, design_test = _prepare(
            features[training], features[held_out], probe
        )
        penalty = np.zeros(design_train.shape[1], dtype=np.float64)
        penalty[1:] = l2 / max(1, design_train.shape[0])
        weights, iterations, converged = fit_ridge_logistic(
            design_train, labels[training].astype(np.float64), penalty
        )
        held_out_probability = predict_ridge_logistic(design_test, weights)
        probability[held_out] = held_out_probability

        # The threshold comes from scores on specimens the fold did not train on
        # either, so it is neither fitted on the scored bone nor on the fold's own
        # training rows.
        groups = inner_threshold_groups(specimens[training], inner_folds, seed + index)
        inner_probability = np.full(int(training.sum()), np.nan, dtype=np.float64)
        for group in np.unique(groups):
            inner_test = groups == group
            inner_train = ~inner_test
            if inner_train.sum() == 0 or np.unique(labels[training][inner_train]).size < 2:
                continue
            a, b = _prepare(features[training][inner_train], features[training][inner_test], probe)
            p = np.zeros(a.shape[1], dtype=np.float64)
            p[1:] = l2 / max(1, a.shape[0])
            w, _, _ = fit_ridge_logistic(a, labels[training][inner_train].astype(np.float64), p)
            inner_probability[inner_test] = predict_ridge_logistic(b, w)
        usable = ~np.isnan(inner_probability)
        if usable.sum() > 0 and np.unique(labels[training][usable]).size >= 2:
            scan = threshold_scan(
                labels[training][usable],
                np.column_stack([1.0 - inner_probability[usable], inner_probability[usable]]),
            )
            chosen = float(scan["threshold"])
            scan_source = "inner-folds"
        else:
            in_sample = threshold_scan(
                labels[training],
                np.column_stack(
                    [1.0 - predict_ridge_logistic(design_train, weights), predict_ridge_logistic(design_train, weights)]
                ),
            )
            chosen = float(in_sample["threshold"])
            scan_source = "in-fold-train-scores"
        threshold[held_out] = chosen
        folds.append(
            {
                "specimen": str(bone),
                "points": int(held_out.sum()),
                "positive": int(labels[held_out].sum()),
                "threshold": chosen,
                "threshold_source": scan_source,
                "iterations": int(iterations),
                "converged": bool(converged),
                "training_points": int(training.sum()),
                "probability": [float(value) for value in held_out_probability],
            }
        )
    return {"probability": probability, "threshold": threshold, "folds": folds}


def _specimen_table(
    specimens: np.ndarray, labels: np.ndarray, probability: np.ndarray
) -> dict[str, np.ndarray]:
    """Average the point scores of one bone into one decision about that bone.

    The rounding is load-bearing, not cosmetic.  If every point shares one score,
    every bone must end up with exactly that score, or the ROC's tie handling
    stops seeing the predictor as constant.  Floating-point averaging does **not**
    guarantee this: summing k copies of ``x`` and dividing by k can land one ULP
    away from ``x``, and one ULP is enough to break a tie.  Measured on this
    dataset it moved the constant majority-class fence's specimen AUC from a
    correct 0.500 to a meaningless 0.615, which is the kind of number that looks
    like a real effect and gets quoted.  Nine decimals is far finer than any
    probability this project reports and far coarser than the noise.
    """

    names = np.unique(specimens)
    scores = np.round(
        np.asarray([probability[specimens == bone].mean() for bone in names]), 9
    )
    truth = np.asarray(
        [1 if labels[specimens == bone].mean() >= 0.5 else 0 for bone in names], dtype=np.int64
    )
    counts = np.asarray([int((specimens == bone).sum()) for bone in names], dtype=np.int64)
    return {"names": names, "scores": scores, "labels": truth, "counts": counts}


def score_run(result: Mapping[str, Any], labels: np.ndarray, specimens: np.ndarray) -> dict[str, Any]:
    """Every number the report quotes, in one place."""

    probability = np.asarray(result["probability"], dtype=np.float64)
    threshold = np.asarray(result["threshold"], dtype=np.float64)
    scored = ~np.isnan(probability)
    if scored.sum() == 0:
        raise RuntimeError("no sample was scored; every fold was skipped")
    point_columns = np.column_stack([1.0 - probability, probability])
    # The fold threshold varies per bone, so the "in-fold threshold" accuracy is
    # the mean over points of each point's own fold's decision, not a single
    # global cutoff -- reported next to the single-cutoff accord for contrast.
    per_fold_decision = (probability >= threshold).astype(np.int64)
    oracle_scan = threshold_scan(labels[scored], point_columns[scored])
    points = {
        "n": int(scored.sum()),
        "positives": int(labels[scored].sum()),
        "auc": binary_metrics(labels[scored], point_columns[scored], 0.5)["auc"],
        "accuracy_at_0.5": float(np.mean((probability[scored] >= 0.5) == (labels[scored] == 1))),
        "accuracy_at_fold_threshold": float(np.mean(per_fold_decision[scored] == labels[scored])),
        "accuracy_at_oracle_threshold": float(oracle_scan["accuracy"]),
        "oracle_threshold": float(oracle_scan["threshold"]),
        "threshold_mean": float(np.nanmean(threshold)),
        "threshold_min": float(np.nanmin(threshold)),
        "threshold_max": float(np.nanmax(threshold)),
        "balanced_accuracy_at_fold_threshold": 0.5
        * (
            float(np.mean(per_fold_decision[labels == 1] == 1))
            + float(np.mean(per_fold_decision[labels == 0] == 0))
        ),
    }
    table = _specimen_table(specimens, labels, np.where(np.isnan(probability), 0.5, probability))
    specimen_columns = np.column_stack([1.0 - table["scores"], table["scores"]])
    specimen_scan = threshold_scan(table["labels"], specimen_columns)
    specimen_level = {
        "n": int(table["names"].size),
        "positives": int(table["labels"].sum()),
        "auc": binary_metrics(table["labels"], specimen_columns, 0.5)["auc"],
        "accuracy_at_0.5": float(np.mean((table["scores"] >= 0.5) == (table["labels"] == 1))),
        "accuracy_at_oracle_threshold": float(specimen_scan["accuracy"]),
        "oracle_threshold": float(specimen_scan["threshold"]),
    }
    # A bone with a single point can only score 0 or 1, so the per-fold spread is
    # reported together with the fold sizes rather than as a bare mean.
    fold_accuracy: list[float] = []
    sizes: list[int] = []
    for fold in result["folds"]:
        if "probability" not in fold:
            continue
        mask = specimens == fold["specimen"]
        decision = np.asarray(fold["probability"]) >= float(fold["threshold"])
        fold_accuracy.append(float(np.mean(decision == (labels[mask] == 1))))
        sizes.append(int(mask.sum()))
    fold_sizes = np.asarray(sizes, dtype=np.int64)
    per_fold = {
        "n": int(len(fold_accuracy)),
        "mean_accuracy": float(np.mean(fold_accuracy)) if fold_accuracy else None,
        "std_accuracy": float(np.std(fold_accuracy)) if fold_accuracy else None,
        "perfect_folds": int(np.sum(np.asarray(fold_accuracy) == 1.0)) if fold_accuracy else 0,
        "median_points": float(np.median(fold_sizes)) if fold_sizes.size else None,
        "single_point_folds": int(np.sum(fold_sizes == 1)) if fold_sizes.size else 0,
    }
    skipped = [fold for fold in result["folds"] if "probability" not in fold]
    return {
        "points": points,
        "specimens": specimen_level,
        "per_fold": per_fold,
        "skipped_folds": len(skipped),
        "specimen_table": {
            "names": [str(value) for value in table["names"]],
            "scores": [float(value) for value in table["scores"]],
            "labels": [int(value) for value in table["labels"]],
            "counts": [int(value) for value in table["counts"]],
        },
    }


# --------------------------------------------------------------------------- #
# Control predictors, evaluated inside the same protocol
# --------------------------------------------------------------------------- #

FENCE_KINDS = ("majority_class", "specimen_fingerprint", "specimen_oracle")


def _prior_table(labels: np.ndarray, specimens: np.ndarray) -> dict[str, np.ndarray]:
    """Per-bone class prior, floored so a pure bone is not a hard 0/1 vote."""

    table: dict[str, np.ndarray] = {}
    for bone in np.unique(specimens):
        mask = specimens == bone
        counts = np.bincount(labels[mask], minlength=2).astype(np.float64) + 1e-9
        table[str(bone)] = counts / counts.sum()
    return table


def fence_scores(
    labels: np.ndarray, specimens: np.ndarray, kind: str
) -> np.ndarray:
    """Out-of-fold scores for the three reference predictors.

    Under leave-one-specimen-out the specimen-fingerprint control is *supposed* to
    be useless, and that is exactly what makes it worth computing.  The control
    memorises a histogram per bone and can only score bones it has seen; the
    held-out bone has been seen zero times, so **every** one of its points falls
    back to the prior and the control reduces exactly to the majority-class
    predictor.  The two fence lines therefore coincide, by construction, and that
    coincidence is the evidence that the protocol removed the memory channel -- it
    is not an assumption.  This is the very control that reached 0.657 accuracy on
    the shipped point-level split, where the same bones appear on both sides; if it
    ever rises above the majority line again, the split has sprung a leak and this
    number will say so.

    The fallback is the pooled prior rather than each fold's own training prior.
    That is not cosmetic: the training prior moves with whichever bone was removed
    -- dropping a positive-heavy bone lowers it -- so a fold-wise fallback is
    *almost* constant and its residual variation is anti-correlated with the
    held-out labels.  Measured on this dataset that artefact put the fingerprint
    AUC at 0.070, which would read as "the fingerprint is worse than chance"
    instead of "the fingerprint knows nothing".  A fence has to be a fixed,
    interpretable line to be comparable.

    ``specimen_oracle`` keeps the held-out bone's own labels, so it is the ceiling
    that includes label noise: two points of one bone can carry different labels
    while the classifier can only see one trace.  It is a diagnostic, never a
    result.
    """

    if kind not in FENCE_KINDS:
        raise ValueError(f"unknown fence {kind!r}")
    pooled_prior = float(np.mean(labels))
    if kind == "majority_class":
        # A constant score, so the ROC AUC is exactly 0.500 by the tie rule.
        return np.full(labels.size, pooled_prior, dtype=np.float64)
    probability = np.zeros(labels.size, dtype=np.float64)
    for bone in np.unique(specimens):
        held_out = specimens == bone
        training = ~held_out
        if kind == "specimen_fingerprint":
            table = _prior_table(labels[training], specimens[training])
        else:
            table = _prior_table(labels[held_out], specimens[held_out])
        entry = table.get(str(bone))
        # "No information about this bone" is a constant, not a slightly moving
        # number -- see the docstring for why a fold-wise fallback is an artefact.
        prior = pooled_prior if entry is None else float(entry[1])
        probability[held_out] = prior
    return probability


def fence_report(
    labels: np.ndarray,
    specimens: np.ndarray,
    run_scores: Mapping[str, Any],
    probe: str = "logistic",
) -> dict[str, Any]:
    """The four lines every report has to show together."""

    lines: dict[str, Any] = {}
    for kind in FENCE_KINDS:
        scores = fence_scores(labels, specimens, kind)
        columns = np.column_stack([1.0 - scores, scores])
        table = _specimen_table(specimens, labels, scores)
        specimen_columns = np.column_stack([1.0 - table["scores"], table["scores"]])
        lines[kind] = {
            "points": {
                "accuracy_at_0.5": float(np.mean((scores >= 0.5) == (labels == 1))),
                "auc": binary_metrics(labels, columns, 0.5)["auc"],
                "accuracy_at_oracle_threshold": float(
                    threshold_scan(labels, columns)["accuracy"]
                ),
            },
            "specimens": {
                "accuracy_at_0.5": float(
                    np.mean((table["scores"] >= 0.5) == (table["labels"] == 1))
                ),
                "auc": binary_metrics(table["labels"], specimen_columns, 0.5)["auc"],
            },
        }
    points = run_scores["points"]
    specimens_metrics = run_scores["specimens"]
    lines["this_run"] = {
        "label": f"本方法（{probe} 探针，LOSO）",
        "points": {
            "accuracy_at_fold_threshold": points["accuracy_at_fold_threshold"],
            "accuracy_at_0.5": points["accuracy_at_0.5"],
            "auc": points["auc"],
        },
        "specimens": {
            "accuracy_at_0.5": specimens_metrics["accuracy_at_0.5"],
            "auc": specimens_metrics["auc"],
        },
    }
    return lines


def format_fences(fences: Mapping[str, Any]) -> list[str]:
    """Console lines placing this run between the three reference predictors."""

    order = (
        ("majority_class", "多数类(什么都不学)"),
        ("specimen_fingerprint", "标本指纹(只记忆、不看信号)"),
        ("this_run", "本方法"),
        ("specimen_oracle", "标本oracle(含标签噪声的作弊上界)"),
    )
    lines: list[str] = []
    for key, label in order:
        entry = fences.get(key)
        if not isinstance(entry, dict):
            continue
        point = entry["points"]
        specimen = entry["specimens"]
        threshold_accuracy = point.get(
            "accuracy_at_fold_threshold", point.get("accuracy_at_oracle_threshold")
        )
        lines.append(
            f"  [fence] {label:<26} 点 acc(折内阈值)={float(threshold_accuracy):.3f} "
            f"点 acc@0.5={float(point['accuracy_at_0.5']):.3f} "
            f"点 AUC={float(point['auc']):.3f} | "
            f"标本 acc={float(specimen['accuracy_at_0.5']):.3f} "
            f"标本 AUC={float(specimen['auc']):.3f}"
        )
    majority = fences.get("majority_class")
    fingerprint = fences.get("specimen_fingerprint")
    if isinstance(majority, dict) and isinstance(fingerprint, dict):
        same = (
            majority["points"]["accuracy_at_0.5"] == fingerprint["points"]["accuracy_at_0.5"]
            and majority["points"]["auc"] == fingerprint["points"]["auc"]
        )
        lines.append(
            "  [note]   标本指纹与多数类完全重合 -> 留一标本下指纹表对本折永远为空，"
            "记忆通道已被协议本身消除"
            if same
            else "  [!]      标本指纹高于多数类 -> 协议仍有标本重叠，先修划分再读指标"
        )
    return lines


def _print_dry_run(pool: Mapping[str, Any], args: argparse.Namespace, form: str, region_name: str) -> None:
    depth = pool["depth"]
    specimens = pool["specimens"]
    labels = labels_from_depth(depth, args.threshold_mm, "ge")
    unique, counts = np.unique(specimens, return_counts=True)
    convention = detect_convention(depth, pool["labels_stored"])
    print(f"data_dir      : {args.data_dir}")
    print(f"pooled rows   : {pool['x'].shape}  per split {pool['per_split']}")
    print(f"specimens     : {unique.size}")
    print(f"label rule    : depth >= {args.threshold_mm:g} mm")
    print(
        f"label balance : positive {int(labels.sum())} / {labels.size} "
        f"({float(labels.mean()):.3f})"
    )
    print(f"depth range   : {depth.min():.3f} .. {depth.max():.3f} mm")
    print(
        "stored y.npy  : reproduced by "
        + (
            f">= {convention['ge']:g} mm"
            if convention["ge"] is not None
            else (
                f"> {convention['gt']:g} mm"
                if convention["gt"] is not None
                else "neither boundary rule (stored labels do not follow depth_value)"
            )
        )
    )
    print(f"points/bone   : {counts.mean():.2f} (min {counts.min()}, max {counts.max()})")
    mixed = sum(1 for bone in unique if np.unique(labels[specimens == bone]).size > 1)
    print(f"mixed bones   : {mixed} of {unique.size} carry both labels")
    print(f"protocol      : form={form} region={region_name} source={args.feature_source}")
    print(f"probe         : {args.probe}, l2={args.l2:g}, inner folds={args.inner_folds}")
    print()
    print("fold composition (first 10 bones by size):")
    order = np.argsort(-counts)
    for index in order[:10]:
        bone = unique[index]
        mask = specimens == bone
        print(
            f"  {bone:<12} points={int(counts[index]):>3} "
            f"positive={int(labels[mask].sum()):>3} "
            f"depth {depth[mask].min():.3f}..{depth[mask].max():.3f} mm"
        )


def main() -> None:
    args = build_parser_loso().parse_args()
    start = time.time()
    form = _resolve_form(args)
    region_name, regions = _resolve_regions(args)
    pool = load_pool(args.data_dir)
    if args.dry_run:
        _print_dry_run(pool, args, form, region_name)
        print(f"\n[dry-run] {time.time() - start:.1f}s, nothing written")
        return
    labels = labels_from_depth(pool["depth"], args.threshold_mm, "ge")
    if np.unique(labels).size < 2:
        raise SystemExit(
            f"--threshold-mm {args.threshold_mm:g} puts every sample in one class "
            f"(depths run {pool['depth'].min():.3f}..{pool['depth'].max():.3f} mm)"
        )
    convention = detect_convention(pool["depth"], pool["labels_stored"])
    print(f"[loso] pooled {pool['x'].shape[0]} samples from {pool['per_split']}")
    print(
        f"[loso] labels from depth >= {args.threshold_mm:g} mm; "
        f"positives {int(labels.sum())}/{labels.size}"
    )
    print(
        "[loso] stored y.npy uses "
        + (
            f">= {convention['ge']:g} mm"
            if convention["ge"] is not None
            else (
                f"> {convention['gt']:g} mm"
                if convention["gt"] is not None
                else "neither boundary rule -- check samples.json"
            )
        )
    )
    features, names = build_features(args, form, region_name, regions, pool)
    print(
        f"[loso] features: {features.shape[1]} dimensions "
        f"(source={args.feature_source}, first column {names[0]!r})"
    )
    result = leave_one_specimen_out(
        np.asarray(features, dtype=np.float64),
        labels,
        pool["specimens"],
        l2=float(args.l2),
        inner_folds=int(args.inner_folds),
        seed=int(args.seed),
        probe=args.probe,
    )
    scores = score_run(result, labels, pool["specimens"])
    fences = fence_report(labels, pool["specimens"], scores, probe=args.probe)

    run_tag = args.run_tag or _default_run_tag(args, region_name, form)
    output_dir = Path(args.output_dir) / run_tag
    experiment_config = {
        "data_dir": str(args.data_dir),
        "threshold_mm": float(args.threshold_mm),
        "feature_source": args.feature_source,
        "form": form,
        "region_name": region_name,
        "regions": [
            [float(item.start_mm), float(item.end_mm), item.name] for item in regions
        ]
        if regions
        else [],
        "feature_dimension": int(features.shape[1]),
        "features": list(names),
        "probe": args.probe,
        "l2": float(args.l2),
        "inner_folds": int(args.inner_folds),
        "seed": int(args.seed),
        "channel_mode": int(args.channel_mode),
        "channel_aggregation": args.channel_aggregation,
        "feature_set": args.feature_set,
        "locator_features": args.locator_features,
        "channel_contrast": args.channel_contrast,
        "branch_ratio": args.branch_ratio,
        "stream_aggregation": args.stream_aggregation,
        "max_imfs": int(args.max_imfs),
        "target_length": int(args.target_length),
        "tukey_alpha": float(args.tukey_alpha),
        "protocol": "leave-one-specimen-out",
        "group_key": "point_id[0]",
    }
    guard_output_dir(output_dir, experiment_config, bool(args.allow_config_mismatch))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", experiment_config)
    write_json(
        output_dir / "metrics.json",
        {
            "config": experiment_config,
            "protocol": {
                "name": "leave-one-specimen-out",
                "folds": int(len(result["folds"])),
                "scored_folds": int(scores["per_fold"]["n"]),
                "skipped_folds": int(scores["skipped_folds"]),
                "scaler": "fitted on the training specimens of each fold",
                "threshold": "fitted on inner folds of the training specimens",
                "notes": [
                    "point level = every sample scored by a model that never saw its bone",
                    "specimen level = the point scores of one bone averaged into one decision",
                    "auc is threshold-free and is the primary number",
                    "accuracy_at_oracle_threshold is the best achievable on the scored "
                    "fold and is reportable only as an upper bound",
                ],
            },
            "pool": {
                "rows": int(pool["x"].shape[0]),
                "specimens": int(np.unique(pool["specimens"]).size),
                "per_split": pool["per_split"],
                "positives": int(labels.sum()),
                "label_rule": f"depth_value >= {args.threshold_mm:g}",
                "stored_label_convention": convention,
            },
            "metrics": scores,
            "fences": fences,
            "probe_diagnostics": design_diagnostics(
                args, int(features.shape[1]), int(pool["x"].shape[0]), pool["specimens"]
            ),
        },
    )
    write_json(
        output_dir / "folds.json",
        {
            "folds": result["folds"],
            "threshold_scan_grid": 201,
        },
    )
    write_scores(output_dir / "point_scores.csv", pool, labels, result)
    print_report(
        args, form, region_name, pool, labels, scores, fences, features.shape[1], start
    )
    print(f"  artefact: {output_dir}")


if __name__ == "__main__":
    main()

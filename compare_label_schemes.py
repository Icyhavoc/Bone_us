"""Compare bone-thickness label schemes on a fixed, scheme-independent split.

Why this exists
---------------
`raw_data/relabel_by_thickness.py` defaults to `stratified_random_split`, which
draws a *fresh* train/val/test assignment for every threshold scheme.  Training
on such a directory therefore changes two things at once -- the label rule and
the evaluation set -- so the resulting test accuracy cannot be compared with an
older run.  This script removes the split from the equation:

* it reuses the feature matrix already stored in an experiment directory
  (`features_{train,val,test}.npy` + `samples_{train,val,test}.json`), so no
  feature extraction is repeated;
* it pools every sample back into one matrix and folds that matrix by
  *depth quantile*, which does not depend on the labels at all;
* it then scores each label scheme on the *identical* folds.

Anything that differs between two schemes here is attributable to the label
rule, not to which samples happened to land in the test set.

Labels come from `depth_value` alone, using the same convention as
`relabel_by_thickness.py`: `np.digitize(depth, thresholds)`, thresholds are
inclusive lower bounds, so `n` thresholds give `n + 1` classes.

The unified protocol
--------------------
This script is the project's single comparison harness.  Every claim of the
form "X beats Y" is expected to go through it, because only here do the three
things that can change -- the label rule, the preprocessing, and the feature
columns -- vary *on one fixed fold set*:

* **label scheme** is `--schemes`, a list of comma-separated thresholds;
* **preprocessing** is `--scaler`, see :data:`SCALER_NAMES`;
* **feature columns** are `--feature-sets`, named column subsets such as
  `always3=15,14,8`;
* **model** is `--models`.

The cartesian product of the first three becomes one *variant* per row.  Because
all variants share the same folds, the report's delta section is **paired**: it
differences the two variants fold by fold and prints the mean difference with a
paired 95% interval.  Two numbers that differ by less than that interval are
not a result, and a single 67-sample test split cannot resolve them either --
that is the whole reason this script exists.

Usage
-----
    python compare_label_schemes.py \
        --experiment experiments/form_top3_mean__regions_dyn_envelope__channels_1 \
        --schemes 1.0 1.3 --folds 5 --repeats 5

    # label rule x preprocessing x feature subset, all paired
    python compare_label_schemes.py \
        --experiment experiments/form_top3_mean__regions_dyn_envelope__channels_1 \
        --schemes 1.3 --scalers standard,rank \
        --feature-sets all always3=15,14,8 --models lda,l1_logistic

Add `--models mlp,l2_logistic,lda,prior` to widen the comparison.  Paired
intervals need `--repeats` > 1 so the fold count is large enough to estimate a
standard error; one repeat is usually too few to call anything significant.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from baseline_evaluation import METRIC_KEYS, PriorClassifier, _metrics_from_probabilities
from emd_pipeline import MLPConfig, NumpyMLPClassifier, StandardScaler
from shallow_models import SHALLOW_MODELS

SPLITS = ("train", "val", "test")
DEFAULT_MODELS = ("mlp", "l2_logistic", "lda", "prior")

#: Metrics the paired delta block reports.  ``accuracy`` and
#: ``balanced_accuracy`` are thresholded at 0.5 and therefore move with class
#: balance; ``auc`` is the one that survives a label-rule change most intact.
PAIRED_METRIC_KEYS = ("auc", "balanced_accuracy", "accuracy")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class PooledFeatures:
    """One feature matrix plus the metadata needed to relabel it."""

    features: np.ndarray
    depth: np.ndarray
    point_ids: list[str]
    stored_labels: np.ndarray
    feature_dim: int

    @property
    def size(self) -> int:
        return int(self.features.shape[0])


def load_pooled_features(experiment_dir: Path) -> PooledFeatures:
    """Pool the stored per-split features back into a single matrix.

    Rows are ordered by ``point_id`` so the result is independent of the order
    the splits happen to be written in.
    """

    features: list[np.ndarray] = []
    depth: list[float] = []
    ids: list[str] = []
    labels: list[int] = []
    for split in SPLITS:
        matrix_path = experiment_dir / f"features_{split}.npy"
        samples_path = experiment_dir / f"samples_{split}.json"
        if not matrix_path.is_file():
            raise FileNotFoundError(f"missing feature matrix: {matrix_path}")
        if not samples_path.is_file():
            raise FileNotFoundError(f"missing sample metadata: {samples_path}")
        matrix = np.load(matrix_path)
        records = json.loads(samples_path.read_text(encoding="utf-8"))
        if matrix.shape[0] != len(records):
            raise ValueError(
                f"{matrix_path.name} has {matrix.shape[0]} rows but {samples_path.name} "
                f"describes {len(records)} samples"
            )
        for index, record in enumerate(records):
            if "point_id" not in record or "depth_value" not in record:
                raise ValueError(
                    f"{samples_path.name} row {index} lacks point_id/depth_value; "
                    "cannot pool samples without them"
                )
        features.append(np.asarray(matrix, dtype=np.float64))
        depth.extend(float(record["depth_value"]) for record in records)
        ids.extend(str(record["point_id"]) for record in records)
        labels.extend(int(record["label"]) for record in records)

    order = np.argsort(np.asarray(ids), kind="stable")
    pooled = np.concatenate(features, axis=0)[order]
    return PooledFeatures(
        features=pooled,
        depth=np.asarray(depth, dtype=np.float64)[order],
        point_ids=[ids[index] for index in order],
        stored_labels=np.asarray(labels, dtype=np.int64)[order],
        feature_dim=int(pooled.shape[1]),
    )


# --------------------------------------------------------------------------- #
# Labels and folds
# --------------------------------------------------------------------------- #
def labels_for_thresholds(depth: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """`np.digitize(depth, thresholds)`, matching relabel_by_thickness.py."""

    ordered = sorted(float(value) for value in thresholds)
    return np.digitize(np.asarray(depth, dtype=np.float64), ordered).astype(np.int64)


def scheme_name(thresholds: Sequence[float]) -> str:
    return f"thr{'/'.join(f'{float(value):g}' for value in thresholds)}"


def depth_quantile_folds(
    depth: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold on depth quantiles so the partition never sees the labels.

    Samples are sorted by depth and dealt round-robin into folds from a random
    offset.  Every fold therefore covers the same thickness range, which keeps
    the comparison between label schemes paired: the same sample sits on the
    same side of the split no matter which threshold is being tested.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    values = np.asarray(depth, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    all_indices = np.arange(values.size)
    order = np.argsort(values, kind="stable")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_repeats):
        offset = int(rng.integers(n_splits))
        assignment = np.empty(values.size, dtype=np.int64)
        for position, index in enumerate(order):
            assignment[index] = (position + offset) % n_splits
        for fold in range(n_splits):
            held_out = all_indices[assignment == fold]
            if held_out.size == 0:
                continue
            folds.append((all_indices[assignment != fold], held_out))
    return folds


# --------------------------------------------------------------------------- #
# Scalers
# --------------------------------------------------------------------------- #
def _norm_ppf(probability: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Implemented here because the project is NumPy-only and SciPy's ``ndtri`` is
    not available.  Relative error is below 1.2e-9 across the whole (0, 1)
    range, which is far tighter than anything a 161-sample fold can resolve.
    """

    p = np.asarray(probability, dtype=np.float64)
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    low, high = 0.02425, 1.0 - 0.02425
    result = np.empty_like(p)
    lower = p < low
    upper = p > high
    middle = ~(lower | upper)
    if lower.any():
        q = np.sqrt(-2.0 * np.log(p[lower]))
        result[lower] = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if upper.any():
        q = np.sqrt(-2.0 * np.log(1.0 - p[upper]))
        result[upper] = -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if middle.any():
        q = p[middle] - 0.5
        r = q * q
        result[middle] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return result


class RobustScaler:
    """Column-wise median / (IQR / 1.349).

    The IQR divisor makes the scale *equal* to the standard deviation for a
    normal column, so `robust` and `standard` are directly comparable on
    well-behaved data and only diverge where outliers land.  That is the point:
    several of the amplitude columns have a long right tail, and ``standard``
    lets a handful of large samples set the scale for everybody.
    """

    NORMAL_IQR = 1.3489795003921634

    def __init__(self, degenerate_tolerance: float = 1e-12) -> None:
        self.degenerate_tolerance = float(degenerate_tolerance)
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.degenerate_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "RobustScaler":
        values = np.asarray(values, dtype=np.float64)
        self.center_ = np.median(values, axis=0)
        lower, upper = np.percentile(values, [25.0, 75.0], axis=0)
        scale = (upper - lower) / self.NORMAL_IQR
        self.degenerate_ = np.asarray(scale <= self.degenerate_tolerance, dtype=bool)
        self.scale_ = np.where(self.degenerate_, 1.0, scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("RobustScaler must be fitted before transform")
        scaled = (np.asarray(values, dtype=np.float64) - self.center_) / self.scale_
        if self.degenerate_ is not None and self.degenerate_.any():
            scaled[:, self.degenerate_] = 0.0
        return scaled

    @property
    def degenerate_indices(self) -> list[int]:
        if self.degenerate_ is None:
            return []
        return np.flatnonzero(self.degenerate_).astype(int).tolist()


class RankScaler:
    """Column-wise rank transform mapped through the inverse normal CDF.

    Every column is replaced by the position of its value inside the *training*
    fold's own distribution, then pushed through ``norm_ppf``.  The map is
    strictly monotone, so a model that only reads ordering is unchanged, while a
    model that reads distances stops seeing the raw amplitude scale at all.

    This is the natural control for the amplitude-statistic columns: if a column
    only helps because it encodes gain/coupling/exponent scale, rank-scaling
    removes that and the fold scores should collapse.
    """

    def __init__(self, epsilon: float = 1e-6, degenerate_tolerance: float = 1e-12) -> None:
        self.epsilon = float(epsilon)
        self.degenerate_tolerance = float(degenerate_tolerance)
        self.sorted_: np.ndarray | None = None
        self.degenerate_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "RankScaler":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("RankScaler expects a 2-D matrix")
        self.sorted_ = np.sort(values, axis=0)
        spread = self.sorted_[-1] - self.sorted_[0]
        self.degenerate_ = np.asarray(spread <= self.degenerate_tolerance, dtype=bool)
        return self

    def _column_probabilities(self, column: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Average-rank empirical CDF of ``column`` in ``reference``."""

        below = np.searchsorted(reference, column, side="left")
        above = np.searchsorted(reference, column, side="right")
        # Average rank + 0.5 makes ties get the midpoint of their block, and the
        # +0.5/(n+1) style shift keeps the extremes strictly inside (0, 1).
        rank = 0.5 * (below + above) + 0.5
        return rank / (reference.size + 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.sorted_ is None:
            raise RuntimeError("RankScaler must be fitted before transform")
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.sorted_.shape[1]:
            raise ValueError("RankScaler.transform received a matrix of the wrong width")
        output = np.empty(values.shape, dtype=np.float64)
        for column in range(values.shape[1]):
            if self.degenerate_ is not None and self.degenerate_[column]:
                output[:, column] = 0.0
                continue
            probabilities = self._column_probabilities(values[:, column], self.sorted_[:, column])
            probabilities = np.clip(probabilities, self.epsilon, 1.0 - self.epsilon)
            output[:, column] = _norm_ppf(probabilities)
        return output

    @property
    def degenerate_indices(self) -> list[int]:
        if self.degenerate_ is None:
            return []
        return np.flatnonzero(self.degenerate_).astype(int).tolist()


SCALER_NAMES: dict[str, Callable[[], object]] = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "rank": RankScaler,
}


def make_scaler(name: str):
    """Instantiate one of :data:`SCALER_NAMES`, or explain the valid choices."""

    if name not in SCALER_NAMES:
        available = ", ".join(SCALER_NAMES)
        raise ValueError(f"unknown scaler {name!r}; available: {available}")
    return SCALER_NAMES[name]()


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class MLPWrapper:
    """Sklearn-shaped wrapper around `NumpyMLPClassifier`.

    The pipeline's MLP stops on a validation split, so the wrapper carves a
    stratified slice out of the fold's training data and early-stops on that.
    The held-out fold is never used for stopping.
    """

    def __init__(self, num_classes: int = 2, seed: int = 42, val_fraction: float = 0.2):
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.val_fraction = float(val_fraction)
        self.model: NumpyMLPClassifier | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "MLPWrapper":
        matrix = np.asarray(features, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64).ravel()
        train_index, val_index = _inner_validation_split(targets, self.val_fraction, self.seed)
        config = MLPConfig(
            hidden_dims=(64, 32),
            num_classes=self.num_classes,
            seed=self.seed,
        )
        self.model = NumpyMLPClassifier(
            input_dim=matrix.shape[1], config=config, group_dims=(matrix.shape[1],)
        )
        self.model.fit(
            matrix[train_index].astype(np.float32),
            targets[train_index],
            matrix[val_index].astype(np.float32),
            targets[val_index],
        )
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLPWrapper must be fitted before predict_proba")
        return self.model.predict_proba(np.asarray(features, dtype=np.float32))


def _inner_validation_split(
    labels: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified inner train/val split that always leaves both sides non-empty."""

    targets = np.asarray(labels).ravel()
    rng = np.random.default_rng(seed)
    val_chunks: list[np.ndarray] = []
    train_chunks: list[np.ndarray] = []
    for value in np.unique(targets):
        members = np.flatnonzero(targets == value)
        members = members[rng.permutation(members.size)]
        n_val = int(round(members.size * fraction))
        n_val = max(1, min(n_val, members.size - 1))
        val_chunks.append(members[:n_val])
        train_chunks.append(members[n_val:])
    train_index = np.sort(np.concatenate(train_chunks))
    val_index = np.sort(np.concatenate(val_chunks))
    return train_index, val_index


def _model_factory(name: str, num_classes: int, seed: int) -> Callable[[], object]:
    if name == "mlp":
        return lambda: MLPWrapper(num_classes=num_classes, seed=seed)
    if name == "prior":
        return lambda: PriorClassifier()
    if name in SHALLOW_MODELS:
        return lambda: SHALLOW_MODELS[name](seed=seed)
    available = ", ".join(["mlp", "prior", *SHALLOW_MODELS])
    raise ValueError(f"unknown model {name!r}; available: {available}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def cross_validate_scheme(
    features: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    model_names: Sequence[str],
    seed: int,
    scaler_name: str = "standard",
) -> dict[str, dict[str, object]]:
    """Score every model on identical folds with one preprocessing choice.

    Returns ``model -> {"summary": ..., "folds": [...]}``.  The fold list has
    one entry per element of ``folds`` and uses ``None`` where a fold was
    skipped (a single-class training side), so two runs over the same fold list
    stay index-aligned and can be difference fold by fold.
    """

    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).ravel()
    num_classes = int(np.unique(targets).size)
    results: dict[str, dict[str, object]] = {}
    for name in model_names:
        fold_metrics: list[dict[str, object] | None] = []
        for train_index, validation_index in folds:
            if np.unique(targets[train_index]).size < num_classes:
                fold_metrics.append(None)
                continue
            train_features = matrix[train_index]
            validation_features = matrix[validation_index]
            scaler = make_scaler(scaler_name).fit(train_features)
            train_features = scaler.transform(train_features)
            validation_features = scaler.transform(validation_features)
            model = _model_factory(name, num_classes, seed)()
            model.fit(train_features, targets[train_index])
            probabilities = model.predict_proba(validation_features)
            fold_metrics.append(
                _metrics_from_probabilities(targets[validation_index], probabilities)
            )
        results[name] = {
            "summary": summarise_folds(fold_metrics),
            "folds": fold_metrics,
        }
    return results


def summarise_folds(records: Sequence[dict[str, object] | None]) -> dict[str, object]:
    """Mean/std/count for each metric, skipping metrics that are all None."""

    present = [record for record in records if record is not None]
    summary: dict[str, object] = {
        "n_folds": len(present),
        "n_folds_skipped": len(records) - len(present),
    }
    for key in METRIC_KEYS:
        values = [record.get(key) for record in present]
        numbers = np.asarray(
            [float(value) for value in values if value is not None], dtype=np.float64
        )
        if numbers.size == 0:
            summary[key] = None
            summary[f"{key}_std"] = None
            summary[f"{key}_n"] = 0
            continue
        summary[key] = float(numbers.mean())
        summary[f"{key}_std"] = float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0
        summary[f"{key}_n"] = int(numbers.size)
    return summary


def paired_delta(
    base_folds: Sequence[dict[str, object] | None],
    other_folds: Sequence[dict[str, object] | None],
    key: str,
) -> dict[str, object] | None:
    """Fold-by-fold difference of one metric between two variants.

    Differencing inside the fold removes the fold-to-fold spread, which for a
    rubric with 5-8 validation samples per fold is far larger than the effect
    being measured.  The returned interval is a paired 95% interval using the
    Student factor for the actual number of paired folds; it is deliberately
    conservative rather than approximate-normal.
    """

    differences: list[float] = []
    for base, other in zip(base_folds, other_folds):
        if base is None or other is None:
            continue
        a, b = base.get(key), other.get(key)
        if a is None or b is None:
            continue
        differences.append(float(b) - float(a))
    if len(differences) < 2:
        return None
    values = np.asarray(differences, dtype=np.float64)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    half_width = _student95(values.size - 1) * standard_error
    return {
        "n_paired": int(values.size),
        "mean": mean,
        "std": float(values.std(ddof=1)),
        "standard_error": standard_error,
        "half_width_95": half_width,
        "separated": bool(abs(mean) > half_width),
    }


# Two-sided 95% Student factor by degrees of freedom.  Fold counts here are
# always small (5 x repeats), so the normal 1.96 understates the interval.
_STUDENT95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _student95(degrees_of_freedom: int) -> float:
    """Student factor for 95% two-sided coverage, clamped to the table range."""

    if degrees_of_freedom <= 1:
        return _STUDENT95[1]
    if degrees_of_freedom >= 30:
        # Converges to 1.9600; 1.96 is within 0.2% of the true value by df=30.
        return 1.960
    return _STUDENT95[degrees_of_freedom]


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSet:
    """A named subset of the pooled feature columns.

    ``columns is None`` means "every column as stored", which is the historical
    behaviour and the right default for a label-scheme-only comparison.
    """

    name: str
    columns: tuple[int, ...] | None = None

    def select(self, matrix: np.ndarray) -> np.ndarray:
        if self.columns is None:
            return matrix
        return matrix[:, list(self.columns)]

    @property
    def width(self) -> int | None:
        return None if self.columns is None else len(self.columns)


@dataclass(frozen=True)
class Variant:
    """One point in the label x preprocessing x feature-column grid."""

    name: str
    scheme: str
    thresholds: tuple[float, ...]
    scaler: str
    feature_set: FeatureSet


def build_variants(
    schemes: Sequence[tuple[str, list[float]]],
    scaler_names: Sequence[str],
    feature_sets: Sequence[FeatureSet],
) -> list[Variant]:
    """Cartesian product of the three axes, with a minimal readable name.

    The scaler and feature-set suffixes are only appended when that axis
    actually has more than one setting, so a plain label-scheme run keeps the
    terse historical names (`thr1.3`).
    """

    show_scaler = len(scaler_names) > 1
    show_feature = len(feature_sets) > 1
    variants: list[Variant] = []
    for (scheme, thresholds), scaler, feature_set in product(schemes, scaler_names, feature_sets):
        parts = [scheme]
        if show_scaler:
            parts.append(f"@{scaler}")
        if show_feature:
            parts.append(f"[{feature_set.name}]")
        variants.append(
            Variant(
                name="".join(parts),
                scheme=scheme,
                thresholds=tuple(thresholds),
                scaler=scaler,
                feature_set=feature_set,
            )
        )
    return variants


def parse_feature_sets(raw: Iterable[str], feature_dim: int) -> list[FeatureSet]:
    """Parse ``all`` / ``15,14,8`` / ``always3=15,14,8`` specifications.

    Bare column lists get an auto-generated name so a quick one-off subset does
    not need a label.  Indices are validated against the stored width, because
    a silently out-of-range column would otherwise only show up as a confusing
    shape error deep inside a model.
    """

    parsed: list[FeatureSet] = []
    for item in raw:
        text = item.strip()
        if not text:
            continue
        if text.lower() in {"all", "*"}:
            parsed.append(FeatureSet(name="all", columns=None))
            continue
        if "=" in text:
            name, _, column_text = text.partition("=")
            name = name.strip()
            if not name:
                raise ValueError(f"feature set {item!r} has an empty name")
        else:
            name, column_text = "", text
        columns = tuple(int(part) for part in column_text.split(",") if part.strip())
        if not columns:
            raise ValueError(f"feature set {item!r} lists no columns")
        if len(set(columns)) != len(columns):
            raise ValueError(f"feature set {item!r} repeats a column index")
        out_of_range = [index for index in columns if not 0 <= index < feature_dim]
        if out_of_range:
            raise ValueError(
                f"feature set {item!r} references columns {out_of_range} but the "
                f"stored matrix only has {feature_dim} columns"
            )
        parsed.append(FeatureSet(name=name or ",".join(str(index) for index in columns), columns=columns))
    if not parsed:
        raise ValueError("at least one feature set is required")
    return parsed


def _dedupe(preserving: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in preserving:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(value: object, width: int = 7, digits: int = 3) -> str:
    if value is None:
        return " " * (width - 3) + "n/a"
    return f"{float(value):{width}.{digits}f}"


def format_report(
    variants: dict[str, dict[str, dict[str, object]]],
    features: PooledFeatures,
    variant_labels: dict[str, np.ndarray],
    base_name: str,
) -> str:
    """Tabulate every variant and pair each one against ``base_name``.

    Every variant shares the fold list, so the delta block is a genuine paired
    comparison even when the two variants differ in the label rule, the scaler
    and the feature columns at the same time.
    """

    name_width = max(16, max((len(name) for name in variants), default=16))
    lines: list[str] = []
    lines.append("=" * 108)
    lines.append("Paired comparison on a scheme-independent fold set")
    lines.append("=" * 108)
    lines.append(f"samples pooled      : {features.size}")
    lines.append(f"feature dimension   : {features.feature_dim}")
    lines.append(f"depth range         : {features.depth.min():.3f} - {features.depth.max():.3f} mm")
    lines.append(
        "folds               : stratified by depth quantile (labels never enter the split)"
    )
    lines.append("variants            : label scheme x scaler x feature columns")
    lines.append("")

    header = (
        f"{'variant':>{name_width}s} {'model':>13s} {'cols':>5s} {'floor':>7s} "
        f"{'acc':>7s} {'acc_std':>8s} {'bal_acc':>8s} {'auc':>7s} {'auc_std':>8s} {'f1':>7s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, per_model in variants.items():
        targets = np.asarray(variant_labels[name], dtype=np.int64).ravel()
        counts = np.bincount(targets)
        floor = float(counts.max() / targets.size)
        for model_name, payload in per_model.items():
            summary = payload["summary"]  # type: ignore[index]
            width = payload.get("feature_width")  # type: ignore[union-attr]
            lines.append(
                f"{name:>{name_width}s} {model_name:>13s} {str(width):>5s} {floor:7.3f} "
                f"{_fmt(summary.get('accuracy'))} {_fmt(summary.get('accuracy_std'), 8)} "
                f"{_fmt(summary.get('balanced_accuracy'), 8)} "
                f"{_fmt(summary.get('auc'))} {_fmt(summary.get('auc_std'), 8)} "
                f"{_fmt(summary.get('f1_score'))}"
            )
        lines.append("")

    ordered = list(variants)
    if len(ordered) > 1:
        lines.append("-" * len(header))
        lines.append(
            f"paired deltas vs '{base_name}'  (same folds; delta = other - base; "
            "'*' = 95% paired interval excludes 0)"
        )
        lines.append("")
        for other in ordered:
            if other == base_name:
                continue
            lines.append(f"  {base_name} -> {other}")
            for model_name in variants[base_name]:
                base_payload = variants[base_name][model_name]
                other_payload = variants[other][model_name]
                base_folds = base_payload.get("folds")  # type: ignore[union-attr]
                other_folds = other_payload.get("folds")  # type: ignore[union-attr]
                for key in PAIRED_METRIC_KEYS:
                    base_summary = base_payload["summary"]  # type: ignore[index]
                    other_summary = other_payload["summary"]  # type: ignore[index]
                    a, b = base_summary.get(key), other_summary.get(key)
                    if a is None or b is None:
                        continue
                    delta = paired_delta(base_folds, other_folds, key)  # type: ignore[arg-type]
                    if delta is None:
                        lines.append(
                            f"    {model_name:>13s} {key:<18s} {float(a):7.3f} -> "
                            f"{float(b):7.3f}  (paired interval unavailable)"
                        )
                        continue
                    marker = "*" if delta["separated"] else " "
                    lines.append(
                        f"    {model_name:>13s} {key:<18s} {float(a):7.3f} -> {float(b):7.3f}  "
                        f"({float(delta['mean']):+.3f} +- {float(delta['half_width_95']):.3f}) "
                        f"{marker} n={delta['n_paired']}"
                    )
            lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score label scheme x preprocessing x feature-subset variants on one "
            "depth-stratified fold set, reusing features already stored in an "
            "experiment directory."
        )
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="实验目录，需含 features_{train,val,test}.npy 与 samples_{train,val,test}.json",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["1.0", "1.3"],
        help="每个方案是一组逗号分隔的阈值，如 1.0 1.3 或 0.8,1.2",
    )
    parser.add_argument(
        "--scalers",
        default="standard",
        help=(
            "逗号分隔的预处理器，可选 "
            f"{'/'.join(SCALER_NAMES)}；给多个时会在同一折集上配对对比"
        ),
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=None,
        help=(
            "逗号分隔的列下标，可写成 name=0,1,2 或直接 0,1,2；all 表示全部列。"
            "给多个时会在同一折集上配对对比"
        ),
    )
    parser.add_argument("--folds", type=int, default=5, help="折数（默认 5）")
    parser.add_argument("--repeats", type=int, default=5, help="重复次数（默认 5）")
    parser.add_argument("--seed", type=int, default=42, help="划分与模型随机种子")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"逗号分隔的模型名（默认 {','.join(DEFAULT_MODELS)}）",
    )
    parser.add_argument(
        "--base-variant",
        default=None,
        help="配对对比的基准变体名（默认第一个）；名字即报告第一列的内容",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="写出 comparison_report.txt 与 comparison_results.json 的目录",
    )
    return parser


def parse_schemes(raw: Iterable[str]) -> list[tuple[str, list[float]]]:
    schemes: list[tuple[str, list[float]]] = []
    for item in raw:
        thresholds = [float(part) for part in item.replace("/", ",").split(",") if part.strip()]
        if not thresholds:
            raise ValueError(f"empty threshold list in {item!r}")
        schemes.append((scheme_name(thresholds), thresholds))
    return schemes


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features = load_pooled_features(args.experiment)
    schemes = parse_schemes(args.schemes)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    scaler_names = _dedupe([name.strip() for name in args.scalers.split(",") if name.strip()])
    if not scaler_names:
        raise ValueError("at least one scaler is required")
    for scaler_name in scaler_names:
        if scaler_name not in SCALER_NAMES:
            available = ", ".join(SCALER_NAMES)
            raise ValueError(f"unknown scaler {scaler_name!r}; available: {available}")
    feature_sets = parse_feature_sets(
        args.feature_sets if args.feature_sets is not None else ["all"], features.feature_dim
    )

    variants = build_variants(schemes, scaler_names, feature_sets)
    folds = depth_quantile_folds(features.depth, args.folds, args.repeats, args.seed)
    pool_labels = {
        name: labels_for_thresholds(features.depth, thresholds) for name, thresholds in schemes
    }

    results: dict[str, object] = {
        "experiment": str(args.experiment),
        "num_samples": features.size,
        "feature_dim": features.feature_dim,
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "models": model_names,
        "scalers": scaler_names,
        "feature_sets": {
            item.name: (None if item.columns is None else list(item.columns))
            for item in feature_sets
        },
        "schemes": {name: {"thresholds_mm": thresholds} for name, thresholds in schemes},
        "variants": {variant.name: _variant_metadata(variant) for variant in variants},
        "results": {},
        "paired_deltas": {},
    }

    print(f"[pool] {features.size} samples, {features.feature_dim} features")
    print(f"[folds] {len(folds)} folds ({args.folds} x {args.repeats}), depth-stratified")
    print(f"[grid] {len(variants)} variants = {len(schemes)} schemes x {len(scaler_names)} scalers x {len(feature_sets)} feature sets")

    per_variant: dict[str, dict[str, dict[str, object]]] = {}
    for variant in variants:
        targets = pool_labels[variant.scheme]
        matrix = variant.feature_set.select(features.features)
        counts = np.bincount(targets)
        print(
            f"[variant] {variant.name}: cols={matrix.shape[1]} scaler={variant.scaler} "
            f"classes={'/'.join(str(int(value)) for value in counts)}"
        )
        payload = cross_validate_scheme(
            matrix, targets, folds, model_names, args.seed, scaler_name=variant.scaler
        )
        for model_payload in payload.values():
            model_payload["feature_width"] = int(matrix.shape[1])
        per_variant[variant.name] = payload
        results["results"][variant.name] = {  # type: ignore[index]
            model: dict(payload_entry) for model, payload_entry in payload.items()
        }

    base_name = args.base_variant or variants[0].name
    if base_name not in per_variant:
        available = ", ".join(per_variant)
        raise ValueError(f"unknown --base-variant {base_name!r}; available: {available}")
    variant_labels = {variant.name: pool_labels[variant.scheme] for variant in variants}

    # Persist the paired deltas as data, not just as report text, so a later
    # script can read the interval without re-running the folds.
    paired: dict[str, dict[str, dict[str, object]]] = {}
    for other in per_variant:
        if other == base_name:
            continue
        per_model: dict[str, dict[str, object]] = {}
        for model_name in per_variant[base_name]:
            per_model[model_name] = {
                key: paired_delta(
                    per_variant[base_name][model_name].get("folds"),  # type: ignore[arg-type]
                    per_variant[other][model_name].get("folds"),  # type: ignore[arg-type]
                    key,
                )
                for key in PAIRED_METRIC_KEYS
            }
        paired[other] = per_model
    results["paired_deltas"] = {"base": base_name, "comparisons": paired}

    report = format_report(per_variant, features, variant_labels, base_name)
    print()
    print(report)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "comparison_report.txt").write_text(
            report + "\n", encoding="utf-8"
        )
        (args.output_dir / "comparison_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print()
        print(f"written: {args.output_dir / 'comparison_report.txt'}")
        print(f"written: {args.output_dir / 'comparison_results.json'}")
    return 0


def _variant_metadata(variant: Variant) -> dict[str, object]:
    return {
        "scheme": variant.scheme,
        "thresholds_mm": list(variant.thresholds),
        "scaler": variant.scaler,
        "feature_set": variant.feature_set.name,
        "columns": None if variant.feature_set.columns is None else list(variant.feature_set.columns),
    }


if __name__ == "__main__":
    raise SystemExit(main())

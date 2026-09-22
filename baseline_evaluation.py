"""Cross-validated evaluation harness for the shallow baselines.

The point of this module is to make one comparison trustworthy: **do the hand-made
EMD features support a simple model as well as the MLP, once everything is
measured the same way?**

Three rules make the numbers comparable to the stored MLP results:

1. The scaler is re-fitted **inside every fold** on that fold's training part.
   The stored ``features_*.npy`` are raw (the pipeline's ``StandardScaler`` is
   applied internally at training time and only the fitted statistics are
   written out as ``standard_scaler.npz``), so a baseline that read those columns
   unscaled would be comparing "raw features + linear model" against "scaled
   features + MLP" and would look far worse than the protocol deserves.
2. For the final number the scaler is fitted on the **whole training split** and
   the model is fitted on a single pass over it.  That is exactly what
   ``run_emd_experiments.py`` does before scoring ``val``/``test``, so the
   baseline and the MLP face the same fitted-once model.
3. ``val`` and ``test`` are read only at the end.  Anything that looks at a
   score in order to make a choice -- hyper-parameters, the feature subset -- is
   driven by folds inside ``train`` only.

Every metric is returned per split, and the fold results are summarised with a
mean, a standard deviation, and the number of folds that produced a value, so a
metric that silently became undefined (AUC with one class present) is visible
instead of being quietly averaged away.

All of this is class-count agnostic: the metrics come from
``emd_pipeline.classification_metrics``, which reproduces the historical
``binary_metrics`` values verbatim for two classes and switches to the macro
one-vs-rest statistics (``f1_score``, ``sensitivity``, ``specificity``, ``auc``)
beyond two.  A model therefore only has to return one probability column per
class for the three-class datasets to be scored by the same code path.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

from emd_pipeline import StandardScaler, classification_metrics

__all__ = [
    "PriorClassifier",
    "stratified_folds",
    "cross_validate",
    "holdout_evaluate",
    "summarise",
    "METRIC_KEYS",
]

#: Reported in this order everywhere, so tables diff cleanly between runs.
METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "auc",
    "f1_score",
    "sensitivity",
    "specificity",
)


class PriorClassifier:
    """Predict the training prior for every sample: the "no features" floor.

    Its score is constant, so its AUC is 0.5 by the tie-handling rule in
    ``_roc_auc`` -- which is the honest value for a model that cannot rank at
    all, and a useful check that a reported AUC is not accidentally measuring
    something else.

    Unlike the five learners in ``shallow_models`` this one is not a binary
    model wearing a multiclass adapter: the prior *is* a k-vector of class
    frequencies, so a three-class run uses the class-frequency columns directly
    and ``argmax`` lands on the majority class.  The two-class case keeps the
    historical scalar ``1 - p`` / ``p`` columns.
    """

    name = "prior"

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.prior_: float = 0.5
        self.priors_: np.ndarray = np.asarray([0.5, 0.5], dtype=np.float64)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PriorClassifier":
        labels = np.asarray(y, dtype=np.float64).ravel()
        self.prior_ = float(np.mean(labels))
        if labels.size == 0:
            self.priors_ = np.asarray([], dtype=np.float64)
        else:
            integers = labels.astype(np.int64)
            counts = np.bincount(integers, minlength=int(integers.max()) + 1)
            self.priors_ = counts.astype(np.float64) / float(labels.size)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        features = np.asarray(x)
        if self.priors_.size <= 2:
            positives = np.full(features.shape[0], self.prior_, dtype=np.float64)
            return np.column_stack([1.0 - positives, positives])
        return np.tile(self.priors_, (features.shape[0], 1))


def stratified_folds(
    labels: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated stratified K-fold, as ``(train_index, validation_index)`` pairs.

    Class proportions are preserved exactly wherever the counts allow: each
    class's indices are shuffled independently and dealt out round-robin, so a
    fold differs from the overall proportion by at most one sample.  With 73/88
    samples that matters -- an unstratified fold can easily land at 60/40 and
    make the accuracy of the small folds meaningless.
    """

    labels = np.asarray(labels).ravel()
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    classes = np.unique(labels)
    if classes.size > n_splits:
        raise ValueError("n_splits cannot exceed the number of classes present")
    rng = np.random.default_rng(seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_repeats):
        assignment = np.full(labels.size, -1, dtype=np.int64)
        for value in classes:
            members = np.flatnonzero(labels == value)
            members = members[rng.permutation(members.size)]
            # Round-robin from a random offset keeps the folds balanced even when
            # a class count is not a multiple of n_splits.
            offset = int(rng.integers(n_splits))
            for position, index in enumerate(members):
                assignment[index] = (position + offset) % n_splits
        if np.any(assignment < 0):
            raise AssertionError("every sample must be assigned to a fold")
        all_indices = np.arange(labels.size)
        for fold in range(n_splits):
            held_out = all_indices[assignment == fold]
            if held_out.size == 0:
                continue
            folds.append((all_indices[assignment != fold], held_out))
    return folds


def _metrics_from_probabilities(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | int | None]:
    """Score one prediction matrix, inferring the class count from its width.

    ``classification_metrics`` returns ``balanced_accuracy`` itself for more than
    two classes, and for two classes it delegates to ``binary_metrics``, so the
    two-class numbers here are the same as before; only the extra macro keys are
    new.  ``balanced_accuracy`` is therefore computed explicitly only when it is
    still missing, which is the binary case.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    num_classes = int(probabilities.shape[1]) if probabilities.ndim == 2 else 2
    record = dict(classification_metrics(labels, probabilities, num_classes))
    if record.get("balanced_accuracy") is None:
        sensitivity = record.get("sensitivity")
        specificity = record.get("specificity")
        record["balanced_accuracy"] = (
            None
            if sensitivity is None or specificity is None
            else 0.5 * (float(sensitivity) + float(specificity))
        )
    return record


def cross_validate(
    model_factory: Callable[[], Any],
    features: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    standardize: bool = True,
) -> dict[str, Any]:
    """Fit and score ``model_factory()`` on every fold.

    Returns ``{"folds": [...], "summary": {...}}`` where each fold record carries
    its metrics plus the fitted model and scaler, so a caller can inspect
    coefficients without refitting.
    """

    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels).ravel()
    if matrix.shape[0] != targets.size:
        raise ValueError("features and labels disagree on the number of samples")
    records: list[dict[str, Any]] = []
    for train_index, validation_index in folds:
        train_features, validation_features = matrix[train_index], matrix[validation_index]
        scaler = None
        if standardize:
            scaler = StandardScaler().fit(train_features)
            train_features = scaler.transform(train_features)
            validation_features = scaler.transform(validation_features)
        model = model_factory()
        model.fit(train_features, targets[train_index])
        probabilities = model.predict_proba(validation_features)
        records.append(
            {
                "metrics": _metrics_from_probabilities(targets[validation_index], probabilities),
                "n_train": int(train_index.size),
                "n_validation": int(validation_index.size),
                "model": model,
                "scaler": scaler,
                "validation_index": validation_index,
            }
        )
    return {"folds": records, "summary": summarise([record["metrics"] for record in records])}


def holdout_evaluate(
    model_factory: Callable[[], Any],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    standardize: bool = True,
) -> dict[str, Any]:
    """Fit once on the whole training split and score every named split.

    ``splits`` maps a split name to ``(features, labels)``.  The scaler and model
    are returned so the caller can persist or inspect them, and so the same
    fitted object can be reused (e.g. to add ``test`` to an already scored run).
    """

    matrix = np.asarray(train_features, dtype=np.float64)
    targets = np.asarray(train_labels).ravel()
    scaler = None
    fitted_on = matrix
    if standardize:
        scaler = StandardScaler().fit(matrix)
        fitted_on = scaler.transform(matrix)
    model = model_factory()
    model.fit(fitted_on, targets)
    scored: dict[str, dict[str, float | int | None]] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, (split_features, split_labels) in splits.items():
        values = np.asarray(split_features, dtype=np.float64)
        if scaler is not None:
            values = scaler.transform(values)
        probability = model.predict_proba(values)
        probabilities[name] = probability
        scored[name] = _metrics_from_probabilities(np.asarray(split_labels).ravel(), probability)
    return {
        "metrics": scored,
        "probabilities": probabilities,
        "model": model,
        "scaler": scaler,
    }


def summarise(records: Iterable[dict[str, Any]], keys: Sequence[str] = METRIC_KEYS) -> dict[str, Any]:
    """Mean/standard deviation/count per metric, skipping undefined entries."""

    records = list(records)
    summary: dict[str, Any] = {"n": len(records)}
    for key in keys:
        values = [
            float(record[key])
            for record in records
            if record.get(key) is not None
        ]
        if not values:
            summary[key] = {"mean": None, "std": None, "n": 0}
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }
    return summary


def format_summary(summary: dict[str, Any], keys: Sequence[str] = METRIC_KEYS) -> str:
    """One compact line per metric, safe for a GBK console (ASCII only)."""

    parts = []
    for key in keys:
        entry = summary.get(key) or {}
        mean, std = entry.get("mean"), entry.get("std")
        if mean is None:
            parts.append(f"{key}=n/a")
        elif std is None or entry.get("n", 0) < 2:
            parts.append(f"{key}={mean:.4f}")
        else:
            parts.append(f"{key}={mean:.4f}+-{std:.4f}")
    return "  ".join(parts)

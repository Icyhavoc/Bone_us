"""Feature ranking and greedy selection, NumPy only.

Two questions are answered here, both needed before blaming the MLP for the flat
error rate:

1. *Which columns carry signal at all?*  :func:`univariate_auc_scores` ranks each
   feature on its own, so a column that is dead, near-collinear or noisy shows up
   immediately.  Because a single feature is trivially monotone in its own value,
   the score is ``|AUC - 0.5|`` and the direction is reported separately.  With
   more than two labels the same score is computed from the average pairwise
   AUC, so a three-class run ranks its columns with the same call and the sign
   keeps meaning "larger value -> higher label".
2. *Does a small subset beat the full set?*  :func:`greedy_forward_selection`
   and :func:`greedy_backward_elimination` search subsets.  They are deliberately
   decoupled from any model: the caller passes ``score_fn(indices) -> float``,
   which in practice is a cross-validated score, so the selection can never see
   the test split as long as the caller builds ``score_fn`` from train folds only.

Caveat that matters for how the results get reported: greedy selection optimises
whatever ``score_fn`` returns, and a cross-validated score on 161 samples is
noisy, so forward selection *will* eventually admit a pure-noise column for a
fraction of a point of CV AUC.  That is a property of the search, not a bug.  It
is why the evaluation harness reports the full set and the selected set **on the
held-out test split** side by side, and why ``min_gain`` exists: a positive value
makes each step pay for itself instead of accumulating noise.  A subset score
that only looks good inside the search loop is worth nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from emd_pipeline import _roc_auc

__all__ = [
    "univariate_auc_scores",
    "rank_features_by_auc",
    "correlated_groups",
    "SelectionResult",
    "greedy_forward_selection",
    "greedy_backward_elimination",
    "ColumnSelector",
]


def _validate(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels).ravel()
    if matrix.ndim != 2:
        raise ValueError("features must be 2-D")
    if matrix.shape[0] != targets.size:
        raise ValueError("features and labels disagree on the number of samples")
    return matrix, targets


def _column_separation(values: np.ndarray, targets: np.ndarray) -> float | None:
    """How well one raw column separates the labels, as a single AUC.

    Two classes use the plain AUC of ``values`` (larger values score the larger
    label as positive), exactly as before.  For ``k > 2`` classes the value is
    the **average pairwise AUC** (Hand & Till): the mean, over every pair of
    classes ``(a, b)`` with ``a < b``, of ``P(value of an a-sample < value of a
    b-sample)``.

    Pairwise is used rather than macro one-vs-rest because one-vs-rest is blind
    to the very pattern these features should show.  Take a perfectly ordered
    three-class problem and a feature that increases with the label: the OvR
    AUCs are 0.0 for the low class, 0.5 for the middle class and 1.0 for the high
    class, whose mean is exactly 0.5 -- the best possible column would rank last.
    The pairwise average gives 1.0 there, and it also keeps the sign meaningful
    again: a positive deviation means larger values belong to higher labels.

    Returns ``None`` when the column cannot be scored (a constant column, or a
    label set with a single class).
    """

    present = np.unique(targets)
    if present.size < 2:
        return None
    if present.size == 2:
        return _roc_auc((targets == present[1]).astype(np.int64), values)
    per_pair: list[float] = []
    for position, low in enumerate(present):
        low_mask = targets == low
        for high in present[position + 1 :]:
            mask = low_mask | (targets == high)
            auc = _roc_auc((targets[mask] == high).astype(np.int64), values[mask])
            if auc is not None:
                per_pair.append(float(auc))
    return None if not per_pair else float(np.mean(per_pair))


def univariate_auc_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-column ``|AUC - 0.5|``; 0.0 means the column is useless on its own.

    A column with no variation returns 0.0 (its AUC is undefined, and calling it
    a coin flip is the honest reading), which is how the dead ``mean`` column
    would have been caught automatically.  For more than two classes the column
    score comes from the average pairwise AUC (see :func:`_column_separation`),
    which is still ``0.5`` for a column that carries no class information.  Note
    that a single column of an ordered problem is capped well below 1.0: it has
    to order every pair of classes with one threshold, so ``0.5`` per middle
    class is the honest ceiling of what one column can do.
    """

    matrix, targets = _validate(features, labels)
    scores = np.zeros(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        if np.ptp(values) <= 0.0:
            continue
        auc = _column_separation(values, targets)
        if auc is None:
            continue
        scores[column] = abs(float(auc) - 0.5)
    return scores


def univariate_auc_directions(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Signed AUC deviation per column: >0 means larger values are the higher label.

    With two classes this is the signed direction the binary ranking used all
    along.  The average pairwise AUC keeps that reading for more than two
    classes: the pairwise comparisons are oriented by the label order, so a
    positive value still means larger feature values belong to higher labels.
    """

    matrix, targets = _validate(features, labels)
    directions = np.zeros(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        if np.ptp(values) <= 0.0:
            continue
        auc = _column_separation(values, targets)
        if auc is None:
            continue
        directions[column] = float(auc) - 0.5
    return directions


def rank_features_by_auc(features: np.ndarray, labels: np.ndarray) -> list[int]:
    """Column indices from most to least informative on their own."""

    scores = univariate_auc_scores(features, labels)
    # ``-score`` with a stable sort keeps the original column order among ties, so
    # the ranking is reproducible.
    return np.argsort(-scores, kind="mergesort").astype(int).tolist()


def correlated_groups(features: np.ndarray, threshold: float = 0.98) -> list[list[int]]:
    """Group columns whose absolute Pearson correlation exceeds ``threshold``.

    Union-find over the correlation matrix, so the groups are transitive: with
    ``a~b`` and ``b~c`` all three end up together even if ``a`` and ``c`` are
    below the threshold.  This is the cheap way to find the ``rms``/``energy``
    style redundancy that no single-column ranking can see.
    """

    matrix, _ = _validate(features, np.zeros(features.shape[0]))
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    n_features = matrix.shape[1]
    centered = matrix - matrix.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    safe = norms > 0
    normalized = np.zeros_like(centered)
    normalized[:, safe] = centered[:, safe] / norms[safe]
    correlation = np.abs(normalized.T @ normalized)
    parent = list(range(n_features))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(n_features):
        for right in range(left + 1, n_features):
            if correlation[left, right] >= threshold:
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parent[max(root_left, root_right)] = min(root_left, root_right)
    groups: dict[int, list[int]] = {}
    for index in range(n_features):
        groups.setdefault(find(index), []).append(index)
    return sorted(groups.values(), key=lambda group: group[0])


@dataclass
class SelectionResult:
    """Outcome of a greedy search, with the path that produced it."""

    selected: list[int]
    score: float
    #: ``(added_or_removed_index, score_after_the_step)`` per accepted step.
    steps: list[tuple[int, float]] = field(default_factory=list)
    #: Every candidate that was evaluated, ``(indices, score)``, for auditing.
    evaluations: list[tuple[tuple[int, ...], float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "selected": [int(index) for index in self.selected],
            "score": float(self.score),
            "steps": [[int(index), float(value)] for index, value in self.steps],
            "n_evaluations": len(self.evaluations),
        }


def greedy_forward_selection(
    score_fn: Callable[[Sequence[int]], float],
    n_features: int,
    candidate_order: Sequence[int] | None = None,
    max_features: int | None = None,
    min_gain: float = 0.0,
) -> SelectionResult:
    """Add the single best column repeatedly, stopping when nothing helps.

    ``min_gain`` is the improvement in ``score_fn`` required to accept a step, so
    a positive value stops the search at a genuine plateau instead of adding
    every column down to ``max_features``.
    """

    if n_features <= 0:
        raise ValueError("n_features must be positive")
    order = list(range(n_features)) if candidate_order is None else [
        int(index) for index in candidate_order
    ]
    if len(set(order)) != len(order) or any(not 0 <= index < n_features for index in order):
        raise ValueError("candidate_order must be a permutation of valid column indices")
    limit = n_features if max_features is None else min(int(max_features), n_features)

    evaluations: list[tuple[tuple[int, ...], float]] = []
    selected: list[int] = []
    steps: list[tuple[int, float]] = []
    best_score = -np.inf

    def evaluate(indices: Sequence[int]) -> float:
        key = tuple(sorted(int(index) for index in indices))
        value = float(score_fn(key))
        evaluations.append((key, value))
        return value

    while len(selected) < limit:
        remaining = [index for index in order if index not in selected]
        if not remaining:
            break
        best_candidate, best_candidate_score = None, -np.inf
        for index in remaining:
            value = evaluate([*selected, index])
            if value > best_candidate_score:
                best_candidate, best_candidate_score = index, value
        if best_candidate is None:
            break
        if best_candidate_score <= best_score + min_gain:
            break
        selected.append(best_candidate)
        best_score = best_candidate_score
        steps.append((best_candidate, best_candidate_score))
    return SelectionResult(
        selected=sorted(selected),
        score=float(best_score),
        steps=steps,
        evaluations=evaluations,
    )


def greedy_backward_elimination(
    score_fn: Callable[[Sequence[int]], float],
    n_features: int,
    tolerance: float = 0.0,
    min_features: int = 1,
) -> SelectionResult:
    """Start from every column and drop the least useful one while the score holds.

    ``tolerance`` is how much cross-validated score a drop may cost and still be
    taken, so ``tolerance=0`` accepts any drop that does not hurt and therefore
    strips every redundant column, which is the behaviour wanted here.  This is
    the mirror image of :func:`greedy_forward_selection`'s ``min_gain``, which
    requires a step to *earn* its place; the two are deliberately separate
    parameters rather than one shared name with two meanings.
    """

    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if not 1 <= min_features <= n_features:
        raise ValueError("min_features must be in [1, n_features]")

    evaluations: list[tuple[tuple[int, ...], float]] = []
    selected = list(range(n_features))
    steps: list[tuple[int, float]] = []

    def evaluate(indices: Sequence[int]) -> float:
        key = tuple(sorted(int(index) for index in indices))
        value = float(score_fn(key))
        evaluations.append((key, value))
        return value

    best_score = evaluate(selected)
    while len(selected) > min_features:
        best_drop, best_drop_score = None, -np.inf
        for index in selected:
            trial = [column for column in selected if column != index]
            value = evaluate(trial)
            if value > best_drop_score:
                best_drop, best_drop_score = index, value
        if best_drop is None:
            break
        if best_drop_score < best_score - tolerance:
            break
        selected.remove(best_drop)
        best_score = best_drop_score
        steps.append((best_drop, best_drop_score))
    return SelectionResult(
        selected=sorted(selected),
        score=float(best_score),
        steps=steps,
        evaluations=evaluations,
    )


class ColumnSelector:
    """Wrap a classifier so it only ever sees a chosen subset of columns."""

    def __init__(self, base_model, indices: Sequence[int]) -> None:
        indices = [int(index) for index in indices]
        if not indices:
            raise ValueError("indices cannot be empty")
        if len(set(indices)) != len(indices):
            raise ValueError("indices must not repeat a column")
        self.base_model = base_model
        self.indices = indices
        self.name = f"{getattr(base_model, 'name', type(base_model).__name__)}[{len(indices)}d]"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ColumnSelector":
        features = np.asarray(x, dtype=np.float64)
        self.base_model.fit(features[:, self.indices], y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        features = np.asarray(x, dtype=np.float64)
        return self.base_model.predict_proba(features[:, self.indices])

    @property
    def inner_model(self):
        return self.base_model

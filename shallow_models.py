"""NumPy-only shallow classifiers, used as a reference point for the MLP.

The whole project deliberately avoids scikit-learn, so the models that answer
"could a simple, well-regularised linear model do as well as the MLP?" have to
be written here.  Every model exposes the same tiny interface::

    model = SomeClassifier(seed=42)
    model.fit(x_train, y_train)          # x already standardised by the caller
    probabilities = model.predict_proba(x_test)   # [n, 2], column 1 = P(y=1)

Every learner below is strictly binary: ``_as_binary`` rejects any other label
set.  For a dataset relabelled into three classes the caller wraps one of them in
:class:`OneVsRestClassifier`, which fits one learner per class and returns
``[n, k]``; the two-class path is a plain direct call and is untouched by that.

The purpose of these models is *diagnostic*, not competitive: if a shrunken LDA
reaches the MLP's test AUC on 16 dimensions, the MLP is not the bottleneck and
more capacity or more epochs will not help.

Implementations and why they were chosen:

* ``L2LogisticRegression`` -- iteratively reweighted least squares (IRLS) with a
  ridge penalty.  Exact Newton steps, so no learning-rate tuning and no risk of
  a run that silently failed to converge on a small dataset.
* ``L1LogisticRegression`` -- FISTA with soft-thresholding on the coefficients,
  i.e. a genuine sparse solution that can drop features outright.
* ``LinearDiscriminantAnalysis`` -- closed form, with Ledoit-Wolf style shrinkage
  of the covariance towards its diagonal.  With 16 features and 161 samples the
  un-shrunk covariance is nearly singular, so shrinkage is what makes it usable.
* ``RBFLSSVM`` -- **least-squares** SVM, not SMO.  The LS-SVM dual is a linear
  system, which keeps the whole thing in ``numpy.linalg`` while still giving a
  non-linear RBF decision boundary.
* ``NearestCentroid`` -- the floor.  Effectively "pick the class whose mean is
  closer"; if nothing beats this, the features carry almost no signal.

Probability calibration: IRLS/LDA/LS-SVM/an L2 logistic all output here return an
uncalibrated decision score, and a raw sigmoid of it is badly scaled (LS-SVM
scores in particular).  ``_PlattCalibrator`` re-fits a one-dimensional L2
logistic on the training scores, which is cheap and makes AUC unaffected while
keeping the reported probabilities (and thus the 0.5-threshold accuracy)
honest.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__ = [
    "L2LogisticRegression",
    "L1LogisticRegression",
    "LinearDiscriminantAnalysis",
    "RBFLSSVM",
    "NearestCentroid",
    "OneVsRestClassifier",
    "SHALLOW_MODELS",
]


def _as_binary(y: np.ndarray) -> np.ndarray:
    """Return ``y`` as float64 0/1 and reject anything that is not binary."""

    labels = np.asarray(y, dtype=np.float64).ravel()
    unique = np.unique(labels)
    if unique.size != 2 or not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError(f"this project's models are binary; got labels {unique.tolist()}")
    return labels


def _design_matrix(x: np.ndarray) -> np.ndarray:
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("features must be 2-D")
    return matrix


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * values))


class _PlattCalibrator:
    """Map an unbounded decision score onto a probability.

    Fitting ``p = sigmoid(a * score + b)`` on the *training* scores only adds two
    parameters, so it cannot leak test information; it exists so that
    ``accuracy`` at a 0.5 threshold means the same thing for every model here.
    """

    def __init__(self, max_iterations: int = 200) -> None:
        self.max_iterations = int(max_iterations)
        self.a_ = 1.0
        self.b_ = 0.0

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "_PlattCalibrator":
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = _as_binary(y)
        # A tiny ridge keeps the fit finite on perfectly separable scores.
        self.a_, self.b_ = 1.0, 0.0
        design = np.column_stack([scores, np.ones_like(scores)])
        ridge = 1e-6 * np.eye(2)
        for _ in range(self.max_iterations):
            probabilities = _sigmoid(design @ np.array([self.a_, self.b_]))
            weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
            gradient = design.T @ (probabilities - labels) + ridge @ np.array([self.a_, self.b_])
            hessian = (design * weights[:, None]).T @ design + ridge
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break
            self.a_ -= step[0]
            self.b_ -= step[1]
            if np.max(np.abs(step)) < 1e-10:
                break
        # Guard against a flipped sign on degenerate, constant scores.
        if not np.isfinite(self.a_) or abs(self.a_) < 1e-12:
            self.a_, self.b_ = 1.0, 0.0
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        positive = _sigmoid(self.a_ * np.asarray(scores, dtype=np.float64).ravel() + self.b_)
        positive = np.clip(positive, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - positive, positive])


class L2LogisticRegression:
    """Ridge-penalised logistic regression fitted by IRLS (Newton steps)."""

    name = "l2_logistic"

    def __init__(
        self,
        l2: float = 1.0,
        max_iterations: int = 100,
        tolerance: float = 1e-10,
        seed: int = 42,
    ) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.l2 = float(l2)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.seed = int(seed)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.calibrator_ = _PlattCalibrator()
        self.n_iter_ = 0

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit the model before predicting")
        return _design_matrix(x) @ self.coef_ + self.intercept_

    def fit(self, x: np.ndarray, y: np.ndarray) -> "L2LogisticRegression":
        features = _design_matrix(x)
        labels = _as_binary(y)
        n_samples, n_features = features.shape
        weights = np.zeros(n_features)
        intercept = 0.0
        penalty = self.l2 * np.eye(n_features)
        # Newton steps on the *mean* negative log-likelihood, so the meaning of
        # ``l2`` does not drift with the number of samples.  (Minimising the sum
        # instead would make the same ``l2`` effectively n times weaker here than
        # in any other model in this module, which silently changes the model.)
        for iteration in range(1, self.max_iterations + 1):
            scores = features @ weights + intercept
            probabilities = _sigmoid(scores)
            variance = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
            gradient = features.T @ (probabilities - labels) / n_samples + self.l2 * weights
            gradient_intercept = float(np.sum(probabilities - labels) / n_samples)
            hessian = (features * variance[:, None]).T @ features / n_samples + penalty
            hessian_intercept = float(np.sum(variance) / n_samples)
            # Solve the (F+1)-dimensional Newton system by eliminating the
            # intercept, so no (F+1) matrix is ever formed.
            cross = features.T @ variance / n_samples
            try:
                coef_step = np.linalg.solve(
                    hessian - np.outer(cross, cross) / hessian_intercept,
                    gradient - cross * (gradient_intercept / hessian_intercept),
                )
            except np.linalg.LinAlgError:
                break
            intercept_step = (gradient_intercept - cross @ coef_step) / hessian_intercept
            weights -= coef_step
            intercept -= intercept_step
            self.n_iter_ = iteration
            if max(np.max(np.abs(coef_step)), abs(intercept_step)) < self.tolerance:
                break
        self.coef_, self.intercept_ = weights, float(intercept)
        self.calibrator_.fit(self.decision_function(features), labels)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.calibrator_.predict_proba(self.decision_function(x))

    def feature_importance(self) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit the model before reading importances")
        return np.abs(self.coef_)


class L1LogisticRegression:
    """Logistic regression with an L1 penalty, solved by FISTA + soft-thresholding.

    The L1 penalty is what can set a coefficient exactly to zero, so this model
    doubles as a feature selector: ``selected_features`` lists the surviving
    columns.
    """

    name = "l1_logistic"

    def __init__(
        self,
        l1: float = 0.05,
        max_iterations: int = 3000,
        tolerance: float = 1e-9,
        seed: int = 42,
    ) -> None:
        if l1 < 0:
            raise ValueError("l1 must be non-negative")
        self.l1 = float(l1)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.seed = int(seed)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.calibrator_ = _PlattCalibrator()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit the model before predicting")
        return _design_matrix(x) @ self.coef_ + self.intercept_

    def fit(self, x: np.ndarray, y: np.ndarray) -> "L1LogisticRegression":
        features = _design_matrix(x)
        labels = _as_binary(y)
        n_samples = features.shape[0]
        weights = np.zeros(features.shape[1])
        momentum = weights.copy()
        # Lipschitz constant of the smooth part; 0.25 bounds sigmoid'' and the
        # spectral norm bounds X'X, so this step size cannot diverge.
        lipschitz = 0.25 * float(np.linalg.norm(features, 2)) ** 2 / max(1, n_samples)
        step_size = 1.0 / max(lipschitz, 1e-12)
        intercept = 0.0
        momentum_scale = 1.0
        for _ in range(self.max_iterations):
            gradient = features.T @ (_sigmoid(features @ momentum + intercept) - labels) / n_samples
            candidate = _soft_threshold(momentum - step_size * gradient, self.l1 * step_size)
            gradient_intercept = float(np.mean(_sigmoid(features @ momentum + intercept) - labels))
            next_momentum_scale = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum_scale**2))
            momentum = candidate + ((momentum_scale - 1.0) / next_momentum_scale) * (candidate - weights)
            shift = np.max(np.abs(candidate - weights))
            weights = candidate
            intercept -= step_size * gradient_intercept
            momentum_scale = next_momentum_scale
            if shift < self.tolerance:
                break
        self.coef_, self.intercept_ = weights, float(intercept)
        self.calibrator_.fit(self.decision_function(features), labels)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.calibrator_.predict_proba(self.decision_function(x))

    def feature_importance(self) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit the model before reading importances")
        return np.abs(self.coef_)

    @property
    def selected_features(self) -> list[int]:
        """Indices whose coefficient survived the L1 penalty."""

        if self.coef_ is None:
            raise RuntimeError("fit the model before reading selected features")
        return np.flatnonzero(np.abs(self.coef_) > 0.0).astype(int).tolist()


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


class LinearDiscriminantAnalysis:
    """Shrinkage LDA, closed form.

    ``shrinkage`` interpolates the pooled covariance towards its diagonal
    (Ledoit-Wolf style).  With ``shrinkage=0`` this is textbook LDA; the default
    is deliberately higher because 16 features / 161 samples makes the raw
    covariance close to singular.
    """

    name = "lda"

    def __init__(self, shrinkage: float = 0.3, seed: int = 42) -> None:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        self.shrinkage = float(shrinkage)
        self.seed = int(seed)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.calibrator_ = _PlattCalibrator()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit the model before predicting")
        return _design_matrix(x) @ self.coef_ + self.intercept_

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearDiscriminantAnalysis":
        features = _design_matrix(x)
        labels = _as_binary(y)
        positive = labels == 1.0
        mean_positive = features[positive].mean(axis=0)
        mean_negative = features[~positive].mean(axis=0)
        centered = features - np.where(positive[:, None], mean_positive, mean_negative)
        pooled = centered.T @ centered / max(1, features.shape[0] - 2)
        target = np.diag(np.diag(pooled))
        covariance = (1.0 - self.shrinkage) * pooled + self.shrinkage * target
        covariance += 1e-10 * np.eye(covariance.shape[0])
        difference = mean_positive - mean_negative
        weights = np.linalg.solve(covariance, difference)
        self.coef_ = weights
        self.intercept_ = float(-0.5 * (mean_positive + mean_negative) @ weights)
        self.calibrator_.fit(self.decision_function(features), labels)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.calibrator_.predict_proba(self.decision_function(x))

    def feature_importance(self) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit the model before reading importances")
        return np.abs(self.coef_)


class RBFLSSVM:
    """Least-squares SVM with an RBF kernel.

    The LS-SVM dual solves

        [[0, 1^T], [1, K + I/gamma]] @ [b, alpha] = [0, y]

    with ``y`` in {-1, +1}, so the training is exact and needs no iterates.  The
    trade-off parameter is called ``gamma`` (LS-SVM convention) and is the
    inverse of the usual ``C``.
    """

    name = "rbf_lssvm"

    def __init__(
        self,
        gamma: float = 1.0,
        sigma: float = 1.0,
        seed: int = 42,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.support_: np.ndarray | None = None
        self.dual_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.calibrator_ = _PlattCalibrator()

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        difference = left[:, None, :] - right[None, :, :]
        return np.exp(-np.sum(difference**2, axis=2) / (2.0 * self.sigma**2))

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.support_ is None or self.dual_ is None:
            raise RuntimeError("fit the model before predicting")
        return self._kernel(_design_matrix(x), self.support_) @ self.dual_ + self.intercept_

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RBFLSSVM":
        features = _design_matrix(x)
        labels = _as_binary(y)
        signed = np.where(labels == 1.0, 1.0, -1.0)
        n_samples = features.shape[0]
        kernel = self._kernel(features, features)
        system = np.zeros((n_samples + 1, n_samples + 1))
        system[0, 1:] = 1.0
        system[1:, 0] = 1.0
        system[1:, 1:] = kernel + np.eye(n_samples) / self.gamma
        target = np.concatenate([[0.0], signed])
        try:
            solution = np.linalg.solve(system, target)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(system, target, rcond=None)[0]
        self.intercept_ = float(solution[0])
        self.dual_ = solution[1:]
        self.support_ = features
        self.calibrator_.fit(self.decision_function(features), labels)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.calibrator_.predict_proba(self.decision_function(x))

    def feature_importance(self) -> np.ndarray:
        """RBF models have no per-feature weights; report the input scale used."""

        if self.support_ is None:
            raise RuntimeError("fit the model before reading importances")
        return self.support_.std(axis=0)


class NearestCentroid:
    """Distance-to-class-mean classifier: the simplest defensible baseline."""

    name = "nearest_centroid"

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.centroids_: np.ndarray | None = None
        self.calibrator_ = _PlattCalibrator()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("fit the model before predicting")
        features = _design_matrix(x)
        distance = np.linalg.norm(features[:, None, :] - self.centroids_[None, :, :], axis=2)
        return distance[:, 0] - distance[:, 1]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        features = _design_matrix(x)
        labels = _as_binary(y)
        self.centroids_ = np.vstack([features[labels == 0.0].mean(axis=0), features[labels == 1.0].mean(axis=0)])
        self.calibrator_.fit(self.decision_function(features), labels)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.calibrator_.predict_proba(self.decision_function(x))


class OneVsRestClassifier:
    """Turn any binary learner in this module into a k-class one.

    ``k`` independent one-vs-rest learners are fitted, ``column c`` is the
    calibrated probability of class ``c``, and the prediction is ``argmax`` over
    the columns.  This is deliberately the *adapter* route rather than a
    multinomial rewrite of each model: the binary learners stay bit-for-bit
    what they were, so adding this class cannot move any stored two-class
    number, and the five baselines can still answer "does a simple model do as
    well as the MLP?" on a three-class relabelling without five new
    derivations.

    Two consequences worth stating, because they show up in the report:

    * The columns need not sum to one -- each is calibrated against its own
      one-vs-rest problem.  That is fine for ``argmax`` and for the per-class
      one-vs-rest AUC, but these columns are not a normalised softmax and should
      not be read as one.
    * A class with no samples in the training part of a fold cannot be fitted
      one-vs-rest, so this raises instead of quietly emitting a constant column.
      ``stratified_folds`` keeps at least one sample of every class in every
      training part as long as each class has at least ``n_splits`` samples.
    """

    name = "one_vs_rest"

    def __init__(
        self,
        base_factory: Callable[[], Any],
        num_classes: int,
        seed: int = 42,
    ) -> None:
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least 2")
        self.base_factory = base_factory
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.models_: list[Any] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OneVsRestClassifier":
        features = _design_matrix(x)
        labels = np.asarray(y).ravel()
        if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= self.num_classes):
            raise ValueError(
                f"labels run from {int(labels.min())} to {int(labels.max())} "
                f"but num_classes={self.num_classes}"
            )
        missing = [
            label for label in range(self.num_classes) if not int(np.sum(labels == label))
        ]
        if missing:
            raise ValueError(
                f"class(es) {missing} have no training samples, so a one-vs-rest learner "
                "cannot be fitted; reduce --folds or use a dataset with more samples "
                "per class"
            )
        self.models_ = []
        for label in range(self.num_classes):
            model = self.base_factory()
            model.fit(features, (labels == label).astype(np.float64))
            self.models_.append(model)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("fit the model before predicting")
        features = _design_matrix(x)
        columns = [model.predict_proba(features)[:, 1] for model in self.models_]
        return np.column_stack(columns)

    def feature_importance(self) -> np.ndarray:
        """Mean of the per-class importances, i.e. the average OvR relevance."""

        if not self.models_:
            raise RuntimeError("fit the model before reading importances")
        return np.vstack([model.feature_importance() for model in self.models_]).mean(axis=0)

    @property
    def selected_features(self) -> list[int]:
        """Union of the per-class sparse selections (sparse learners only)."""

        selected: set[int] = set()
        for model in self.models_:
            if not hasattr(model, "selected_features"):
                raise RuntimeError(
                    f"{type(model).__name__} has no selected_features; "
                    "only the L1 model is sparse"
                )
            selected.update(int(index) for index in model.selected_features)
        return sorted(selected)


#: Ordered from most to least structured, so a report reads as a ladder.
SHALLOW_MODELS: dict[str, type[Any]] = {
    "nearest_centroid": NearestCentroid,
    "lda": LinearDiscriminantAnalysis,
    "l2_logistic": L2LogisticRegression,
    "l1_logistic": L1LogisticRegression,
    "rbf_lssvm": RBFLSSVM,
}

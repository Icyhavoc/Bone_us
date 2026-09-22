"""Run the shallow baselines against already-extracted EMD features.

This is change #3 of the flat-error investigation: before adding capacity, more
epochs or a bigger grid to the MLP, establish what a well-regularised linear model
reaches on the *same* columns with the *same* train/val/test split.  If a
shrunken LDA matches the MLP's test accuracy, the MLP is not the bottleneck and
tuning it is wasted effort.

What the script does per experiment directory:

1. loads ``features_{train,val,test}.npy`` and their labels (the features are
   stored **raw**; the pipeline's ``StandardScaler`` is applied inside each fit);
2. prints the label balance and the majority-class floor, so every accuracy below
   has a reference point;
3. ranks every column by its own univariate ``|AUC - 0.5|`` and groups
   near-collinear columns, naming them from ``metrics.json`` when available;
4. tunes each model on a repeated stratified K-fold **inside train only**,
   picking the configuration with the best mean validation AUC;
5. scores the tuned model once on train/val/test by fitting on the whole train
   split, exactly the protocol ``run_emd_experiments.py`` uses;
6. optionally searches a feature subset (forward or backward) driven by the same
   train-only CV score, and reports the subset's held-out numbers next to the
   full set's;
7. records the stored MLP metrics from ``metrics.json`` for a direct comparison.

Only step 5 and step 6's final scoring touch ``val``/``test``, and nothing is
chosen from what they show.

The class count is inferred from the stored labels, so the same script scores the
historical two-class split and a three-class relabelling.  With two classes every
number and every line of output is what it was before; with more, each binary
learner is fitted one-vs-rest (``OneVsRestClassifier``) and the reported ``auc``
/ ``f1_score`` / ``sensitivity`` / ``specificity`` are the macro one-vs-rest
averages, which is also what the MLP records for a three-class run.

Usage::

    python run_baseline_models.py --experiments bin/_ab2/compact_core_s42 \\
        --experiments bin/_ab2/legacy_none_s42 \\
        --output-dir baseline_results --feature-selection forward
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from baseline_evaluation import (
    PriorClassifier,
    cross_validate,
    holdout_evaluate,
    stratified_folds,
)
from emd_pipeline import label_scheme_tag, read_label_scheme
from feature_selection import (
    ColumnSelector,
    correlated_groups,
    rank_features_by_auc,
    univariate_auc_directions,
    univariate_auc_scores,
    greedy_backward_elimination,
    greedy_forward_selection,
)
from shallow_models import SHALLOW_MODELS, OneVsRestClassifier

SPLITS = ("train", "val", "test")

#: Candidate configurations per model.  Kept deliberately small: the point is a
#: defensible reference point, not a leaderboard, and every extra configuration
#: is another chance to fit the CV score's noise.
MODEL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "nearest_centroid": [{}],
    "lda": [{"shrinkage": value} for value in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)],
    "l2_logistic": [{"l2": value} for value in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)],
    "l1_logistic": [{"l1": value} for value in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)],
    "rbf_lssvm": [
        {"gamma": gamma, "sigma": sigma}
        for gamma in (0.3, 1.0, 3.0, 10.0)
        for sigma in (0.5, 1.0, 2.0, 4.0)
    ],
}


def _auc_text(value: Any) -> str:
    """``0.8123``, or ``n/a`` when the AUC is undefined for a split.

    A three-class split that happens to contain a single class has no one-vs-rest
    AUC at all, and a report full of ``None`` would say less than an explicit
    ``n/a``.  The caller right-aligns the result, so a defined value occupies the
    same columns as the historical fixed-width format.
    """

    return "n/a" if value is None else f"{float(value):.4f}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--experiments",
        action="append",
        required=True,
        help="experiment directory, or a directory holding several; repeatable",
    )
    parser.add_argument("--output-dir", default="baseline_results", help="where reports are written")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["prior", "nearest_centroid", "lda", "l2_logistic", "l1_logistic", "rbf_lssvm"],
        choices=["prior", *sorted(SHALLOW_MODELS)],
    )
    parser.add_argument("--folds", type=int, default=5, help="stratified folds inside train")
    parser.add_argument("--repeats", type=int, default=5, help="repeats of the fold split")
    parser.add_argument("--seed", type=int, default=0, help="fold-split seed")
    parser.add_argument(
        "--feature-selection",
        choices=("none", "forward", "backward"),
        default="none",
        help="search a column subset too (driven by train-only CV)",
    )
    parser.add_argument(
        "--selection-model",
        default="lda",
        help="model whose CV AUC drives the subset search",
    )
    parser.add_argument("--selection-folds", type=int, default=5)
    parser.add_argument("--selection-repeats", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=8, help="forward-search ceiling")
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.002,
        help="CV AUC a forward step must add to be accepted",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="CV AUC a backward drop may cost",
    )
    return parser.parse_args(argv)


def discover_experiments(paths: Sequence[str]) -> list[Path]:
    """Expand directories into complete experiment directories, sorted."""

    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_dir():
            raise SystemExit(f"not a directory: {path}")
        candidates = [path, *sorted(child for child in path.iterdir() if child.is_dir())]
        for candidate in candidates:
            if all((candidate / f"features_{split}.npy").is_file() for split in SPLITS):
                found.append(candidate)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise SystemExit("no complete experiment directory found")
    return unique


def load_experiment(directory: Path) -> dict[str, Any]:
    features = {
        split: np.load(directory / f"features_{split}.npy").astype(np.float64) for split in SPLITS
    }
    labels = {
        split: np.load(directory / f"labels_{split}.npy").astype(np.int64).ravel()
        for split in SPLITS
    }
    if any(int(values.min()) < 0 for values in labels.values() if values.size):
        raise SystemExit(f"negative labels in {directory}; the labels must run 0..k-1")
    # The labels are the authority on the class count; taking the max over every
    # split keeps the number right even if a tiny split happens to miss the top
    # class.
    num_classes = max(
        (int(values.max()) for values in labels.values() if values.size), default=1
    ) + 1
    names = dimension_names(directory, features["train"].shape[1])
    stored = None
    metrics_path = directory / "metrics.json"
    if metrics_path.is_file():
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    scheme = experiment_label_scheme(directory, stored)
    return {
        "directory": directory,
        "features": features,
        "labels": labels,
        "names": names,
        "stored": stored,
        "num_classes": num_classes,
        "label_scheme": scheme,
        "label_scheme_tag": label_scheme_tag(scheme),
    }


def experiment_label_scheme(
    directory: Path, stored: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Recover the relabelling behind this directory, best effort.

    Runs written before the label scheme was recorded in ``config.json`` still
    point at their dataset through ``config.json['data_dir']``, so the scheme is
    looked up there as a fallback.  Never fatal: an old directory simply reports
    no thresholds.
    """

    config: dict[str, Any] = {}
    config_path = directory / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
    for payload in (stored, config):
        scheme = (payload or {}).get("label_scheme")
        if isinstance(scheme, dict) and scheme:
            return scheme
    data_dir = config.get("data_dir")
    return read_label_scheme(data_dir) if data_dir else None


def dimension_names(directory: Path, width: int) -> list[str]:
    """Column names from ``metrics.json`` when the run recorded them."""

    for candidate in (directory / "metrics.json", directory / "baseline_metrics.json"):
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            names = payload.get("feature_dimension_names")
            if names and len(names) == width:
                return [str(name) for name in names]
    return [f"dim{index}" for index in range(width)]


def tune(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    seed: int,
    num_classes: int = 2,
) -> dict[str, Any]:
    """Pick the configuration with the best mean CV AUC, on train folds only.

    Ties are broken towards the *first* grid entry, which for every model is the
    most strongly regularised end of the grid, so a flat CV surface does not
    silently select the least stable model in it.  The objective is the macro
    one-vs-rest AUC for a three-class dataset, which is the same objective
    ``run_emd_experiments.py`` uses for its early stopping there.
    """

    grid = MODEL_GRIDS.get(model_name, [{}])
    best: dict[str, Any] | None = None
    for params in grid:
        make = build_factory(model_name, params, seed, num_classes)
        result = cross_validate(make, features, labels, folds)
        auc = result["summary"]["auc"]["mean"]
        accuracy = result["summary"]["accuracy"]["mean"]
        score = -1.0 if auc is None else float(auc)
        entry = {
            "params": dict(params),
            "cv": result["summary"],
            "cv_objective": score,
            "cv_accuracy": accuracy,
        }
        if best is None or score > best["cv_objective"]:
            best = entry
    assert best is not None
    return best


def build_factory(
    model_name: str, params: dict[str, Any], seed: int, num_classes: int = 2
) -> Callable[[], Any]:
    """A zero-argument factory for one fitted model instance.

    ``prior`` needs no adapter: its columns are class frequencies, which are a
    k-vector already.  Every other model here is a binary learner, so for
    ``k > 2`` it is wrapped in ``OneVsRestClassifier`` (one learner per class)
    rather than rewritten; that is what keeps the two-class numbers untouched.
    """

    base = PriorClassifier if model_name == "prior" else SHALLOW_MODELS[model_name]
    if int(num_classes) <= 2 or model_name == "prior":
        return lambda: base(seed=seed, **params)
    return lambda: OneVsRestClassifier(
        lambda: base(seed=seed, **params), num_classes=num_classes, seed=seed
    )


def univariate_table(
    features: np.ndarray, labels: np.ndarray, names: Sequence[str], threshold: float = 0.98
) -> dict[str, Any]:
    scores = univariate_auc_scores(features, labels)
    directions = univariate_auc_directions(features, labels)
    spread = features.std(axis=0)
    order = rank_features_by_auc(features, labels)
    groups = correlated_groups(features, threshold)
    return {
        "order": [int(index) for index in order],
        "columns": [
            {
                "index": int(index),
                "name": names[index],
                "abs_auc_deviation": float(scores[index]),
                "signed_auc_deviation": float(directions[index]),
                "std": float(spread[index]),
                "raw_std": float(spread[index]),
            }
            for index in order
        ],
        "correlated_groups": [[int(index) for index in group] for group in groups],
    }


def evaluate_subset(
    model_name: str,
    params: dict[str, Any],
    indices: Sequence[int],
    experiment: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Score one column subset with the fold-fitted-once-train protocol."""

    indices = [int(index) for index in indices]
    factory = build_factory(
        model_name, params, seed, experiment.get("num_classes", 2)
    )
    wrapped = lambda: ColumnSelector(factory(), indices)  # noqa: E731
    return holdout_evaluate(
        wrapped,
        experiment["features"]["train"],
        experiment["labels"]["train"],
        {split: (experiment["features"][split], experiment["labels"][split]) for split in SPLITS},
    )


def run_experiment(experiment: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    directory: Path = experiment["directory"]
    train_features = experiment["features"]["train"]
    train_labels = experiment["labels"]["train"]
    names = experiment["names"]
    num_classes = int(experiment.get("num_classes", 2))
    folds = stratified_folds(train_labels, args.folds, args.repeats, args.seed)

    print(f"\n{'=' * 78}\n{directory}\n{'=' * 78}")
    print(f"features train/val/test: "
          f"{train_features.shape} / {experiment['features']['val'].shape} / {experiment['features']['test'].shape}")
    if num_classes > 2:
        tag = experiment.get("label_scheme_tag")
        print(f"classes: {num_classes}" + (f"  ({tag})" if tag else ""))
    for split in SPLITS:
        labels = experiment["labels"][split]
        if num_classes == 2:
            print(f"  {split:5s} n={labels.size:4d}  positive={int(labels.sum()):3d} "
                  f"({labels.mean():.4f})  majority floor={max(labels.mean(), 1 - labels.mean()):.4f}")
        else:
            counts = np.bincount(labels, minlength=num_classes)
            rates = "  ".join(
                f"c{label}={count}({count / max(1, labels.size):.4f})"
                for label, count in enumerate(counts)
            )
            majority = float(counts.max() / max(1, labels.size)) if labels.size else 0.0
            print(f"  {split:5s} n={labels.size:4d}  {rates}  majority floor={majority:.4f}")
    print(f"train folds: {len(folds)} ({args.folds}-fold x {args.repeats}, stratified, seed {args.seed})")

    univariate = univariate_table(train_features, train_labels, names)
    print("\nunivariate ranking (train only), strongest first:")
    for entry in univariate["columns"]:
        print(f"  [{entry['index']:2d}] {entry['name']:<38s} "
              f"|AUC-.5|={entry['abs_auc_deviation']:.4f} "
              f"sign={entry['signed_auc_deviation']:+.4f} raw_std={entry['raw_std']:.3e}")
    collinear = [group for group in univariate["correlated_groups"] if len(group) > 1]
    if collinear:
        print("near-collinear groups (|r| >= 0.98):")
        for group in collinear:
            print("  " + " ~ ".join(names[index] for index in group))

    record: dict[str, Any] = {
        "directory": str(directory),
        "num_classes": num_classes,
        "n_features": int(train_features.shape[1]),
        "names": list(names),
        "split_sizes": {split: int(experiment["labels"][split].size) for split in SPLITS},
        "split_positive_rate": {split: float(experiment["labels"][split].mean()) for split in SPLITS},
        "majority_floor": {split: float(max(experiment["labels"][split].mean(),
                                            1 - experiment["labels"][split].mean())) for split in SPLITS},
        "univariate": univariate,
        "models": {},
    }
    if num_classes > 2:
        # ``split_positive_rate`` cannot describe three classes, so the counts are
        # recorded next to it instead of replacing it (the binary JSON stays
        # byte-identical).
        record["split_class_counts"] = {
            split: np.bincount(
                experiment["labels"][split], minlength=num_classes
            ).astype(int).tolist()
            for split in SPLITS
        }
        record["label_scheme"] = experiment.get("label_scheme")
        record["label_scheme_tag"] = experiment.get("label_scheme_tag")
    if experiment["stored"] is not None:
        stored_metrics = experiment["stored"].get("metrics")
        if stored_metrics:
            record["stored_mlp"] = {
                split: {
                    key: stored_metrics[split].get(key)
                    for key in ("accuracy", "auc", "f1_score", "sensitivity", "specificity")
                }
                for split in SPLITS
                if split in stored_metrics
            }

    print(f"\n{'model':<18} {'params':<34} {'CV acc':>8} {'CV auc':>8} | "
          f"{'test acc':>9} {'test auc':>9} {'test F1':>8}")
    for model_name in args.models:
        tuned = tune(model_name, train_features, train_labels, folds, args.seed, num_classes)
        factory = build_factory(model_name, tuned["params"], args.seed, num_classes)
        held_out = holdout_evaluate(
            factory,
            train_features,
            train_labels,
            {split: (experiment["features"][split], experiment["labels"][split]) for split in SPLITS},
        )
        cv_auc = tuned["cv"]["auc"]["mean"]
        print(f"{model_name:<18} {str(tuned['params']):<34} "
              f"{tuned['cv_accuracy']:>8.4f} {(-1.0 if cv_auc is None else cv_auc):>8.4f} | "
              f"{held_out['metrics']['test']['accuracy']:>9.4f} "
              f"{_auc_text(held_out['metrics']['test']['auc']):>9s} "
              f"{held_out['metrics']['test']['f1_score']:>8.4f}")
        record["models"][model_name] = {
            "params": tuned["params"],
            "cv": tuned["cv"],
            "cv_objective": tuned["cv_objective"],
            "metrics": held_out["metrics"],
        }

    if args.feature_selection != "none":
        selected = run_selection(experiment, args, names, folds)
        record["feature_selection"] = selected
    return record


def run_selection(
    experiment: dict[str, Any],
    args: argparse.Namespace,
    names: Sequence[str],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Search a subset on train-only CV, then score it on the held-out splits."""

    model_name = args.selection_model
    if model_name not in SHALLOW_MODELS and model_name != "prior":
        raise SystemExit(f"--selection-model must be one of {sorted(SHALLOW_MODELS)}")
    train_features = experiment["features"]["train"]
    train_labels = experiment["labels"]["train"]
    num_classes = int(experiment.get("num_classes", 2))
    width = train_features.shape[1]

    # A separate, smaller fold set keeps the search affordable; it is still built
    # from train only.
    search_folds = stratified_folds(
        train_labels, args.selection_folds, args.selection_repeats, args.seed + 1000
    )
    tuned = tune(model_name, train_features, train_labels, search_folds, args.seed, num_classes)
    params = tuned["params"]
    factory = build_factory(model_name, params, args.seed, num_classes)

    def score(indices: Sequence[int]) -> float:
        indices = [int(index) for index in indices]
        wrapped = lambda: ColumnSelector(factory(), indices)  # noqa: E731
        result = cross_validate(wrapped, train_features, train_labels, search_folds)
        auc = result["summary"]["auc"]["mean"]
        return -1.0 if auc is None else float(auc)

    order = rank_features_by_auc(train_features, train_labels)
    if args.feature_selection == "forward":
        search = greedy_forward_selection(
            score,
            width,
            candidate_order=order,
            max_features=args.max_features,
            min_gain=args.min_gain,
        )
    else:
        search = greedy_backward_elimination(
            score, width, tolerance=args.tolerance, min_features=1
        )

    full = evaluate_subset(model_name, params, list(range(width)), experiment, args.seed)
    subset = evaluate_subset(model_name, params, search.selected, experiment, args.seed)

    print(f"\nfeature selection ({args.feature_selection}, model={model_name} {params}, "
          f"driven by {len(search_folds)} train-only folds, {len(search.evaluations)} evaluations)")
    print(f"  search CV AUC : {search.score:.4f}")
    print(f"  full-set CV   : {tuned['cv']['auc']['mean']:.4f}")
    print("  order of accepted steps (column, CV AUC after):")
    for index, value in search.steps:
        print(f"    + [{index:2d}] {names[index]:<38s} -> {value:.4f}")
    print(f"  selected {len(search.selected)}/{width}: "
          + ", ".join(f"[{index}]{names[index]}" for index in search.selected))
    print(f"  {'set':<10} {'test acc':>9} {'test auc':>9} {'test F1':>8}")
    for label, result in (("full", full), ("selected", subset)):
        print(f"  {label:<10} {result['metrics']['test']['accuracy']:>9.4f} "
              f"{_auc_text(result['metrics']['test']['auc']):>9s} "
              f"{result['metrics']['test']['f1_score']:>8.4f}")
    print("  (a subset that only wins inside the search loop is worth nothing; the "
          "test columns above are the honest comparison)")

    return {
        "mode": args.feature_selection,
        "model": model_name,
        "params": params,
        "search_cv_auc": float(search.score),
        "selected": [int(index) for index in search.selected],
        "selected_names": [names[index] for index in search.selected],
        "steps": [[int(index), float(value)] for index, value in search.steps],
        "n_evaluations": len(search.evaluations),
        "full": {"metrics": full["metrics"], "selection": list(range(width))},
        "subset": {"metrics": subset["metrics"], "selection": [int(i) for i in search.selected]},
    }


def write_report(output_dir: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_results.json").write_text(
        json.dumps({"arguments": vars(args), "experiments": records}, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("SHALLOW BASELINE REPORT (change #3)")
    lines.append("=" * 78)
    lines.append("Features are the stored raw EMD columns. The scaler is re-fitted inside")
    lines.append("every CV fold, and once on the whole train split for the reported")
    lines.append("train/val/test numbers, which is the protocol run_emd_experiments.py uses.")
    lines.append(f"CV: {args.folds}-fold x {args.repeats} repeats, stratified, seed {args.seed}")
    lines.append("")

    for record in records:
        lines.append("-" * 78)
        lines.append(record["directory"])
        lines.append(f"columns: {record['n_features']}")
        num_classes = int(record.get("num_classes", 2))
        if num_classes > 2:
            tag = record.get("label_scheme_tag")
            lines.append(f"classes: {num_classes}" + (f"  ({tag})" if tag else ""))
        for split in SPLITS:
            if num_classes > 2:
                counts = record["split_class_counts"][split]
                total = max(1, record["split_sizes"][split])
                rates = "  ".join(
                    f"c{label}={count}({count / total:.4f})" for label, count in enumerate(counts)
                )
                lines.append(
                    f"  {split:5s} n={record['split_sizes'][split]:4d}  {rates}  "
                    f"majority floor={record['majority_floor'][split]:.4f}"
                )
            else:
                lines.append(
                    f"  {split:5s} n={record['split_sizes'][split]:4d}  "
                    f"positive={record['split_positive_rate'][split]:.4f}  "
                    f"majority floor={record['majority_floor'][split]:.4f}"
                )
        if "stored_mlp" in record:
            lines.append("  stored MLP (same split):")
            for split in SPLITS:
                entry = record["stored_mlp"].get(split)
                if not entry:
                    continue
                lines.append(
                    f"    {split:5s} accuracy={entry['accuracy']:.4f}  auc={entry['auc']:.4f}  "
                    f"f1={entry['f1_score']:.4f}"
                )
        lines.append("")
        lines.append("  baseline models (test = held-out, never used for tuning):")
        lines.append(f"  {'model':<18} {'params':<30} {'CV auc':>8} {'test acc':>9} {'test auc':>9} "
                     f"{'test bal-acc':>13} {'test F1':>8}")
        for model_name, entry in record["models"].items():
            cv_auc = entry["cv"]["auc"]["mean"]
            test = entry["metrics"]["test"]
            lines.append(
                f"  {model_name:<18} {str(entry['params']):<30} "
                f"{(-1.0 if cv_auc is None else cv_auc):>8.4f} "
                f"{test['accuracy']:>9.4f} {_auc_text(test['auc']):>9s} "
                f"{test['balanced_accuracy']:>13.4f} {test['f1_score']:>8.4f}"
            )
        lines.append("")
        lines.append("  strongest single columns (train-only |AUC-0.5|):")
        for entry in record["univariate"]["columns"][:5]:
            lines.append(f"    [{entry['index']:2d}] {entry['name']:<38s} "
                         f"{entry['abs_auc_deviation']:.4f}")
        weakest = record["univariate"]["columns"][-3:]
        lines.append("  weakest columns (candidates to drop):")
        for entry in weakest:
            lines.append(f"    [{entry['index']:2d}] {entry['name']:<38s} "
                         f"{entry['abs_auc_deviation']:.4f}")
        if "feature_selection" in record:
            selection = record["feature_selection"]
            lines.append("")
            lines.append(f"  feature selection ({selection['mode']}, model={selection['model']} "
                         f"{selection['params']}):")
            lines.append(f"    selected {len(selection['selected'])}/{record['n_features']}: "
                         + ", ".join(selection["selected_names"]))
            for label, key in (("full set", "full"), ("selected", "subset")):
                test = selection[key]["metrics"]["test"]
                lines.append(f"    {label:<10} test acc={test['accuracy']:.4f}  auc={test['auc']:.4f}  "
                             f"f1={test['f1_score']:.4f}")
        lines.append("")

    lines.append("=" * 78)
    lines.append("Reading the table: compare each 'test acc' against the majority floor and")
    lines.append("against the stored MLP row. A baseline within noise of the MLP means more")
    lines.append("MLP capacity will not help; the limit is in the features or the data.")
    (output_dir / "baseline_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {output_dir / 'baseline_report.txt'} and {output_dir / 'baseline_results.json'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    directories = discover_experiments(args.experiments)
    print(f"running {len(directories)} experiment directory(ies)")
    records = [run_experiment(load_experiment(directory), args) for directory in directories]
    write_report(Path(args.output_dir), records, args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

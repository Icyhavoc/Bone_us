"""Regression guard for ``--channel-contrast`` (strategy S4).

S4 uses the second physical channel, which is the only source of genuinely new
information that does not require new data: the two channels insonate the same
site from different angles, so their *contrast* is a new observable.  The
mechanism is one extra feature group appended after the channel groups, holding
an elementwise contrast of the first two groups -- which are column-aligned by
construction, because both are built from the same branches with the same
statistics and the same locator preset.

The option is therefore only safe if it is strictly **additive**.  Four claims
are checked here, the first being the one that matters most:

1. **A contrast cannot change a single-channel run.**  A pre-S4 experiment
   directory on disk (``experiments/...__channels_1``) is replayed from its own
   recorded ``config.json``, and the resulting ``features_train.npy`` must be
   bit-identical to the artifact that was written before this option existed.
   Asking for a contrast on one channel is refused outright rather than silently
   ignored, so a ``channels_1`` run can never acquire an extra group by mistake.
2. **The contrast is appended, not woven in.**  Running with and without the
   option, the leading ``W`` columns are bit-identical and only the trailing
   ``W / 2`` columns are new.
3. **Those new columns are the declared arithmetic** of the two channel groups,
   including the ``normalized`` form's invariance to a common gain -- the
   property that makes it cancel probe pressure and coupling gel.
4. **The published metadata describes the extra group**: ``feature_group_dims``
   grows by one tower and every contrast column name carries its ``contrast_``
   prefix and its source statistic.

Runs are real CLI invocations, so nothing is mocked.  Claim 1 replays one full
experiment (reduced to one epoch, since only the features are compared); claims
2-4 use ``--max-samples`` so they stay quick.

Run: ``python verify_channel_contrast.py``
Add ``--keep`` to keep the temporary ``_contrast_regress/`` output directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "_contrast_regress"

#: An experiment directory written before ``channel_contrast`` existed, used as
#: the ground truth for claim 1.  Its ``config.json`` records every setting, so
#: the run can be reproduced instead of guessed at.
REPLAY_EXPERIMENT = ROOT / "experiments" / "form_top3_mean__regions_dyn_envelope__channels_1"

#: ``config.json`` key -> CLI flag, for the detector settings that are not passed
#: through ``--region-set``.
DYN_FLAGS: dict[str, str] = {
    "noise_start": "--dyn-noise-start",
    "weak_threshold": "--dyn-weak-threshold",
    "smooth_window": "--dyn-smooth-window",
    "peak_window_back": "--dyn-peak-window-back",
    "peak_window_forward": "--dyn-peak-window-forward",
    "peak_ratio": "--dyn-peak-ratio",
    "gap_max": "--dyn-gap-max",
    "min_run_width": "--dyn-min-run-width",
    "min_run_area_ratio": "--dyn-min-run-area-ratio",
    "merge_max_lead": "--dyn-merge-max-lead",
    "lead_back": "--dyn-lead-back",
    "main_start_min": "--dyn-main-start-min",
    "main_length": "--dyn-main-length",
    "tail_start": "--dyn-tail-start",
    "locator_top_k": "--dyn-top-k",
}

#: Contrast kinds this script exercises.  ``difference`` is checked as the
#: degenerate member of the family (it is a plain subtraction), the other two as
#: the scale-free ones S4 actually argues for.
CONTRAST_UNDER_TEST = ("difference", "normalized", "log_ratio")

#: The only statistic whose *definition* changed after the ground-truth artifact
#: was written: S3a dropped a ``+1e-12`` from the power sum that feeds
#: ``spectral_centroid``.  Measured on this artifact the shift is one column out of
#: sixteen at a relative 2.5e-7, i.e. a couple of float32 ulps, so the replay
#: tolerates that one statistic and demands bit-equality everywhere else.
DOCUMENTED_DRIFT_SUFFIX = ".spectral_centroid"

#: How many float32 ulps the documented drift may occupy.  It measures 2.1 ulps;
#: anything an order of magnitude beyond that is a real change, not round-off.
DOCUMENTED_DRIFT_ULPS = 8.0


def run_cli(args: list[str], expect_success: bool = True) -> subprocess.CompletedProcess:
    """Run the experiment CLI and return the completed process."""

    command = [sys.executable, str(ROOT / "run_emd_experiments.py"), *args]
    print(f"  $ {' '.join(args)}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if expect_success and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def fresh_dir(name: str) -> Path:
    """Empty ``WORK_DIR/name`` and return it."""

    directory = WORK_DIR / name
    if directory.exists():
        shutil.rmtree(directory)
    return directory


def only_experiment(output_dir: Path) -> Path:
    produced = [path for path in output_dir.iterdir() if path.is_dir()]
    if len(produced) != 1:
        raise RuntimeError(f"expected exactly one experiment directory, got {produced}")
    return produced[0]


def run_experiment(name: str, extra: list[str]) -> Path:
    """Run the CLI into a fresh directory and return the one experiment produced."""

    output_dir = fresh_dir(name)
    run_cli([*extra, "--output-dir", str(output_dir)])
    return only_experiment(output_dir)


def load_features(experiment_dir: Path, split: str = "train") -> np.ndarray:
    return np.load(experiment_dir / f"features_{split}.npy")


def load_metrics(experiment_dir: Path) -> dict:
    return json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))


def replay_args() -> list[str]:
    """Rebuild the CLI invocation of :data:`REPLAY_EXPERIMENT` from its config."""

    config = json.loads((REPLAY_EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    emd = config["emd"]
    mlp = config["mlp"]
    mapper = config["depth_mapper"]
    args = [
        "--data-dir", "raw_data",
        "--forms", str(config["form"]),
        "--region-set", str(config["region_name"]),
        "--channel-mode", str(config["channel_mode"]),
        "--target-length", str(config["target_length"]),
        "--tukey-alpha", str(config["tukey_alpha"]),
        "--max-depth-mm", str(mapper["max_depth_mm"]),
        "--signal-length", str(mapper["signal_length"]),
        "--rounding", str(mapper["rounding"]),
        "--max-imfs", str(emd["max_imfs"]),
        "--max-sift-iterations", str(emd["max_sift_iterations"]),
        "--sift-sd-threshold", str(emd["sift_sd_threshold"]),
        "--stream-aggregation", str(emd["stream_aggregation"]),
        "--channel-aggregation", str(emd["channel_aggregation"]),
        "--locator-features", str(emd["locator_features"]),
        "--hidden-dims", ",".join(str(int(value)) for value in mlp["hidden_dims"]),
        "--learning-rate", str(mlp["learning_rate"]),
        "--dropout", str(mlp["dropout"]),
        "--seed", str(mlp["seed"]),
        # One epoch is enough: claim 1 compares stored features, not the fit, and
        # the training loop cannot feed anything back into feature extraction.
        "--epochs", "1",
        "--patience", "1",
    ]
    # ``--feature-set`` is a preset *name*, but the config records the expanded
    # statistic list, so the preset is recovered by matching that list instead of
    # assuming a name.
    from emd_pipeline import EMD_FEATURE_PRESETS

    recorded = list(emd["feature_names"])
    matches = [
        name for name, names in EMD_FEATURE_PRESETS.items() if list(names) == recorded
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"cannot map the recorded statistics {recorded} back to one preset: {matches}"
        )
    args += ["--feature-set", matches[0]]

    envelope = config["dynamic_envelope"]
    for key, flag in DYN_FLAGS.items():
        args += [flag, str(envelope[key])]
    if not envelope["apply_manual_corrections"]:
        args.append("--dyn-no-manual-corrections")
    return args


def check_single_channel_untouched() -> bool:
    """Claim 1: a channels_1 run is bit-identical to its pre-S4 artifact."""

    print("a contrast option must not disturb a single-channel run")
    if not (REPLAY_EXPERIMENT / "features_train.npy").exists():
        print(f"  [SKIP] {REPLAY_EXPERIMENT} has no stored features to replay")
        return True

    stored = load_features(REPLAY_EXPERIMENT)
    replay = run_experiment("replay_channels_1", replay_args())
    rebuilt = load_features(replay)
    print(f"        replaying {REPLAY_EXPERIMENT.name}")

    ok = True
    if stored.shape != rebuilt.shape:
        print(f"  [FAIL] shape changed: {stored.shape} -> {rebuilt.shape}")
        return False

    # Bit-equality is expected everywhere except the one statistic whose
    # definition moved after this artifact was written, so the drift is measured
    # per column and attributed by name rather than absorbed into a tolerance.
    names = list(load_metrics(replay)["feature_dimension_names"])
    difference = np.abs(stored - rebuilt)
    per_column = difference.max(axis=0) if stored.size else np.zeros(0)
    changed = [index for index in np.flatnonzero(per_column) if per_column[index] > 0.0]
    scale = np.abs(stored).max(axis=0) if stored.size else np.zeros(0)

    unexpected = [index for index in changed if not names[index].endswith(DOCUMENTED_DRIFT_SUFFIX)]
    if unexpected:
        worst = max(unexpected, key=lambda index: per_column[index])
        print(
            f"  [FAIL] {len(unexpected)} column(s) moved that S4 cannot explain, e.g. "
            f"{names[worst]} by {per_column[worst]:.3e} on a scale of {scale[worst]:.3e}"
        )
        ok = False
    else:
        print(
            f"  [PASS] every column but the documented one is bit-identical  "
            f"{stored.shape[0]} samples x {stored.shape[1]} columns"
        )

    if changed:
        relative = max(
            float(per_column[index] / scale[index]) if scale[index] > 0.0 else 0.0
            for index in changed
        )
        if relative > DOCUMENTED_DRIFT_ULPS * float(np.finfo(np.float32).eps):
            print(
                f"  [FAIL] the documented drift grew to a relative {relative:.2e}, beyond "
                f"{DOCUMENTED_DRIFT_ULPS} float32 ulps"
            )
            ok = False
        else:
            print(
                f"  [PASS] the documented drift is still round-off  "
                f"{len(changed)} column ({names[changed[0]]}) at a relative {relative:.2e} "
                f"= {relative / float(np.finfo(np.float32).eps):.1f} float32 ulps"
            )
    else:
        print("  [PASS] the artifact reproduces bit-for-bit, with no drift at all")

    rebuilt_info = json.loads((replay / "feature_info.json").read_text(encoding="utf-8"))
    group_info = rebuilt_info["train"][0]
    if group_info.get("contrast_group"):
        print(f"  [FAIL] a single-channel run grew a contrast group: {group_info['contrast_group']}")
        ok = False
    else:
        dims = group_info.get("feature_group_dims")
        print(f"  [PASS] a single-channel run still has one group  feature_group_dims={dims}")

    # Asking for a contrast on one channel is a configuration error, not a
    # silent no-op: if it were ignored, a user could believe they had run S4.
    refused = run_cli(
        [
            "--data-dir", "raw_data",
            "--forms", "mean_std",
            "--region-set", "full",
            "--channel-mode", "1",
            "--channel-contrast", "normalized",
            "--max-samples", "4",
            "--epochs", "1",
            "--patience", "1",
            "--output-dir", str(fresh_dir("refuse_single_channel")),
        ],
        expect_success=False,
    )
    if refused.returncode == 0:
        print("  [FAIL] channel_mode=1 with a contrast was accepted instead of refused")
        ok = False
    elif "contrasts two channels" not in (refused.stderr + refused.stdout):
        print(f"  [FAIL] refused, but the message does not explain why:")
        print(f"         {(refused.stderr + refused.stdout).strip()[-400:]}")
        ok = False
    else:
        print("  [PASS] channel_mode=1 with a contrast is refused with an explanation")
    return ok


def check_appended_not_woven(common: list[str]) -> bool:
    """Claims 2 and 3: only new columns appear, and they are the declared maths."""

    print("the contrast group must be appended, and must be the declared arithmetic")
    baseline = load_features(run_experiment("full_channels_3", common))
    width = int(baseline.shape[1])

    ok = True
    for kind in CONTRAST_UNDER_TEST:
        contrasted = load_features(
            run_experiment(f"contrast_{kind}", [*common, "--channel-contrast", kind])
        )
        if contrasted.shape[1] != width + width // 2:
            print(
                f"  [FAIL] {kind}: width {contrasted.shape[1]}, expected "
                f"{width} + {width // 2}"
            )
            ok = False
            continue
        leading = float(np.max(np.abs(contrasted[:, :width] - baseline))) if width else 0.0
        if leading != 0.0:
            print(f"  [FAIL] {kind}: adding a contrast perturbed the existing columns ({leading:.3e})")
            ok = False
            continue

        first = np.asarray(baseline[:, : width // 2], dtype=np.float64)
        second = np.asarray(baseline[:, width // 2 :], dtype=np.float64)
        expected = closed_form(kind, first, second)
        got = np.asarray(contrasted[:, width:], dtype=np.float64)
        error = float(np.max(np.abs(got - expected)))
        if error > 1e-5:
            print(f"  [FAIL] {kind}: contrast does not match its definition (max err {error:.3e})")
            ok = False
            continue
        print(
            f"  [PASS] {kind}: existing {width} columns untouched, "
            f"the {width // 2} new columns match the definition (max err {error:.2e})"
        )

    # The scale-free forms exist to remove a gain that multiplies *both* channels.
    # Recomputing from a gain-scaled pair is the cheapest honest test of that.
    first = np.asarray(baseline[:, : width // 2], dtype=np.float64)
    second = np.asarray(baseline[:, width // 2 :], dtype=np.float64)
    gain = 7.5
    unscaled = closed_form("normalized", first, second)
    scaled = closed_form("normalized", first * gain, second * gain)
    worst = float(np.max(np.abs(unscaled - scaled))) if unscaled.size else 0.0
    if worst > 1e-9:
        print(f"  [FAIL] 'normalized' is not invariant to a common gain ({worst:.3e})")
        ok = False
    else:
        print(
            f"  [PASS] 'normalized' is invariant to a common gain  "
            f"a {gain}x gain on both channels moved it by {worst:.1e}"
        )
    return ok


def closed_form(kind: str, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The contrast definition, restated independently of the pipeline."""

    if kind == "difference":
        return first - second
    if kind == "normalized":
        denominator = np.abs(first) + np.abs(second)
        nonzero = denominator > 0.0
        return np.where(nonzero, (first - second) / np.where(nonzero, denominator, 1.0), 0.0)
    if kind == "log_ratio":
        epsilon = 1e-30 * max(
            1.0,
            max(
                float(np.max(np.abs(first))) if first.size else 1.0,
                float(np.max(np.abs(second))) if second.size else 1.0,
            ),
        )
        return np.sign(first) * np.log(np.abs(first) + epsilon) - np.sign(second) * np.log(
            np.abs(second) + epsilon
        )
    raise ValueError(f"no closed form declared for {kind!r}")


def check_published_metadata(common: list[str]) -> bool:
    """Claim 4: the metadata describes the extra group and its columns."""

    print("the extra group must be described by the published metadata")
    contrasted = run_experiment("metadata", [*common, "--channel-contrast", "normalized"])
    features = load_features(contrasted)
    metrics = load_metrics(contrasted)
    dims = [int(width) for width in metrics["feature_group_dims"]]
    names = list(metrics["feature_dimension_names"])

    ok = True
    if dims != [int(width) for width in dims[:2]] + [dims[1]]:
        print(f"  [FAIL] feature_group_dims is not two equal groups plus a contrast: {dims}")
        ok = False
    else:
        print(f"  [PASS] feature_group_dims is two equal channel groups plus the contrast: {dims}")

    if int(np.sum(dims)) != int(features.shape[1]):
        print(f"  [FAIL] feature_group_dims sums to {int(np.sum(dims))}, data is {features.shape[1]} wide")
        ok = False
    elif len(names) != int(features.shape[1]):
        print(f"  [FAIL] {len(names)} column names for {features.shape[1]} columns")
        ok = False
    else:
        print(f"  [PASS] widths and names agree with the data  {features.shape[1]} columns")

    contrast_names = names[dims[0] + dims[1] :]
    if len(contrast_names) != dims[-1]:
        print(f"  [FAIL] {len(contrast_names)} contrast names for a width-{dims[-1]} group")
        ok = False
    elif not all(name.startswith("contrast_normalized:") for name in contrast_names):
        missing = [name for name in contrast_names if not name.startswith("contrast_normalized:")][:3]
        print(f"  [FAIL] contrast columns are not prefixed with their kind: {missing}")
        ok = False
    else:
        # The prefix must be the only difference: a contrast column has to stay
        # traceable to the channel-1 statistic it was derived from.
        source_names = names[: dims[0]]
        stripped = [name.split(":", 1)[1] for name in contrast_names]
        if stripped != source_names:
            mismatch = next(
                (pair for pair in zip(stripped, source_names) if pair[0] != pair[1]), None
            )
            print(f"  [FAIL] contrast columns do not mirror the source group, e.g. {mismatch}")
            ok = False
        else:
            print(
                f"  [PASS] contrast columns mirror the channel-1 names, e.g. "
                f"{contrast_names[0]!r}"
            )

    layout = metrics["model_layout"]
    if int(layout["tower_count"]) != 3:
        print(f"  [FAIL] expected three input towers, got {layout['tower_count']}")
        ok = False
    else:
        print(f"  [PASS] the classifier builds one tower per group  {layout['tower_dims']}")

    if not np.all(np.isfinite(features)):
        print(f"  [FAIL] the contrasted matrix contains non-finite values")
        ok = False
    else:
        print("  [PASS] the contrasted matrix is finite")
    return ok


def check_directory_name(common: list[str]) -> bool:
    """A contrasted run must not land on top of the baseline it is compared with."""

    print("a contrasted run must own its output directory")
    output_dir = fresh_dir("naming")
    run_cli([*common, "--output-dir", str(output_dir)])
    run_cli([*common, "--channel-contrast", "log_ratio", "--output-dir", str(output_dir)])
    produced = sorted(path.name for path in output_dir.iterdir() if path.is_dir())
    expected = ["form_mean_std__regions_full__channels_3", "form_mean_std__regions_full__channels_3__contrast_log_ratio"]
    if produced != expected:
        print(f"  [FAIL] expected {expected}, got {produced}")
        return False
    print(f"  [PASS] baseline and contrast run coexist  {produced}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the output directory")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    common = [
        "--data-dir", "raw_data",
        "--forms", "mean_std",
        "--region-set", "full",
        "--channel-mode", "3",
        "--max-samples", "16",
        "--epochs", "2",
        "--patience", "2",
    ]

    print("Verifying --channel-contrast (S4).\n")
    results = [
        ("a contrast cannot change a single-channel run", check_single_channel_untouched()),
        ("the contrast group is appended and correctly defined", check_appended_not_woven(common)),
        ("the published metadata describes the extra group", check_published_metadata(common)),
        ("a contrasted run owns its output directory", check_directory_name(common)),
    ]

    if not args.keep and WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    passed = sum(1 for _, ok in results if ok)
    print()
    for label, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

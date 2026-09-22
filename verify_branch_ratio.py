"""Regression guard for ``--branch-ratio`` (strategy S3b).

S3a added fits that measure how a statistic *changes with depth inside one
window*; S3b measures the change **between** the two windows.  Under the
``dyn_envelope`` region plan branch 0 is the shallow envelope window and branch 1
the deeper tail window, so a ratio of the two is the attenuation itself.

Like the cross-channel contrast of S4 the option has to be strictly **additive**,
and unlike S4 it also has a physical normalisation to get right.  Five claims:

1. **A ratio cannot change a run that does not ask for one.**  A pre-S3b
   experiment directory on disk (``experiments/...__channels_1``) is replayed
   from its own recorded ``config.json`` and must reproduce bit-for-bit.  Asking
   for a ratio on a region plan that yields a single window is refused with an
   explanation rather than silently producing a meaningless column.
2. **The ratio block is appended, not woven in.**  Running with and without the
   option, every pre-existing column is bit-identical and the width grows by
   exactly one five-column block per feature group.
3. **Those columns are the declared arithmetic** of the two windows: the
   ``ratio`` and ``attenuation`` forms are recomputed from the baseline columns
   by an independent restatement of the definitions, including the separation
   normalisation -- and the attenuation is shown to be invariant to a common gain
   on both windows, which is what makes it a medium property rather than a
   coupling property.
4. **The published metadata describes the extra block**: ``feature_group_dims``
   grows per group, every new name carries its ``branch_ratio[<preset>].``
   prefix, and the derived branch publishes its own ``feature_names``, its
   numerator/denominator branches and the per-sample window separations it used.
5. **A ratio run owns its output directory**, so it cannot overwrite the baseline
   it is meant to be compared against.

Runs are real CLI invocations, so nothing is mocked.  Claim 1 replays one full
experiment (reduced to one epoch, since only the features are compared); the rest
use ``--max-samples`` so they stay quick.

The checks are individually invocable so a long verification can be run in small
batches:

    python verify_branch_ratio.py            # all checks
    python verify_branch_ratio.py --list     # print the catalogue
    python verify_branch_ratio.py --check 3  # only check 3 (repeatable)

Add ``--keep`` to keep the temporary ``_ratio_regress/`` output directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from verify_channel_contrast import (
    DOCUMENTED_DRIFT_SUFFIX,
    DOCUMENTED_DRIFT_ULPS,
    REPLAY_EXPERIMENT,
    load_features,
    load_metrics,
    replay_args,
    run_cli,
)

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "_ratio_regress"

#: Presets this script exercises.  ``ratio`` is the plain multiplicative form of
#: the S3 proposal, ``attenuation`` the additive (nepers) one.
PRESETS_UNDER_TEST = ("ratio", "attenuation")

#: The paired statistics S3b declares, restated here so a change to the pipeline's
#: list shows up as a verification failure rather than as a silent agreement.
PAIRED_STATISTICS = ("std", "peak_abs", "zero_crossing_rate", "spectral_centroid")

#: Width of one ratio block per feature group: one column per pair plus the
#: separation-normalised attenuation coefficient.
BLOCK_WIDTH = len(PAIRED_STATISTICS) + 1

#: The export name of the gain-invariant attenuation coefficient.
ATTENUATION_FEATURE = "attenuation_per_mm"

#: The floor used inside the signed logarithm, mirrored from
#: :data:`emd_pipeline.CHANNEL_CONTRAST_EPSILON`.
SIGNED_LOG_EPSILON = 1e-30


# --------------------------------------------------------------------------- #
# Independent restatement of the S3b definitions                              #
# --------------------------------------------------------------------------- #


def signed_log(value: float) -> float:
    """``sign(x) * log(|x| + eps)`` with the pipeline's amplitude-tied floor."""

    epsilon = SIGNED_LOG_EPSILON * max(1.0, abs(value))
    return float(np.sign(value) * np.log(abs(value) + epsilon))


def ratio_value(upper: float, lower: float, form: str) -> float:
    """One declared contrast of two window statistics."""

    if form == "ratio":
        # An exactly zero denominator carries no ratio and the pipeline writes
        # zero rather than inventing a magnitude.
        return upper / lower if lower != 0.0 else 0.0
    if form == "log_ratio":
        return signed_log(upper) - signed_log(lower)
    raise ValueError(f"no closed form declared for {form!r}")


def declared_columns(
    upper_window: np.ndarray,
    lower_window: np.ndarray,
    feature_names: Sequence[str],
    separation_mm: float,
    form: str,
) -> list[float]:
    """The whole ratio block for one sample, from the two raw window vectors."""

    upper = {
        name: float(np.mean(upper_window[index :: len(feature_names)]))
        for index, name in enumerate(feature_names)
    }
    lower = {
        name: float(np.mean(lower_window[index :: len(feature_names)]))
        for index, name in enumerate(feature_names)
    }
    values = [
        ratio_value(upper[statistic], lower[statistic], form)
        for statistic in PAIRED_STATISTICS
    ]
    values.append(
        (signed_log(upper["std"]) - signed_log(lower["std"])) / separation_mm
    )
    return values


# --------------------------------------------------------------------------- #
# Harness                                                                     #
# --------------------------------------------------------------------------- #


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


def load_info(experiment_dir: Path, split: str = "train") -> dict:
    return json.loads((experiment_dir / "feature_info.json").read_text(encoding="utf-8"))[split]


def ratio_branch(sample_info: dict) -> dict | None:
    """The derived branch, if this run asked for one."""

    for branch in sample_info["branches"]:
        if str(branch.get("name", "")).startswith("branch_ratio["):
            return branch
    return None


def window_layout(sample_info: dict) -> dict:
    """Where each window's columns sit inside one feature group.

    A group is assembled as ``[branch 0][locators][branch 1]`` -- the locator
    block is appended to branch 0 because it describes where the windows are --
    so the two windows are *not* contiguous and the offsets are read from the
    published metadata rather than assumed.
    """

    bridges = int(sample_info["feature_groups"])
    branch0 = int(sample_info["branches"][0]["group_widths"][0])
    branch1 = int(sample_info["branches"][1]["group_widths"][0])
    group_widths = [int(width) for width in sample_info["feature_group_dims"]]
    if len(group_widths) != bridges:
        raise RuntimeError(f"{len(group_widths)} group widths for {bridges} groups")
    locator = group_widths[0] - branch0 - branch1
    if locator < 0:
        raise RuntimeError(f"group of {group_widths[0]} is too small for 2 windows")
    return {"groups": bridges, "branch0": branch0, "branch1": branch1, "locator": locator}


# --------------------------------------------------------------------------- #
# Claim 1                                                                     #
# --------------------------------------------------------------------------- #


def check_additive() -> bool:
    """Claim 1: a default run is unchanged, and an impossible ratio is refused."""

    print("a ratio option must not disturb a run that does not ask for one")
    if not (REPLAY_EXPERIMENT / "features_train.npy").exists():
        print(f"  [SKIP] {REPLAY_EXPERIMENT} has no stored features to replay")
        return True

    stored = load_features(REPLAY_EXPERIMENT)
    replay = run_experiment("replay_default", replay_args())
    rebuilt = load_features(replay)
    print(f"        replaying {REPLAY_EXPERIMENT.name}")

    ok = True
    if stored.shape != rebuilt.shape:
        print(f"  [FAIL] shape changed: {stored.shape} -> {rebuilt.shape}")
        return False

    # Bit-equality everywhere except the one statistic whose definition moved
    # after the ground-truth artifact was written (S3a dropped a ``+1e-12`` from
    # the power sum feeding ``spectral_centroid``), so the drift is attributed by
    # column name instead of being absorbed into a blanket tolerance.
    names = list(load_metrics(replay)["feature_dimension_names"])
    difference = np.abs(stored - rebuilt)
    per_column = difference.max(axis=0) if stored.size else np.zeros(0)
    changed = [int(index) for index in np.flatnonzero(per_column) if per_column[index] > 0.0]
    scale = np.abs(stored).max(axis=0) if stored.size else np.zeros(0)

    unexpected = [index for index in changed if not names[index].endswith(DOCUMENTED_DRIFT_SUFFIX)]
    if unexpected:
        worst = max(unexpected, key=lambda index: per_column[index])
        print(
            f"  [FAIL] {len(unexpected)} column(s) moved that S3b cannot explain, e.g. "
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

    info = load_info(replay)[0]
    if ratio_branch(info) is not None:
        print(f"  [FAIL] a default run grew a ratio branch")
        ok = False
    else:
        print(
            f"  [PASS] a default run has no derived branch  "
            f"feature_group_dims={info['feature_group_dims']}"
        )

    # A single-window plan has nothing to ratio against.  Silently emitting a
    # zero column would let a user believe they had run S3b, so it must fail.
    refused = run_cli(
        [
            "--data-dir", "raw_data",
            "--forms", "mean_std",
            "--region-set", "full",
            "--channel-mode", "1",
            "--branch-ratio", "ratio",
            "--max-samples", "4",
            "--epochs", "1",
            "--patience", "1",
            "--output-dir", str(fresh_dir("refuse_single_window")),
        ],
        expect_success=False,
    )
    if refused.returncode == 0:
        print("  [FAIL] a single-window region plan with a ratio was accepted")
        ok = False
    elif "contrasts two windows" not in (refused.stderr + refused.stdout):
        print("  [FAIL] refused, but the message does not explain why:")
        print(f"         {(refused.stderr + refused.stdout).strip()[-400:]}")
        ok = False
    else:
        print("  [PASS] a single-window plan with a ratio is refused with an explanation")
    return ok


# --------------------------------------------------------------------------- #
# Claim 2                                                                     #
# --------------------------------------------------------------------------- #


def check_appended(common: list[str]) -> bool:
    """Claim 2: only new columns appear, and the width grows by one block per group."""

    print("the ratio block must be appended, not woven into the existing columns")
    baseline_dir = run_experiment("appended_baseline", common)
    baseline = load_features(baseline_dir)
    base_dims = [int(width) for width in load_info(baseline_dir)[0]["feature_group_dims"]]
    layout = window_layout(load_info(baseline_dir)[0])
    print(
        f"        baseline is {baseline.shape[1]} columns as {base_dims}; "
        f"expecting {baseline.shape[1] + layout['groups'] * BLOCK_WIDTH}"
    )

    ok = True
    for preset in PRESETS_UNDER_TEST:
        run = run_experiment(f"appended_{preset}", [*common, "--branch-ratio", preset])
        wide = load_features(run)
        wide_info = load_info(run)[0]
        wide_dims = [int(width) for width in wide_info["feature_group_dims"]]

        if wide.shape[0] != baseline.shape[0]:
            print(
                f"  [FAIL] {preset}: the run has {wide.shape[0]} samples, "
                f"the baseline {baseline.shape[0]}"
            )
            ok = False
            continue
        if [width - base for width, base in zip(wide_dims, base_dims)] != [
            BLOCK_WIDTH
        ] * layout["groups"]:
            print(f"  [FAIL] {preset}: group widths did not each grow by {BLOCK_WIDTH}: {wide_dims}")
            ok = False
            continue

        # The block lands *after* each group's own content, so the existing
        # columns are only identical if the insertion point is per group rather
        # than at the very end of the matrix.  Comparing group by group, in order,
        # is what makes that distinction visible.
        worst = 0.0
        base_offset = 0
        wide_offset = 0
        for group, base_width in enumerate(base_dims):
            moved = float(
                np.max(
                    np.abs(
                        wide[:, wide_offset : wide_offset + base_width]
                        - baseline[:, base_offset : base_offset + base_width]
                    )
                )
            )
            if moved != 0.0:
                print(
                    f"  [FAIL] {preset}: adding a ratio perturbed group {group} "
                    f"by {moved:.3e}"
                )
                ok = False
                break
            worst = max(worst, moved)
            base_offset += base_width
            wide_offset += wide_dims[group]
        else:
            print(
                f"  [PASS] {preset}: all {baseline.shape[1]} existing columns are bit-identical "
                f"and each of the {layout['groups']} group(s) grew by {BLOCK_WIDTH}"
            )
    return ok


# --------------------------------------------------------------------------- #
# Claim 3                                                                     #
# --------------------------------------------------------------------------- #


def check_arithmetic(common: list[str]) -> bool:
    """Claim 3: the new columns are the declared arithmetic of the two windows."""

    print("the ratio columns must be the declared arithmetic of the two windows")
    baseline_dir = run_experiment("arith_baseline", common)
    baseline = load_features(baseline_dir)
    baseline_info = load_info(baseline_dir)[0]
    layout = window_layout(baseline_info)
    feature_names = [str(name).rsplit(".", 1)[-1] for name in baseline_info["component_features"]]
    base_dims = [int(width) for width in baseline_info["feature_group_dims"]]
    print(f"        {layout['groups']} group(s); statistics per window: {feature_names}")

    def offsets(dims: Sequence[int]) -> list[int]:
        running = 0
        result = []
        for width in dims:
            result.append(running)
            running += int(width)
        return result

    base_offsets = offsets(base_dims)
    ok = True
    for preset in PRESETS_UNDER_TEST:
        form = "ratio" if preset == "ratio" else "log_ratio"
        run = run_experiment(f"arith_{preset}", [*common, "--branch-ratio", preset])
        wide = load_features(run)
        samples = load_info(run)
        wide_dims = [int(width) for width in samples[0]["feature_group_dims"]]
        wide_offsets = offsets(wide_dims)

        # The published geometry has to agree before the numbers can be trusted:
        # one five-column block per group, sitting after that group's own content.
        published = [int(width) for width in ratio_branch(samples[0])["group_widths"]]
        if published != [BLOCK_WIDTH] * layout["groups"]:
            print(f"  [FAIL] {preset}: the derived branch publishes widths {published}")
            ok = False
            continue

        expected = np.empty((wide.shape[0], layout["groups"] * BLOCK_WIDTH), dtype=np.float64)
        got = np.empty_like(expected)
        for row in range(wide.shape[0]):
            branch = ratio_branch(samples[row])
            for group in range(layout["groups"]):
                base_start = base_offsets[group]
                upper = baseline[
                    row, base_start : base_start + layout["branch0"]
                ]
                lower_start = base_start + layout["branch0"] + layout["locator"]
                lower = baseline[
                    row, lower_start : base_start + base_dims[group]
                ]
                separation = float(branch["sample_spacing_mm_by_channel"][group])
                block = declared_columns(
                    upper, lower, feature_names, separation, form
                )
                start = group * BLOCK_WIDTH
                expected[row, start : start + BLOCK_WIDTH] = block
                wide_end = wide_offsets[group] + wide_dims[group]
                got[row, start : start + BLOCK_WIDTH] = wide[
                    row, wide_end - BLOCK_WIDTH : wide_end
                ]

        error = float(np.max(np.abs(got - expected))) if got.size else 0.0
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        # float32 storage against a float64 restatement, so the only meaningful
        # tolerance is a relative one.
        relative = error / scale if scale > 0.0 else 0.0
        if relative > 1e-5:
            _, column = np.unravel_index(int(np.argmax(np.abs(got - expected))), got.shape)
            print(
                f"  [FAIL] {preset}: columns do not match the definition "
                f"(max err {error:.3e} on a scale of {scale:.3e}, worst at "
                f"group {column // BLOCK_WIDTH} column {column % BLOCK_WIDTH})"
            )
            ok = False
            continue
        print(
            f"  [PASS] {preset}: all {got.shape[1]} new columns match the definition "
            f"(max err {error:.2e}, relative {relative:.2e} on a scale of {scale:.2f})"
        )

    # The point of dividing by the separation is that the answer is an attenuation
    # coefficient in 1/mm -- a property of the medium, not of how the detector
    # happened to place the two windows.  A common gain on both windows is the
    # other nuisance factor, and the signed-log form must cancel it.
    upper = np.asarray([3.0, 1.5, 0.4, 2.0])
    lower = np.asarray([0.3, 0.2, 0.9, 0.7])
    base = declared_columns(upper, lower, PAIRED_STATISTICS, 2.0, "log_ratio")
    gain = 7.5
    scaled = declared_columns(
        upper * gain, lower * gain, PAIRED_STATISTICS, 2.0, "log_ratio"
    )
    drift = float(np.max(np.abs(np.asarray(base) - np.asarray(scaled))))
    if drift > 1e-9:
        print(f"  [FAIL] the attenuation is not invariant to a common gain ({drift:.3e})")
        ok = False
    else:
        print(
            f"  [PASS] the attenuation is invariant to a common gain  "
            f"a {gain}x gain on both windows moved it by {drift:.1e}"
        )
    return ok


# --------------------------------------------------------------------------- #
# Claim 4                                                                     #
# --------------------------------------------------------------------------- #


def check_metadata(common: list[str]) -> bool:
    """Claim 4: the metadata describes the derived branch and its geometry."""

    print("the derived branch must be described by the published metadata")
    baseline_dir = run_experiment("meta_baseline", common)
    run = run_experiment("meta_ratio", [*common, "--branch-ratio", "ratio"])
    features = load_features(run)
    metrics = load_metrics(run)
    dims = [int(width) for width in metrics["feature_group_dims"]]
    names = list(metrics["feature_dimension_names"])
    samples = load_info(run)
    layout = window_layout(samples[0])
    base_dims = [int(width) for width in load_info(baseline_dir)[0]["feature_group_dims"]]

    ok = True
    if len(dims) != len(base_dims):
        print(f"  [FAIL] the number of groups changed: {base_dims} -> {dims}")
        ok = False
    elif [width - base for width, base in zip(dims, base_dims)] != [BLOCK_WIDTH] * len(dims):
        print(f"  [FAIL] groups did not each grow by {BLOCK_WIDTH}: {base_dims} -> {dims}")
        ok = False
    else:
        print(f"  [PASS] each group grew by one ratio block  {base_dims} -> {dims}")

    if int(np.sum(dims)) != int(features.shape[1]):
        print(f"  [FAIL] feature_group_dims sums to {int(np.sum(dims))}, data is {features.shape[1]} wide")
        ok = False
    elif len(names) != int(features.shape[1]):
        print(f"  [FAIL] {len(names)} column names for {features.shape[1]} columns")
        ok = False
    else:
        print(f"  [PASS] widths and names agree with the data  {features.shape[1]} columns")

    # A derived branch publishes its *block* names and ``feature_dimension_names``
    # prefixes them with the branch name, exactly as it does for a component
    # branch.  Checking both halves separately is what keeps a column traceable
    # back to the statistic pair it came from.
    expected_block = [
        *(f"ratio:{name}/{name}" for name in PAIRED_STATISTICS),
        f"{ATTENUATION_FEATURE}:std/std",
    ]
    recorded = [str(name) for name in ratio_branch(samples[0])["feature_names"]]
    if recorded != expected_block:
        print(f"  [FAIL] the derived branch publishes unexpected columns: {recorded}")
        ok = False
    else:
        print(f"  [PASS] the derived branch publishes {len(expected_block)} block names")

    if not ok:
        return False

    for group in range(layout["groups"]):
        suffix = f"[ch{group + 1}]" if layout["groups"] > 1 else ""
        branch_name = f"branch_ratio[ratio]{suffix}"
        # The block sits directly after the group's own columns, which is where the
        # group-major assembly puts an appended branch -- so locating it by index
        # verifies the position and the naming in one step.
        start = sum(dims[:group]) + base_dims[group]
        actual = names[start : start + BLOCK_WIDTH]
        expected_names = [f"{branch_name}.{name}" for name in expected_block]
        if actual != expected_names:
            print(
                f"  [FAIL] group {group}: columns {start}..{start + BLOCK_WIDTH - 1} are "
                f"{actual}, expected {expected_names}"
            )
            ok = False
        else:
            print(f"  [PASS] group {group}: the block is at {start} and named {actual[0]!r} .. {actual[-1]!r}")

    prefixed = [name for name in names if "branch_ratio[" in name]
    if len(prefixed) != BLOCK_WIDTH * layout["groups"]:
        print(
            f"  [FAIL] {len(prefixed)} ratio column(s) for "
            f"{BLOCK_WIDTH} x {layout['groups']} group(s)"
        )
        ok = False
    else:
        print(f"  [PASS] all {len(prefixed)} new columns carry their branch prefix")

    # The normalisation is only meaningful if the recorded separation is the real
    # distance between the two window centres, which under dyn_envelope varies
    # from sample to sample.  A constant would mean the geometry was assumed.
    separations = [
        float(ratio_branch(sample)["sample_spacing_mm_by_channel"][0])
        for sample in samples
    ]
    if not all(value > 0.0 for value in separations):
        print(f"  [FAIL] a recorded window separation is not positive: {min(separations)}")
        ok = False
    elif len(set(separations)) == 1:
        print(
            f"  [FAIL] the window separation is constant at {separations[0]:.4f} mm, "
            f"but dyn_envelope places the windows per sample"
        )
        ok = False
    else:
        print(
            f"  [PASS] the separation is per-sample and positive  "
            f"{min(separations):.3f}-{max(separations):.3f} mm over {len(separations)} samples"
        )

    for sample in samples:
        branch = ratio_branch(sample)
        if branch["numerator_branch"] != "dynamic_main_window":
            print(f"  [FAIL] numerator is {branch['numerator_branch']}, not the shallow window")
            ok = False
            break
        if branch["denominator_branch"] != "dynamic_tail_window":
            print(f"  [FAIL] denominator is {branch['denominator_branch']}, not the deeper window")
            ok = False
            break
    else:
        print("  [PASS] the shallow window is the numerator and the deeper one the denominator")

    layout_metrics = metrics["model_layout"]
    if int(layout_metrics["tower_count"]) != layout["groups"]:
        print(
            f"  [FAIL] expected {layout['groups']} tower(s), got {layout_metrics['tower_count']}"
        )
        ok = False
    else:
        print(
            f"  [PASS] the classifier still builds one tower per group  "
            f"{layout_metrics['tower_dims']}"
        )

    if not np.all(np.isfinite(features)):
        print("  [FAIL] the matrix with the ratio block contains non-finite values")
        ok = False
    else:
        print("  [PASS] the matrix with the ratio block is finite")
    return ok


# --------------------------------------------------------------------------- #
# Claim 5                                                                     #
# --------------------------------------------------------------------------- #


def check_directory_name(common: list[str]) -> bool:
    """A ratio run must not land on top of the baseline it is compared with."""

    print("a ratio run must own its output directory")
    output_dir = fresh_dir("naming")
    run_cli([*common, "--output-dir", str(output_dir)])
    run_cli([*common, "--branch-ratio", "ratio", "--output-dir", str(output_dir)])
    produced = sorted(path.name for path in output_dir.iterdir() if path.is_dir())
    stem = produced[0].split("__branchratio_")[0] if produced else ""
    expected = [stem, f"{stem}__branchratio_ratio"]
    if produced != expected:
        print(f"  [FAIL] expected {expected}, got {produced}")
        return False
    print(f"  [PASS] baseline and ratio run coexist  {produced}")
    return True


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

#: ``(label, callable, needs_common_settings)``.
CHECKS: list[tuple[str, Callable[..., bool], bool]] = [
    ("a ratio cannot change a run that does not ask for one", check_additive, False),
    ("the ratio block is appended, not woven in", check_appended, True),
    ("the new columns are the declared arithmetic", check_arithmetic, True),
    ("the published metadata describes the derived branch", check_metadata, True),
    ("a ratio run owns its output directory", check_directory_name, True),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="keep the output directory")
    parser.add_argument("--list", action="store_true", help="print the check catalogue and exit")
    parser.add_argument(
        "--check",
        type=int,
        action="append",
        default=None,
        metavar="N",
        help="run only check N (1-based, repeatable); default is all checks",
    )
    args = parser.parse_args()

    if args.list:
        for index, (label, _, _) in enumerate(CHECKS, start=1):
            print(f"{index}  {label}")
        return 0

    selected = list(range(1, len(CHECKS) + 1)) if not args.check else sorted(set(args.check))
    unknown = [index for index in selected if not 1 <= index <= len(CHECKS)]
    if unknown:
        parser.error(f"no such check: {unknown} (see --list)")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    common = [
        "--data-dir", "raw_data",
        "--forms", "mean_std",
        "--region-set", "dyn_envelope",
        "--channel-mode", "3",
        "--max-samples", "16",
        "--epochs", "2",
        "--patience", "2",
    ]

    print("Verifying --branch-ratio (S3b).\n")
    results = []
    for index in selected:
        label, function, needs_common = CHECKS[index - 1]
        print(f"[{index}/{len(CHECKS)}] {label}")
        ok = function(common) if needs_common else function()
        results.append((label, ok))
        print()

    if not args.keep and WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    passed = sum(1 for _, ok in results if ok)
    for label, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

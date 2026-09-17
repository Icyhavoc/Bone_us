"""Regression guard for ``--channel-aggregation`` and the two-tower MLP head.

Three claims are checked, each by running the real CLI so nothing is mocked:

1. With a **single** selected channel (``--channel-mode 1`` / ``2``) the
   ``per_channel`` and ``pooled`` settings must be byte-identical.  A single
   channel yields exactly one feature group, so the two code paths have to agree
   on features, on the resulting MLP and therefore on every metric.  This is the
   regression that proves the channel-3 rewrite did not disturb modes 1 and 2.
2. With ``--channel-mode 3`` the ``per_channel`` feature vector must be exactly
   twice as long as the ``pooled`` one and must expose two feature groups.  The
   old behaviour averaged the two physical channels away, so an unchanged length
   would mean the rewrite silently did nothing.
3. ``--channel-mode 3 --channel-aggregation per_channel`` must build an input
   tower per channel (two towers) while ``pooled`` builds one.

Runs are small but real: ``mean_std`` form, ``full`` region, five epochs.  The
``--max-samples`` runs are only used for the structural claims, where the exact
numbers do not matter.

Run: ``python verify_channel_aggregation.py``
Add ``--keep`` to keep the temporary ``_agg_regress/`` output directory.
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
WORK_DIR = ROOT / "_agg_regress"
COMMON = [
    "--data-dir",
    "raw_data",
    "--forms",
    "mean_std",
    "--region-set",
    "full",
]


def run_pipeline(name: str, extra: list[str]) -> Path:
    """Run the experiment CLI into ``WORK_DIR/name`` and return that directory."""
    output_dir = WORK_DIR / name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    command = [
        sys.executable,
        str(ROOT / "run_emd_experiments.py"),
        *COMMON,
        "--output-dir",
        str(output_dir),
        *extra,
    ]
    print(f"  $ {' '.join(extra)}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    produced = [path for path in output_dir.iterdir() if path.is_dir()]
    if len(produced) != 1:
        raise RuntimeError(f"expected exactly one experiment directory, got {produced}")
    return produced[0]


def load_metrics(experiment_dir: Path) -> dict:
    return json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))


def compare_identical(label: str, left: Path, right: Path) -> bool:
    """Assert two experiment runs agree bit-for-bit on features and metrics."""
    ok = True
    for split in ("train", "val", "test"):
        for prefix in ("features", "labels", "probabilities"):
            left_array = np.load(left / f"{prefix}_{split}.npy")
            right_array = np.load(right / f"{prefix}_{split}.npy")
            if left_array.shape != right_array.shape:
                print(f"  [FAIL] {prefix}_{split}: shape {left_array.shape} != {right_array.shape}")
                ok = False
                continue
            worst = float(np.max(np.abs(left_array - right_array))) if left_array.size else 0.0
            if worst != 0.0:
                print(f"  [FAIL] {prefix}_{split}: max_abs_diff={worst:.3e}")
                ok = False
    left_metrics = load_metrics(left)
    right_metrics = load_metrics(right)
    for split in ("train", "val", "test"):
        left_split = left_metrics["metrics"][split]
        right_split = right_metrics["metrics"][split]
        for key, left_value in left_split.items():
            right_value = right_split.get(key)
            if isinstance(left_value, float) and isinstance(right_value, float):
                if left_value != right_value:
                    print(f"  [FAIL] metrics[{split}].{key}: {left_value} != {right_value}")
                    ok = False
            elif left_value != right_value:
                print(f"  [FAIL] metrics[{split}].{key}: {left_value} != {right_value}")
                ok = False
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {label}: features, labels, probabilities and metrics are identical")
    return ok


def check_single_channel(channel_mode: int) -> bool:
    print(f"channel mode {channel_mode}: per_channel must equal pooled")
    per_channel = run_pipeline(
        f"ch{channel_mode}_per_channel",
        ["--channel-mode", str(channel_mode), "--channel-aggregation", "per_channel",
         "--epochs", "5", "--patience", "5"],
    )
    pooled = run_pipeline(
        f"ch{channel_mode}_pooled",
        ["--channel-mode", str(channel_mode), "--channel-aggregation", "pooled",
         "--epochs", "5", "--patience", "5"],
    )
    per_channel_metrics = load_metrics(per_channel)
    pooled_metrics = load_metrics(pooled)
    ok = compare_identical(f"channel mode {channel_mode}", per_channel, pooled)
    if len(per_channel_metrics["feature_group_dims"]) != 1:
        print(f"  [FAIL] expected one feature group, got {per_channel_metrics['feature_group_dims']}")
        ok = False
    else:
        print(f"  [OK]   one feature group: {per_channel_metrics['feature_group_dims']}")
    if pooled_metrics["feature_dim"] != per_channel_metrics["feature_dim"]:
        print("  [FAIL] feature_dim differs between settings")
        ok = False
    return ok


def check_channel_three_structure() -> bool:
    print("channel mode 3: per_channel must expose two groups and double the width")
    per_channel = run_pipeline(
        "ch3_per_channel",
        ["--channel-mode", "3", "--channel-aggregation", "per_channel",
         "--max-samples", "32", "--epochs", "5", "--patience", "5"],
    )
    pooled = run_pipeline(
        "ch3_pooled",
        ["--channel-mode", "3", "--channel-aggregation", "pooled",
         "--max-samples", "32", "--epochs", "5", "--patience", "5"],
    )
    per_channel_metrics = load_metrics(per_channel)
    pooled_metrics = load_metrics(pooled)

    ok = True
    per_channel_groups = per_channel_metrics["feature_group_dims"]
    pooled_groups = pooled_metrics["feature_group_dims"]
    if per_channel_groups != [pooled_groups[0], pooled_groups[0]]:
        print(f"  [FAIL] expected two equal halves of {pooled_groups[0]}, got {per_channel_groups}")
        ok = False
    else:
        print(f"  [OK]   groups {pooled_groups} -> {per_channel_groups}")
    if per_channel_metrics["feature_dim"] != 2 * pooled_metrics["feature_dim"]:
        print(
            f"  [FAIL] feature_dim {pooled_metrics['feature_dim']} -> "
            f"{per_channel_metrics['feature_dim']} (expected double)"
        )
        ok = False
    else:
        print(
            f"  [OK]   feature_dim {pooled_metrics['feature_dim']} -> "
            f"{per_channel_metrics['feature_dim']}"
        )
    per_channel_towers = per_channel_metrics["model_layout"]["tower_count"]
    pooled_towers = pooled_metrics["model_layout"]["tower_count"]
    if per_channel_towers != 2 or pooled_towers != 1:
        print(
            f"  [FAIL] input towers: per_channel={per_channel_towers} "
            f"pooled={pooled_towers} (expected 2 and 1)"
        )
        ok = False
    else:
        print(f"  [OK]   input towers: per_channel={per_channel_towers} pooled={pooled_towers}")
    per_channel_layout = per_channel_metrics["model_layout"]
    pooled_layout = pooled_metrics["model_layout"]
    per_channel_head = per_channel_layout["head_layers"]
    pooled_head = pooled_layout["head_layers"]
    # The head must keep its depth and its output widths; only the first layer's
    # input grows, because it now reads the concatenated towers.
    per_channel_outputs = [layer["fan_out"] for layer in per_channel_head]
    pooled_outputs = [layer["fan_out"] for layer in pooled_head]
    expected_fan_in = per_channel_layout["tower_count"] * per_channel_layout["tower_outputs"]
    if per_channel_outputs != pooled_outputs:
        print(f"  [FAIL] head output widths differ: {per_channel_outputs} != {pooled_outputs}")
        ok = False
    elif per_channel_head[0]["fan_in"] != expected_fan_in:
        print(
            f"  [FAIL] head fan_in={per_channel_head[0]['fan_in']}, "
            f"expected towers x width = {expected_fan_in}"
        )
        ok = False
    else:
        print(
            f"  [OK]   head depth/widths unchanged {pooled_outputs}; "
            f"fan_in {pooled_head[0]['fan_in']} -> {per_channel_head[0]['fan_in']} "
            f"(= {per_channel_layout['tower_count']} towers x "
            f"{per_channel_layout['tower_outputs']})"
        )
    print(
        f"  [note] test AUC: per_channel={per_channel_metrics['metrics']['test']['auc']:.4f} "
        f"pooled={pooled_metrics['metrics']['test']['auc']:.4f} (expected to differ)"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the _agg_regress directory")
    args = parser.parse_args()

    results = [
        check_single_channel(1),
        check_single_channel(2),
        check_channel_three_structure(),
    ]
    if not args.keep:
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    print()
    if all(results):
        print("channel aggregation regression guard passed")
        return 0
    print(f"{results.count(False)} of {len(results)} checks FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())

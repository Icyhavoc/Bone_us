"""Side-by-side view of the ``full`` and ``dyn_envelope`` cuts.

Focuses on the three train/val points whose Hilbert envelope never exceeds
``DynamicEnvelopeConfig.weak_threshold`` on one channel:

* ``N35_P1_01`` (train) -- channel 1 sub-threshold, so single-channel mode 1
  falls back to the degenerate joint-RMS rule and lands at sample 71
* ``N2_P2_01``  (val)   -- channel 1 sub-threshold, same degenerate result
* ``N5_P7_01``  (train) -- channel 2 sub-threshold, and the degenerate
  single-channel (mode 2) fallback lands at sample 893, pushing the main window
  past the end of the trace

Left column: the fixed-region ``full`` preset (one region, 0-5 mm == samples
``[0, 896)``).  Right column: ``dyn_envelope``.  On the right the *solid* bands
are the two-channel (mode 3) result, where a missing channel borrows its
partner's crossing, and the *hatched red* band is the same point evaluated with
only the sub-threshold channel selected -- which is what ``--channel-mode 1`` /
``--channel-mode 2`` actually feed to the pipeline.

Usage::

    python bin/visualize_dyn_envelope_cases.py     # run from the project root

This script was archived into ``bin/`` (it is not part of the GUI / documented
pipeline), so it bootstraps ``sys.path`` to import the project's ``emd_pipeline``
and writes to ``visualizations/`` relative to the current working directory.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

# Archived under ``bin/``: put the project root (parent of this file's folder)
# back on ``sys.path`` so ``emd_pipeline`` still resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import emd_pipeline as E

# (split, point id, single-channel mode that isolates the sub-threshold channel)
CASES: list[tuple[str, str, int]] = [
    ("train", "N35_P1_01", 1),
    ("val", "N2_P2_01", 1),
    ("train", "N5_P7_01", 2),
]
CH_COLORS = ("#1f6fd0", "#e07b1a")
OUT_PATH = Path("visualizations") / "dyn_envelope_cases.png"
FULL_REGION = E.RegionSpec(0.0, 5.0)


def smoothed_envelope(locator: np.ndarray, config: E.DynamicEnvelopeConfig) -> np.ndarray:
    """Hilbert envelope of the locator, box-smoothed past ``noise_start``."""

    envelope = E._hilbert_envelope(locator).astype(np.float32)
    smoothed = np.zeros_like(envelope)
    smoothed[:, config.noise_start :] = E._moving_average_nearest(
        envelope[:, config.noise_start :], config.smooth_window
    )
    return smoothed


def bar(ax, y0, height, segments, color, alpha=0.85, hatched=False, zorder=4):
    ax.broken_barh(
        [(start, stop - start) for start, stop in segments],
        (y0, height),
        facecolors="none" if hatched else color,
        edgecolors=color if hatched else "black",
        hatch="///" if hatched else None,
        linewidth=1.3 if hatched else 0.7,
        alpha=1.0 if hatched else alpha,
        zorder=zorder,
    )


def box(ax, x, y, text, fontsize=7.0, ha="center", color="black"):
    ax.text(
        x, y, text, ha=ha, va="center", fontsize=fontsize, color=color, zorder=8,
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="#999999", pad=2.0),
    )


def load_case(
    split: str,
    point_id: str,
    single_mode: int,
    mapper: E.DepthMapper,
    config: E.DynamicEnvelopeConfig,
) -> dict:
    frames, labels, samples = E.load_split("raw_data", split)
    ids = [str(record.get("sample_id", "")) for record in samples]
    index = ids.index(point_id)
    selected = E.select_channels(frames[index], 3)

    locator = E.locator_mean_signal(selected, config.locator_top_k)
    smoothed = smoothed_envelope(locator, config)
    processed, chosen = E.process_frames(selected, "top3_mean", mapper)
    paired = E.locate_dynamic_span(
        smoothed, config, point_id=point_id, physical_channels=(0, 1)
    )

    # Single-channel run: isolate the sub-threshold channel.  ``main_length`` is
    # shrunk to 2 purely so the window-overflow guard cannot fire; the crossing
    # and rule are unaffected by that field, and the real end is recomputed
    # below so an overflow stays visible instead of aborting this script.
    physical = E.channel_physical_indices(single_mode)
    selected_single = E.select_channels(frames[index], single_mode)
    locator_single = E.locator_mean_signal(selected_single, config.locator_top_k)
    smoothed_single = smoothed_envelope(locator_single, config)
    probe = E.locate_dynamic_span(
        smoothed_single,
        replace(config, main_length=2),
        point_id=point_id,
        physical_channels=physical,
    )
    single_crossing = int(probe["crossing_index"][0])
    single_start = max(single_crossing - config.lead_back, config.main_start_min)
    single_end = single_start + config.main_length
    return {
        "split": split,
        "point_id": point_id,
        "index": index,
        "label": int(labels[index]),
        "locator": locator,
        "smoothed": smoothed,
        "full_signal": processed[0],
        "full_frames": None if chosen is None else np.asarray(chosen).tolist(),
        "paired": paired,
        "single_mode": single_mode,
        "single_channel": physical[0] + 1,
        "single_rule": str(probe["reference_rule"][0]),
        "single_crossing": single_crossing,
        "single_start": single_start,
        "single_end": single_end,
        "single_overflow": single_end > locator.shape[1],
    }


def decorate(ax, case: dict, mapper: E.DepthMapper, sample_count: int, ymax: float, config):
    top = ax.secondary_xaxis(
        "top",
        functions=(
            lambda i: i / sample_count * mapper.max_depth_mm,
            lambda d: d / mapper.max_depth_mm * sample_count,
        ),
    )
    top.set_xlabel("depth (mm)", fontsize=8)
    top.tick_params(labelsize=7.5)
    ax.axhline(config.weak_threshold, color="#444444", lw=1.0, ls=":", zorder=1)
    ax.set_xlim(0, sample_count - 1)
    ax.set_ylim(-0.07 * ymax, 1.78 * ymax)
    ax.set_xlabel("sample index", fontsize=8)
    ax.set_ylabel("amplitude", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.15, lw=0.5)
    ax.set_title(
        f"{case['point_id']}   ({case['split']} #{case['index']}, "
        f"label {case['label']})",
        fontsize=9.5, pad=24,
    )


def plot_full(ax, case: dict, config: E.DynamicEnvelopeConfig, mapper: E.DepthMapper):
    """``full`` preset: a single region spanning the whole 5 mm trace."""

    signal = case["full_signal"]
    locator = case["locator"]
    sample_count = signal.shape[1]
    ymax = float(max(signal.max(), locator.max(), config.weak_threshold))
    decorate(ax, case, mapper, sample_count, ymax, config)

    x = np.arange(sample_count)
    for channel in range(locator.shape[0]):
        ax.plot(x, locator[channel], color=CH_COLORS[channel], lw=0.9,
                alpha=0.3, ls=(0, (4, 2)), zorder=3)
    for channel in range(signal.shape[0]):
        ax.plot(x, signal[channel], color=CH_COLORS[channel], lw=1.6, zorder=5)

    start_index, end_index = mapper.slice_bounds(FULL_REGION)
    y0, height = 1.38 * ymax, 0.16 * ymax
    bar(ax, y0, height, [(start_index, end_index)], "#7a7a7a", alpha=0.5)
    box(ax, sample_count * 0.5, y0 + 0.5 * height,
        f"one region  [{start_index}, {end_index})   ->   resampled to 512 samples")

    box(ax, 0.015, 0.90, f"real input: RMS-selected top-3 mean "
                         f"(frames {case['full_frames']})", ha="left")
    for channel in range(signal.shape[0]):
        peak = float(smoothed_envelope(locator, config)[channel].max())
        box(ax, 0.015, 0.90 - 0.11 * (channel + 1),
            f"ch{channel + 1}  |signal| peak = {float(np.abs(signal[channel]).max()):.4f}"
            f"     envelope peak = {peak:.4f}"
            f"     {'BELOW' if peak < config.weak_threshold else 'above'} "
            f"weak_threshold={config.weak_threshold}",
            fontsize=6.6, ha="left")


def plot_dynamic(ax, case: dict, config: E.DynamicEnvelopeConfig, mapper: E.DepthMapper):
    """``dyn_envelope``: per-channel main window plus the shared fixed tail."""

    locator = case["locator"]
    smoothed = case["smoothed"]
    span = case["paired"]
    sample_count = locator.shape[1]
    ymax = float(max(locator.max(), smoothed.max()))
    decorate(ax, case, mapper, sample_count, ymax, config)

    x = np.arange(sample_count)
    for channel in range(locator.shape[0]):
        ax.plot(x, locator[channel], color=CH_COLORS[channel], lw=0.9,
                alpha=0.3, ls=(0, (4, 2)), zorder=3)
    for channel in range(smoothed.shape[0]):
        ax.plot(x, smoothed[channel], color=CH_COLORS[channel], lw=2.0,
                alpha=0.55, ls=(0, (1, 1)), zorder=4)

    starts = [int(v) for v in span["starts"]]
    ends = [int(v) for v in span["ends"]]
    rules = [str(rule) for rule in span["reference_rule"]]
    crossings = [int(v) for v in span["crossing_index"]]

    # --- two-channel (mode 3) result -----------------------------------
    main_y, main_h = 0.98 * ymax, 0.14 * ymax
    for channel in range(2):
        bar(ax, main_y, main_h, [(starts[channel], ends[channel])],
            CH_COLORS[channel], alpha=0.8)
        box(ax, (starts[channel] + ends[channel]) / 2, main_y + 0.5 * main_h,
            f"mode 3  main ch{channel + 1}  [{starts[channel]}, {ends[channel]})",
            fontsize=6.8)

    # --- single-channel fallback result --------------------------------
    single_y, single_h = 1.20 * ymax, 0.14 * ymax
    clipped_end = min(case["single_end"], sample_count - 1)
    bar(ax, single_y, single_h, [(case["single_start"], clipped_end)],
        "#c0392b", hatched=True)
    label = (
        f"mode {case['single_mode']} (only ch{case['single_channel']})  main "
        f"[{case['single_start']}, {case['single_end']})"
    )
    if case["single_overflow"]:
        label += "   OVERFLOW -> rejected"
    box(ax, (case["single_start"] + clipped_end) / 2, single_y + 0.5 * single_h,
        label, fontsize=6.8, color="#8e2a1e")

    tail_start, tail_end = int(span["tail_start"]), int(span["tail_end"])
    tail_y, tail_h = 1.42 * ymax, 0.14 * ymax
    bar(ax, tail_y, tail_h, [(tail_start, tail_end)], "#4d9a4d", alpha=0.8)
    box(ax, (tail_start + tail_end) / 2, tail_y + 0.5 * tail_h,
        f"tail  [{tail_start}, {tail_end})   -   identical for both channels",
        fontsize=6.8)

    for channel in range(2):
        crossing = crossings[channel]
        if 0 <= crossing < sample_count:
            ax.plot([crossing], [locator[channel, crossing]], marker="v", ms=8,
                    color=CH_COLORS[channel], mec="black", mew=0.5, zorder=9)
        ax.axvline(starts[channel], color=CH_COLORS[channel], lw=1.0,
                   alpha=0.5, ls="--", zorder=2)

    for channel in range(2):
        peak = float(smoothed[channel].max())
        verdict = "BELOW threshold" if peak < config.weak_threshold else "above threshold"
        box(ax, 0.015, 0.90 - 0.115 * (channel + 1),
            f"ch{channel + 1}  envelope peak = {peak:.4f}  ->  {verdict}      "
            f"mode 3 rule = {rules[channel]} @ t={crossings[channel]}",
            fontsize=6.6, ha="left")
    box(ax, 0.015, 0.90,
        f"single channel rule = {case['single_rule']} @ t={case['single_crossing']}"
        f"      (weak_threshold={config.weak_threshold}, "
        f"lead_back={config.lead_back}, main_length={config.main_length})",
        fontsize=6.6, ha="left", color="#8e2a1e")


def main() -> None:
    config = E.DynamicEnvelopeConfig()
    mapper = E.DepthMapper()
    cases = [load_case(split, point_id, mode, mapper, config)
             for split, point_id, mode in CASES]

    figure, axes = plt.subplots(len(cases), 2, figsize=(15.5, 11.0))
    # ``constrained_layout`` collapses these axes because every panel carries a
    # secondary top axis, so the margins are set explicitly instead.
    figure.subplots_adjust(
        left=0.055, right=0.985, top=0.900, bottom=0.055, hspace=0.62, wspace=0.16
    )
    for row, case in enumerate(cases):
        plot_full(axes[row, 0], case, config, mapper)
        plot_dynamic(axes[row, 1], case, config, mapper)

    axes[0, 0].set_title(
        "full   --   one fixed region [0, 896), covering all of 0-5 mm",
        fontsize=11, pad=28,
    )
    axes[0, 1].set_title(
        "dyn_envelope   --   per-channel main window  +  shared fixed tail",
        fontsize=11, pad=28,
    )
    axes[0, 1].legend(
        handles=[
            Patch(facecolor="#7a7a7a", alpha=0.5, label="full preset region"),
            Patch(facecolor=CH_COLORS[0], label="mode 3 main window, channel 1"),
            Patch(facecolor=CH_COLORS[1], label="mode 3 main window, channel 2"),
            Patch(facecolor="none", edgecolor="#c0392b", hatch="///",
                  label="single-channel main (degenerate fallback)"),
            Patch(facecolor="#4d9a4d", label="tail window (shared)"),
            plt.Line2D([], [], color="black", lw=1.6,
                       label="solid = this method's real input"),
            plt.Line2D([], [], color="black", lw=0.9, alpha=0.3, ls=(0, (4, 2)),
                       label="faint dash = the other method's input"),
            plt.Line2D([], [], color="black", lw=2.0, alpha=0.55, ls=(0, (1, 1)),
                       label="smoothed Hilbert envelope"),
            plt.Line2D([], [], color="black", lw=1.0, ls=":",
                       label="weak_threshold"),
            plt.Line2D([], [], color="black", marker="v", ls="none", ms=8,
                       label="located reference crossing"),
        ],
        loc="upper right", fontsize=6.8, framealpha=0.95,
    )
    figure.suptitle(
        "full vs dyn_envelope on the three sub-threshold points",
        fontsize=12.5,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

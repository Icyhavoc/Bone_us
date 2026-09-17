"""Visualize the four frame-processing outputs.

The script reuses ``emd_pipeline.py`` for loading, channel selection, frame
processing, depth mapping, and resampling.  It creates one PNG sheet for each
form/region/class combination.  Each sheet contains ten randomly selected
samples from the requested class.

Pillow is used instead of Matplotlib so the script can run in the current
workspace without additional plotting dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from emd_pipeline import (
    AFTER_SPLIT_BRANCHES,
    FORM_ORDER,
    DatasetSpec,
    DepthMapper,
    DynamicEnvelopeConfig,
    RegionSpec,
    _gather_branch,
    _resample_branch,
    _select_branch_indices,
    canonical_form,
    detect_dataset_kind,
    json_ready,
    load_after_split_pair,
    load_dataset_split,
    load_split,
    parse_regions,
    prepare_branch_signals,
    prepare_dynamic_envelope_branches,
    process_frames,
    select_channels,
    tukey_window,
)
from run_emd_experiments import (
    AFTER_SPLIT_PAIR_NAME,
    DYNAMIC_REGION_NAME,
    REGION_CHOICES,
    REGION_PRESETS,
)


CLASS_NAMES = {0: "即将穿透", 1: "安全"}
CHANNEL_COLORS = [(44, 95, 154), (192, 57, 43)]
STAT_COLORS = [(44, 95, 154), (128, 128, 128)]
DEFAULT_Y_LIMITS = (-1.0, 1.0)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
        ]
        if bold
        else [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value)


def load_split_records(
    data_dir: Path | str, split: str = "all"
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[str]]:
    """Load one legacy split or concatenate train/val/test preserving split IDs."""

    splits = ["train", "val", "test"] if split == "all" else [split]
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    split_names: list[str] = []
    for split_name in splits:
        x, y, samples = load_split(data_dir, split_name)
        arrays.append(x)
        labels.append(y)
        records.extend(samples)
        split_names.extend([split_name] * len(samples))
    return np.concatenate(arrays), np.concatenate(labels), records, split_names


def load_pair_records(
    data_dir: Path | str, split: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[str], np.ndarray, np.ndarray]:
    """Load the reconstructed full axis and both branch index maps."""

    splits = ["train", "val", "test"] if split == "all" else [split]
    fulls: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    split_names: list[str] = []
    main_indices: list[np.ndarray] = []
    tail_indices: list[np.ndarray] = []
    for split_name in splits:
        pair = load_after_split_pair(data_dir, split_name)
        fulls.append(pair.full)
        labels.append(pair.y)
        records.extend(pair.samples)
        split_names.extend([split_name] * pair.count)
        main_indices.append(pair.main_index)
        tail_indices.append(pair.tail_index)
    return (
        np.concatenate(fulls),
        np.concatenate(labels),
        records,
        split_names,
        np.concatenate(main_indices),
        np.concatenate(tail_indices),
    )


def prepare_pair_branch_items(
    full_sample: np.ndarray,
    main_index: np.ndarray,
    tail_index: np.ndarray,
    form: str,
    channel_mode: int,
    mapper: DepthMapper,
    target_length: int,
    tukey_alpha: float,
    score_region: RegionSpec | None = None,
    apply_tukey: bool = True,
) -> list[tuple[np.ndarray, float, float]]:
    """Process the full axis once, then cut both branches from those streams.

    This mirrors the training path: frame processing (including the shared
    top-3 frame selection) happens on the reconstructed 896-sample axis, so
    both branches always come from the same processed streams.

    ``full_sample`` must be the **raw** ``[frames, 2, samples]`` array; channel
    selection happens here so callers never pre-select.
    """

    if score_region is None:
        score_region = RegionSpec(0.0, mapper.max_depth_mm, "selection_full")
    selected = select_channels(full_sample, channel_mode)
    processed, _ = process_frames(
        selected, form, mapper=mapper, score_region=score_region
    )
    scale = mapper.max_depth_mm / float(mapper.signal_length)
    items: list[tuple[np.ndarray, float, float]] = []
    for absolute_index in (main_index, tail_index):
        branch_index = _select_branch_indices(absolute_index, channel_mode)
        branch_signals = _gather_branch(processed, branch_index)
        resampled = _resample_branch(branch_signals, target_length)
        if apply_tukey:
            resampled = resampled * tukey_window(target_length, tukey_alpha)[None, None, :]
        items.append(
            (
                resampled,
                float(int(np.min(branch_index)) * scale),
                float((int(np.max(branch_index)) + 1) * scale),
            )
        )
    return items


def load_branch_records(
    data_dir: Path | str, split: str, branch: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[str]]:
    """Load one ``after_split_data`` branch across the requested splits."""

    splits = ["train", "val", "test"] if split == "all" else [split]
    spec = DatasetSpec(kind="after_split", root=data_dir, branch=branch)
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    split_names: list[str] = []
    for split_name in splits:
        x, y, samples = load_dataset_split(spec, split_name)
        arrays.append(x)
        labels.append(y)
        records.extend(samples)
        split_names.extend([split_name] * len(samples))
    return np.concatenate(arrays), np.concatenate(labels), records, split_names


def select_class_indices(
    labels: np.ndarray,
    requested_labels: Sequence[int],
    samples_per_class: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Select the same fixed sample set for every form and region."""

    rng = np.random.default_rng(seed)
    selected: dict[int, np.ndarray] = {}
    for label in requested_labels:
        candidates = np.flatnonzero(labels == label)
        if candidates.size < samples_per_class:
            raise ValueError(
                f"Label {label} has {candidates.size} samples, "
                f"but {samples_per_class} are required."
            )
        selected[label] = np.sort(rng.choice(candidates, size=samples_per_class, replace=False))
    return selected


def _resolve_region_experiments(
    region_set: str, regions_json: str | None, dataset_kind: str = "legacy"
) -> list[tuple[str, list[RegionSpec]]]:
    if dataset_kind == "after_split":
        if regions_json is not None:
            raise ValueError("--regions-json does not apply to after_split_data")
        valid = [AFTER_SPLIT_PAIR_NAME, *AFTER_SPLIT_BRANCHES]
        if region_set == "all":
            # The combined two-branch flow is the main path for this dataset.
            return [(AFTER_SPLIT_PAIR_NAME, [])]
        if region_set not in valid:
            raise ValueError(f"after_split_data supports {valid}, got {region_set!r}")
        return [(region_set, [])]
    if regions_json is not None:
        return [("custom", parse_regions(regions_json))]
    names = [*REGION_PRESETS, DYNAMIC_REGION_NAME] if region_set == "all" else [region_set]
    return [
        (name, [] if name == DYNAMIC_REGION_NAME else parse_regions(REGION_PRESETS[name]))
        for name in names
    ]


def prepare_branch_items(
    sample: np.ndarray,
    form: str,
    channel_mode: int,
    region_name: str,
    regions: Sequence[RegionSpec],
    mapper: DepthMapper,
    target_length: int,
    tukey_alpha: float,
    dynamic_envelope_config: DynamicEnvelopeConfig | None,
    branch_depth_range: tuple[float, float] | None = None,
    apply_tukey: bool = True,
) -> list[tuple[np.ndarray, float, float]]:
    """Return ``[(streams_flat, start_mm, end_mm)]`` for one plotted sample.

    For ``after_split_data`` the sample already is one pre-cut branch, so it is
    only resampled and windowed.  Its depth axis is derived from the recorded
    absolute index range so the x-axis still refers to the shared 0-5 mm axis.

    ``sample`` must be the **raw** ``[frames, 2, samples]`` array; channel
    selection happens here so callers never pre-select.
    """

    selected = select_channels(sample, channel_mode)
    processed, _ = process_frames(
        selected,
        form,
        mapper=mapper,
        score_region=RegionSpec(0.0, mapper.max_depth_mm, "selection_full"),
    )
    if region_name in AFTER_SPLIT_BRANCHES:
        resampled = _resample_branch(processed[None, ...], target_length)[0]
        if apply_tukey:
            resampled = resampled * tukey_window(target_length, tukey_alpha)[None, None, :]
        if branch_depth_range is None:
            start_mm, end_mm = 0.0, mapper.max_depth_mm
        else:
            start_mm, end_mm = branch_depth_range
        return [(resampled, float(start_mm), float(end_mm))]
    if dynamic_envelope_config is None:
        items: list[tuple[np.ndarray, float, float]] = []
        for region in regions:
            items.append(
                (
                    prepare_branch_signals(
                        processed,
                        region,
                        mapper,
                        target_length,
                        tukey_alpha=tukey_alpha,
                        apply_tukey=apply_tukey,
                    ),
                    region.start_mm,
                    region.end_mm,
                )
            )
        return items
    dynamic_branches, dynamic_info = prepare_dynamic_envelope_branches(
        processed,
        config=dynamic_envelope_config,
        target_length=target_length,
        tukey_alpha=tukey_alpha,
        max_depth_mm=mapper.max_depth_mm,
        apply_tukey=apply_tukey,
    )
    items = []
    for branch_flat, branch_info in zip(dynamic_branches, dynamic_info["branches"]):
        items.append(
            (
                branch_flat,
                mapper.index_to_depth(int(branch_info["start_index"])),
                mapper.index_to_depth(int(branch_info["end_index_exclusive"])),
            )
        )
    return items


def resolve_sample_mapper(sample: np.ndarray, args: argparse.Namespace) -> DepthMapper:
    """Build a mapper whose signal length matches the plotted sample."""

    return DepthMapper(args.max_depth_mm, int(sample.shape[-1]), args.rounding)


def branch_depth_range_from_record(
    record: dict[str, Any], absolute_signal_length: int, max_depth_mm: float
) -> tuple[float, float] | None:
    """Convert a recorded absolute index range into a physical depth range."""

    index_range = record.get("absolute_index_range")
    if not index_range:
        return None
    start_index, end_index = (float(index_range[0]), float(index_range[1]))
    scale = max_depth_mm / float(absolute_signal_length)
    return (start_index * scale, end_index * scale)



def _curve_data(
    processed_resampled: np.ndarray,
    form: str,
    channel_mode: int,
) -> list[tuple[np.ndarray, tuple[int, int, int], int]]:
    """Return curves as ``(values, color, width)`` for one plotted sample."""

    streams, channels, length = processed_resampled.shape
    curves: list[tuple[np.ndarray, tuple[int, int, int], int]] = []
    form = canonical_form(form)
    if form == "mean_std":
        for stat_index in range(streams):
            for channel in range(channels):
                color = STAT_COLORS[stat_index % len(STAT_COLORS)]
                if channels == 2:
                    color = CHANNEL_COLORS[channel] if stat_index == 0 else tuple(
                        min(255, int(v * 0.65 + 255 * 0.35)) for v in CHANNEL_COLORS[channel]
                    )
                curves.append((processed_resampled[stat_index, channel], color, 2))
        return curves
    if form == "raw50":
        for frame in range(streams):
            for channel in range(channels):
                curves.append((processed_resampled[frame, channel], CHANNEL_COLORS[channel], 1))
        return curves
    for channel in range(channels):
        curves.append((processed_resampled[0, channel], CHANNEL_COLORS[channel], 2))
    return curves


def _draw_plot_cell(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    curves: Sequence[tuple[np.ndarray, tuple[int, int, int], int]],
    depth_axis: np.ndarray,
    y_limits: tuple[float, float],
    title: str,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(170, 170, 170), width=1)
    plot_left, plot_top = left + 48, top + 26
    plot_right, plot_bottom = right - 10, bottom - 30
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(70, 70, 70), width=1)
    y_min, y_max = y_limits
    if y_max <= y_min:
        y_min, y_max = y_min - 1.0, y_max + 1.0

    for fraction in (0.0, 0.5, 1.0):
        y = int(plot_bottom - fraction * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill=(225, 225, 225), width=1)
        tick = y_min + fraction * (y_max - y_min)
        draw.text((left + 2, y - 7), f"{tick:.1f}", fill=(70, 70, 70), font=_font(11))
    for fraction in (0.0, 1.0):
        x = int(plot_left + fraction * (plot_right - plot_left))
        draw.line((x, plot_top, x, plot_bottom), fill=(225, 225, 225), width=1)
        tick = depth_axis[0] + fraction * (depth_axis[-1] - depth_axis[0])
        draw.text((x - 20, plot_bottom + 4), f"{tick:.2f}", fill=(70, 70, 70), font=_font(11))

    for values, color, width in curves:
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        values = np.nan_to_num(values, nan=0.0, posinf=y_max, neginf=y_min)
        values = np.clip(values, y_min, y_max)
        x_positions = np.linspace(plot_left, plot_right, values.size)
        y_positions = plot_bottom - (values - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        points = [(int(x), int(y)) for x, y in zip(x_positions, y_positions)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)
    draw.text((left + 5, top + 4), title, fill=(20, 20, 20), font=_font(13, bold=True))
    draw.text(
        (plot_left + 5, bottom - 20),
        "Depth / mm",
        fill=(70, 70, 70),
        font=_font(11),
    )


def _branch_title(region_name: str, column: int, regions: Sequence[RegionSpec]) -> str:
    if region_name == AFTER_SPLIT_PAIR_NAME:
        return (
            "Branch 1 (main): peak window -> 512"
            if column == 0
            else "Branch 2 (tail): remaining -> 512"
        )
    if region_name == "main":
        return "Branch 1 (main): peak window -> 512"
    if region_name == "tail":
        return "Branch 2 (tail): remaining -> 512"
    if regions:
        return f"Branch {column + 1}: {regions[column].label}"
    if column == 0:
        return "Branch 1: first envelope-peak window (a samples)"
    return "Branch 2: remaining signal"


def branch_count_for(
    region_name: str, regions: Sequence[RegionSpec], dynamic_envelope: bool
) -> int:
    """How many branch panels one sheet shows for a given target."""

    if region_name == AFTER_SPLIT_PAIR_NAME:
        return 2
    if region_name in AFTER_SPLIT_BRANCHES:
        return 1
    return 2 if dynamic_envelope else len(regions)


def _make_sheet(
    form: str,
    region_name: str,
    regions: Sequence[RegionSpec],
    class_label: int,
    selected_indices: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    records: Sequence[dict[str, Any]],
    split_names: Sequence[str],
    channel_mode: int,
    mapper: DepthMapper,
    target_length: int,
    tukey_alpha: float,
    dynamic_envelope_config: DynamicEnvelopeConfig | None,
    y_limits: tuple[float, float],
    output_path: Path,
    main_index: np.ndarray | None = None,
    tail_index: np.ndarray | None = None,
) -> None:
    cell_width = 470
    cell_height = 205
    left_margin = 150
    top_margin = 88
    pair_mode = region_name == AFTER_SPLIT_PAIR_NAME
    branch_count = branch_count_for(
        region_name, regions, dynamic_envelope_config is not None
    )
    width = left_margin + cell_width * branch_count
    height = top_margin + cell_height * len(selected_indices)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    class_name = CLASS_NAMES.get(class_label, f"label={class_label}")
    draw.text(
        (20, 16),
        f"{form} | {region_name} | {class_name} | channel_mode={channel_mode}",
        fill=(15, 15, 15),
        font=_font(20, bold=True),
    )
    draw.text(
        (20, 48),
        "RF Amplitude (normalized; fixed y-range [-1, 1])",
        fill=(70, 70, 70),
        font=_font(13),
    )
    for column in range(branch_count):
        x0 = left_margin + column * cell_width
        draw.text(
            (x0 + 8, 58),
            _branch_title(region_name, column, regions),
            fill=(20, 20, 20),
            font=_font(14, bold=True),
        )

    for row, sample_index in enumerate(selected_indices):
        record = records[sample_index]
        sample_id = record.get("sample_id", f"sample_{sample_index}")
        depth = record.get("depth_value", "?")
        split = split_names[sample_index]
        y0 = top_margin + row * cell_height
        draw.text(
            (8, y0 + 40),
            f"{row + 1:02d}. {sample_id}",
            fill=(20, 20, 20),
            font=_font(13, bold=True),
        )
        draw.text(
            (8, y0 + 65),
            f"depth={depth} mm | {split}",
            fill=(80, 80, 80),
            font=_font(12),
        )
        # Pass the raw sample: the prepare helpers do channel selection
        # themselves, so pre-selecting here would double-apply it (and crash
        # for channel 1/2 on an already single-channel array).
        raw_sample = x[sample_index]
        sample_mapper = DepthMapper(
            mapper.max_depth_mm, int(raw_sample.shape[-1]), mapper.rounding
        )
        if pair_mode:
            if main_index is None or tail_index is None:
                raise ValueError("pair mode requires main_index and tail_index")
            low = min(int(np.min(main_index[sample_index])), int(np.min(tail_index[sample_index])))
            high = max(int(np.max(main_index[sample_index])), int(np.max(tail_index[sample_index])))
            scale = mapper.max_depth_mm / float(mapper.signal_length)
            branch_items = prepare_pair_branch_items(
                raw_sample,
                main_index[sample_index],
                tail_index[sample_index],
                form,
                channel_mode,
                sample_mapper,
                target_length,
                tukey_alpha,
                score_region=RegionSpec(low * scale, (high + 1) * scale, "covered"),
                apply_tukey=False,
            )
        else:
            branch_items = prepare_branch_items(
                raw_sample,
                form,
                channel_mode,
                region_name,
                regions,
                sample_mapper,
                target_length,
                tukey_alpha,
                dynamic_envelope_config,
                branch_depth_range=branch_depth_range_from_record(
                    record, mapper.signal_length, mapper.max_depth_mm
                ),
                apply_tukey=False,
            )
        for column, (branch_flat, start_mm, end_mm) in enumerate(branch_items):
            channels = 1 if channel_mode in (1, 2) else 2
            streams = int(np.asarray(branch_flat).size) // (target_length * channels)
            branch = np.asarray(branch_flat).reshape(streams, channels, target_length)
            depth_axis = np.linspace(start_mm, end_mm, target_length)
            box = (
                left_margin + column * cell_width + 4,
                y0 + 4,
                left_margin + (column + 1) * cell_width - 4,
                y0 + cell_height - 4,
            )
            _draw_plot_cell(
                draw,
                box,
                _curve_data(branch, form, channel_mode),
                depth_axis,
                y_limits,
                f"{sample_id}",
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize four preprocessing forms")
    parser.add_argument("--data-dir", type=Path, default=Path("raw_data"))
    parser.add_argument(
        "--dataset",
        choices=["auto", "legacy", "after_split"],
        default="auto",
        help="auto detects after_split_data branches, otherwise uses legacy raw_data",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/preprocessing"))
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument(
        "--forms",
        default="all",
        help="all or comma-separated forms; raw50 is much slower to render",
    )
    parser.add_argument("--region-set", choices=REGION_CHOICES, default="all")
    parser.add_argument("--regions-json", default=None)
    parser.add_argument("--channel-mode", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--imminent-label", type=int, default=0)
    parser.add_argument("--safe-label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--tukey-alpha", type=float, default=0.3)
    parser.add_argument(
        "--dyn-a", "--dyn-a-mm", "--dyn-branch-length-mm",
        dest="dyn_a_mm", type=float,
        default=DynamicEnvelopeConfig().branch_length_mm,
    )
    parser.add_argument("--dyn-smooth-window", type=int, default=21)
    parser.add_argument("--dyn-prominence-sigma", type=float, default=2.0)
    parser.add_argument("--dyn-min-peak-width", type=int, default=12)
    parser.add_argument("--max-depth-mm", type=float, default=5.0)
    parser.add_argument(
        "--signal-length",
        type=int,
        default=896,
        help="legacy raw axis length; after_split branches use their own length",
    )
    parser.add_argument("--rounding", choices=["round", "floor", "ceil"], default="round")
    parser.add_argument("--y-min", type=float, default=-1.0)
    parser.add_argument("--y-max", type=float, default=1.0)
    return parser.parse_args()


def _resolve_dataset_kind(args: argparse.Namespace) -> str:
    if args.dataset == "auto":
        return detect_dataset_kind(args.data_dir)
    return args.dataset


def main() -> None:
    args = _parse_args()
    dataset_kind = _resolve_dataset_kind(args)
    forms = list(FORM_ORDER) if args.forms.strip().lower() == "all" else [
        canonical_form(item) for item in args.forms.split(",") if item.strip()
    ]
    regions_to_run = _resolve_region_experiments(args.region_set, args.regions_json, dataset_kind)
    output_dir = args.output_dir.resolve()
    dynamic_envelope_config = DynamicEnvelopeConfig(
        branch_length_mm=args.dyn_a_mm,
        smooth_window=args.dyn_smooth_window,
        prominence_sigma=args.dyn_prominence_sigma,
        min_peak_width=args.dyn_min_peak_width,
    )
    selection_manifest: dict[str, Any] = {
        "data_dir": args.data_dir.resolve(),
        "dataset": dataset_kind,
        "split": args.split,
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "class_labels": {
            "imminent": args.imminent_label,
            "safe": args.safe_label,
        },
        "selections": {},
    }

    for region_name, regions in regions_to_run:
        main_index = tail_index = None
        if dataset_kind == "after_split" and region_name == AFTER_SPLIT_PAIR_NAME:
            x, labels, records, split_names, main_index, tail_index = load_pair_records(
                args.data_dir, args.split
            )
            mapper = DepthMapper(args.max_depth_mm, int(x.shape[-1]), args.rounding)
            effective_regions: list[RegionSpec] = []
        elif dataset_kind == "after_split":
            x, labels, records, split_names = load_branch_records(
                args.data_dir, args.split, region_name
            )
            mapper = DepthMapper(args.max_depth_mm, int(x.shape[-1]), args.rounding)
            effective_regions = [RegionSpec(0.0, args.max_depth_mm, region_name)]
        else:
            x, labels, records, split_names = load_split_records(args.data_dir, args.split)
            mapper = DepthMapper(args.max_depth_mm, args.signal_length, args.rounding)
            effective_regions = regions
        # Keep a fixed sample set per class so every form/target is comparable,
        # and reuse the same rows across targets of the same dataset.
        selected = select_class_indices(
            labels,
            [args.imminent_label, args.safe_label],
            args.samples_per_class,
            args.seed,
        )
        if not selection_manifest["selections"]:
            for label, indices in selected.items():
                name = "imminent" if label == args.imminent_label else "safe"
                selection_manifest["selections"][name] = [
                    {
                        "index": int(index),
                        "sample_id": records[index].get("sample_id"),
                        "depth_value": records[index].get("depth_value"),
                        "split": split_names[index],
                    }
                    for index in indices
                ]
        for form in forms:
            for label, indices in selected.items():
                class_name = "imminent" if label == args.imminent_label else "safe"
                file_name = (
                    f"form_{form}__regions_{_safe_name(region_name)}__"
                    f"channels_{args.channel_mode}__class_{class_name}.png"
                )
                _make_sheet(
                    form,
                    region_name,
                    effective_regions,
                    label,
                    indices,
                    x,
                    labels,
                    records,
                    split_names,
                    args.channel_mode,
                    mapper,
                    args.target_length,
                    args.tukey_alpha,
                    dynamic_envelope_config
                    if dataset_kind == "legacy" and region_name == DYNAMIC_REGION_NAME
                    else None,
                    (args.y_min, args.y_max),
                    output_dir / file_name,
                    main_index=main_index,
                    tail_index=tail_index,
                )
    selection_manifest["forms"] = forms
    selection_manifest["regions"] = {
        name: [{"start_mm": r.start_mm, "end_mm": r.end_mm, "name": r.name} for r in regions]
        for name, regions in regions_to_run
    }
    selection_manifest["target_length"] = args.target_length
    selection_manifest["tukey_alpha"] = args.tukey_alpha
    selection_manifest["dynamic_envelope"] = dynamic_envelope_config
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(json_ready(selection_manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Preprocessing visualizations written to {output_dir}")


if __name__ == "__main__":
    main()

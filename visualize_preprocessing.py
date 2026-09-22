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
    FORM_ORDER,
    LOCATOR_BRANCH_NAME,
    TAIL_BRANCH_NAME,
    DepthMapper,
    DynamicEnvelopeConfig,
    RegionSpec,
    canonical_form,
    channel_physical_indices,
    class_display_names,
    json_ready,
    load_split,
    parse_regions,
    prepare_branch_signals,
    prepare_dynamic_envelope_branches,
    process_frames,
    read_label_scheme,
    select_channels,
)
from dyn_cli import add_dyn_arguments, dyn_config_from_args
from run_emd_experiments import DYNAMIC_REGION_NAME, REGION_CHOICES, REGION_PRESETS


# Historical binary names, used whenever the dataset ships no label_scheme.json.
# ``raw_data`` is exactly that case, so every stored binary sheet keeps its title.
CLASS_NAMES = {0: "即将穿透", 1: "安全"}
CHANNEL_COLORS = [(44, 95, 154), (192, 57, 43)]
STAT_COLORS = [(44, 95, 154), (128, 128, 128)]
DEFAULT_Y_LIMITS = (-1.0, 1.0)


def class_names_for_data_dir(data_dir: Path | str, num_classes: int) -> list[str]:
    """One display name per label, preferring the dataset's ``label_scheme.json``.

    A relabelled dataset records names such as ``"depth < 0.8"``; without it the
    historical Chinese binary names are used, which is what ``raw_data`` needs.
    """

    return class_display_names(read_label_scheme(data_dir), num_classes)


def _parse_optional_labels(value: str | None) -> list[int] | None:
    """Parse ``--class-labels``; ``None`` means "use the imminent/safe pair"."""

    if value is None or not value.strip():
        return None
    labels: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            labels.append(int(item))
        except ValueError as error:
            raise SystemExit(f"--class-labels expects integers, got {item!r}") from error
    return labels or None


def resolve_class_labels(
    labels: np.ndarray,
    requested: Sequence[int] | None,
    imminent: int,
    safe: int,
) -> list[int]:
    """The class labels to draw, in order.

    ``--class-labels`` wins; otherwise the historical two-label pair is used, so
    a binary run needs no new arguments.  The order also fixes the file-name
    tokens (``imminent``/``safe`` for the historical pair, else ``label{k}``).
    """

    ordered = [int(label) for label in requested] if requested else [int(imminent), int(safe)]
    if len(set(ordered)) != len(ordered):
        raise SystemExit(f"duplicate class labels requested: {ordered}")
    highest = int(labels.max()) + 1 if labels.size else 0
    for label in ordered:
        if not 0 <= label < max(highest, 1):
            raise SystemExit(f"class label {label} is out of range for labels 0..{highest - 1}")
        if not np.any(labels == label):
            raise SystemExit(f"class label {label} has no samples in this split")
    return ordered


def class_file_token(label: int, imminent: int, safe: int, num_classes: int = 2) -> str:
    """File-name token for one class.

    ``imminent``/``safe`` are kept verbatim for a binary dataset so the
    historical ``...__class_imminent.png`` / ``...__class_safe.png`` files keep
    their names.  A k-class dataset has no such heritage -- calling its middle
    group "safe" would be actively misleading -- so every label becomes
    ``label{k}`` instead.
    """

    if int(num_classes) > 2:
        return f"label{label}"
    if label == int(imminent):
        return "imminent"
    if label == int(safe):
        return "safe"
    return f"label{label}"


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


def _short_branch_name(name: str) -> str:
    """``dynamic_main_window`` -> ``main``, so a title stays inside its cell."""

    label = str(name)
    if label.startswith("dynamic_"):
        label = label[len("dynamic_"):]
    if label.endswith("_window"):
        label = label[: -len("_window")]
    return label.replace("_", " ") or str(name)


def _fit_title(
    draw: "ImageDraw.ImageDraw",
    segments: Sequence[str],
    font: Any,
    max_width: float,
) -> str:
    """Join ``segments`` with " | ", dropping optional ones until the text fits.

    The first segment is always kept: it identifies the branch, and a clipped
    identifier is worse than missing detail.  Cell widths are fixed while branch
    geometry is configurable, so no single format fits every configuration.
    """

    for count in range(len(segments), 0, -1):
        candidate = " | ".join(segments[:count])
        if draw.textlength(candidate, font=font) <= max_width:
            return candidate
    return segments[0]


def dynamic_branch_title(
    column: int,
    name: str,
    span_samples: int,
    mm_per_index: float,
    target_length: int,
) -> str:
    """Describe a dynamic branch by its real extent and physical resolution.

    Both branches are resampled to ``target_length`` samples, so one point on
    the main branch covers far less depth than one point on the tail branch.
    Printing the span and the per-point spacing is what makes that visible: the
    hardcoded "(170 samples)" / "[251, 896)" strings that used to sit here
    described the default configuration, not the one that was actually run.
    """

    span_mm = float(span_samples) * float(mm_per_index)
    spacing_mm = span_mm / target_length if target_length else 0.0
    return " | ".join(
        [
            f"Branch {column + 1}: {_short_branch_name(name)}",
            f"{int(span_samples)} pt = {span_mm:.2f} mm",
            f"{spacing_mm:.5f} mm/pt",
        ]
    )


def load_split_records(
    data_dir: Path | str, split: str = "all"
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[str]]:
    """Load one split or concatenate train/val/test while preserving split IDs."""

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
    region_set: str, regions_json: str | None
) -> list[tuple[str, list[RegionSpec]]]:
    if regions_json is not None:
        return [("custom", parse_regions(regions_json))]
    names = [*REGION_PRESETS, DYNAMIC_REGION_NAME] if region_set == "all" else [region_set]
    return [
        (name, [] if name == DYNAMIC_REGION_NAME else parse_regions(REGION_PRESETS[name]))
        for name in names
    ]


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


def _make_sheet(
    form: str,
    region_name: str,
    regions: Sequence[RegionSpec],
    class_label: int,
    class_name: str,
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
) -> None:
    cell_width = 470
    cell_height = 205
    left_margin = 150
    top_margin = 88
    branch_count = 2 if dynamic_envelope_config is not None else len(regions)
    width = left_margin + cell_width * branch_count
    height = top_margin + cell_height * len(selected_indices)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    branch_font = _font(14, bold=True)
    mm_per_index = mapper.max_depth_mm / mapper.signal_length
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
        if dynamic_envelope_config is None:
            branch_title = f"Branch {column + 1}: {regions[column].label}"
        elif column == 0:
            # The main window has a fixed *length* config.main_length; only the
            # offset the detector finds moves, so the config describes the span.
            span = int(dynamic_envelope_config.main_length)
            branch_title = _fit_title(
                draw,
                [
                    f"Branch 1: {_short_branch_name(LOCATOR_BRANCH_NAME)}",
                    f"{span} pt = {span * mm_per_index:.2f} mm",
                    f"{mm_per_index / target_length:.5f} mm/pt",
                ],
                branch_font,
                cell_width - 16,
            )
        else:
            tail_start = int(dynamic_envelope_config.tail_start)
            tail_end = int(dynamic_envelope_config.signal_length)
            branch_title = _fit_title(
                draw,
                [
                    f"Branch 2: {_short_branch_name(TAIL_BRANCH_NAME)}",
                    f"[{tail_start}, {tail_end}) = {tail_end - tail_start} pt",
                    f"{(tail_end - tail_start) * mm_per_index:.2f} mm",
                    f"{mm_per_index / target_length:.5f} mm/pt",
                ],
                branch_font,
                cell_width - 16,
            )
        draw.text(
            (x0 + 8, 58),
            branch_title,
            fill=(20, 20, 20),
            font=branch_font,
        )

    for row, sample_index in enumerate(selected_indices):
        sample_id = records[sample_index].get("sample_id", f"sample_{sample_index}")
        depth = records[sample_index].get("depth_value", "?")
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
        selected = select_channels(x[sample_index], channel_mode)
        physical_channels = channel_physical_indices(channel_mode)
        processed, _ = process_frames(
            selected,
            form,
            mapper=mapper,
            score_region=RegionSpec(0.0, mapper.max_depth_mm, "selection_full"),
        )
        if dynamic_envelope_config is None:
            branch_items = [
                (
                    prepare_branch_signals(
                        processed,
                        region,
                        mapper,
                        target_length,
                        tukey_alpha=tukey_alpha,
                        apply_tukey=False,
                    ),
                    region.start_mm,
                    region.end_mm,
                )
                for region in regions
            ]
        else:
            dynamic_branches, dynamic_info = prepare_dynamic_envelope_branches(
                processed,
                locator_frames=selected,
                config=dynamic_envelope_config,
                target_length=target_length,
                tukey_alpha=tukey_alpha,
                apply_tukey=False,
                point_id=str(sample_id),
                physical_channels=physical_channels,
            )
            branch_items = []
            for branch_flat, branch_info in zip(
                dynamic_branches, dynamic_info["branches"]
            ):
                branch_items.append(
                    (
                        branch_flat,
                        mapper.index_to_depth(int(branch_info["start_index"])),
                        mapper.index_to_depth(int(branch_info["end_index_exclusive"])),
                    )
                )
        for column, (branch_flat, start_mm, end_mm) in enumerate(branch_items):
            branch = branch_flat.reshape(processed.shape[0], processed.shape[1], target_length)
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
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/preprocessing"))
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--forms", default="all", help="all or comma-separated forms")
    parser.add_argument("--region-set", choices=REGION_CHOICES, default="all")
    parser.add_argument("--regions-json", default=None)
    parser.add_argument("--channel-mode", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--imminent-label", type=int, default=0)
    parser.add_argument("--safe-label", type=int, default=1)
    parser.add_argument(
        "--class-labels",
        default=None,
        help=(
            "comma-separated class labels to draw, e.g. \"0,1,2\" for a 3-class "
            "dataset; defaults to --imminent-label/--safe-label"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--tukey-alpha", type=float, default=0.3)
    add_dyn_arguments(parser)
    parser.add_argument("--max-depth-mm", type=float, default=5.0)
    parser.add_argument("--signal-length", type=int, default=896)
    parser.add_argument("--rounding", choices=["round", "floor", "ceil"], default="round")
    parser.add_argument("--y-min", type=float, default=-1.0)
    parser.add_argument("--y-max", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    forms = list(FORM_ORDER) if args.forms.strip().lower() == "all" else [
        canonical_form(item) for item in args.forms.split(",") if item.strip()
    ]
    regions_to_run = _resolve_region_experiments(args.region_set, args.regions_json)
    x, labels, records, split_names = load_split_records(args.data_dir, args.split)
    num_classes = int(labels.max()) + 1 if labels.size else 2
    class_names = class_names_for_data_dir(args.data_dir, num_classes)
    class_labels = resolve_class_labels(
        labels,
        _parse_optional_labels(args.class_labels),
        args.imminent_label,
        args.safe_label,
    )
    file_tokens = {
        label: class_file_token(label, args.imminent_label, args.safe_label, num_classes)
        for label in class_labels
    }
    selected = select_class_indices(
        labels,
        class_labels,
        args.samples_per_class,
        args.seed,
    )
    mapper = DepthMapper(args.max_depth_mm, args.signal_length, args.rounding)
    dynamic_envelope_config = dyn_config_from_args(args)
    output_dir = args.output_dir.resolve()
    selection_manifest: dict[str, Any] = {
        "data_dir": args.data_dir.resolve(),
        "split": args.split,
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "num_classes": num_classes,
        "label_scheme": read_label_scheme(args.data_dir),
        "class_labels": {
            "imminent": args.imminent_label,
            "safe": args.safe_label,
        },
        "classes": [
            {
                "label": label,
                "name": class_names[label],
                "file_token": file_tokens[label],
            }
            for label in class_labels
        ],
        "selections": {},
    }
    for label, indices in selected.items():
        name = file_tokens[label]
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
        for region_name, regions in regions_to_run:
            for label, indices in selected.items():
                class_name = file_tokens[label]
                file_name = (
                    f"form_{form}__regions_{_safe_name(region_name)}__"
                    f"channels_{args.channel_mode}__class_{class_name}.png"
                )
                _make_sheet(
                    form,
                    region_name,
                    regions,
                    label,
                    class_names[label],
                    indices,
                    x,
                    labels,
                    records,
                    split_names,
                    args.channel_mode,
                    mapper,
                    args.target_length,
                    args.tukey_alpha,
                    dynamic_envelope_config if region_name == DYNAMIC_REGION_NAME else None,
                    (args.y_min, args.y_max),
                    output_dir / file_name,
                )
    selection_manifest["forms"] = forms
    selection_manifest["regions"] = {
        name: regions for name, regions in regions_to_run
    }
    selection_manifest["mapper"] = mapper
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

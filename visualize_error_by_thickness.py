"""Render per-thickness misclassification histograms from saved experiment results.

Companion to ``visualize_confusion_matrix.py``.  That script answers *how many*
errors a run makes; this one answers *where* those errors sit on the bone
thickness axis.

The horizontal axis is the sample's intrinsic thickness (the ``depth_value``
field of ``samples_*.json``, sourced from the ``总厚度`` column of the raw
``aa_summary.xlsx`` table).  This is a fixed measured property of the scanned
point and is **not** the 896-point 0-5 mm acquisition axis used to slice the
EMD branches.  It is also the quantity the binary label is thresholded on
(``label = thickness >= 1.0 mm``).

Each experiment yields one image holding all three splits at once:

* wide light bar  -> how many samples of that thickness bin exist,
* narrow dark bar -> how many of them the model misclassified, stacked by
  split with ``test`` at the bottom, then ``val``, then ``train`` on top.

Everything is read back from the arrays the training run already saved
(``probabilities_{split}.npy``, ``labels_{split}.npy``, ``samples_{split}.json``),
so no inference is re-run.
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
    canonical_form,
    label_scheme_tag,
    label_thresholds,
    read_label_scheme,
)


SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
# Stacking order inside one bin, drawn from the baseline upwards.
STACK_ORDER = ("test", "val", "train")
SPLIT_COLORS = {"test": "#2f6f9f", "val": "#3d8b62", "train": "#c45a32"}
TOTAL_FILL = "#dfe5ea"
TOTAL_OUTLINE = "#b3bfc8"
# The historical binary split (``raw_data`` predates ``label_scheme.json``), used
# when a run records no thresholds of its own.
DEFAULT_LABEL_THRESHOLDS: tuple[float, ...] = (1.0,)

BACKGROUND = "#f7f9fb"
INK = "#1f2d36"
MUTED = "#52616b"
GRID = "#d5dce1"
REFERENCE = "#7a4b50"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _parse_forms(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    forms = {canonical_form(item.strip()) for item in value.split(",") if item.strip()}
    return forms or set(FORM_ORDER)


def _load_split(
    experiment_dir: Path, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return ``(thickness_mm, labels, probabilities)`` for one split, or None.

    Runs that only wrote ``config.json``/``metrics.json`` (and no arrays) are
    reported as missing instead of aborting the whole batch.
    """

    probability_path = experiment_dir / f"probabilities_{split}.npy"
    label_path = experiment_dir / f"labels_{split}.npy"
    sample_path = experiment_dir / f"samples_{split}.json"
    if not (probability_path.exists() and label_path.exists() and sample_path.exists()):
        return None
    try:
        probabilities = np.load(probability_path)
        labels = np.load(label_path)
    except (OSError, ValueError):
        return None
    records = _read_json(sample_path, [])
    if not isinstance(records, list) or not records:
        return None
    try:
        thickness = np.asarray([float(record["depth_value"]) for record in records], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if thickness.shape[0] != labels.shape[0] or probabilities.shape[0] != labels.shape[0]:
        return None
    if not np.all(np.isfinite(thickness)):
        return None
    return thickness, labels, probabilities


def _error_flags(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Misclassification flags, matching the model that produced the scores.

    Two columns reproduce the decision rule of ``emd_pipeline.binary_metrics``
    exactly (score = ``probabilities[:, 1]`` compared against ``threshold``).
    More columns mean a softmax head, whose prediction is the argmax, so the
    stored ``--threshold`` does not apply there.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 2 and probabilities.shape[1] > 2:
        predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    elif probabilities.ndim == 2 and probabilities.shape[1] > 1:
        predictions = (probabilities[:, 1] >= threshold).astype(np.int64)
    else:
        predictions = (probabilities.reshape(-1) >= threshold).astype(np.int64)
    return (predictions != labels).reshape(-1)


def _bin_edges(values: np.ndarray, bin_width: float) -> np.ndarray:
    """Cover ``values`` with edges snapped outwards onto the bin-width grid."""

    low = float(np.min(values))
    high = float(np.max(values))
    start = float(np.floor((low + 1e-9) / bin_width) * bin_width)
    stop = float(np.ceil((high - 1e-9) / bin_width) * bin_width)
    if stop <= start:
        stop = start + bin_width
    count = int(round((stop - start) / bin_width))
    return np.round(start + np.arange(count + 1) * bin_width, 6)


def _bin_index(values: np.ndarray, start: float, bin_width: float, bin_count: int) -> np.ndarray:
    """Left-closed, right-open binning; the last edge belongs to the last bin."""

    index = np.floor((values - start) / bin_width + 1e-9).astype(np.int64)
    return np.clip(index, 0, bin_count - 1)


def _nice_step(maximum: int, divisions: int = 6) -> int:
    if maximum <= 0:
        return 1
    raw = maximum / float(divisions)
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if step >= raw:
            return step
    return int(np.ceil(raw))


def _paste_rotated_text(
    image: Image.Image, center: tuple[float, float], text: str, font: ImageFont.ImageFont, fill: str
) -> None:
    """Pillow's ``angle`` argument is version dependent, so rotate manually."""

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    box = measure.textbbox((0, 0), text, font=font)
    label = Image.new("RGBA", (box[2] - box[0] + 6, box[3] - box[1] + 10), (0, 0, 0, 0))
    ImageDraw.Draw(label).text((3 - box[0], 5 - box[1]), text, font=font, fill=fill)
    label = label.rotate(90, expand=True)
    image.paste(
        label,
        (int(center[0] - label.width / 2), int(center[1] - label.height / 2)),
        label,
    )


def render_error_by_thickness(
    per_split: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    output_path: Path,
    bin_width: float = 0.05,
    threshold: float = 0.5,
    label_thresholds: Sequence[float] = (),
) -> dict[str, Any]:
    """Draw the stacked misclassification histogram and return its statistics.

    ``label_thresholds`` are the thicknesses that define the label boundaries.
    One dashed reference line is drawn per entry, so a 3-class dataset with
    0.8 mm / 1.2 mm boundaries shows both instead of a single 1.0 mm line that
    does not exist in its labelling.
    """

    ordered = {split: per_split[split] for split in SPLIT_LABELS if split in per_split}
    all_thickness = np.concatenate([thickness for thickness, _flags in ordered.values()])
    edges = _bin_edges(all_thickness, bin_width)
    start = float(edges[0])
    bin_count = int(len(edges) - 1)

    totals = np.zeros(bin_count, dtype=np.int64)
    errors: dict[str, np.ndarray] = {}
    for split, (thickness, flags) in ordered.items():
        index = _bin_index(thickness, start, bin_width, bin_count)
        totals += np.bincount(index, minlength=bin_count)
        errors[split] = np.bincount(index[flags], minlength=bin_count)

    error_stack = np.zeros(bin_count, dtype=np.int64)
    for split in ordered:
        error_stack += errors[split]
    peak = int(max(totals.max(initial=0), error_stack.max(initial=0)))
    y_step = _nice_step(peak)
    y_top = max(y_step, int(np.ceil(peak / y_step) * y_step))

    width, height = 1280, 720
    left, top, right, bottom = 96, 168, 44, 106
    plot_x0, plot_y0 = left, top
    plot_x1, plot_y1 = width - right, height - bottom
    plot_width = plot_x1 - plot_x0
    plot_height = plot_y1 - plot_y0

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 38), title, fill=INK, font=_font(24, bold=True), anchor="mm")
    draw.text(
        (width // 2, 74),
        f"错分样本的骨头厚度分布 | 柱宽 {bin_width:g} mm | 判定阈值 {threshold:g} "
        f"| 浅色柱 = 该厚度区间总样本数",
        fill=MUTED,
        font=_font(15),
        anchor="mm",
    )

    legend_font = _font(13)
    legend = [("该厚度区间总样本数", TOTAL_FILL, TOTAL_OUTLINE)]
    legend += [(f"{SPLIT_LABELS[split]} 错分", SPLIT_COLORS[split], None) for split in STACK_ORDER]
    swatch, inner_gap, between_gap = 20, 7, 24
    legend_width = sum(swatch + inner_gap + draw.textlength(text, font=legend_font) for text, _f, _o in legend)
    legend_width += between_gap * (len(legend) - 1)
    cursor = (width - legend_width) / 2
    for text, fill, outline in legend:
        draw.rectangle((cursor, 104, cursor + swatch, 118), fill=fill, outline=outline or fill, width=1)
        draw.text((cursor + swatch + inner_gap, 111), text, fill=INK, font=legend_font, anchor="lm")
        cursor += swatch + inner_gap + draw.textlength(text, font=legend_font) + between_gap

    for index in range(y_top // y_step + 1):
        value = index * y_step
        y = plot_y1 - value / y_top * plot_height
        draw.line((plot_x0, y, plot_x1, y), fill=GRID, width=1)
        draw.text((plot_x0 - 10, y), str(value), fill=MUTED, font=_font(12), anchor="rm")
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=MUTED, width=1)
    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=MUTED, width=1)

    bin_pixel = plot_width / bin_count
    total_width = max(2.0, bin_pixel - 1.0)
    error_width = max(2.0, bin_pixel * 0.62)
    for index in range(bin_count):
        center = plot_x0 + (index + 0.5) * bin_pixel
        total_value = int(totals[index])
        if total_value:
            bar_height = total_value / y_top * plot_height
            x0, x1 = center - total_width / 2, center + total_width / 2
            y = plot_y1 - bar_height
            draw.rectangle(
                (x0, y, x1, plot_y1),
                fill=TOTAL_FILL,
                outline=TOTAL_OUTLINE if total_width >= 5 else None,
                width=1,
            )
        cursor_y = plot_y1
        for split in STACK_ORDER:
            if split not in errors:
                continue
            value = int(errors[split][index])
            if not value:
                continue
            bar_height = value / y_top * plot_height
            x0, x1 = center - error_width / 2, center + error_width / 2
            y = cursor_y - bar_height
            draw.rectangle((x0, y, x1, cursor_y), fill=SPLIT_COLORS[split])
            cursor_y = y

    for reference in label_thresholds:
        reference = float(reference)
        if not start <= reference <= float(edges[-1]):
            continue
        x = plot_x0 + (reference - start) / (bin_count * bin_width) * plot_width
        y = plot_y0
        while y < plot_y1:
            draw.line((x, y, x, min(y + 6.0, plot_y1)), fill=REFERENCE, width=1)
            y += 12.0
        # The single-threshold caption is the historical one, verbatim, so the
        # stored binary images stay byte-identical.
        if len(label_thresholds) == 1:
            caption = f"label 分界 {reference:g} mm（左=即将穿透，右=安全）"
        else:
            caption = f"label 分界 {reference:g} mm"
        draw.text(
            (x, plot_y0 - 10),
            caption,
            fill=REFERENCE,
            font=_font(12),
            anchor="ms",
        )

    tick_font = _font(12)
    tick_value = float(np.ceil(start / 0.5 - 1e-9) * 0.5)
    tick_values = [start]
    while tick_value <= float(edges[-1]) + 1e-9:
        if tick_value - start > 1e-6:
            tick_values.append(round(tick_value, 6))
        tick_value += 0.5
    if float(edges[-1]) - tick_values[-1] > 1e-6:
        tick_values.append(float(edges[-1]))
    for value in tick_values:
        x = plot_x0 + (value - start) / (bin_count * bin_width) * plot_width
        draw.line((x, plot_y1, x, plot_y1 + 5), fill=MUTED, width=1)
        draw.text((x, plot_y1 + 10), f"{value:.2f}", fill=MUTED, font=tick_font, anchor="ma")
    draw.text(
        ((plot_x0 + plot_x1) / 2, plot_y1 + 44),
        "骨头厚度 (mm)",
        fill=INK,
        font=_font(14),
        anchor="mm",
    )
    _paste_rotated_text(image, (30, (plot_y0 + plot_y1) / 2), "样本个数", _font(14), INK)

    total_samples = int(totals.sum())
    total_errors = int(error_stack.sum())
    rate = 0.0 if total_samples == 0 else total_errors / total_samples * 100.0
    breakdown = "    ".join(
        f"{SPLIT_LABELS[split]} 错 {int(errors[split].sum())}/{len(ordered[split][0])}"
        for split in STACK_ORDER
        if split in ordered
    )
    draw.text(
        (width // 2, height - 52),
        f"合计 {total_samples} 个样本，错分 {total_errors} 个（{rate:.1f}%）    {breakdown}",
        fill=MUTED,
        font=_font(14),
        anchor="mm",
    )
    draw.text(
        (width // 2, height - 26),
        "纵轴同时是“该区间样本数”和“其中错分个数”，因此深色堆叠柱恒不高于浅色柱。",
        fill=MUTED,
        font=_font(12),
        anchor="mm",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "bin_width": bin_width,
        "threshold": threshold,
        "bin_start": start,
        "bin_count": bin_count,
        "total_samples": total_samples,
        "total_errors": total_errors,
        "error_rate": rate,
        "errors_by_split": {split: int(errors[split].sum()) for split in errors},
        "samples_by_split": {split: int(len(thickness)) for split, (thickness, _f) in ordered.items()},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize misclassified sample counts against bone thickness"
    )
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("visualizations/error_by_thickness")
    )
    parser.add_argument("--forms", default="all", help="all or comma-separated forms")
    parser.add_argument("--region-set", default="all")
    parser.add_argument("--channel-mode", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument(
        "--splits",
        default="test,val,train",
        help=(
            "comma-separated split subsets to include; the stacking order inside "
            "every bin is always test -> val -> train from the bottom up"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="positive-class decision threshold")
    parser.add_argument("--bin-width", type=float, default=0.05, help="thickness bin width in mm")
    parser.add_argument(
        "--label-thresholds",
        default=None,
        help=(
            "comma-separated thickness thresholds drawn as reference lines; "
            "defaults to the experiment's label_scheme (or 1.0 mm for the "
            "historical binary runs that predate label_scheme.json)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("raw_data"),
        help=(
            "dataset whose label scheme selects which experiments to draw; "
            "raw_data (no label_scheme.json) keeps the historical binary runs"
        ),
    )
    return parser.parse_args()


def _parse_threshold_override(value: str | None) -> list[float] | None:
    """Parse ``--label-thresholds``; ``None`` means "use the recorded scheme"."""

    if value is None or not value.strip():
        return None
    try:
        return [float(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit(f"--label-thresholds expects numbers, got {value!r}") from error


def _scheme_of(config: dict[str, Any]) -> dict[str, Any] | None:
    """A run's label scheme: recorded inline, else from its ``data_dir``.

    ``raw_data`` has no ``label_scheme.json``, so this returns None there, which
    is exactly the signal that the run used the historical 1.0 mm binary split.
    """

    scheme = config.get("label_scheme")
    if isinstance(scheme, dict):
        return scheme
    data_dir = config.get("data_dir")
    if not data_dir:
        return None
    found = read_label_scheme(data_dir)
    return found if isinstance(found, dict) else None


def main() -> None:
    args = _parse_args()
    if args.bin_width <= 0:
        raise SystemExit("--bin-width must be positive")
    form_filter = _parse_forms(args.forms)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    unknown = [split for split in splits if split not in SPLIT_LABELS]
    if unknown:
        raise SystemExit(f"unknown splits: {', '.join(unknown)}")
    experiments_dir = args.experiments_dir.resolve()
    output_dir = args.output_dir.resolve()
    dataset_tag = label_scheme_tag(read_label_scheme(args.data_dir))
    manifest: dict[str, Any] = {
        "bin_width": args.bin_width,
        "threshold": args.threshold,
        "label_thresholds_override": _parse_threshold_override(args.label_thresholds),
        "splits": splits,
        "output_dir": output_dir,
        "data_dir": str(Path(args.data_dir).resolve()),
        "label_scheme_tag": dataset_tag,
        "files": [],
        "skipped": [],
    }
    override = manifest["label_thresholds_override"]
    generated = 0
    directories = (
        sorted(path for path in experiments_dir.iterdir() if path.is_dir())
        if experiments_dir.exists()
        else []
    )
    for directory in directories:
        config = _read_json(directory / "config.json", {})
        if not isinstance(config, dict) or not config:
            continue
        # A run belongs to exactly one dataset; drawing a 3-class run into the
        # binary chart folder would mix two different label definitions.
        if label_scheme_tag(_scheme_of(config)) != dataset_tag:
            continue
        form = str(config.get("form", ""))
        region = str(config.get("region_name", ""))
        channel = str(config.get("channel_mode", ""))
        if form_filter is not None and form not in form_filter:
            continue
        if args.region_set != "all" and region != args.region_set:
            continue
        if args.channel_mode is not None and channel != str(args.channel_mode):
            continue

        per_split: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        widths: set[int] = set()
        for split in splits:
            loaded = _load_split(directory, split)
            if loaded is None:
                continue
            thickness, labels, probabilities = loaded
            widths.add(int(probabilities.shape[1]) if probabilities.ndim == 2 else 1)
            per_split[split] = (thickness, _error_flags(probabilities, labels, args.threshold))
        if not per_split:
            manifest["skipped"].append({"experiment": directory.name, "reason": "no saved arrays"})
            print(f"skipped {directory.name}: no saved probabilities/labels/samples")
            continue

        scheme = _scheme_of(config)
        if override is not None:
            reference_thresholds = list(override)
        else:
            reference_thresholds = label_thresholds(scheme)
        num_classes = max(widths)
        if not reference_thresholds and num_classes == 2:
            reference_thresholds = list(DEFAULT_LABEL_THRESHOLDS)

        label_tag = label_scheme_tag(scheme)
        file_name = (
            f"form_{_safe_name(form)}__regions_{_safe_name(region)}__"
            f"channels_{_safe_name(channel)}"
        )
        if label_tag:
            file_name = f"{file_name}__{_safe_name(label_tag)}"
        file_name = f"{file_name}.png"
        stats = render_error_by_thickness(
            per_split,
            f"{form} | {region} | channel_mode={channel}",
            output_dir / file_name,
            bin_width=args.bin_width,
            threshold=args.threshold,
            label_thresholds=reference_thresholds,
        )
        manifest["files"].append(
            {
                "file": file_name,
                "experiment": directory.name,
                "num_classes": num_classes,
                "label_scheme_tag": label_tag,
                "label_thresholds": reference_thresholds,
                **stats,
            }
        )
        generated += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"Error-by-thickness visualizations written to {output_dir} ({generated} files)")


if __name__ == "__main__":
    main()

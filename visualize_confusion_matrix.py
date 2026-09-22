"""Render confusion matrices from saved experiment metrics.

The drawings are Pillow-only, like the other ``visualize_*.py`` scripts.  Two
datasets are supported without any special-casing:

* the historical binary experiments, whose ``metrics.json`` stores the
  ``tn``/``fp``/``fn``/``tp`` quadruple;
* k-class experiments, whose ``metrics.json`` stores a full ``confusion_matrix``
  (rows = true label, columns = predicted label).

Class names come from the ``label_scheme`` block that ``run_emd_experiments.py``
records, falling back to the historical Chinese binary names, so the original
binary images stay byte-identical.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from emd_pipeline import (
    FORM_ORDER,
    canonical_form,
    class_display_names,
    config_label_scheme_tag,
    label_scheme_tag,
    read_label_scheme,
)


SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
# Two-class fills, kept exactly as they were so the stored binary images do not change.
BINARY_FILLS = (("#d9eaf7", "#f5d7d0"), ("#f5d7d0", "#dcefe3"))
# k > 2: the diagonal is correct by construction, off-diagonal cells are errors.
DIAGONAL_FILL = "#dcefe3"
OFF_DIAGONAL_FILL = "#f5d7d0"
EMPTY_FILL = "#eef1f4"


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


def _confusion_payload(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    split: str,
) -> dict[str, Any] | None:
    """Collect the matrix, the class names and the dataset tag for one split.

    k-class runs are read straight from the saved ``confusion_matrix``; older
    binary runs, which never stored one, are rebuilt from ``tn``/``fp``/``fn``/
    ``tp``.  Both routes yield the same rows=true / columns=predicted layout, so
    one renderer covers them.
    """

    split_metrics = metrics.get("metrics", {}).get(split, {})
    if not isinstance(split_metrics, dict):
        return None

    matrix: list[list[int]] | None = None
    recorded = split_metrics.get("confusion_matrix")
    if isinstance(recorded, list) and recorded and all(isinstance(row, list) for row in recorded):
        try:
            candidate = [[int(value) for value in row] for row in recorded]
        except (TypeError, ValueError):
            candidate = None
        if candidate and all(len(row) == len(candidate) for row in candidate) and len(candidate) >= 2:
            matrix = candidate

    if matrix is None:
        try:
            tn = int(split_metrics["tn"])
            fp = int(split_metrics["fp"])
            fn = int(split_metrics["fn"])
            tp = int(split_metrics["tp"])
        except (KeyError, TypeError, ValueError):
            return None
        matrix = [[tn, fp], [fn, tp]]

    num_classes = len(matrix)
    try:
        recorded_classes = int(split_metrics["num_classes"])
    except (KeyError, TypeError, ValueError):
        recorded_classes = num_classes
    if recorded_classes != num_classes:
        # A k-class matrix that does not match its advertised class count cannot
        # be labelled correctly, so trust the matrix and say so.
        recorded_classes = num_classes

    scheme = metrics.get("label_scheme")
    if not isinstance(scheme, dict):
        scheme = config.get("label_scheme")
    if not isinstance(scheme, dict):
        data_dir = config.get("data_dir")
        scheme = read_label_scheme(data_dir) if data_dir else None
    if not isinstance(scheme, dict):
        scheme = None

    return {
        "matrix": matrix,
        "num_classes": num_classes,
        "class_names": class_display_names(scheme, num_classes),
        "label_tag": label_scheme_tag(scheme),
        "split_metrics": split_metrics,
    }


def _class_caption(label: int, name: str) -> str:
    """Two-line axis caption: the numeric label, then its physical meaning."""

    return f"label={label}\n{name}"


def _render_binary_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    title: str,
    split: str,
    output_path: Path,
) -> None:
    """The original 2x2 layout.  Unchanged rather than merely equivalent: the
    binary images in ``visualizations/`` are the reference for every
    k-class comparison, so they must not shift by a pixel."""

    tn, fp = matrix[0]
    fn, tp = matrix[1]
    row_totals = (tn + fp, fn + tp)
    width, height = 760, 660
    left, top = 160, 190
    cell = 170
    image = Image.new("RGB", (width, height), "#f7f9fb")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 42), title, fill="#1f2d36", font=_font(26, bold=True), anchor="mm")
    draw.text(
        (width // 2, 82),
        f"{SPLIT_LABELS.get(split, split)} confusion matrix | rows=true, columns=predicted",
        fill="#52616b",
        font=_font(16),
        anchor="mm",
    )

    for row in range(2):
        for column in range(2):
            x0 = left + column * cell
            y0 = top + row * cell
            x1, y1 = x0 + cell, y0 + cell
            value = matrix[row][column]
            total = row_totals[row]
            percentage = 0.0 if total == 0 else value / total * 100.0
            draw.rectangle(
                (x0, y0, x1, y1), fill=BINARY_FILLS[row][column], outline="#52616b", width=2
            )
            draw.text(
                ((x0 + x1) // 2, y0 + 66),
                str(value),
                fill="#1f2d36",
                font=_font(32, bold=True),
                anchor="mm",
            )
            draw.text(
                ((x0 + x1) // 2, y0 + 116),
                f"{percentage:.1f}% of true class",
                fill="#52616b",
                font=_font(14),
                anchor="mm",
            )

    class_labels = tuple(_class_caption(index, name) for index, name in enumerate(class_names))
    for index, label in enumerate(class_labels):
        x = left + index * cell + cell // 2
        draw.text((x, top - 24), label, fill="#1f2d36", font=_font(16, bold=True), anchor="ms", align="center")
        y = top + index * cell + cell // 2
        draw.text((left - 22, y), label, fill="#1f2d36", font=_font(16, bold=True), anchor="rm", align="right")
    draw.text((left + cell, top - 58), "Predicted label", fill="#1f2d36", font=_font(17), anchor="mm")
    draw.text((52, top + cell), "True label", fill="#1f2d36", font=_font(17), anchor="mm")
    draw.text(
        (width // 2, height - 58),
        f"TN={tn}   FP={fp}   FN={fn}   TP={tp}",
        fill="#52616b",
        font=_font(17),
        anchor="mm",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _render_multi_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    split_metrics: Mapping[str, Any],
    title: str,
    split: str,
    output_path: Path,
) -> None:
    """k x k grid for k >= 3, annotated with the macro statistics."""

    num_classes = len(matrix)
    cell = 170 if num_classes <= 4 else max(96, 640 // num_classes)
    left, top = 250, 210
    grid = cell * num_classes
    width = left + grid + 60
    height = top + grid + 130
    image = Image.new("RGB", (width, height), "#f7f9fb")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 42), title, fill="#1f2d36", font=_font(26, bold=True), anchor="mm")
    draw.text(
        (width // 2, 82),
        f"{SPLIT_LABELS.get(split, split)} confusion matrix | rows=true, columns=predicted"
        f" | {num_classes} classes",
        fill="#52616b",
        font=_font(16),
        anchor="mm",
    )

    value_font = _font(max(16, cell // 5), bold=True)
    share_font = _font(max(11, cell // 9))
    axis_font = _font(14, bold=True)
    for row in range(num_classes):
        row_total = sum(int(value) for value in matrix[row]) or 1
        for column in range(num_classes):
            value = int(matrix[row][column])
            x0 = left + column * cell
            y0 = top + row * cell
            x1, y1 = x0 + cell, y0 + cell
            if value == 0:
                fill = EMPTY_FILL
            elif row == column:
                fill = DIAGONAL_FILL
            else:
                fill = OFF_DIAGONAL_FILL
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline="#52616b", width=2)
            draw.text(
                ((x0 + x1) // 2, y0 + cell // 2 - 8),
                str(value),
                fill="#1f2d36",
                font=value_font,
                anchor="mm",
            )
            draw.text(
                ((x0 + x1) // 2, y0 + cell // 2 + 22),
                f"{value / row_total * 100.0:.1f}%",
                fill="#52616b",
                font=share_font,
                anchor="mm",
            )

    for index in range(num_classes):
        caption = _class_caption(index, class_names[index])
        draw.text(
            (left + index * cell + cell // 2, top - 26),
            caption,
            fill="#1f2d36",
            font=axis_font,
            anchor="ms",
            align="center",
        )
        draw.text(
            (left - 26, top + index * cell + cell // 2),
            caption,
            fill="#1f2d36",
            font=axis_font,
            anchor="rm",
            align="right",
        )
    draw.text((left + grid // 2, top - 76), "Predicted label", fill="#1f2d36", font=_font(17), anchor="mm")
    draw.text((70, top + grid // 2), "True label", fill="#1f2d36", font=_font(17), anchor="mm")

    def _number(name: str) -> str:
        value = split_metrics.get(name)
        return "n/a" if value is None else f"{float(value):.3f}"

    draw.text(
        (width // 2, top + grid + 34),
        "macro over present classes:"
        f"  accuracy={_number('accuracy')}  precision={_number('precision')}"
        f"  recall={_number('sensitivity')}  f1={_number('f1_score')}"
        f"  auc={_number('auc')}",
        fill="#52616b",
        font=_font(16),
        anchor="mm",
    )
    support = "   ".join(
        f"{index}:{sum(int(value) for value in matrix[index])}" for index in range(num_classes)
    )
    draw.text(
        (width // 2, top + grid + 66),
        f"support per class  {support}",
        fill="#52616b",
        font=_font(14),
        anchor="mm",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    title: str,
    split: str,
    output_path: Path,
    split_metrics: Mapping[str, Any] | None = None,
) -> None:
    """Dispatch on the number of classes; both layouts share rows=true."""

    if len(matrix) == 2:
        _render_binary_matrix(matrix, class_names, title, split, output_path)
        return
    _render_multi_matrix(matrix, class_names, split_metrics or {}, title, split, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize experiment confusion matrices")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/confusion_matrices"))
    parser.add_argument("--forms", default="all", help="all or comma-separated forms")
    parser.add_argument("--region-set", default="all")
    parser.add_argument("--channel-mode", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
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


def main() -> None:
    args = _parse_args()
    form_filter = _parse_forms(args.forms)
    dataset_tag = label_scheme_tag(read_label_scheme(args.data_dir))
    experiments_dir = args.experiments_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest: dict[str, Any] = {
        "split": args.split,
        "files": [],
        "generated": [],
        "output_dir": output_dir,
        "data_dir": str(Path(args.data_dir).resolve()),
        "label_scheme_tag": dataset_tag,
    }
    generated = 0
    if experiments_dir.exists():
        directories = sorted(path for path in experiments_dir.iterdir() if path.is_dir())
    else:
        directories = []
    for directory in directories:
        config = _read_json(directory / "config.json", {})
        metrics = _read_json(directory / "metrics.json", {})
        if not isinstance(config, dict) or not isinstance(metrics, dict):
            continue
        # A 3-class run must never be drawn into (or overwrite) the historical
        # binary image of the same form/region/channel combination.
        if config_label_scheme_tag(config) != dataset_tag:
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
        payload = _confusion_payload(config, metrics, args.split)
        if payload is None:
            continue
        file_name = (
            f"form_{_safe_name(form)}__regions_{_safe_name(region)}__"
            f"channels_{_safe_name(channel)}__split_{args.split}"
        )
        label_tag = payload["label_tag"]
        if label_tag:
            file_name = f"{file_name}__{_safe_name(label_tag)}"
        file_name = f"{file_name}.png"
        render_confusion_matrix(
            payload["matrix"],
            payload["class_names"],
            f"{form} | {region} | channel_mode={channel}",
            args.split,
            output_dir / file_name,
            payload["split_metrics"],
        )
        manifest["files"].append(file_name)
        manifest["generated"].append(
            {
                "file": file_name,
                "experiment": directory.name,
                "num_classes": payload["num_classes"],
                "label_scheme_tag": label_tag,
            }
        )
        generated += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"Confusion matrix visualizations written to {output_dir} ({generated} files)")


if __name__ == "__main__":
    main()

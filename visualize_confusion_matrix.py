"""Render confusion matrices from saved experiment metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from emd_pipeline import FORM_ORDER, canonical_form


SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}


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


def _confusion_counts(metrics: dict[str, Any], split: str) -> tuple[int, int, int, int] | None:
    split_metrics = metrics.get("metrics", {}).get(split, {})
    if not isinstance(split_metrics, dict):
        return None
    try:
        return (
            int(split_metrics["tn"]),
            int(split_metrics["fp"]),
            int(split_metrics["fn"]),
            int(split_metrics["tp"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def render_confusion_matrix(
    counts: tuple[int, int, int, int],
    title: str,
    split: str,
    output_path: Path,
) -> None:
    tn, fp, fn, tp = counts
    matrix = ((tn, fp), (fn, tp))
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

    colors = (("#d9eaf7", "#f5d7d0"), ("#f5d7d0", "#dcefe3"))
    for row in range(2):
        for column in range(2):
            x0 = left + column * cell
            y0 = top + row * cell
            x1, y1 = x0 + cell, y0 + cell
            value = matrix[row][column]
            total = row_totals[row]
            percentage = 0.0 if total == 0 else value / total * 100.0
            draw.rectangle((x0, y0, x1, y1), fill=colors[row][column], outline="#52616b", width=2)
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

    class_labels = ("label=0\n即将穿透", "label=1\n安全")
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize experiment confusion matrices")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/confusion_matrices"))
    parser.add_argument("--forms", default="all", help="all or comma-separated forms")
    parser.add_argument("--region-set", default="all")
    parser.add_argument("--channel-mode", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    form_filter = _parse_forms(args.forms)
    experiments_dir = args.experiments_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest: dict[str, Any] = {"split": args.split, "files": [], "output_dir": output_dir}
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
        form = str(config.get("form", ""))
        region = str(config.get("region_name", ""))
        channel = str(config.get("channel_mode", ""))
        if form_filter is not None and form not in form_filter:
            continue
        if args.region_set != "all" and region != args.region_set:
            continue
        if args.channel_mode is not None and channel != str(args.channel_mode):
            continue
        counts = _confusion_counts(metrics, args.split)
        if counts is None:
            continue
        file_name = (
            f"form_{_safe_name(form)}__regions_{_safe_name(region)}__"
            f"channels_{_safe_name(channel)}__split_{args.split}.png"
        )
        render_confusion_matrix(
            counts,
            f"{form} | {region} | channel_mode={channel}",
            args.split,
            output_dir / file_name,
        )
        manifest["files"].append(file_name)
        generated += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"Confusion matrix visualizations written to {output_dir} ({generated} files)")


if __name__ == "__main__":
    main()

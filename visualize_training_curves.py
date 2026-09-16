"""Render validation and test loss histories saved by EMD experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from emd_pipeline import FORM_ORDER, canonical_form


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _parse_forms(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    forms = {canonical_form(item.strip()) for item in value.split(",") if item.strip()}
    return forms or set(FORM_ORDER)


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


def _history_points(history: Any, key: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    if not isinstance(history, list):
        return points
    for record in history:
        if not isinstance(record, dict):
            continue
        try:
            epoch = int(record["epoch"])
            loss = float(record[key])
        except (KeyError, TypeError, ValueError):
            continue
        if epoch > 0 and loss == loss:
            points.append((epoch, loss))
    return sorted(points)


def _draw_dashed_vertical(draw: ImageDraw.ImageDraw, x: int, top: int, bottom: int, fill: str) -> None:
    for start in range(top, bottom, 16):
        draw.line((x, start, x, min(start + 8, bottom)), fill=fill, width=2)


def render_history(
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    title: str,
    output_path: Path,
) -> bool:
    val_points = _history_points(history, "val_loss")
    test_points = _history_points(history, "test_loss")
    if not val_points and not test_points:
        return False

    width, height = 1400, 850
    left, top, right, bottom = 120, 92, 330, 115
    plot_left, plot_top = left, top
    plot_right, plot_bottom = width - right, height - bottom
    image = Image.new("RGB", (width, height), "#f7f9fb")
    draw = ImageDraw.Draw(image)
    title_font = _font(28, bold=True)
    label_font = _font(20)
    tick_font = _font(17)
    legend_font = _font(18)
    draw.text((plot_left, 28), title, fill="#1f2d36", font=title_font)

    all_points = val_points + test_points
    max_epoch = max(epoch for epoch, _loss in all_points)
    min_loss = min(loss for _epoch, loss in all_points)
    max_loss = max(loss for _epoch, loss in all_points)
    if abs(max_loss - min_loss) < 1e-12:
        padding = max(abs(max_loss) * 0.05, 0.05)
    else:
        padding = (max_loss - min_loss) * 0.05
    min_loss -= padding
    max_loss += padding

    def point_xy(epoch: int, loss: float) -> tuple[int, int]:
        x_fraction = 0.5 if max_epoch == 1 else (epoch - 1) / (max_epoch - 1)
        y_fraction = (loss - min_loss) / (max_loss - min_loss)
        return (
            int(plot_left + x_fraction * (plot_right - plot_left)),
            int(plot_bottom - y_fraction * (plot_bottom - plot_top)),
        )

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#52616b", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#52616b", width=2)
    for index in range(7):
        fraction = index / 6
        y = int(plot_bottom - fraction * (plot_bottom - plot_top))
        value = min_loss + fraction * (max_loss - min_loss)
        draw.line((plot_left, y, plot_right, y), fill="#d7dee3", width=1)
        draw.text((plot_left - 14, y - 12), f"{value:.3f}", fill="#52616b", font=tick_font, anchor="ra")
    x_ticks = min(8, max_epoch)
    for index in range(x_ticks + 1):
        fraction = index / max(x_ticks, 1)
        epoch = 1 if max_epoch == 1 else int(round(1 + fraction * (max_epoch - 1)))
        x = plot_left if max_epoch == 1 else int(plot_left + fraction * (plot_right - plot_left))
        draw.line((x, plot_bottom, x, plot_bottom + 8), fill="#52616b", width=2)
        draw.text((x, plot_bottom + 16), str(epoch), fill="#52616b", font=tick_font, anchor="ma")

    draw.text(((plot_left + plot_right) // 2, height - 48), "Epoch", fill="#1f2d36", font=label_font, anchor="mm")
    draw.text((30, (plot_top + plot_bottom) // 2), "Loss", fill="#1f2d36", font=label_font, anchor="mm")

    series = ((val_points, "#2f6f9f", "Validation loss"), (test_points, "#c45a32", "Test loss"))
    for points, color, label in series:
        if not points:
            continue
        coordinates = [point_xy(epoch, loss) for epoch, loss in points]
        if len(coordinates) > 1:
            draw.line(coordinates, fill=color, width=4, joint="curve")
        for x, y in coordinates:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="#1f2d36", width=1)

    best_epoch = metrics.get("best_epoch")
    try:
        best_epoch = int(best_epoch)
    except (TypeError, ValueError):
        best_epoch = None
    if best_epoch is not None and val_points:
        best_point = next(((epoch, loss) for epoch, loss in val_points if epoch == best_epoch), None)
        if best_point is not None:
            x, _y = point_xy(*best_point)
            _draw_dashed_vertical(draw, x, plot_top, plot_bottom, "#6b7280")
            draw.text((x + 8, plot_top + 10), f"Best epoch={best_epoch}", fill="#4b5563", font=legend_font)

    legend_x, legend_y = plot_right + 34, plot_top + 24
    for _points, color, label in series:
        draw.line((legend_x, legend_y, legend_x + 28, legend_y), fill=color, width=4)
        draw.text((legend_x + 40, legend_y - 12), label, fill="#1f2d36", font=legend_font)
        legend_y += 38
    if not test_points:
        draw.multiline_text(
            (legend_x, legend_y + 12),
            "Test loss unavailable in this history;\nrerun training to add it.",
            fill="#7a4b50",
            font=_font(16),
            spacing=5,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize validation and test loss by epoch")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/training_curves"))
    parser.add_argument("--forms", default="all", help="all or comma-separated forms")
    parser.add_argument("--region-set", default="all")
    parser.add_argument("--channel-mode", type=int, choices=[1, 2, 3], default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    form_filter = _parse_forms(args.forms)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"files": [], "output_dir": output_dir}
    generated = 0
    for directory in sorted(args.experiments_dir.resolve().iterdir()):
        if not directory.is_dir():
            continue
        config = _read_json(directory / "config.json", {})
        history = _read_json(directory / "history.json", [])
        metrics = _read_json(directory / "metrics.json", {})
        if not isinstance(config, dict) or not isinstance(history, list):
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
        file_name = (
            f"form_{_safe_name(form)}__regions_{_safe_name(region)}__channels_{_safe_name(channel)}.png"
        )
        output_path = output_dir / file_name
        title = f"{form} | {region} | channel_mode={channel}"
        if render_history(history, metrics, title, output_path):
            generated += 1
            manifest["files"].append(file_name)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"Training curve visualizations written to {output_dir} ({generated} files)")


if __name__ == "__main__":
    main()

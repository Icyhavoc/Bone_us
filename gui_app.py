"""Tkinter viewer for EMD experiments and generated visualizations.

The GUI reads ``experiments/`` and ``visualizations/``.  Select ``all`` to
compare available experiment summaries, or select one frame-processing form
to inspect its metrics, training curve, preprocessing plots, and EMD component
plots.  The training button runs the existing training script in a background
worker for the selected concrete channel.
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import tkinter as tk
import threading
from pathlib import Path
from tkinter import ttk
from typing import Any, Iterable

from PIL import Image, ImageTk


APP_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = APP_DIR / "experiments"
VISUALIZATIONS_DIR = APP_DIR / "visualizations"

FORM_LABELS = {
    "all": "全部预处理方式",
    "mean_std": "Mean + Std",
    "raw50": "Raw 50 Frames",
    "max1": "Max 1 Frame",
    "top3_mean": "Top-3 Mean",
}
REGION_LABELS = {
    "all": "全部区域组合",
    "full": "0-5 mm",
    "bone": "1-3 mm",
    "bone_plus_post": "1-3 mm + 3-5 mm",
    "dyn_envelope": "Dynamic Envelope",
    "custom": "自定义区域",
}
def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def _best_history_info(
    history: list[dict[str, Any]], min_delta: float = 1e-4
) -> tuple[int | None, int | None]:
    """Return best epoch and last epoch, matching the training checkpoint rule."""

    if not history:
        return None, None
    best_record = None
    best_auc = float("-inf")
    best_loss = float("inf")
    for record in history:
        auc_value = record.get("val_auc")
        auc = float("-inf") if auc_value is None else float(auc_value)
        loss = float(record.get("val_loss", float("inf")))
        improved = auc > best_auc + min_delta
        if not improved and auc == best_auc and loss < best_loss - min_delta:
            improved = True
        if improved:
            best_record = record
            best_auc = auc
            best_loss = loss
    best_epoch = None if best_record is None else int(best_record.get("epoch", 0))
    return best_epoch, int(history[-1].get("epoch", 0))


class ExperimentCatalog:
    """Read summary/metric files and find matching generated image files."""

    def __init__(self, experiments_dir: Path, visualizations_dir: Path) -> None:
        self.experiments_dir = experiments_dir
        self.visualizations_dir = visualizations_dir
        self.summary: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []

    def reload(self) -> None:
        self.summary = _read_json(self.experiments_dir / "summary.json", {})
        summary_results = self.summary.get("results", {})
        rows: list[dict[str, Any]] = []
        if self.experiments_dir.exists():
            directories = sorted(p for p in self.experiments_dir.iterdir() if p.is_dir())
        else:
            directories = []
        for directory in directories:
            config = _read_json(directory / "config.json", {})
            metrics_file = _read_json(directory / "metrics.json", {})
            history = _read_json(directory / "history.json", [])
            summary_item = summary_results.get(directory.name, {})
            summary_metrics = summary_item.get("metrics", {})
            metrics = summary_metrics or metrics_file.get("metrics", {})
            if not metrics:
                continue
            mlp_config = config.get("mlp", {})
            min_delta = float(mlp_config.get("min_delta", 1e-4)) if isinstance(mlp_config, dict) else 1e-4
            fallback_best_epoch, fallback_trained_epochs = _best_history_info(history, min_delta)
            rows.append(
                {
                    "name": directory.name,
                    "directory": directory,
                    "form": config.get("form", self._parse_name(directory.name, "form")),
                    "region": config.get(
                        "region_name", self._parse_name(directory.name, "region")
                    ),
                    "channel": str(
                        config.get("channel_mode", self._parse_name(directory.name, "channel"))
                    ),
                    "feature_dim": metrics_file.get(
                        "feature_dim", summary_item.get("feature_dim", "-")
                    ),
                    "best_epoch": metrics_file.get(
                        "best_epoch", summary_item.get("best_epoch", fallback_best_epoch)
                    ),
                    "trained_epochs": metrics_file.get(
                        "trained_epochs", summary_item.get("trained_epochs", fallback_trained_epochs)
                    ),
                    "history": history,
                    "metrics": metrics,
                }
            )
        self.rows = rows

    @staticmethod
    def _parse_name(name: str, part: str) -> str:
        if part == "form":
            match = re.search(r"^form_(.*?)__regions_", name)
            return match.group(1) if match else ""
        if part == "region":
            match = re.search(r"__regions_(.*?)__channels_", name)
            return match.group(1) if match else ""
        # The name may carry a trailing aggregation suffix (``__pooled``).
        match = re.search(r"__channels_(\d+)", name)
        return match.group(1) if match else ""

    def filtered_rows(self, form: str, region: str, channel: str) -> list[dict[str, Any]]:
        rows = self.rows
        if form != "all":
            rows = [row for row in rows if row["form"] == form]
        if region != "all":
            rows = [row for row in rows if row["region"] == region]
        if channel != "all":
            rows = [row for row in rows if row["channel"] == channel]
        return rows

    def image_files(
        self,
        kind: str,
        form: str,
        region: str,
        channel: str,
        class_filter: str,
    ) -> list[Path]:
        """Find new channel-aware files, with fallback to legacy filenames."""

        root = self.visualizations_dir / kind
        if not root.exists() or form == "all":
            return []
        classes = ["imminent", "safe"] if class_filter == "all" else [class_filter]
        regions = [region]
        if region == "all":
            regions = ["*"]
        found: list[Path] = []
        for class_name in classes:
            for region_name in regions:
                if region_name == "*":
                    channel_part = "*" if channel == "all" else channel
                    new_pattern = f"form_{form}__regions_*__channels_{channel_part}__class_{class_name}*.png"
                    legacy_pattern = f"form_{form}__regions_*__class_{class_name}*.png"
                else:
                    channel_part = "*" if channel == "all" else channel
                    new_pattern = f"form_{form}__regions_{region_name}__channels_{channel_part}__class_{class_name}*.png"
                    legacy_pattern = f"form_{form}__regions_{region_name}__class_{class_name}*.png"
                new_matches = sorted(root.glob(new_pattern))
                legacy_matches = sorted(root.glob(legacy_pattern))
                found.extend(new_matches or legacy_matches)
        return sorted(set(found))

    def preprocessing_options(
        self, form_filter: str, region_filter: str, channel_filter: str
    ) -> dict[str, dict[str, Path]]:
        """Return selectable preprocessing image pairs keyed by configuration."""

        root = self.visualizations_dir / "preprocessing"
        pattern = re.compile(
            r"^form_(?P<form>.+?)__regions_(?P<region>.+?)"
            r"(?:__channels_(?P<channel>[123]))?__class_(?P<class>imminent|safe)\.png$"
        )
        options: dict[str, dict[str, Path]] = {}
        if not root.exists():
            return options
        for path in sorted(root.glob("*.png")):
            match = pattern.match(path.name)
            if match is None:
                continue
            form = match.group("form")
            region = match.group("region")
            file_channel = match.group("channel")
            if form_filter != "all" and form != form_filter:
                continue
            if region_filter != "all" and region != region_filter:
                continue
            if channel_filter != "all" and file_channel not in (None, channel_filter):
                continue
            display_channel = file_channel or (channel_filter if channel_filter != "all" else "legacy")
            key = f"form={form} | region={region} | channel={display_channel}"
            class_name = match.group("class")
            pair = options.setdefault(key, {})
            # Prefer channel-aware files over legacy files when both exist.
            if class_name not in pair or file_channel is not None:
                pair[class_name] = path
        return options

    def training_curve_files(self, form: str, region: str, channel: str) -> list[Path]:
        root = self.visualizations_dir / "training_curves"
        if not root.exists():
            return []
        form_part = "*" if form == "all" else form
        region_part = "*" if region == "all" else region
        channel_part = "*" if channel == "all" else channel
        pattern = (
            f"form_{form_part}__regions_{region_part}__channels_{channel_part}.png"
        )
        return sorted(root.glob(pattern))

    def confusion_matrix_files(
        self, form: str, region: str, channel: str, split: str = "test"
    ) -> list[Path]:
        root = self.visualizations_dir / "confusion_matrices"
        if not root.exists() or form == "all":
            return []
        form_part = form
        region_part = "*" if region == "all" else region
        channel_part = "*" if channel == "all" else channel
        pattern = (
            f"form_{form_part}__regions_{region_part}__channels_{channel_part}"
            f"__split_{split}.png"
        )
        return sorted(root.glob(pattern))

    def confusion_matrix_options(self, split: str = "test") -> dict[str, Path]:
        """Return all generated confusion matrices for the target selector."""

        root = self.visualizations_dir / "confusion_matrices"
        pattern = re.compile(
            rf"^form_(?P<form>.+?)__regions_(?P<region>.+?)"
            rf"__channels_(?P<channel>[123])__split_{re.escape(split)}\.png$"
        )
        options: dict[str, Path] = {}
        if not root.exists():
            return options
        for path in sorted(root.glob(f"*__split_{split}.png")):
            match = pattern.match(path.name)
            if match is None:
                continue
            key = (
                f"form={match.group('form')} | region={match.group('region')} | "
                f"channel={match.group('channel')}"
            )
            options[key] = path
        return options

    def error_by_thickness_files(self, form: str, region: str, channel: str) -> list[Path]:
        root = self.visualizations_dir / "error_by_thickness"
        if not root.exists():
            return []
        form_part = "*" if form == "all" else form
        region_part = "*" if region == "all" else region
        channel_part = "*" if channel == "all" else channel
        pattern = f"form_{form_part}__regions_{region_part}__channels_{channel_part}.png"
        return sorted(root.glob(pattern))

    def error_by_thickness_options(self) -> dict[str, Path]:
        """Return all generated thickness-vs-error charts for the target selector."""

        root = self.visualizations_dir / "error_by_thickness"
        pattern = re.compile(
            r"^form_(?P<form>.+?)__regions_(?P<region>.+?)"
            r"__channels_(?P<channel>[123])\.png$"
        )
        options: dict[str, Path] = {}
        if not root.exists():
            return options
        for path in sorted(root.glob("*.png")):
            match = pattern.match(path.name)
            if match is None:
                continue
            key = (
                f"form={match.group('form')} | region={match.group('region')} | "
                f"channel={match.group('channel')}"
            )
            options[key] = path
        return options


class ScrollableImagePanel(ttk.Frame):
    """A vertical image browser used by both visualization tabs."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#f4f6f8")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.inner.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.window_id, width=event.width),
        )
        self._photo_refs: list[ImageTk.PhotoImage] = []

    def clear(self, message: str | None = None) -> None:
        for child in self.inner.winfo_children():
            child.destroy()
        self._photo_refs.clear()
        if message:
            ttk.Label(self.inner, text=message, padding=20).pack(anchor="w")

    def show_images(self, paths: Iterable[Path]) -> int:
        self.clear()
        paths = list(paths)
        if not paths:
            self.clear("当前筛选条件下没有找到对应图片。请先运行可视化脚本生成结果。")
            return 0
        for path in paths:
            try:
                image = Image.open(path).convert("RGB")
                image.thumbnail((1100, 850), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
            except (OSError, ValueError) as exc:
                ttk.Label(self.inner, text=f"无法读取 {path.name}: {exc}").pack(anchor="w")
                continue
            self._photo_refs.append(photo)
            frame = ttk.Frame(self.inner, padding=(10, 10, 10, 4))
            frame.pack(fill="x", anchor="w")
            ttk.Label(frame, text=path.name).pack(anchor="w")
            ttk.Label(frame, image=photo).pack(anchor="w")
        return len(self._photo_refs)


class SynchronizedImageColumns(ttk.Frame):
    """Show four equal-width image columns with synchronized scrolling."""

    def __init__(self, parent: tk.Misc, titles: Iterable[str]) -> None:
        super().__init__(parent)
        self._titles = list(titles)
        if len(self._titles) != 4:
            raise ValueError("Exactly four image columns are required")
        self._paths: list[list[Path]] = [[] for _ in self._titles]
        self._message: str | None = None
        self._photo_refs: list[ImageTk.PhotoImage] = []
        self._render_pending = False

        row = ttk.Frame(self)
        row.pack(fill="both", expand=True)
        viewport = ttk.Frame(row)
        viewport.pack(side="left", fill="both", expand=True)
        vertical_scroll = ttk.Scrollbar(row, orient="vertical", command=self._yview)
        vertical_scroll.pack(side="right", fill="y")
        horizontal_scroll = ttk.Scrollbar(self, orient="horizontal", command=self._xview)
        horizontal_scroll.pack(fill="x")

        self._viewport = viewport
        self._horizontal_scroll = horizontal_scroll
        self._vertical_scroll = vertical_scroll
        self._canvases: list[tk.Canvas] = []
        for index, title in enumerate(self._titles):
            viewport.columnconfigure(index, weight=1, uniform="preprocessing_columns")
            frame = ttk.LabelFrame(viewport, text=title)
            frame.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 2, 0))
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            canvas = tk.Canvas(frame, highlightthickness=0, background="#f4f6f8")
            canvas.grid(row=0, column=0, sticky="nsew")
            self._canvases.append(canvas)
            canvas.bind("<Configure>", lambda _event: self._schedule_render())
        viewport.rowconfigure(0, weight=1)

        self._canvases[0].configure(
            xscrollcommand=self._sync_horizontal_scroll,
            yscrollcommand=self._sync_vertical_scroll,
        )

    def _sync_horizontal_scroll(self, first: str, last: str) -> None:
        self._horizontal_scroll.set(first, last)

    def _sync_vertical_scroll(self, first: str, last: str) -> None:
        self._vertical_scroll.set(first, last)

    def _xview(self, *args: str) -> None:
        for canvas in self._canvases:
            canvas.xview(*args)

    def _yview(self, *args: str) -> None:
        for canvas in self._canvases:
            canvas.yview(*args)

    def _schedule_render(self) -> None:
        if self._render_pending:
            return
        self._render_pending = True
        self.after_idle(self._render)

    def clear(self, message: str | None = None) -> None:
        self._paths = [[] for _ in self._titles]
        self._message = message
        self._schedule_render()

    def show_columns(self, paths_by_column: Iterable[Iterable[Path]]) -> None:
        columns = [list(paths) for paths in paths_by_column]
        if len(columns) != 4:
            raise ValueError("Exactly four image columns are required")
        self._paths = columns
        self._message = None
        self._schedule_render()

    def _render(self) -> None:
        self._render_pending = False
        if not self._canvases or self._canvases[0].winfo_width() <= 20:
            # The Configure event will schedule another render after the
            # window receives its real size. Do not enqueue an idle callback
            # repeatedly while the window is still being laid out.
            return
        column_width = max(120, min(canvas.winfo_width() for canvas in self._canvases))
        caption_height = 28
        max_count = max((len(paths) for paths in self._paths), default=0)
        slot_count = max(1, max_count)
        content_width = column_width * slot_count
        content_height = 80
        self._photo_refs.clear()
        for canvas in self._canvases:
            canvas.delete("all")
            canvas.configure(xscrollincrement=column_width)

        for column_index, (canvas, paths) in enumerate(zip(self._canvases, self._paths)):
            if not paths:
                message = self._message or "当前没有对应图片。"
                canvas.create_text(18, 18, anchor="nw", text=message, fill="#263238")
                continue
            for image_index, path in enumerate(paths):
                x = image_index * column_width
                try:
                    image = Image.open(path).convert("RGB")
                    image_width, image_height = image.size
                    scaled_height = max(1, int(image_height * column_width / max(image_width, 1)))
                    image = image.resize((column_width, scaled_height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    self._photo_refs.append(photo)
                    canvas.create_text(
                        x + 8,
                        6,
                        anchor="nw",
                        text=path.name,
                        fill="#263238",
                    )
                    canvas.create_image(x, caption_height, image=photo, anchor="nw")
                    content_height = max(content_height, caption_height + scaled_height)
                except (OSError, ValueError) as exc:
                    canvas.create_text(
                        x + 12,
                        caption_height + 12,
                        anchor="nw",
                        text=f"无法读取 {path.name}: {exc}",
                        fill="#7a4b50",
                    )

        for canvas in self._canvases:
            canvas.configure(scrollregion=(0, 0, content_width, content_height))
            canvas.xview_moveto(0)
            canvas.yview_moveto(0)


class TrainingCurvePanel(ttk.Frame):
    """Draw validation loss against epoch for the currently selected rows."""

    _COLORS = ("#2f6f9f", "#c45a32", "#3d8b62", "#7b5aa6", "#b08a2e", "#7a4b50")

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#f4f6f8")
        self.canvas.pack(fill="both", expand=True)
        self._rows: list[dict[str, Any]] = []
        self.canvas.bind("<Configure>", lambda _event: self._draw())

    def clear(self, message: str | None = None) -> None:
        self._rows = []
        self.canvas.delete("all")
        if message:
            self.canvas.create_text(
                24,
                24,
                anchor="nw",
                text=message,
                fill="#263238",
                font=("Microsoft YaHei", 11),
            )

    def show_histories(self, rows: Iterable[dict[str, Any]]) -> None:
        self._rows = [row for row in rows if self._history_points(row.get("history", []))]
        self._draw()

    @staticmethod
    def _history_points(history: Any) -> list[tuple[int, float]]:
        points: list[tuple[int, float]] = []
        if not isinstance(history, list):
            return points
        for record in history:
            if not isinstance(record, dict):
                continue
            try:
                epoch = int(record["epoch"])
                loss = float(record["val_loss"])
            except (KeyError, TypeError, ValueError):
                continue
            if epoch > 0 and loss == loss:
                points.append((epoch, loss))
        return sorted(points)

    def _draw(self) -> None:
        self.canvas.delete("all")
        histories = [(row, self._history_points(row.get("history", []))) for row in self._rows]
        histories = [(row, points) for row, points in histories if points]
        if not histories:
            self.canvas.create_text(
                24,
                24,
                anchor="nw",
                text="No training history is available for the current selection.",
                fill="#263238",
                font=("Microsoft YaHei", 11),
            )
            return

        width = max(self.canvas.winfo_width(), 900)
        height = max(self.canvas.winfo_height(), 520)
        left, top, bottom = 82, 54, 72
        right = 260 if len(histories) > 1 else 150
        plot_width = max(width - left - right, 420)
        plot_height = max(height - top - bottom, 300)
        x0, y0 = left, top
        x1, y1 = left + plot_width, top + plot_height
        max_epoch = max(points[-1][0] for _row, points in histories)
        all_losses = [loss for _row, points in histories for _epoch, loss in points]
        min_loss, max_loss = min(all_losses), max(all_losses)
        if abs(max_loss - min_loss) < 1e-12:
            padding = max(abs(max_loss) * 0.05, 0.05)
            min_loss -= padding
            max_loss += padding
        else:
            padding = (max_loss - min_loss) * 0.05
            min_loss -= padding
            max_loss += padding

        self.canvas.create_text(
            x0,
            18,
            anchor="w",
            text="Validation Loss vs Epoch",
            fill="#263238",
            font=("Microsoft YaHei", 13, "bold"),
        )
        self.canvas.create_line(x0, y1, x1, y1, fill="#52616b", width=1)
        self.canvas.create_line(x0, y0, x0, y1, fill="#52616b", width=1)

        tick_count = 6
        for index in range(tick_count + 1):
            fraction = index / tick_count
            y = y1 - fraction * plot_height
            value = min_loss + fraction * (max_loss - min_loss)
            self.canvas.create_line(x0, y, x1, y, fill="#d5dce1", width=1)
            self.canvas.create_text(
                x0 - 8,
                y,
                anchor="e",
                text=f"{value:.3f}",
                fill="#52616b",
                font=("Microsoft YaHei", 9),
            )
        x_ticks = min(6, max_epoch)
        for index in range(x_ticks + 1):
            fraction = index / max(x_ticks, 1)
            epoch = 1 if max_epoch == 1 else int(round(1 + fraction * (max_epoch - 1)))
            x = x0 if max_epoch == 1 else x0 + fraction * plot_width
            self.canvas.create_line(x, y1, x, y1 + 5, fill="#52616b", width=1)
            self.canvas.create_text(
                x,
                y1 + 18,
                text=str(epoch),
                fill="#52616b",
                font=("Microsoft YaHei", 9),
            )

        self.canvas.create_text(
            (x0 + x1) / 2,
            y1 + 48,
            text="Epoch",
            fill="#263238",
            font=("Microsoft YaHei", 10),
        )
        self.canvas.create_text(
            18,
            (y0 + y1) / 2,
            text="Validation loss",
            angle=90,
            fill="#263238",
            font=("Microsoft YaHei", 10),
        )

        for index, (row, points) in enumerate(histories):
            color = self._COLORS[index % len(self._COLORS)]
            coordinates: list[float] = []
            for epoch, loss in points:
                x = x0 if max_epoch == 1 else x0 + (epoch - 1) / (max_epoch - 1) * plot_width
                y = y1 - (loss - min_loss) / (max_loss - min_loss) * plot_height
                coordinates.extend((x, y))
            if len(coordinates) >= 4:
                self.canvas.create_line(*coordinates, fill=color, width=2, smooth=False)
            best_epoch = row.get("best_epoch")
            try:
                best_epoch_int = int(best_epoch)
            except (TypeError, ValueError):
                best_epoch_int = None
            if best_epoch_int is not None:
                best_points = [(epoch, loss) for epoch, loss in points if epoch == best_epoch_int]
                if best_points:
                    epoch, loss = best_points[0]
                    x = x0 if max_epoch == 1 else x0 + (epoch - 1) / (max_epoch - 1) * plot_width
                    y = y1 - (loss - min_loss) / (max_loss - min_loss) * plot_height
                    self.canvas.create_oval(
                        x - 4,
                        y - 4,
                        x + 4,
                        y + 4,
                        fill=color,
                        outline="#263238",
                        width=1,
                    )
            legend_y = top + index * 24
            self.canvas.create_line(x1 + 12, legend_y, x1 + 34, legend_y, fill=color, width=3)
            name = str(row.get("name", "experiment"))
            if len(name) > 30:
                name = name[:27] + "..."
            best_text = "-" if best_epoch_int is None else str(best_epoch_int)
            self.canvas.create_text(
                x1 + 42,
                legend_y,
                anchor="w",
                text=f"{name} (best={best_text})",
                fill="#263238",
                font=("Microsoft YaHei", 9),
            )


class EMDViewerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EMD + MLP 实验查看器")
        self.geometry("1450x920")
        self.minsize(1100, 700)
        self.catalog = ExperimentCatalog(EXPERIMENTS_DIR, VISUALIZATIONS_DIR)
        self.training_active = False
        self.training_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._build_style()
        self._build_controls()
        self._build_tabs()
        self.refresh()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Microsoft YaHei", 16, "bold"))
        style.configure("Hint.TLabel", foreground="#5f6b76")

    def _build_controls(self) -> None:
        header = ttk.Frame(self, padding=(16, 12, 16, 6))
        header.pack(fill="x")
        ttk.Label(header, text="EMD + MLP 实验查看器", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="选择全部方法查看汇总；选择单一方法查看对应指标、预处理信号和 EMD 分量。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        controls = ttk.Frame(self, padding=(16, 4, 16, 10))
        controls.pack(fill="x")
        self.form_var = tk.StringVar(value="all")
        self.region_var = tk.StringVar(value="all")
        self.channel_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="")
        self._add_combo(controls, "预处理方式", self.form_var, list(FORM_LABELS), 0, "form")
        self._add_combo(controls, "区域组合", self.region_var, list(REGION_LABELS), 2, "region")
        self._add_combo(controls, "通道", self.channel_var, ["all", "1", "2", "3"], 4, "channel")
        ttk.Button(controls, text="刷新", command=self.refresh).grid(row=0, column=6, padx=(18, 4))
        ttk.Button(controls, text="打开实验目录", command=self._open_experiment_dir).grid(
            row=0, column=7, padx=4
        )
        self.train_button = ttk.Button(controls, text="开始训练", command=self.start_training)
        self.train_button.grid(row=0, column=8, padx=4)
        self.auto_visualize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="训练后生成图像",
            variable=self.auto_visualize_var,
        ).grid(row=0, column=9, padx=(4, 0))
        for column in (1, 3, 5):
            controls.columnconfigure(column, weight=1)
        self.form_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        self.region_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        self.channel_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        ttk.Label(controls, textvariable=self.status_var, style="Hint.TLabel").grid(
            row=1, column=0, columnspan=10, sticky="w", pady=(8, 0)
        )

    def _add_combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        column: int,
        attribute_name: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=0, column=column, sticky="w", padx=(0, 6))
        combo = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
            width=20,
        )
        combo.grid(row=0, column=column + 1, sticky="ew", padx=(0, 10))
        setattr(self, f"{attribute_name}_combo", combo)

    def _build_tabs(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.summary_tab = ttk.Frame(notebook, padding=8)
        self.preprocessing_tab = ttk.Frame(notebook)
        self.training_curve_tab = ttk.Frame(notebook)
        self.confusion_matrix_tab = ttk.Frame(notebook)
        self.error_by_thickness_tab = ttk.Frame(notebook)
        self.emd_tab = ttk.Frame(notebook)
        notebook.add(self.summary_tab, text="Summary")
        notebook.add(self.preprocessing_tab, text="预处理可视化")
        notebook.add(self.training_curve_tab, text="Training Curve")
        notebook.add(self.confusion_matrix_tab, text="Confusion Matrix")
        notebook.add(self.error_by_thickness_tab, text="错分厚度分布")
        notebook.add(self.emd_tab, text="EMD 分解")

        self.summary_hint = ttk.Label(self.summary_tab, text="", style="Hint.TLabel")
        self.summary_hint.pack(anchor="w", pady=(0, 8))
        columns = (
            "experiment",
            "feature_dim",
            "best_epoch",
            "trained_epochs",
            "train_auc",
            "val_auc",
            "test_auc",
            "test_accuracy",
            "test_f1",
            "test_sensitivity",
            "test_specificity",
        )
        self.summary_tree = ttk.Treeview(self.summary_tab, columns=columns, show="headings", height=15)
        headings = {
            "experiment": "实验",
            "feature_dim": "特征维数",
            "best_epoch": "Best Epoch",
            "trained_epochs": "Trained Epochs",
            "train_auc": "Train AUC",
            "val_auc": "Val AUC",
            "test_auc": "Test AUC",
            "test_accuracy": "Test Accuracy",
            "test_f1": "Test F1",
            "test_sensitivity": "Test Sensitivity",
            "test_specificity": "Test Specificity",
        }
        widths = {
            "experiment": 330,
            "feature_dim": 80,
            "best_epoch": 85,
            "trained_epochs": 100,
            "train_auc": 85,
            "val_auc": 85,
            "test_auc": 85,
            "test_accuracy": 105,
            "test_f1": 85,
            "test_sensitivity": 115,
            "test_specificity": 115,
        }
        for column in columns:
            self.summary_tree.heading(column, text=headings[column])
            self.summary_tree.column(column, width=widths[column], anchor="center")
        summary_scroll = ttk.Scrollbar(self.summary_tab, orient="vertical", command=self.summary_tree.yview)
        self.summary_tree.configure(yscrollcommand=summary_scroll.set)
        self.summary_tree.pack(side="left", fill="both", expand=True)
        summary_scroll.pack(side="right", fill="y")

        log_frame = ttk.LabelFrame(self.summary_tab, text="运行日志", padding=4)
        log_frame.pack(fill="x", expand=False, pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=8, wrap="none", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        selection_frame = ttk.Frame(self.preprocessing_tab, padding=(8, 8, 8, 4))
        selection_frame.pack(fill="x")
        ttk.Label(selection_frame, text="选择右侧预处理可视化组合：").pack(side="left")
        self.preprocessing_choice_var = tk.StringVar(value="")
        self.preprocessing_choice = ttk.Combobox(
            selection_frame,
            textvariable=self.preprocessing_choice_var,
            state="readonly",
            width=50,
        )
        self.preprocessing_choice.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.preprocessing_choice.bind("<<ComboboxSelected>>", lambda _event: self._refresh_preprocessing_choice())
        self.preprocessing_columns = SynchronizedImageColumns(
            self.preprocessing_tab,
            (
                "当前组合：label=0（即将穿透）",
                "当前组合：label=1（安全）",
                "目标组合：label=0（即将穿透）",
                "目标组合：label=1（安全）",
            ),
        )
        self.preprocessing_columns.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.preprocessing_option_map: dict[str, dict[str, Path]] = {}
        self._current_preprocessing_paths: tuple[list[Path], list[Path]] = ([], [])

        self.training_curve_images = ScrollableImagePanel(self.training_curve_tab)
        self.training_curve_images.pack(fill="both", expand=True)
        confusion_selection = ttk.Frame(self.confusion_matrix_tab, padding=(8, 8, 8, 4))
        confusion_selection.pack(fill="x")
        ttk.Label(confusion_selection, text="选择右侧混淆矩阵组合：").pack(side="left")
        self.confusion_choice_var = tk.StringVar(value="")
        self.confusion_choice = ttk.Combobox(
            confusion_selection,
            textvariable=self.confusion_choice_var,
            state="readonly",
            width=50,
        )
        self.confusion_choice.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.confusion_choice.bind("<<ComboboxSelected>>", lambda _event: self._refresh_confusion_matrix_selection())
        self.confusion_split = ttk.PanedWindow(self.confusion_matrix_tab, orient="horizontal")
        self.confusion_split.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        current_confusion_frame = ttk.LabelFrame(self.confusion_split, text="当前组合")
        target_confusion_frame = ttk.LabelFrame(self.confusion_split, text="目标组合")
        self.confusion_split.add(current_confusion_frame)
        self.confusion_split.add(target_confusion_frame)
        self.current_confusion_images = ScrollableImagePanel(current_confusion_frame)
        self.current_confusion_images.pack(fill="both", expand=True)
        self.target_confusion_images = ScrollableImagePanel(target_confusion_frame)
        self.target_confusion_images.pack(fill="both", expand=True)
        self.confusion_option_map: dict[str, Path] = {}
        self._current_confusion_paths: list[Path] = []

        thickness_selection = ttk.Frame(self.error_by_thickness_tab, padding=(8, 8, 8, 4))
        thickness_selection.pack(fill="x")
        ttk.Label(thickness_selection, text="选择右侧错分厚度分布组合：").pack(side="left")
        self.thickness_choice_var = tk.StringVar(value="")
        self.thickness_choice = ttk.Combobox(
            thickness_selection,
            textvariable=self.thickness_choice_var,
            state="readonly",
            width=50,
        )
        self.thickness_choice.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.thickness_choice.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_error_by_thickness_selection()
        )
        self.thickness_split = ttk.PanedWindow(self.error_by_thickness_tab, orient="horizontal")
        self.thickness_split.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        current_thickness_frame = ttk.LabelFrame(self.thickness_split, text="当前组合")
        target_thickness_frame = ttk.LabelFrame(self.thickness_split, text="目标组合")
        self.thickness_split.add(current_thickness_frame)
        self.thickness_split.add(target_thickness_frame)
        self.current_thickness_images = ScrollableImagePanel(current_thickness_frame)
        self.current_thickness_images.pack(fill="both", expand=True)
        self.target_thickness_images = ScrollableImagePanel(target_thickness_frame)
        self.target_thickness_images.pack(fill="both", expand=True)
        self.thickness_option_map: dict[str, Path] = {}
        self._current_thickness_paths: list[Path] = []

        self.emd_images = ScrollableImagePanel(self.emd_tab)
        self.emd_images.pack(fill="both", expand=True)

    def _refresh_preprocessing_choice(self) -> None:
        selected = self.preprocessing_choice_var.get()
        pair = self.preprocessing_option_map.get(selected, {})
        imminent = pair.get("imminent")
        safe = pair.get("safe")
        current_imminent, current_safe = self._current_preprocessing_paths
        self.preprocessing_columns.show_columns(
            (
                current_imminent,
                current_safe,
                [imminent] if imminent else [],
                [safe] if safe else [],
            )
        )

    def _refresh_preprocessing_layout(self, form: str, region: str, channel: str) -> tuple[int, int]:
        imminent_paths = self.catalog.image_files("preprocessing", form, region, channel, "imminent")
        safe_paths = self.catalog.image_files("preprocessing", form, region, channel, "safe")
        self._current_preprocessing_paths = (imminent_paths, safe_paths)
        left_imminent_count = len(imminent_paths)
        left_safe_count = len(safe_paths)

        # The target selector is intentionally independent of the top-level
        # current-result filters, so it can compare any generated form,
        # region, and channel combination against the current columns.
        self.preprocessing_option_map = self.catalog.preprocessing_options("all", "all", "all")
        options = sorted(self.preprocessing_option_map)
        self.preprocessing_choice.configure(values=options)
        if self.preprocessing_choice_var.get() not in self.preprocessing_option_map:
            self.preprocessing_choice_var.set(options[0] if options else "")
        self._refresh_preprocessing_choice()
        return left_imminent_count, left_safe_count

    def _refresh_confusion_matrix_layout(self, form: str, region: str, channel: str) -> int:
        current_paths = self.catalog.confusion_matrix_files(form, region, channel, split="test")
        self._current_confusion_paths = current_paths
        current_count = self.current_confusion_images.show_images(current_paths)
        self.confusion_option_map = self.catalog.confusion_matrix_options(split="test")
        options = sorted(self.confusion_option_map)
        self.confusion_choice.configure(values=options)
        if self.confusion_choice_var.get() not in self.confusion_option_map:
            self.confusion_choice_var.set(options[0] if options else "")
        selected_path = self.confusion_option_map.get(self.confusion_choice_var.get())
        self.target_confusion_images.show_images([selected_path] if selected_path else [])
        return current_count

    def _refresh_confusion_matrix_selection(self) -> None:
        selected_path = self.confusion_option_map.get(self.confusion_choice_var.get())
        self.target_confusion_images.show_images([selected_path] if selected_path else [])

    def _refresh_error_by_thickness_layout(self, form: str, region: str, channel: str) -> int:
        current_paths = self.catalog.error_by_thickness_files(form, region, channel)
        self._current_thickness_paths = current_paths
        current_count = self.current_thickness_images.show_images(current_paths)
        self.thickness_option_map = self.catalog.error_by_thickness_options()
        options = sorted(self.thickness_option_map)
        self.thickness_choice.configure(values=options)
        if self.thickness_choice_var.get() not in self.thickness_option_map:
            self.thickness_choice_var.set(options[0] if options else "")
        selected_path = self.thickness_option_map.get(self.thickness_choice_var.get())
        self.target_thickness_images.show_images([selected_path] if selected_path else [])
        return current_count

    def _refresh_error_by_thickness_selection(self) -> None:
        selected_path = self.thickness_option_map.get(self.thickness_choice_var.get())
        self.target_thickness_images.show_images([selected_path] if selected_path else [])

    def refresh(self) -> None:
        self.catalog.reload()
        form = self.form_var.get()
        region = self.region_var.get()
        channel = self.channel_var.get()
        rows = self.catalog.filtered_rows(form, region, channel)
        self._fill_summary(rows)
        pre_imminent_count, pre_safe_count = self._refresh_preprocessing_layout(form, region, channel)
        confusion_count = self._refresh_confusion_matrix_layout(form, region, channel)
        thickness_count = self._refresh_error_by_thickness_layout(form, region, channel)
        if form == "all":
            self.emd_images.clear("请选择一种具体预处理方式后查看 EMD 分量图。")
            curve_paths = self.catalog.training_curve_files(form, region, channel)
            curve_count = self.training_curve_images.show_images(curve_paths)
            self.status_var.set(
                f"已加载 {len(rows)} 组实验摘要；预处理图 label=0/{pre_imminent_count}、"
                f"label=1/{pre_safe_count}，训练曲线 {curve_count} 张，错分厚度分布 {thickness_count} 张。"
            )
            self.summary_hint.configure(text="当前显示所有可用实验的 summary；右侧可通过选择栏查看具体预处理组合。")
            self._update_train_button()
            return
        emd_paths = self.catalog.image_files("emd", form, region, channel, "all")
        curve_paths = self.catalog.training_curve_files(form, region, channel)
        curve_count = self.training_curve_images.show_images(curve_paths)
        emd_count = self.emd_images.show_images(emd_paths)
        self.status_var.set(
            f"当前筛选 {len(rows)} 组实验；预处理图 label=0/{pre_imminent_count}、"
            f"label=1/{pre_safe_count}，训练曲线 {curve_count} 张、EMD 图 {emd_count} 张，"
            f"错分厚度分布 {thickness_count} 张。"
        )
        self.summary_hint.configure(
            text="左侧显示当前筛选条件下的 label=0/1；右侧可通过选择栏查看任意已生成组合。"
        )
        self._update_train_button()

    def _update_train_button(self) -> None:
        if self.training_active or self.channel_var.get() == "all":
            self.train_button.configure(state="disabled")
        else:
            self.train_button.configure(state="normal")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_controls_during_training(self, active: bool) -> None:
        self.training_active = active
        state = "disabled" if active else "readonly"
        for combo in (self.form_combo, self.region_combo, self.channel_combo):
            combo.configure(state=state)
        self._update_train_button()

    def _build_training_commands(self) -> list[list[str]]:
        form = self.form_var.get()
        region = self.region_var.get()
        channel = self.channel_var.get()
        if channel == "all":
            raise ValueError("训练时请选择具体通道 1、2 或 3。")
        training_command = [
            sys.executable,
            str(APP_DIR / "run_emd_experiments.py"),
            "--forms",
            form,
            "--region-set",
            region,
            "--channel-mode",
            channel,
            "--output-dir",
            str(EXPERIMENTS_DIR),
        ]
        commands = [training_command]
        if self.auto_visualize_var.get():
            commands.extend(
                [
                    [
                        sys.executable,
                        str(APP_DIR / "visualize_preprocessing.py"),
                        "--forms",
                        form,
                        "--region-set",
                        region,
                        "--channel-mode",
                        channel,
                        "--output-dir",
                        str(VISUALIZATIONS_DIR / "preprocessing"),
                    ],
                    [
                        sys.executable,
                        str(APP_DIR / "visualize_emd_components.py"),
                        "--forms",
                        form,
                        "--region-set",
                        region,
                        "--channel-mode",
                        channel,
                        "--output-dir",
                        str(VISUALIZATIONS_DIR / "emd"),
                    ],
                    [
                        sys.executable,
                        str(APP_DIR / "visualize_training_curves.py"),
                        "--experiments-dir",
                        str(EXPERIMENTS_DIR),
                        "--output-dir",
                        str(VISUALIZATIONS_DIR / "training_curves"),
                        "--forms",
                        form,
                        "--region-set",
                        region,
                        "--channel-mode",
                        channel,
                    ],
                    [
                        sys.executable,
                        str(APP_DIR / "visualize_confusion_matrix.py"),
                        "--experiments-dir",
                        str(EXPERIMENTS_DIR),
                        "--output-dir",
                        str(VISUALIZATIONS_DIR / "confusion_matrices"),
                        "--forms",
                        form,
                        "--region-set",
                        region,
                        "--channel-mode",
                        channel,
                        "--split",
                        "test",
                    ],
                    [
                        sys.executable,
                        str(APP_DIR / "visualize_error_by_thickness.py"),
                        "--experiments-dir",
                        str(EXPERIMENTS_DIR),
                        "--output-dir",
                        str(VISUALIZATIONS_DIR / "error_by_thickness"),
                        "--forms",
                        form,
                        "--region-set",
                        region,
                        "--channel-mode",
                        channel,
                    ],
                ]
            )
        return commands

    def start_training(self) -> None:
        if self.training_active:
            return
        try:
            commands = self._build_training_commands()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        self._set_controls_during_training(True)
        self._append_log("=" * 70)
        self._append_log(
            f"开始训练：form={self.form_var.get()}, region={self.region_var.get()}, "
            f"channel={self.channel_var.get()}"
        )
        if self.auto_visualize_var.get():
            self._append_log("训练完成后将自动生成预处理图、EMD 分量图、训练曲线图和混淆矩阵图。")
        threading.Thread(
            target=self._training_worker,
            args=(commands,),
            daemon=True,
        ).start()
        self.after(100, self._poll_training_queue)

    def _training_worker(self, commands: list[list[str]]) -> None:
        for command_index, command in enumerate(commands, start=1):
            display_command = " ".join(f'"{part}"' if " " in part else part for part in command)
            self.training_queue.put(("log", f"[{command_index}/{len(commands)}] {display_command}"))
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(APP_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                self.training_queue.put(("error", f"无法启动命令：{exc}"))
                return
            assert process.stdout is not None
            for line in process.stdout:
                self.training_queue.put(("log", line.rstrip()))
            return_code = process.wait()
            if return_code != 0:
                self.training_queue.put(("error", f"命令失败，退出码：{return_code}"))
                return
        self.training_queue.put(("done", "训练及后处理完成。"))

    def _poll_training_queue(self) -> None:
        finished = False
        try:
            while True:
                kind, payload = self.training_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "error":
                    self._append_log(f"ERROR: {payload}")
                    self.status_var.set(str(payload))
                    finished = True
                elif kind == "done":
                    self._append_log(str(payload))
                    self.status_var.set(str(payload))
                    finished = True
        except queue.Empty:
            pass
        if finished:
            self._set_controls_during_training(False)
            self.catalog.reload()
            self.refresh()
            return
        if self.training_active:
            self.after(100, self._poll_training_queue)

    def _fill_summary(self, rows: list[dict[str, Any]]) -> None:
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        for row in rows:
            metrics = row["metrics"]
            test = metrics.get("test", {})
            train = metrics.get("train", {})
            val = metrics.get("val", {})
            values = (
                row["name"],
                row["feature_dim"],
                "-" if row["best_epoch"] is None else row["best_epoch"],
                "-" if row["trained_epochs"] is None else row["trained_epochs"],
                _metric(train, "auc"),
                _metric(val, "auc"),
                _metric(test, "auc"),
                _metric(test, "accuracy"),
                _metric(test, "f1_score"),
                _metric(test, "sensitivity"),
                _metric(test, "specificity"),
            )
            self.summary_tree.insert("", "end", values=values)

    def _open_experiment_dir(self) -> None:
        try:
            import os

            os.startfile(str(self.catalog.experiments_dir))
        except (AttributeError, OSError):
            self.status_var.set(f"实验目录：{self.catalog.experiments_dir}")


def main() -> None:
    app = EMDViewerApp()
    app.mainloop()


if __name__ == "__main__":
    main()

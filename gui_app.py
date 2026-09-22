"""Tkinter viewer for EMD experiments and generated visualizations.

The GUI reads ``experiments/`` and ``visualizations/``.  Select ``all`` to
compare available experiment summaries, or select one frame-processing form
to inspect its metrics, training curve, preprocessing plots, and EMD component
plots.  The training button runs the existing training script in a background
worker for the selected concrete channel.

A dataset selector scopes everything to one label scheme.  ``raw_data`` is the
historical 1.0 mm binary split; datasets written under ``raw_data_relabeled/``
(2-class 1.3 mm, 3-class 0.8/1.2 mm, ...) are discovered from their
``label_scheme.json`` and trained, visualised, and browsed independently.
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import tkinter as tk
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Any, Iterable, Sequence

from PIL import Image, ImageTk

from emd_pipeline import (
    class_display_names,
    label_scheme_tag,
    label_thresholds,
    read_label_scheme,
)


APP_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = APP_DIR / "experiments"
VISUALIZATIONS_DIR = APP_DIR / "visualizations"
RAW_DATA_DIR = APP_DIR / "raw_data"
RELABELED_DATA_DIR = APP_DIR / "raw_data_relabeled"
# ``raw_data`` predates ``label_scheme.json``; its stored rule is
# ``label = int(depth_value >= 1.0)``, re-verified against the stored labels.
DEFAULT_BINARY_THRESHOLDS: tuple[float, ...] = (1.0,)
CLASS_NUMBER_WORDS = {2: "二分类", 3: "三分类", 4: "四分类"}

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


@dataclass(frozen=True)
class DatasetProfile:
    """One selectable training dataset, described by its label scheme.

    ``tag`` is the dataset's label-scheme marker: ``None`` for the historical
    ``raw_data`` binary split, otherwise the tag stored in
    ``label_scheme.json``.  It is also the sub-directory name used to keep the
    per-dataset visualisations apart and the suffix appended to experiment and
    image file names by the command-line scripts.
    """

    data_dir: Path
    num_classes: int
    thresholds: tuple[float, ...]
    tag: str | None

    @property
    def is_default(self) -> bool:
        """True when the dataset has no label scheme (the historical split)."""

        return self.tag is None

    @property
    def display(self) -> str:
        """Combo-box label, e.g. ``三分类 · 阈值 0.8 / 1.2 mm · cls3_thr0.8_1.2``."""

        kind = CLASS_NUMBER_WORDS.get(self.num_classes, f"{self.num_classes} 分类")
        thresholds = " / ".join(str(float(value)) for value in self.thresholds)
        if self.is_default:
            return f"默认 · {kind} · depth >= {thresholds} mm"
        return f"{kind} · 阈值 {thresholds} mm · {self.data_dir.name}"

    @property
    def class_names(self) -> list[str]:
        """One display name per label, preferring the recorded scheme."""

        return class_display_names(read_label_scheme(self.data_dir), self.num_classes)

    @property
    def class_tokens(self) -> list[str]:
        """File-name tokens used by the preprocessing/EMD visualisations.

        Mirrors ``visualize_preprocessing.class_file_token``: the historical
        binary pair keeps ``imminent``/``safe``, a k-class dataset uses
        ``label0``..``label{k-1}``.
        """

        if self.num_classes <= 2:
            return ["imminent", "safe"]
        return [f"label{index}" for index in range(self.num_classes)]


def discover_datasets() -> list[DatasetProfile]:
    """Every dataset the GUI can train and browse, historical default first.

    Candidates are ``raw_data`` plus each sub-directory of
    ``raw_data_relabeled/`` that carries a ``label_scheme.json``.  Directory
    order is stable so the default dataset stays selected across reloads.
    """

    candidates: list[Path] = [RAW_DATA_DIR]
    if RELABELED_DATA_DIR.exists():
        candidates.extend(
            sorted(
                directory
                for directory in RELABELED_DATA_DIR.iterdir()
                if directory.is_dir() and (directory / "label_scheme.json").exists()
            )
        )
    profiles: list[DatasetProfile] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        scheme = read_label_scheme(resolved)
        thresholds = tuple(label_thresholds(scheme)) if scheme else DEFAULT_BINARY_THRESHOLDS
        num_classes = len(thresholds) + 1 if thresholds else 2
        profiles.append(
            DatasetProfile(
                data_dir=resolved,
                num_classes=num_classes,
                thresholds=thresholds,
                tag=label_scheme_tag(scheme),
            )
        )
    return profiles


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


def _row_num_classes(config: Any, metrics: Any) -> int:
    """Class count of a saved run, defaulting to the historical binary pair.

    Newer ``config.json``/``metrics.json`` files record ``num_classes``
    directly; older binary runs only carry the ``tn``/``fp``/``fn``/``tp``
    quadruple, which is why the fallback is 2 rather than an error.
    """

    if isinstance(config, dict) and config.get("num_classes") is not None:
        try:
            return max(2, int(config["num_classes"]))
        except (TypeError, ValueError):
            pass
    if isinstance(metrics, dict):
        if metrics.get("num_classes") is not None:
            try:
                return max(2, int(metrics["num_classes"]))
            except (TypeError, ValueError):
                pass
        for block in metrics.values():
            if not isinstance(block, dict):
                continue
            matrix = block.get("confusion_matrix")
            if isinstance(matrix, (list, tuple)) and len(matrix) >= 2:
                return len(matrix)
    return 2


class ExperimentCatalog:
    """Read summary/metric files and find matching generated image files.

    Every lookup is scoped to :attr:`dataset`: experiments recorded with a
    different ``label_scheme`` tag are invisible, and (when the dataset is not
    the historical default) images are read from the dataset's own
    ``<kind>/<tag>/`` sub-directory, so nothing leaks between label schemes.
    """

    def __init__(
        self,
        experiments_dir: Path,
        visualizations_dir: Path,
        dataset: DatasetProfile | None = None,
    ) -> None:
        self.experiments_dir = experiments_dir
        self.visualizations_dir = visualizations_dir
        self.dataset = dataset
        self.summary: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []

    def set_dataset(self, dataset: DatasetProfile | None) -> None:
        self.dataset = dataset

    @property
    def _dataset_tag(self) -> str | None:
        return None if self.dataset is None else self.dataset.tag

    def _kind_root(self, kind: str) -> Path:
        """Image root for one visualisation kind, scoped to the dataset.

        The historical binary artefacts live directly under
        ``visualizations/<kind>/`` and keep that location; a relabelled dataset
        gets ``visualizations/<kind>/<tag>/`` so the two never collide.
        """

        root = self.visualizations_dir / kind
        tag = self._dataset_tag
        return root if tag is None else root / tag

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
            label_scheme = config.get("label_scheme") if isinstance(config, dict) else None
            rows.append(
                {
                    "name": directory.name,
                    "directory": directory,
                    "label_tag": label_scheme_tag(
                        label_scheme if isinstance(label_scheme, dict) else None
                    ),
                    "num_classes": _row_num_classes(config, metrics),
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
        # Experiments carry the label-scheme tag of the dataset they were
        # trained on; a 3-class run is not an "alternative" of a binary one.
        if self.dataset is not None:
            rows = [row for row in rows if row["label_tag"] == self.dataset.tag]
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
        class_filter: str = "all",
        class_tokens: Sequence[str] | None = None,
    ) -> list[Path]:
        """Find new channel-aware files, with fallback to legacy filenames.

        ``class_filter`` selects one class token; ``"all"`` expands to every
        token of the current dataset, so a 3-class dataset collects
        ``label0``..``label2`` instead of the historical ``imminent``/``safe``.
        """

        root = self._kind_root(kind)
        if not root.exists() or form == "all":
            return []
        if class_filter == "all":
            classes = list(class_tokens) if class_tokens else ["imminent", "safe"]
        else:
            classes = [class_filter]
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

        root = self._kind_root("preprocessing")
        pattern = re.compile(
            r"^form_(?P<form>.+?)__regions_(?P<region>.+?)"
            r"(?:__channels_(?P<channel>[123]))?"
            r"__class_(?P<class>imminent|safe|label\d+)(?:__.+)?\.png$"
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
        root = self._kind_root("training_curves")
        if not root.exists():
            return []
        form_part = "*" if form == "all" else form
        region_part = "*" if region == "all" else region
        channel_part = "*" if channel == "all" else channel
        tag = self._dataset_tag
        suffix = "" if tag is None else f"__{tag}"
        pattern = (
            f"form_{form_part}__regions_{region_part}__channels_{channel_part}{suffix}.png"
        )
        return sorted(root.glob(pattern))

    def confusion_matrix_files(
        self, form: str, region: str, channel: str, split: str = "test"
    ) -> list[Path]:
        root = self._kind_root("confusion_matrices")
        if not root.exists() or form == "all":
            return []
        form_part = form
        region_part = "*" if region == "all" else region
        channel_part = "*" if channel == "all" else channel
        tag = self._dataset_tag
        suffix = "" if tag is None else f"__{tag}"
        pattern = (
            f"form_{form_part}__regions_{region_part}__channels_{channel_part}"
            f"__split_{split}{suffix}.png"
        )
        return sorted(root.glob(pattern))

    def confusion_matrix_options(self, split: str = "test") -> dict[str, Path]:
        """Return all generated confusion matrices for the target selector."""

        root = self._kind_root("confusion_matrices")
        pattern = re.compile(
            rf"^form_(?P<form>.+?)__regions_(?P<region>.+?)"
            rf"__channels_(?P<channel>[123])__split_{re.escape(split)}(?:__.+)?\.png$"
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

    def error_by_thickness_files(self, form: str, region: str, channel: str) -> list[Path]:
        root = self._kind_root("error_by_thickness")
        if not root.exists():
            return []
        form_part = "*" if form == "all" else form
        region_part = "*" if region == "all" else region
        channel_part = "*" if channel == "all" else channel
        tag = self._dataset_tag
        suffix = "" if tag is None else f"__{tag}"
        pattern = (
            f"form_{form_part}__regions_{region_part}__channels_{channel_part}{suffix}.png"
        )
        return sorted(root.glob(pattern))

    def error_by_thickness_options(self) -> dict[str, Path]:
        """Return all generated thickness-vs-error charts for the target selector."""

        root = self._kind_root("error_by_thickness")
        pattern = re.compile(
            r"^form_(?P<form>.+?)__regions_(?P<region>.+?)"
            r"__channels_(?P<channel>[123])(?:__.+)?\.png$"
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
    """Show current/target image column pairs with synchronized scrolling.

    The binary datasets render the historical four columns; a dataset with
    ``k`` classes renders ``2 * k`` columns.
    """

    def __init__(self, parent: tk.Misc, titles: Iterable[str]) -> None:
        super().__init__(parent)
        self._titles = list(titles)
        if not self._titles:
            raise ValueError("At least one image column is required")
        if len(self._titles) % 2:
            raise ValueError("Image columns must come in current/target pairs")
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
        if len(columns) != len(self._titles):
            raise ValueError(
                f"Expected {len(self._titles)} image columns, got {len(columns)}"
            )
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
        self.geometry("1520x920")
        self.minsize(1100, 700)
        self.datasets = discover_datasets()
        self.dataset_map = {profile.display: profile for profile in self.datasets}
        self.catalog = ExperimentCatalog(
            EXPERIMENTS_DIR,
            VISUALIZATIONS_DIR,
            self.datasets[0] if self.datasets else None,
        )
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
            text=(
                "选择数据集决定使用哪套厚度阈值划分；再选预处理方式/区域/通道查看指标、"
                "预处理信号和 EMD 分量。"
            ),
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        controls = ttk.Frame(self, padding=(16, 4, 16, 10))
        controls.pack(fill="x")
        self.dataset_var = tk.StringVar(
            value=self.datasets[0].display if self.datasets else ""
        )
        self.form_var = tk.StringVar(value="all")
        self.region_var = tk.StringVar(value="all")
        self.channel_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="")
        self._add_combo(
            controls, "数据集", self.dataset_var, list(self.dataset_map), 0, "dataset", width=32
        )
        self._add_combo(controls, "预处理方式", self.form_var, list(FORM_LABELS), 2, "form")
        self._add_combo(controls, "区域组合", self.region_var, list(REGION_LABELS), 4, "region")
        self._add_combo(controls, "通道", self.channel_var, ["all", "1", "2", "3"], 6, "channel")
        ttk.Button(controls, text="刷新", command=self.refresh).grid(row=0, column=8, padx=(18, 4))
        ttk.Button(controls, text="打开实验目录", command=self._open_experiment_dir).grid(
            row=0, column=9, padx=4
        )
        self.train_button = ttk.Button(controls, text="开始训练", command=self.start_training)
        self.train_button.grid(row=0, column=10, padx=4)
        self.auto_visualize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="训练后生成图像",
            variable=self.auto_visualize_var,
        ).grid(row=0, column=11, padx=(4, 0))
        for column in (1, 3, 5, 7):
            controls.columnconfigure(column, weight=1)
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_dataset_change())
        self.form_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        self.region_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        self.channel_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        ttk.Label(controls, textvariable=self.status_var, style="Hint.TLabel").grid(
            row=1, column=0, columnspan=12, sticky="w", pady=(8, 0)
        )

    def _add_combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        column: int,
        attribute_name: str,
        width: int = 20,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=0, column=column, sticky="w", padx=(0, 6))
        combo = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
            width=width,
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
        # Column count follows the dataset: 2 classes -> 4 columns (the
        # historical layout), k classes -> 2k columns.
        self._preprocessing_titles: tuple[str, ...] = tuple(
            self._preprocessing_column_titles(self.catalog.dataset)
        )
        self.preprocessing_columns = SynchronizedImageColumns(
            self.preprocessing_tab,
            self._preprocessing_titles,
        )
        self.preprocessing_columns.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.preprocessing_option_map: dict[str, dict[str, Path]] = {}
        self._current_preprocessing_paths: list[list[Path]] = []

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

    def _preprocessing_column_titles(self, profile: DatasetProfile | None) -> list[str]:
        """Column titles for the current/target pairs of one dataset.

        A binary dataset yields the historical four titles; a k-class dataset
        yields ``当前组合``/``目标组合`` for every label.
        """

        num_classes = 2 if profile is None else profile.num_classes
        names = ["即将穿透", "安全"] if profile is None else profile.class_names
        titles: list[str] = []
        for prefix in ("当前组合", "目标组合"):
            for label in range(num_classes):
                name = names[label] if label < len(names) else f"label={label}"
                titles.append(f"{prefix}：label={label}（{name}）")
        return titles

    def _class_tokens(self) -> list[str]:
        """File-name tokens of the current dataset's classes."""

        profile = self.catalog.dataset
        return ["imminent", "safe"] if profile is None else profile.class_tokens

    def _rebuild_preprocessing_columns(self, profile: DatasetProfile | None) -> None:
        """Recreate the side-by-side panel when the class count changes."""

        titles = tuple(self._preprocessing_column_titles(profile))
        if titles == self._preprocessing_titles:
            return
        self._preprocessing_titles = titles
        self.preprocessing_columns.destroy()
        self.preprocessing_columns = SynchronizedImageColumns(self.preprocessing_tab, titles)
        self.preprocessing_columns.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _on_dataset_change(self) -> None:
        profile = self.dataset_map.get(self.dataset_var.get())
        self.catalog.set_dataset(profile)
        self._rebuild_preprocessing_columns(profile)
        self.refresh()

    def _refresh_preprocessing_choice(self) -> None:
        selected = self.preprocessing_choice_var.get()
        pair = self.preprocessing_option_map.get(selected, {})
        tokens = self._class_tokens()
        current = self._current_preprocessing_paths
        columns: list[list[Path]] = [
            list(current[index]) if index < len(current) else []
            for index in range(len(tokens))
        ]
        for token in tokens:
            path = pair.get(token)
            columns.append([path] if path is not None else [])
        self.preprocessing_columns.show_columns(columns)

    def _refresh_preprocessing_layout(self, form: str, region: str, channel: str) -> list[int]:
        tokens = self._class_tokens()
        per_class = [
            self.catalog.image_files("preprocessing", form, region, channel, token)
            for token in tokens
        ]
        self._current_preprocessing_paths = per_class

        # The target selector is intentionally independent of the top-level
        # current-result filters, so it can compare any generated form,
        # region, and channel combination against the current columns.
        self.preprocessing_option_map = self.catalog.preprocessing_options("all", "all", "all")
        options = sorted(self.preprocessing_option_map)
        self.preprocessing_choice.configure(values=options)
        if self.preprocessing_choice_var.get() not in self.preprocessing_option_map:
            self.preprocessing_choice_var.set(options[0] if options else "")
        self._refresh_preprocessing_choice()
        return [len(paths) for paths in per_class]

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
        pre_counts = self._refresh_preprocessing_layout(form, region, channel)
        pre_summary = "、".join(
            f"label={index}/{count}" for index, count in enumerate(pre_counts)
        )
        confusion_count = self._refresh_confusion_matrix_layout(form, region, channel)
        thickness_count = self._refresh_error_by_thickness_layout(form, region, channel)
        if form == "all":
            self.emd_images.clear("请选择一种具体预处理方式后查看 EMD 分量图。")
            curve_paths = self.catalog.training_curve_files(form, region, channel)
            curve_count = self.training_curve_images.show_images(curve_paths)
            self.status_var.set(
                f"数据集：{self.dataset_var.get()}｜已加载 {len(rows)} 组实验摘要；"
                f"预处理图 {pre_summary}，训练曲线 {curve_count} 张，"
                f"错分厚度分布 {thickness_count} 张。"
            )
            self.summary_hint.configure(
                text="当前显示所选数据集下所有可用实验的 summary；右侧可通过选择栏查看具体预处理组合。"
            )
            self._update_train_button()
            return
        emd_paths = self.catalog.image_files(
            "emd", form, region, channel, "all", self._class_tokens()
        )
        curve_paths = self.catalog.training_curve_files(form, region, channel)
        curve_count = self.training_curve_images.show_images(curve_paths)
        emd_count = self.emd_images.show_images(emd_paths)
        self.status_var.set(
            f"数据集：{self.dataset_var.get()}｜当前筛选 {len(rows)} 组实验；"
            f"预处理图 {pre_summary}，训练曲线 {curve_count} 张、EMD 图 {emd_count} 张，"
            f"错分厚度分布 {thickness_count} 张。"
        )
        self.summary_hint.configure(
            text="左侧显示当前筛选条件下的各类别样本；右侧可通过选择栏查看任意已生成组合。"
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
        for combo in (
            self.dataset_combo,
            self.form_combo,
            self.region_combo,
            self.channel_combo,
        ):
            combo.configure(state=state)
        self._update_train_button()

    def _build_training_commands(self) -> list[list[str]]:
        form = self.form_var.get()
        region = self.region_var.get()
        channel = self.channel_var.get()
        if channel == "all":
            raise ValueError("训练时请选择具体通道 1、2 或 3。")
        profile = self.catalog.dataset
        data_dir = RAW_DATA_DIR if profile is None else profile.data_dir
        num_classes = 2 if profile is None else profile.num_classes
        tag = None if profile is None else profile.tag

        def _scoped(kind: str) -> Path:
            """Output dir for one figure kind, scoped per dataset tag."""

            root = VISUALIZATIONS_DIR / kind
            return root if tag is None else root / tag

        def _class_label_args() -> list[str]:
            """Explicit label list, only needed once there are 3+ classes."""

            if num_classes <= 2:
                return []
            return ["--class-labels", ",".join(str(index) for index in range(num_classes))]

        training_command = [
            sys.executable,
            str(APP_DIR / "run_emd_experiments.py"),
            "--data-dir",
            str(data_dir),
            "--num-classes",
            str(num_classes),
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
                        "--data-dir",
                        str(data_dir),
                        "--forms",
                        form,
                        "--region-set",
                        region,
                        "--channel-mode",
                        channel,
                        "--output-dir",
                        str(_scoped("preprocessing")),
                        *_class_label_args(),
                    ],
                    [
                        sys.executable,
                        str(APP_DIR / "visualize_emd_components.py"),
                        "--data-dir",
                        str(data_dir),
                        "--forms",
                        form,
                        "--region-set",
                        region,
                        "--channel-mode",
                        channel,
                        "--output-dir",
                        str(_scoped("emd")),
                        *_class_label_args(),
                    ],
                    [
                        sys.executable,
                        str(APP_DIR / "visualize_training_curves.py"),
                        "--experiments-dir",
                        str(EXPERIMENTS_DIR),
                        "--data-dir",
                        str(data_dir),
                        "--output-dir",
                        str(_scoped("training_curves")),
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
                        "--data-dir",
                        str(data_dir),
                        "--output-dir",
                        str(_scoped("confusion_matrices")),
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
                        "--data-dir",
                        str(data_dir),
                        "--output-dir",
                        str(_scoped("error_by_thickness")),
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
            f"开始训练：dataset={self.dataset_var.get()}, form={self.form_var.get()}, "
            f"region={self.region_var.get()}, channel={self.channel_var.get()}"
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

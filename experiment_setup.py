"""Input choices shared by ViT and the retained EMD experiment entry points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emd_pipeline import FORM_ORDER, RegionSpec, canonical_form, json_ready, parse_regions


REGION_PRESETS: dict[str, list[list[float]]] = {
    "full": [[0.0, 5.0]],
    "bone": [[1.0, 3.0]],
    "bone_plus_post": [[0.2, 1.5], [1.5, 5.0]],
}
DYNAMIC_REGION_NAME = "dyn_envelope"
REGION_CHOICES = ["all", *REGION_PRESETS, DYNAMIC_REGION_NAME]


def parse_forms(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(FORM_ORDER)
    forms = [canonical_form(item) for item in value.split(",") if item.strip()]
    if not forms:
        raise ValueError("At least one frame form is required")
    return [form for form in FORM_ORDER if form in forms]


def region_experiments(
    region_set: str, regions_json: str | None = None
) -> list[tuple[str, list[RegionSpec]]]:
    if regions_json is not None:
        return [("custom", parse_regions(regions_json))]
    names = [*REGION_PRESETS, DYNAMIC_REGION_NAME] if region_set == "all" else [region_set]
    if any(name not in REGION_CHOICES for name in names):
        raise ValueError(f"unknown region set: {region_set}")
    return [
        (name, [] if name == DYNAMIC_REGION_NAME else parse_regions(REGION_PRESETS[name]))
        for name in names
    ]


def infer_num_classes(data_dir: Path, max_samples: int | None = None) -> int:
    """Infer the class count from development labels, without reading test labels."""
    highest = -1
    for split in ("train", "val"):
        path = data_dir / split / "y.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing label file: {path}")
        labels = np.load(path)
        if max_samples is not None:
            labels = labels[:max_samples]
        if labels.size:
            highest = max(highest, int(labels.max()))
    if highest < 0:
        raise ValueError(f"no train/val labels found under {data_dir}")
    return highest + 1


def guard_existing_experiment(output_dir: Path, config: dict[str, object]) -> None:
    """Reject a different run before overwriting any artifacts."""
    path = output_dir / "config.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != json_ready(config):
        raise SystemExit(
            f"{path} records a different run; select another --output-dir "
            "to keep both results"
        )

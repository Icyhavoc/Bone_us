"""Visualize every EMD component for one sample from each class.

The script reuses the preprocessing/EMD functions in ``emd_pipeline.py`` and
the standard region presets in ``run_emd_experiments.py``.  It selects one
sample from each requested class and creates a component sheet for every
form/region/class combination.  For Raw 50 Frames, EMD is applied to every
frame/channel stream and the streams are overlaid in each component panel;
the dark curve is their mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from emd_pipeline import (
    FORM_ORDER,
    DepthMapper,
    DynamicEnvelopeConfig,
    EMDConfig,
    RegionSpec,
    canonical_form,
    emd_decompose,
    json_ready,
    parse_regions,
    prepare_branch_signals,
    prepare_dynamic_envelope_branches,
    process_frames,
    select_channels,
)
from run_emd_experiments import DYNAMIC_REGION_NAME, REGION_CHOICES, REGION_PRESETS
from visualize_preprocessing import (
    CLASS_NAMES,
    CHANNEL_COLORS,
    _draw_plot_cell,
    _font,
    _resolve_region_experiments,
    _safe_name,
    load_split_records,
    select_class_indices,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize EMD IMFs and residue")
    parser.add_argument("--data-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/emd"))
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--forms", default="all", help="all or comma-separated forms")
    parser.add_argument("--region-set", choices=REGION_CHOICES, default="all")
    parser.add_argument("--regions-json", default=None)
    parser.add_argument("--channel-mode", type=int, choices=[1, 2, 3], default=3)
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
    parser.add_argument("--signal-length", type=int, default=896)
    parser.add_argument("--rounding", choices=["round", "floor", "ceil"], default="round")
    parser.add_argument("--max-imfs", type=int, default=5)
    parser.add_argument("--max-sift-iterations", type=int, default=30)
    parser.add_argument("--sift-sd-threshold", type=float, default=0.2)
    parser.add_argument("--include-residue", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _decompose_streams(streams: np.ndarray, config: EMDConfig) -> np.ndarray:
    """Return components as ``[component, stream, samples]`` with zero padding."""

    component_count = config.max_imfs + int(config.include_residue)
    output = np.zeros((component_count, streams.shape[0], streams.shape[1]), dtype=np.float32)
    for stream_index, stream in enumerate(streams):
        imfs, residue = emd_decompose(
            stream,
            max_imfs=config.max_imfs,
            max_sift_iterations=config.max_sift_iterations,
            sift_sd_threshold=config.sift_sd_threshold,
        )
        for component_index, component in enumerate(imfs[: config.max_imfs]):
            output[component_index, stream_index] = component
        if config.include_residue:
            output[config.max_imfs, stream_index] = residue
    return output


def _component_y_limits(
    component_arrays: Sequence[np.ndarray],
) -> list[tuple[float, float]]:
    """Create symmetric per-component y-ranges shared by both classes."""

    if not component_arrays:
        return []
    component_count = component_arrays[0].shape[0]
    limits: list[tuple[float, float]] = []
    for component_index in range(component_count):
        max_abs = 0.0
        for array in component_arrays:
            max_abs = max(max_abs, float(np.max(np.abs(array[component_index]))))
        if max_abs < 1e-8:
            max_abs = 1.0
        max_abs *= 1.1
        limits.append((-max_abs, max_abs))
    return limits


def _make_component_sheet(
    form: str,
    region_name: str,
    regions: Sequence[RegionSpec],
    class_label: int,
    sample_index: int,
    record: dict[str, Any],
    split_name: str,
    branch_components: Sequence[np.ndarray],
    channel_mode: int,
    mapper: DepthMapper,
    target_length: int,
    component_limits: Sequence[Sequence[tuple[float, float]]],
    include_residue: bool,
    branch_ranges: Sequence[tuple[float, float]] | None,
    output_path: Path,
) -> None:
    cell_width = 500
    cell_height = 205
    left_margin = 115
    top_margin = 100
    component_count = branch_components[0].shape[0]
    branch_count = len(branch_components)
    width = left_margin + cell_width * branch_count
    height = top_margin + cell_height * component_count
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    class_name = CLASS_NAMES.get(class_label, f"label={class_label}")
    sample_id = record.get("sample_id", f"sample_{sample_index}")
    depth_value = record.get("depth_value", "?")
    draw.text(
        (16, 14),
        f"{form} | {region_name} | {class_name} | {sample_id}",
        fill=(15, 15, 15),
        font=_font(20, bold=True),
    )
    draw.text(
        (16, 46),
        f"depth={depth_value} mm | split={split_name} | channel_mode={channel_mode}",
        fill=(70, 70, 70),
        font=_font(13),
    )
    draw.text(
        (16, 68),
        "Light curves: individual streams; dark curve: stream mean",
        fill=(70, 70, 70),
        font=_font(12),
    )
    for column in range(branch_count):
        x0 = left_margin + column * cell_width
        if branch_ranges is None:
            branch_title = f"Branch {column + 1}: {regions[column].label}"
            start_mm, end_mm = regions[column].start_mm, regions[column].end_mm
        else:
            branch_title = (
                "Branch 1: first envelope-peak window (a samples)"
                if column == 0
                else "Branch 2: remaining signal"
            )
            start_mm, end_mm = branch_ranges[column]
        draw.text(
            (x0 + 8, 80),
            branch_title,
            fill=(20, 20, 20),
            font=_font(14, bold=True),
        )
        depth_axis = np.linspace(start_mm, end_mm, target_length)
        components = branch_components[column]
        for component_index in range(component_count):
            curves: list[tuple[np.ndarray, tuple[int, int, int], int]] = []
            streams = components[component_index]
            for stream_index in range(streams.shape[0]):
                if streams.shape[0] == 1:
                    color = CHANNEL_COLORS[0]
                else:
                    color = (172, 198, 220) if stream_index % 2 == 0 else (224, 181, 175)
                curves.append((streams[stream_index], color, 1))
            curves.append((np.mean(streams, axis=0), (20, 20, 20), 2))
            title = f"IMF {component_index + 1}"
            if include_residue and component_index == component_count - 1:
                title = "Residue" if component_count > 1 else "Component 1"
            y0 = top_margin + component_index * cell_height
            box = (
                left_margin + column * cell_width + 4,
                y0 + 4,
                left_margin + (column + 1) * cell_width - 4,
                y0 + cell_height - 4,
            )
            _draw_plot_cell(
                draw,
                box,
                curves,
                depth_axis,
                component_limits[column][component_index],
                title,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = _parse_args()
    forms = list(FORM_ORDER) if args.forms.strip().lower() == "all" else [
        canonical_form(item) for item in args.forms.split(",") if item.strip()
    ]
    region_experiments = _resolve_region_experiments(args.region_set, args.regions_json)
    x, labels, records, split_names = load_split_records(args.data_dir, args.split)
    selected = select_class_indices(
        labels,
        [args.imminent_label, args.safe_label],
        samples_per_class=1,
        seed=args.seed,
    )
    mapper = DepthMapper(args.max_depth_mm, args.signal_length, args.rounding)
    dynamic_envelope_config = DynamicEnvelopeConfig(
        branch_length_mm=args.dyn_a_mm,
        smooth_window=args.dyn_smooth_window,
        prominence_sigma=args.dyn_prominence_sigma,
        min_peak_width=args.dyn_min_peak_width,
    )
    emd_config = EMDConfig(
        max_imfs=args.max_imfs,
        max_sift_iterations=args.max_sift_iterations,
        sift_sd_threshold=args.sift_sd_threshold,
        include_residue=args.include_residue,
    )
    output_dir = args.output_dir.resolve()
    manifest: dict[str, Any] = {
        "data_dir": args.data_dir.resolve(),
        "split": args.split,
        "seed": args.seed,
        "class_labels": {"imminent": args.imminent_label, "safe": args.safe_label},
        "forms": forms,
        "regions": {name: regions for name, regions in region_experiments},
        "mapper": mapper,
        "target_length": args.target_length,
        "tukey_alpha": args.tukey_alpha,
        "dynamic_envelope": dynamic_envelope_config,
        "emd": emd_config,
        "selections": {},
    }
    for label, indices in selected.items():
        name = "imminent" if label == args.imminent_label else "safe"
        index = int(indices[0])
        manifest["selections"][name] = {
            "index": index,
            "sample_id": records[index].get("sample_id"),
            "depth_value": records[index].get("depth_value"),
            "split": split_names[index],
        }

    for form in forms:
        for region_name, regions in region_experiments:
            selected_components: dict[int, list[np.ndarray]] = {}
            selected_ranges: dict[int, list[tuple[float, float]] | None] = {}
            for label, indices in selected.items():
                index = int(indices[0])
                selected_channels = select_channels(x[index], args.channel_mode)
                processed, selected_frames = process_frames(
                    selected_channels,
                    form,
                    mapper=mapper,
                    score_region=RegionSpec(0.0, mapper.max_depth_mm, "selection_full"),
                )
                branch_components: list[np.ndarray] = []
                branch_ranges: list[tuple[float, float]] = []
                if region_name == DYNAMIC_REGION_NAME:
                    dynamic_branches, dynamic_info = prepare_dynamic_envelope_branches(
                        processed,
                        config=dynamic_envelope_config,
                        target_length=args.target_length,
                        tukey_alpha=args.tukey_alpha,
                        max_depth_mm=mapper.max_depth_mm,
                    )
                    branch_items = zip(dynamic_branches, dynamic_info["branches"])
                    for branch_flat, branch_info in branch_items:
                        branch_ranges.append(
                            (
                                mapper.index_to_depth(int(branch_info["start_index"])),
                                mapper.index_to_depth(int(branch_info["end_index_exclusive"])),
                            )
                        )
                        branch_components.append(
                            _decompose_streams(
                                branch_flat.reshape(-1, args.target_length), emd_config
                            )
                        )
                    manifest.setdefault("dynamic_envelope_details", {})[
                        f"{form}/{region_name}/{label}"
                    ] = dynamic_info
                else:
                    branch_items = ((
                        prepare_branch_signals(
                            processed,
                            region,
                            mapper,
                            args.target_length,
                            tukey_alpha=args.tukey_alpha,
                        ),
                        None,
                    ) for region in regions)
                    for branch_flat, _ in branch_items:
                        branch_components.append(
                            _decompose_streams(
                                branch_flat.reshape(-1, args.target_length), emd_config
                            )
                        )
                    branch_ranges = [(region.start_mm, region.end_mm) for region in regions]
                selected_components[label] = branch_components
                selected_ranges[label] = None if region_name != DYNAMIC_REGION_NAME else branch_ranges
                manifest.setdefault("frame_selections", {})[
                    f"{form}/{region_name}/{label}"
                ] = None if selected_frames is None else selected_frames.tolist()

            component_limits = [
                _component_y_limits(
                    [selected_components[label][branch_index] for label in selected_components]
                )
                for branch_index in range(len(selected_components[next(iter(selected_components))]))
            ]
            for label, indices in selected.items():
                class_name = "imminent" if label == args.imminent_label else "safe"
                index = int(indices[0])
                file_name = (
                    f"form_{form}__regions_{_safe_name(region_name)}__"
                    f"channels_{args.channel_mode}__class_{class_name}__"
                    f"sample_{_safe_name(str(records[index].get('sample_id', index)))}.png"
                )
                _make_component_sheet(
                    form,
                    region_name,
                    regions,
                    label,
                    index,
                    records[index],
                    split_names[index],
                    selected_components[label],
                    args.channel_mode,
                    mapper,
                    args.target_length,
                    component_limits,
                    args.include_residue,
                    selected_ranges[label],
                    output_dir / file_name,
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(json_ready(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"EMD component visualizations written to {output_dir}")


if __name__ == "__main__":
    main()

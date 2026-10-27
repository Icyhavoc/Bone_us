"""Arrange the existing signed waveform branches for the ViT input."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from emd_pipeline import (
    CHANNEL3_FRAME_SELECTION,
    RESAMPLING_CUBIC,
    DepthMapper,
    DynamicEnvelopeConfig,
    RegionSpec,
    canonical_form,
    channel_physical_indices,
    prepare_branch_block,
    prepare_dynamic_envelope_branches,
    prepare_frame_input,
)


def prepare_vit_sample(
    sample: np.ndarray,
    *,
    form: str,
    regions: Sequence[RegionSpec],
    channel_mode: int,
    mapper: DepthMapper,
    target_length: int = 512,
    tukey_alpha: float = 0.3,
    dynamic_envelope_config: DynamicEnvelopeConfig | None = None,
    point_id: str | None = None,
    frame_selection: str = CHANNEL3_FRAME_SELECTION,
    resampling_method: str = RESAMPLING_CUBIC,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return [region×channel, streams, length] and the per-sample audit."""

    physical_channels = channel_physical_indices(channel_mode)
    processed, selected_frames, span = prepare_frame_input(
        sample, form, channel_mode, point_id, dynamic_envelope_config, frame_selection
    )
    if dynamic_envelope_config is None:
        if not regions:
            raise ValueError("at least one region is required")
        blocks = [
            prepare_branch_block(
                processed, region, mapper, target_length, tukey_alpha,
                resampling_method=resampling_method,
            )
            for region in regions
        ]
        branch_info = []
        for region in regions:
            start, end = mapper.slice_bounds(region)
            branch_info.append({
                "name": region.label, "start_index": start,
                "end_index_exclusive": end,
            })
        dynamic_info = None
    else:
        if span is None:
            raise AssertionError("dynamic branches require a located span")
        blocks, dynamic_info = prepare_dynamic_envelope_branches(
            processed, span, dynamic_envelope_config, target_length, tukey_alpha,
            point_id=point_id, physical_channels=physical_channels,
            resampling_method=resampling_method,
        )
        branch_info = list(dynamic_info["branches"])
        for region in regions:
            start, end = mapper.slice_bounds(region)
            blocks.append(prepare_branch_block(
                processed, region, mapper, target_length, tukey_alpha,
                resampling_method=resampling_method,
            ))
            branch_info.append({
                "name": region.label, "start_index": start,
                "end_index_exclusive": end,
                "start_index_by_channel": [start] * len(physical_channels),
                "end_index_exclusive_by_channel": [end] * len(physical_channels),
                "input_length_by_channel": [end - start] * len(physical_channels),
            })
    inputs = np.stack(
        [block[:, channel, :] for block in blocks for channel in range(len(physical_channels))],
        axis=0,
    ).astype(np.float32, copy=False)
    if not np.isfinite(inputs).all():
        raise ValueError(f"non-finite ViT input for {point_id}")
    names = [
        f"region{index}_{branch['name']}_ch{physical + 1}"
        for index, branch in enumerate(branch_info)
        for physical in physical_channels
    ]
    audit: dict[str, Any] = {
        "point_id": point_id,
        "form": canonical_form(form),
        "channel_mode": channel_mode,
        "resampling_method": resampling_method,
        "selected_frames": None if selected_frames is None else selected_frames.tolist(),
        "channel3_selection": None if span is None else span.get("channel3_selection"),
        "shared_frame_selection": None if span is None else span.get("shared_frame_selection"),
        "channel4_selection": None if span is None else span.get("channel4_selection"),
        "selection_main_start_index_by_channel": None if span is None else span["starts"].astype(int).tolist(),
        "selection_main_end_exclusive_by_channel": None if span is None else span["ends"].astype(int).tolist(),
        "branch_names": names,
        "branches": branch_info,
    }
    if dynamic_info is not None:
        audit["dynamic_envelope"] = dynamic_info
    return inputs, audit


def build_vit_inputs(
    samples: np.ndarray,
    *,
    sample_ids: Sequence[str],
    form: str,
    regions: Sequence[RegionSpec],
    channel_mode: int,
    mapper: DepthMapper,
    target_length: int = 512,
    tukey_alpha: float = 0.3,
    dynamic_envelope_config: DynamicEnvelopeConfig | None = None,
    frame_selection: str = CHANNEL3_FRAME_SELECTION,
    resampling_method: str = RESAMPLING_CUBIC,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(samples) != len(sample_ids):
        raise ValueError("sample_ids must match the sample count")
    if len(samples) == 0:
        raise ValueError("the selected split is empty")
    prepared = [
        prepare_vit_sample(
            sample, form=form, regions=regions, channel_mode=channel_mode,
            mapper=mapper, target_length=target_length, tukey_alpha=tukey_alpha,
            dynamic_envelope_config=dynamic_envelope_config, point_id=str(point_id),
            frame_selection=frame_selection,
            resampling_method=resampling_method,
        )
        for sample, point_id in zip(samples, sample_ids)
    ]
    return np.stack([input_block for input_block, _ in prepared]), [audit for _, audit in prepared]

"""Compare dynamic main and padded tail geometry with ViT_reference.

All 268 points and both physical channels are checked against the reference's
absolute indices and pre-resampling signal arrays. Run ``python verify_dyn_envelope.py``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from emd_pipeline import (
    detect_dynamic_span,
    prepare_dynamic_envelope_branches,
    replace_adc_dc_level,
    DynamicEnvelopeConfig,
)

APP_DIR = Path(__file__).resolve().parent
RAW_DIR = APP_DIR / "raw_data"
REFERENCE_DIR = Path(
    os.environ.get("DYN_REFERENCE_DIR", APP_DIR.parent / "ViT_reference" / "dataset")
)
TARGET_LENGTH = 512


def load_raw_frames() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        split_dir = RAW_DIR / split
        frames = np.load(split_dir / "X.npy")
        samples = json.loads((split_dir / "samples.json").read_text(encoding="utf-8"))
        for index, meta in enumerate(samples):
            out[str(meta["sample_id"])] = frames[index]
    return out


def load_reference() -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        arrays = REFERENCE_DIR / split / "arrays"
        ids = np.load(arrays / "point_ids.npy", allow_pickle=False)
        main_index = np.load(arrays / "main_absolute_index.npy")
        main_signal = np.load(arrays / "main_signal.npy")
        tail_index = np.load(arrays / "tail_absolute_index.npy")
        tail_signal = np.load(arrays / "tail_signal.npy")
        for index, point in enumerate(ids):
            out[str(point)] = {
                "main_index": main_index[index],
                "main_signal": main_signal[index],
                "tail_index": tail_index[index],
                "tail_signal": tail_signal[index],
            }
    return out


def check_boundaries(
    raw: dict[str, np.ndarray],
    reference: dict[str, dict[str, np.ndarray]],
    config: DynamicEnvelopeConfig,
) -> list[str]:
    """Reproduce the detected boundary through the public locator path."""
    problems: list[str] = []
    for point in sorted(raw):
        span = detect_dynamic_span(replace_adc_dc_level(raw[point]), config, point)
        want_main = reference[point]["main_index"][:, 0]
        want_tail = reference[point]["tail_index"]
        for channel in range(2):
            if int(span["starts"][channel]) != int(want_main[channel]):
                problems.append(
                    f"{point} ch{channel}: start {int(span['starts'][channel])} "
                    f"!= reference {int(want_main[channel])}"
                )
            if int(span["ends"][channel]) != int(want_main[channel]) + config.main_length:
                problems.append(f"{point} ch{channel}: main window length is not fixed")
            expected_tail_index = np.full(config.tail_length, -1, dtype=np.int64)
            valid_length = int(span["tail_valid_ends"][channel] - span["tail_starts"][channel])
            expected_tail_index[:valid_length] = np.arange(
                int(span["tail_starts"][channel]), int(span["tail_valid_ends"][channel])
            )
            if not np.array_equal(want_tail[channel], expected_tail_index):
                problems.append(f"{point} ch{channel}: padded tail indices differ")
    return problems


def check_branch_content(
    raw: dict[str, np.ndarray],
    reference: dict[str, dict[str, np.ndarray]],
    config: DynamicEnvelopeConfig,
) -> list[str]:
    """Check all original slices and the prepared one-frame branch shapes."""
    problems: list[str] = []
    for point in sorted(raw):
        frames = replace_adc_dc_level(raw[point])
        span = detect_dynamic_span(frames, config, point)
        branches, info = prepare_dynamic_envelope_branches(
            frames[:1],
            span=span,
            config=config,
            target_length=TARGET_LENGTH,
            apply_tukey=False,
            point_id=point,
            physical_channels=(0, 1),
        )
        main_block, tail_block = branches
        stream_count, channel_count = 1, 2
        expected_shape = (stream_count, channel_count, TARGET_LENGTH)
        if main_block.shape != expected_shape:
            problems.append(f"{point}: unexpected main branch shape {main_block.shape}")
            continue
        if tail_block.shape != expected_shape:
            problems.append(f"{point}: unexpected tail branch shape {tail_block.shape}")
            continue
        starts = info["main_start_index_by_channel"]
        for channel in range(channel_count):
            start = int(starts[channel])
            if not np.isfinite(main_block[:, channel, :]).all() or not np.isfinite(tail_block[:, channel, :]).all():
                problems.append(f"{point} ch{channel}: non-finite branch samples")
            if not np.array_equal(
                reference[point]["main_signal"][:, channel, :],
                frames[:, channel, start : start + config.main_length],
            ):
                problems.append(f"{point} ch{channel}: reference main slice differs")
            tail_start = int(span["tail_starts"][channel])
            tail_end = int(span["tail_valid_ends"][channel])
            expected_tail = np.pad(
                frames[:, channel, tail_start:tail_end],
                ((0, 0), (0, int(span["tail_padding"][channel]))),
            )
            if not np.array_equal(reference[point]["tail_signal"][:, channel, :], expected_tail):
                problems.append(f"{point} ch{channel}: reference padded tail differs")
    return problems


def main() -> int:
    config = DynamicEnvelopeConfig()
    raw = load_raw_frames()
    reference = load_reference()
    if set(raw) != set(reference):
        print(f"point sets differ between {RAW_DIR.name} and {REFERENCE_DIR}")
        return 1
    print(f"points={len(raw)} channels={len(raw) * 2}")

    problems = check_boundaries(raw, reference, config)
    print(f"boundary check: {'PASS' if not problems else f'{len(problems)} FAIL'}")
    if not problems:
        problems = check_branch_content(raw, reference, config)
        print(f"branch content check: {'PASS' if not problems else f'{len(problems)} FAIL'}")

    for message in problems[:20]:
        print(f"  {message}")
    if problems:
        print(f"total problems: {len(problems)}")
        return 1
    print("dyn_envelope geometry and raw branch content match ViT_reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())

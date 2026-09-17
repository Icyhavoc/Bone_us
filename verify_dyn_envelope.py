"""Verify the ported ``dyn_envelope`` split against the reference output.

``after_split_data/`` was produced by ``reference_code/pipeline.py``.  The two
datasets share the same source traces; ``raw_data`` only differs by not
collapsing the ADC ``127``/``128`` DC pair onto the shared zero level, which
:func:`emd_pipeline.replace_adc_dc_level` reproduces before comparing.

Two levels are checked for all 268 points and both channels:

1. the detected boundary (``main_absolute_index``) is reproduced exactly;
2. the branch content is exactly a resampled slice of the source trace, with no
   zero padding and no window shifting.

Run: ``python verify_dyn_envelope.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from emd_pipeline import (
    _hilbert_envelope,
    _moving_average_nearest,
    prepare_dynamic_envelope_branches,
    locate_dynamic_span,
    locator_mean_signal,
    replace_adc_dc_level,
    resample_signals,
    select_channels,
    DynamicEnvelopeConfig,
)

APP_DIR = Path(__file__).resolve().parent
RAW_DIR = APP_DIR / "raw_data"
REFERENCE_DIR = APP_DIR / "after_split_data"
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
        frames = select_channels(replace_adc_dc_level(raw[point]), 3)
        locator = locator_mean_signal(frames, config.locator_top_k)
        envelope = _hilbert_envelope(locator).astype(np.float32)
        smoothed = np.zeros_like(envelope)
        smoothed[:, config.noise_start :] = _moving_average_nearest(
            envelope[:, config.noise_start :], config.smooth_window
        )
        span = locate_dynamic_span(smoothed, config=config, point_id=point)
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
        if not (want_tail[0, 0] == config.tail_start and want_tail[0, -1] == 895):
            problems.append(f"{point}: reference tail interval is not [251, 896)")
        if int(span["tail_start"]) != config.tail_start or int(span["tail_end"]) != 896:
            problems.append(f"{point}: tail interval does not match the reference rule")
    return problems


def check_branch_content(
    raw: dict[str, np.ndarray],
    reference: dict[str, dict[str, np.ndarray]],
    config: DynamicEnvelopeConfig,
) -> list[str]:
    """Confirm branches are pure resampled slices: no padding, no shifting."""
    problems: list[str] = []
    for point in sorted(raw):
        frames = replace_adc_dc_level(raw[point])
        channels = select_channels(frames, 3)
        branches, info = prepare_dynamic_envelope_branches(
            channels,
            locator_frames=channels,
            config=config,
            target_length=TARGET_LENGTH,
            apply_tukey=False,
            point_id=point,
        )
        main_block, tail_block = branches
        stream_count, channel_count = channels.shape[0], channels.shape[1]
        expected_rows = stream_count * channel_count
        if main_block.shape != (expected_rows, TARGET_LENGTH):
            problems.append(f"{point}: unexpected main branch shape {main_block.shape}")
            continue
        if tail_block.shape != (expected_rows, TARGET_LENGTH):
            problems.append(f"{point}: unexpected tail branch shape {tail_block.shape}")
            continue
        starts = info["main_start_index_by_channel"]
        for channel in range(channel_count):
            start = int(starts[channel])
            # Branch rows are ordered stream-major, so every row of one channel
            # is ``channel::channel_count``.
            expected_main = resample_signals(
                frames[:, channel, start : start + config.main_length], TARGET_LENGTH
            )
            expected_tail = resample_signals(
                frames[:, channel, config.tail_start :], TARGET_LENGTH
            )
            if not np.array_equal(main_block[channel::channel_count], expected_main):
                problems.append(f"{point} ch{channel}: main branch is not a pure slice")
            if not np.array_equal(tail_block[channel::channel_count], expected_tail):
                problems.append(f"{point} ch{channel}: tail branch is not a pure slice")
            if not np.array_equal(
                reference[point]["main_signal"][:, channel, :],
                frames[:, channel, start : start + config.main_length],
            ):
                problems.append(f"{point} ch{channel}: reference main slice differs")
    return problems


def main() -> int:
    config = DynamicEnvelopeConfig()
    raw = load_raw_frames()
    reference = load_reference()
    if set(raw) != set(reference):
        print("point sets differ between raw_data and after_split_data")
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
    print("dyn_envelope matches the reference implementation exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())

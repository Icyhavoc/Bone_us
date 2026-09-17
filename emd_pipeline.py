"""EMD feature extraction and NumPy MLP classification pipeline.

The pipeline is intentionally dependency-light.  It supports:

* four frame-processing forms: mean/std, raw 50 frames, max-1 frame,
  and top-3-frame mean;
* configurable channel selection (1, 2, or both channels);
* arbitrary, overlapping, or nested depth branches;
* configurable depth-to-sample mapping and resampling to a fixed length;
* a small empirical mode decomposition (EMD) implementation;
* fixed-size EMD feature extraction and a NumPy MLP classifier.

The command-line entry point is in ``run_emd_experiments.py``.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# NumPy 2.0 renamed ``trapz`` to ``trapezoid``; support both so the pipeline runs
# unchanged on NumPy 1.x and 2.x.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


FORM_ALIASES = {
    "mean_std": "mean_std",
    "mean+std": "mean_std",
    "meanstd": "mean_std",
    "raw50": "raw50",
    "raw_50": "raw50",
    "raw_50_frames": "raw50",
    "max1": "max1",
    "max_1": "max1",
    "max_1_frame": "max1",
    "top3_mean": "top3_mean",
    "top3": "top3_mean",
    "top_3_mean": "top3_mean",
}
FORM_ORDER = ("mean_std", "raw50", "max1", "top3_mean")


def canonical_form(name: str) -> str:
    """Return the canonical name for one of the four frame forms."""

    key = name.strip().lower().replace(" ", "_")
    if key not in FORM_ALIASES:
        raise ValueError(
            f"Unknown frame form {name!r}. Expected one of: {', '.join(FORM_ORDER)}"
        )
    return FORM_ALIASES[key]


@dataclass(frozen=True)
class RegionSpec:
    """One depth interval in millimetres.

    Intervals are represented as left-closed and right-open ranges.  This
    avoids duplicating a boundary sample when adjacent regions are used.
    """

    start_mm: float
    end_mm: float
    name: str | None = None

    def validate(self, max_depth_mm: float) -> None:
        if not (0.0 <= self.start_mm < self.end_mm <= max_depth_mm):
            raise ValueError(
                f"Invalid depth region [{self.start_mm}, {self.end_mm}) mm; "
                f"expected 0 <= start < end <= {max_depth_mm}."
            )

    @property
    def label(self) -> str:
        return self.name or f"{self.start_mm:g}-{self.end_mm:g}mm"


@dataclass(frozen=True)
class DepthMapper:
    """Map physical depth to sample boundaries.

    The default follows the project convention: ``index = round(depth / 5 *
    896)``.  The endpoint at 5 mm is allowed to equal ``signal_length`` so it
    can be used safely as a Python slice boundary.
    """

    max_depth_mm: float = 5.0
    signal_length: int = 896
    rounding: str = "round"

    def _round(self, value: float) -> int:
        if self.rounding == "floor":
            return int(math.floor(value))
        if self.rounding == "ceil":
            return int(math.ceil(value))
        if self.rounding != "round":
            raise ValueError("rounding must be one of: round, floor, ceil")
        # Explicit half-up rounding avoids Python's banker-rounding behaviour.
        return int(math.floor(value + 0.5))

    def depth_to_index(self, depth_mm: float, allow_endpoint: bool = True) -> int:
        if not (0.0 <= depth_mm <= self.max_depth_mm):
            raise ValueError(
                f"depth_mm={depth_mm} is outside [0, {self.max_depth_mm}]."
            )
        raw = depth_mm / self.max_depth_mm * self.signal_length
        upper = self.signal_length if allow_endpoint else self.signal_length - 1
        return max(0, min(self._round(raw), upper))

    def index_to_depth(self, index: int) -> float:
        if not (0 <= index <= self.signal_length):
            raise ValueError(f"index={index} is outside [0, {self.signal_length}].")
        return float(index) / self.signal_length * self.max_depth_mm

    def slice_bounds(self, region: RegionSpec) -> tuple[int, int]:
        region.validate(self.max_depth_mm)
        start = self.depth_to_index(region.start_mm, allow_endpoint=True)
        end = self.depth_to_index(region.end_mm, allow_endpoint=True)
        if end <= start:
            raise ValueError(f"Region {region.label} maps to an empty interval.")
        return start, end


@dataclass(frozen=True)
class DynamicEnvelopeConfig:
    """Settings for the sample-wise envelope-defined two-branch split.

    The defaults reproduce ``reference_code/pipeline.py``
    (:func:`segment`) exactly, including its index constants.  The detector is
    applied **per channel**: every channel gets its own reference crossing,
    local peak, merged leading packet and therefore its own main/tail
    boundary.  Branches are never re-padded and may overlap.

    ``main_length`` and ``tail_start`` are index constants of the reference
    rule ``start = max(t - lead_back, noise_start); end = start + main_length``
    and ``tail = [tail_start, signal_length)``.
    """

    # Detector constants (identical to the reference implementation).
    noise_start: int = 71
    weak_threshold: float = 0.015
    smooth_window: int = 5
    peak_window_back: int = 50
    peak_window_forward: int = 150
    peak_ratio: float = 0.05
    gap_max: int = 8
    min_run_width: int = 8
    min_run_area_ratio: float = 0.05
    merge_max_lead: int = 96
    lead_back: int = 20
    # Boundary constants.
    # ``main_start_min`` is the clamp floor of ``max(t - lead_back, 70)`` in the
    # reference.  It is deliberately *not* ``noise_start`` (71): the reference
    # lets the main window start one sample into the noise head.
    main_start_min: int = 70
    main_length: int = 170
    tail_start: int = 251
    signal_length: int = 896
    # Per-channel locator signal: mean of the K frames with the largest
    # absolute amplitude, matching ``compute_hilbert_topk_amplitude_mean``.
    locator_top_k: int = 3
    # Apply the small hand-checked correction table ported from the reference.
    # Set to ``False`` to run the detector with no hand tuning at all.
    apply_manual_corrections: bool = True

    def validate(self) -> None:
        if self.noise_start < 0:
            raise ValueError("dynamic envelope noise_start cannot be negative")
        if not 0.0 <= self.weak_threshold:
            raise ValueError("dynamic envelope weak_threshold cannot be negative")
        if self.smooth_window <= 0:
            raise ValueError("dynamic envelope smooth_window must be positive")
        if self.peak_window_back < 0 or self.peak_window_forward <= 0:
            raise ValueError("dynamic envelope peak window must be forward-positive")
        if not 0.0 < self.peak_ratio < 1.0:
            raise ValueError("dynamic envelope peak_ratio must be in (0, 1)")
        if self.gap_max < 0 or self.merge_max_lead < 0 or self.lead_back < 0:
            raise ValueError("dynamic envelope merge limits cannot be negative")
        if self.min_run_width <= 0:
            raise ValueError("dynamic envelope min_run_width must be positive")
        if not 0.0 <= self.min_run_area_ratio <= 1.0:
            raise ValueError("dynamic envelope min_run_area_ratio must be in [0, 1]")
        if self.main_length <= 0:
            raise ValueError("dynamic envelope main_length must be positive")
        if self.main_start_min < 0:
            raise ValueError("dynamic envelope main_start_min cannot be negative")
        if not 0 <= self.tail_start < self.signal_length:
            raise ValueError("dynamic envelope tail_start must be inside the signal")
        if self.locator_top_k <= 0:
            raise ValueError("dynamic envelope locator_top_k must be positive")


@dataclass(frozen=True)
class EMDConfig:
    """EMD and feature-extraction settings."""

    max_imfs: int = 5
    max_sift_iterations: int = 30
    sift_sd_threshold: float = 0.2
    include_residue: bool = True
    # Pooled mode averages component/stream statistics and keeps each branch
    # at len(feature_names) dimensions. Branches remain independent and are
    # concatenated by sample_feature_vector.
    stream_aggregation: str = "pooled"
    feature_names: tuple[str, ...] = (
        "mean",
        "std",
        "rms",
        "energy",
        "abs_mean",
        "peak_abs",
        "zero_crossing_rate",
        "spectral_centroid",
    )

    def validate(self) -> None:
        if self.max_imfs <= 0:
            raise ValueError("max_imfs must be positive")
        if self.max_sift_iterations <= 0:
            raise ValueError("max_sift_iterations must be positive")
        if self.sift_sd_threshold <= 0:
            raise ValueError("sift_sd_threshold must be positive")
        if self.stream_aggregation not in {"pooled", "flatten", "stats"}:
            raise ValueError("stream_aggregation must be 'pooled', 'flatten', or 'stats'")


@dataclass(frozen=True)
class MLPConfig:
    """Small MLP settings for the current small dataset."""

    hidden_dims: tuple[int, ...] = (64, 32)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    batch_size: int = 32
    epochs: int = 200
    patience: int = 40
    min_delta: float = 1e-4
    seed: int = 42

    def validate(self) -> None:
        if not self.hidden_dims or any(v <= 0 for v in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_size <= 0 or self.epochs <= 0 or self.patience <= 0:
            raise ValueError("batch_size, epochs, and patience must be positive")


def parse_regions(value: str | Sequence[Sequence[float]]) -> list[RegionSpec]:
    """Parse a JSON region list such as ``[[0, 5], [1, 3]]``."""

    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    regions: list[RegionSpec] = []
    for item in parsed:
        if isinstance(item, Mapping):
            regions.append(
                RegionSpec(
                    start_mm=float(item["start_mm"]),
                    end_mm=float(item["end_mm"]),
                    name=item.get("name"),
                )
            )
        else:
            if len(item) != 2:
                raise ValueError(f"Each region must have two values, got {item!r}")
            regions.append(RegionSpec(float(item[0]), float(item[1])))
    if not regions:
        raise ValueError("At least one region is required")
    return regions


def select_channels(frames: np.ndarray, channel_mode: int) -> np.ndarray:
    """Select physical channels from ``[frames, channels, samples]`` data."""

    if frames.ndim != 3:
        raise ValueError(f"Expected [frames, channels, samples], got {frames.shape}")
    if frames.shape[1] < 2:
        raise ValueError("The current dataset must contain physical channels 1 and 2")
    if channel_mode == 1:
        return frames[:, 0:1, :]
    if channel_mode == 2:
        return frames[:, 1:2, :]
    if channel_mode == 3:
        return frames[:, 0:2, :]
    raise ValueError("channel_mode must be 1, 2, or 3")


def _rms_scores(frames: np.ndarray, start: int, end: int) -> np.ndarray:
    segment = frames[..., start:end]
    return np.sqrt(np.mean(np.square(segment, dtype=np.float64), axis=(1, 2)))


def process_frames(
    frames: np.ndarray,
    form: str,
    mapper: DepthMapper,
    score_region: RegionSpec = RegionSpec(0.0, 5.0, "selection_full"),
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply one of the four frame forms.

    Returns ``processed, selected_frame_indices`` where processed has shape
    ``[streams_in_time, channels, samples]``.  For mean/std the first axis is
    ``[mean, std]``; for raw50 it is the original 50 frames; for max1/top3 it
    is one processed signal.
    """

    form = canonical_form(form)
    if frames.ndim != 3:
        raise ValueError(f"Expected [frames, channels, samples], got {frames.shape}")
    start, end = mapper.slice_bounds(score_region)

    if form == "mean_std":
        mean = np.mean(frames, axis=0, dtype=np.float32)
        std = np.std(frames, axis=0, dtype=np.float32)
        return np.stack([mean, std], axis=0), None
    if form == "raw50":
        return np.asarray(frames, dtype=np.float32), None

    scores = _rms_scores(frames, start, end)
    if form == "max1":
        selected = int(np.argmax(scores))
        return frames[selected : selected + 1].astype(np.float32), np.array([selected])
    if form == "top3_mean":
        count = min(3, frames.shape[0])
        selected = np.argsort(scores)[-count:][::-1]
        averaged = np.mean(frames[selected], axis=0, dtype=np.float32)
        return averaged[None, ...], selected.astype(np.int64)
    raise AssertionError(f"Unhandled frame form: {form}")


def resample_signals(signals: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a batch of 1-D signals using NumPy only."""

    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 2:
        raise ValueError(f"Expected [num_signals, length], got {signals.shape}")
    if signals.shape[1] <= 0 or target_length <= 0:
        raise ValueError("Signal and target lengths must be positive")
    if signals.shape[1] == target_length:
        return signals.copy()
    positions = np.linspace(0.0, signals.shape[1] - 1.0, target_length)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, signals.shape[1] - 1)
    weight = (positions - left).astype(np.float32)
    return signals[:, left] * (1.0 - weight)[None, :] + signals[:, right] * weight[None, :]


def tukey_window(length: int, alpha: float = 0.3) -> np.ndarray:
    """Return a Tukey window of ``length`` samples without SciPy."""

    if length <= 0:
        raise ValueError("Window length must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Tukey alpha must be in [0, 1]")
    if length == 1 or alpha == 0.0:
        return np.ones(length, dtype=np.float32)
    if alpha == 1.0:
        positions = np.linspace(0.0, 1.0, length)
        return (0.5 * (1.0 - np.cos(2.0 * np.pi * positions))).astype(np.float32)

    positions = np.linspace(0.0, 1.0, length)
    window = np.ones(length, dtype=np.float64)
    taper_width = alpha / 2.0
    left = positions < taper_width
    right = positions > (1.0 - taper_width)
    window[left] = 0.5 * (
        1.0 + np.cos(np.pi / taper_width * (positions[left] - taper_width))
    )
    window[right] = 0.5 * (
        1.0 + np.cos(np.pi / taper_width * (positions[right] - (1.0 - taper_width)))
    )
    return window.astype(np.float32)


def prepare_branch_signals(
    processed: np.ndarray,
    region: RegionSpec,
    mapper: DepthMapper,
    target_length: int = 512,
    tukey_alpha: float = 0.3,
    apply_tukey: bool = True,
) -> np.ndarray:
    """Cut/resample one branch and optionally apply its Tukey window."""

    if processed.ndim != 3:
        raise ValueError(f"Expected [streams, channels, samples], got {processed.shape}")
    start, end = mapper.slice_bounds(region)
    return prepare_interval_signals(
        processed,
        start,
        end,
        target_length=target_length,
        tukey_alpha=tukey_alpha,
        apply_tukey=apply_tukey,
    )


def prepare_interval_signals(
    processed: np.ndarray,
    start: int,
    end: int,
    target_length: int = 512,
    tukey_alpha: float = 0.3,
    apply_tukey: bool = True,
) -> np.ndarray:
    """Cut/resample an explicit interval and optionally apply Tukey."""

    if processed.ndim != 3:
        raise ValueError(f"Expected [streams, channels, samples], got {processed.shape}")
    sample_count = processed.shape[-1]
    if not (0 <= start < end <= sample_count):
        raise ValueError(
            f"Invalid sample interval [{start}, {end}) for length {sample_count}"
        )
    segment = processed[..., start:end]
    streams = segment.reshape(-1, segment.shape[-1])
    resampled = resample_signals(streams, target_length)
    if not apply_tukey:
        return resampled
    return resampled * tukey_window(target_length, tukey_alpha)[None, :]


# Manual per-channel corrections ported verbatim from
# ``reference_code/pipeline.py``.  Keys are ``(point_id, channel_index)`` with
# 0-based channel indices.  The reference refuses to apply a correction when
# the automatic result no longer matches the recorded one, so a changed input
# cannot silently inherit a stale hand fix.
MANUAL_CORRECTIONS: dict[tuple[str, int], tuple[int, int]] = {
    ("N32_P3_01", 0): (249, 163),
    ("N35_P2_01", 0): (324, 272),
    ("N35_P2_01", 1): (365, 279),
}


def _hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    """Magnitude of the analytic signal, mirroring ``scipy.signal.hilbert``.

    Accepts a single ``[samples]`` signal or a ``[rows, samples]`` batch and
    returns the envelope with the same shape.  Unlike the earlier version this
    performs no detrending and no boundary padding, because the reference
    implementation relies on the plain FFT round trip: any smoothing of the
    edges would move the detected crossing.
    """

    values = np.asarray(signal, dtype=np.float32)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D input, got {values.shape}")
    length = values.shape[-1]
    if length == 0:
        return values.astype(np.float64).reshape(np.shape(signal))
    spectrum = np.fft.fft(values, axis=-1)
    mask = np.zeros(length, dtype=np.float64)
    mask[0] = 1.0
    if length % 2 == 0:
        mask[length // 2] = 1.0
        mask[1 : length // 2] = 2.0
    else:
        mask[1 : (length + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * mask, axis=-1)
    envelope = np.abs(analytic)
    return envelope[0] if squeeze else envelope


def _moving_average_nearest(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average with clamped edges, matching ``uniform_filter1d``.

    The reference smooths with
    ``scipy.ndimage.uniform_filter1d(..., mode='nearest')``, which accumulates
    in ``float64`` and writes the result back in the input dtype.  Reproducing
    that accumulation order keeps the later ``> weak_threshold`` comparisons
    identical; accumulating in ``float32`` instead shifts values by ~1e-6 and
    can flip a crossing.
    """

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [rows, samples], got {values.shape}")
    window = int(window)
    if window <= 1:
        return values.copy()
    window = min(window, values.shape[-1])
    radius = window // 2
    # ``uniform_filter1d`` biases an even-sized window towards earlier samples.
    left, right = (radius, radius) if window % 2 else (radius - 1, radius)
    padded = np.pad(values, [(0, 0), (left, right)], mode="edge")
    windowed = np.lib.stride_tricks.sliding_window_view(padded, window, axis=-1)
    return windowed.mean(axis=-1, dtype=np.float64).astype(values.dtype)


def locator_mean_signal(frames: np.ndarray, top_k: int) -> np.ndarray:
    """Average the ``top_k`` frames with the largest peak amplitude per channel.

    Port of ``compute_hilbert_topk_amplitude_mean.select_topk``: each channel
    picks its own frames by ``max(abs(x))`` over the whole trace, ties broken
    towards the earlier frame, and the selection is averaged in ``float64``
    before being stored back as ``float32``.
    """

    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError(f"Expected [frames, channels, samples], got {frames.shape}")
    frame_count, channel_count = frames.shape[0], frames.shape[1]
    top_k = min(int(top_k), frame_count)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    peak_abs = np.max(np.abs(frames), axis=-1)
    order = np.arange(frame_count)
    selected = np.empty((channel_count, top_k, frames.shape[-1]), dtype=np.float32)
    for channel in range(channel_count):
        chosen = np.lexsort((order, -peak_abs[:, channel]))[:top_k]
        selected[channel] = frames[chosen, channel, :]
    return np.mean(selected, axis=1, dtype=np.float64).astype(np.float32)


def _first_crossing(values: np.ndarray, threshold: float, noise_start: int) -> int:
    """First index at or after ``noise_start`` strictly above ``threshold``."""

    hits = np.flatnonzero(values[noise_start:] > threshold)
    return int(hits[0] + noise_start) if hits.size else -1


def _threshold_runs(
    values: np.ndarray, threshold: float, noise_start: int
) -> list[dict[str, float]]:
    """Contiguous runs of ``values > threshold`` ignoring the noise head."""

    mask = values > threshold
    mask[:noise_start] = False
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    # ``area`` keeps the input dtype accumulation of the reference code because
    # it is only ever compared against a ratio of another run's area.
    return [
        {
            "start": int(run_start),
            "end": int(run_end),
            "width": int(run_end - run_start),
            "area": float(values[run_start:run_end].sum()),
        }
        for run_start, run_end in zip(starts, ends)
    ]


def locate_dynamic_span(
    smoothed: np.ndarray,
    config: DynamicEnvelopeConfig = DynamicEnvelopeConfig(),
    point_id: str | None = None,
) -> dict[str, Any]:
    """Port of ``reference_code/pipeline.py::segment`` (per-channel locator).

    ``smoothed`` is the ``[channels, samples]`` smoothed Hilbert envelope.  The
    rules are reproduced exactly:

    * reference crossing: first sample at or after ``noise_start`` strictly
      above ``weak_threshold``; when exactly one channel is missing it borrows
      the other channel's crossing, and when every channel is missing it falls
      back to ``0.5 * max`` of the joint RMS envelope;
    * local peak: maximum of the window
      ``[reference - peak_window_back, reference + peak_window_forward)``;
    * threshold: ``peak_ratio`` times that peak;
    * leading-packet merge: walk backwards while the gap to the running start
      is within ``gap_max``, the run is at least ``min_run_width`` wide, its
      area reaches ``min_run_area_ratio`` of the *original* anchor run, and the
      total lead stays within ``merge_max_lead`` of the anchor start;
    * boundary: ``start = max(t - lead_back, main_start_min)`` and
      ``end = start + main_length``, never silently shifted to fit.

    The tail is the fixed ``[tail_start, signal_length)`` slice, so it may
    overlap the main window or leave a gap.
    """

    config.validate()
    smoothed = np.asarray(smoothed, dtype=np.float32)
    if smoothed.ndim != 2:
        raise ValueError(f"Expected [channels, samples], got {smoothed.shape}")
    channel_count, sample_count = smoothed.shape
    if sample_count != config.signal_length:
        raise ValueError(
            f"dynamic envelope expects {config.signal_length} samples, got {sample_count}"
        )

    direct = np.array(
        [_first_crossing(row, config.weak_threshold, config.noise_start) for row in smoothed],
        dtype=np.int64,
    )
    reference = direct.copy()
    rules = ["direct"] * channel_count
    relative_fallback = 0.0
    missing = np.flatnonzero(reference < 0)
    if missing.size == 0:
        pass
    elif missing.size == 1 and channel_count > 1:
        channel = int(missing[0])
        donor = int(next(i for i in range(channel_count) if i != channel))
        reference[channel] = reference[donor]
        rules[channel] = "paired_fallback"
    elif missing.size == channel_count:
        joint = np.sqrt(np.mean(smoothed.astype(np.float64) ** 2, axis=0))
        relative_fallback = 0.5 * float(joint[config.noise_start :].max())
        reference[:] = _first_crossing(joint, relative_fallback, config.noise_start)
        if reference[0] < 0:
            raise ValueError(
                "dynamic envelope: no valid reference crossing, manual review required"
            )
        rules = ["rms_joint_fallback"] * channel_count
    else:
        raise ValueError(
            "dynamic envelope: partial multi-channel crossing fallback is undefined"
        )

    starts = np.zeros(channel_count, dtype=np.int64)
    ends = np.zeros(channel_count, dtype=np.int64)
    automatic = np.zeros(channel_count, dtype=np.int64)
    peaks = np.zeros(channel_count, dtype=np.int64)
    thresholds = np.zeros(channel_count, dtype=np.float64)
    premerge = np.zeros(channel_count, dtype=np.int64)
    windows = np.zeros((channel_count, 2), dtype=np.int64)
    corrected = np.zeros(channel_count, dtype=bool)
    audits: list[dict[str, Any]] = []

    for channel in range(channel_count):
        row = smoothed[channel]
        low = max(config.noise_start, int(reference[channel]) - config.peak_window_back)
        high = min(sample_count, int(reference[channel]) + config.peak_window_forward)
        peak = int(low + np.argmax(row[low:high]))
        threshold = config.peak_ratio * float(row[peak])
        if threshold <= 0.0:
            raise ValueError("dynamic envelope: non-positive local peak, cannot threshold")
        runs = _threshold_runs(row, threshold, config.noise_start)
        anchor_index = next(
            index for index, run in enumerate(runs) if run["start"] <= peak < run["end"]
        )
        anchor = runs[anchor_index]
        crossing = int(anchor["start"])
        accepted: list[int] = []
        # The area denominator stays the original anchor run and never grows as
        # further packets are merged.
        for index in range(anchor_index - 1, -1, -1):
            run = runs[index]
            if (
                crossing - run["end"] > config.gap_max
                or run["width"] < config.min_run_width
                or run["area"] < config.min_run_area_ratio * anchor["area"]
                or anchor["start"] - run["start"] > config.merge_max_lead
            ):
                break
            crossing = int(run["start"])
            accepted.append(index)

        automatic[channel] = crossing
        peaks[channel] = peak
        thresholds[channel] = threshold
        premerge[channel] = int(anchor["start"])
        windows[channel] = (low, high)
        audits.append(
            {
                "runs": runs,
                "main_run_index": anchor_index,
                "accepted_run_indices": accepted,
            }
        )

    automatic_before_manual = automatic.copy()
    for channel in range(channel_count):
        if point_id is None or not config.apply_manual_corrections:
            continue
        fix = MANUAL_CORRECTIONS.get((point_id, channel))
        if fix is None:
            continue
        recorded, manual = fix
        if int(automatic[channel]) != recorded:
            raise ValueError(
                f"dynamic envelope: {point_id} channel {channel + 1} automatic result "
                f"changed (expected {recorded}, got {int(automatic[channel])}); "
                "the recorded manual correction no longer applies"
            )
        automatic[channel] = manual
        corrected[channel] = True
        audits[channel]["manual_correction"] = {
            "old_threshold_index": recorded,
            "new_threshold_index": manual,
        }

    starts[:] = np.maximum(automatic - config.lead_back, config.main_start_min)
    ends[:] = starts + config.main_length
    if np.any(ends > sample_count):
        raise ValueError(
            "dynamic envelope: main window exceeds the signal, refusing to shift it"
        )
    tail_end = sample_count
    return {
        "starts": starts,
        "ends": ends,
        "tail_start": int(config.tail_start),
        "tail_end": int(tail_end),
        "crossing_index": automatic,
        "automatic_crossing_index": automatic_before_manual,
        "manual_correction_applied": corrected,
        "local_peak_index": peaks,
        "threshold_value": thresholds,
        "premerge_crossing_index": premerge,
        "peak_search_window": windows,
        "reference_absolute_crossing": direct,
        "reference_index": reference,
        "reference_rule": rules,
        "relative_fallback": float(relative_fallback),
        "main_start_clamped": automatic - config.lead_back < config.main_start_min,
        "gap_length": np.maximum((config.tail_start - ends), 0).astype(np.int64),
        "overlap_length": np.maximum(
            np.minimum(ends, tail_end) - np.maximum(starts, config.tail_start), 0
        ).astype(np.int64),
        "audits": audits,
    }


def prepare_dynamic_envelope_branches(
    processed: np.ndarray,
    locator_frames: np.ndarray,
    config: DynamicEnvelopeConfig = DynamicEnvelopeConfig(),
    target_length: int = 512,
    tukey_alpha: float = 0.3,
    apply_tukey: bool = True,
    point_id: str | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Create the two dynamic-envelope branches for one processed sample.

    ``processed`` is the frame-processed ``[streams, channels, samples]`` block
    that the branches are cut from.  ``locator_frames`` is the matching raw
    ``[frames, channels, samples]`` block used only to build the locator signal:
    like the reference, the boundary comes from the top-``locator_top_k`` frame
    average and not from the selected frame form, so switching form does not
    move the split.

    Boundaries are detected **per channel**.  Branch 1 is
    ``[start, start + main_length)`` with ``start = max(t - lead_back,
    main_start_min)``; branch 2 is the fixed ``[tail_start, samples)`` slice.  Both
    are resampled to ``target_length`` and optionally Tukey-windowed by the
    existing feature pipeline, then flattened to
    ``[streams * channels, target_length]``.
    """

    config.validate()
    if processed.ndim != 3:
        raise ValueError(f"Expected [streams, channels, samples], got {processed.shape}")
    channel_count = int(processed.shape[1])
    sample_count = int(processed.shape[-1])
    if locator_frames.ndim != 3:
        raise ValueError(
            f"Expected [frames, channels, samples] locator frames, got {locator_frames.shape}"
        )
    if locator_frames.shape[1] != channel_count or locator_frames.shape[-1] != sample_count:
        raise ValueError(
            "dynamic envelope locator frames must match the processed channel count "
            f"and length; got {locator_frames.shape} against {processed.shape}"
        )

    locator = locator_mean_signal(locator_frames, config.locator_top_k)
    envelope = np.zeros((channel_count, sample_count), dtype=np.float32)
    envelope[:] = _hilbert_envelope(locator).astype(np.float32)
    smoothed = np.zeros_like(envelope)
    # The head of the trace is deliberately zeroed so it can never take part in
    # crossing detection or in the run masks.
    smoothed[:, config.noise_start :] = _moving_average_nearest(
        envelope[:, config.noise_start :], config.smooth_window
    )

    span = locate_dynamic_span(smoothed, config=config, point_id=point_id)
    starts = span["starts"]
    ends = span["ends"]
    tail_start = span["tail_start"]
    tail_end = span["tail_end"]

    def cut(channel: int, start: int, end: int) -> np.ndarray:
        """Resample one channel's interval into ``[streams, target_length]``."""
        block = processed[:, channel : channel + 1, start:end]
        flattened = block.reshape(-1, block.shape[-1])
        resampled = resample_signals(flattened, target_length)
        if not apply_tukey:
            return resampled
        return resampled * tukey_window(target_length, tukey_alpha)[None, :]

    # Channel boundaries differ, so each channel is cut on its own and the
    # pieces are stacked back into a single branch block.
    main_parts = [cut(c, int(starts[c]), int(ends[c])) for c in range(channel_count)]
    tail_parts = [cut(c, int(tail_start), int(tail_end)) for c in range(channel_count)]
    branch_stream_count = int(processed.shape[0])
    main_block = np.stack(
        [part.reshape(branch_stream_count, target_length) for part in main_parts], axis=1
    ).reshape(-1, target_length)
    tail_block = np.stack(
        [part.reshape(branch_stream_count, target_length) for part in tail_parts], axis=1
    ).reshape(-1, target_length)

    branch_info = [
        {
            "name": "dynamic_main_window",
            "start_index": int(starts[0]),
            "end_index_exclusive": int(ends[0]),
            "start_index_by_channel": starts.astype(int).tolist(),
            "end_index_exclusive_by_channel": ends.astype(int).tolist(),
            "input_length_by_channel": (ends - starts).astype(int).tolist(),
            "input_length": int(ends[0] - starts[0]),
            "feature_length": None,
        },
        {
            "name": "dynamic_tail_window",
            "start_index": int(tail_start),
            "end_index_exclusive": int(tail_end),
            "start_index_by_channel": [int(tail_start)] * channel_count,
            "end_index_exclusive_by_channel": [int(tail_end)] * channel_count,
            "input_length_by_channel": [int(tail_end - tail_start)] * channel_count,
            "input_length": int(tail_end - tail_start),
            "feature_length": None,
        },
    ]
    info: dict[str, Any] = {
        "algorithm": "reference_code.pipeline.segment",
        "point_id": point_id,
        "channel_count": channel_count,
        "sample_length": sample_count,
        "locator_top_k": int(config.locator_top_k),
        "main_length": int(config.main_length),
        "main_start_index_by_channel": starts.astype(int).tolist(),
        "main_end_exclusive_by_channel": ends.astype(int).tolist(),
        "tail_interval": [int(tail_start), int(tail_end)],
        "crossing_index": span["crossing_index"].astype(int).tolist(),
        "automatic_crossing_index": span["automatic_crossing_index"].astype(int).tolist(),
        "manual_correction_applied": span["manual_correction_applied"].astype(bool).tolist(),
        "local_peak_index": span["local_peak_index"].astype(int).tolist(),
        "threshold_value": span["threshold_value"].astype(float).tolist(),
        "premerge_crossing_index": span["premerge_crossing_index"].astype(int).tolist(),
        "peak_search_window": span["peak_search_window"].astype(int).tolist(),
        "reference_absolute_crossing": span["reference_absolute_crossing"].astype(int).tolist(),
        "reference_index": span["reference_index"].astype(int).tolist(),
        "reference_rule": span["reference_rule"],
        "relative_fallback": float(span["relative_fallback"]),
        "main_start_clamped": span["main_start_clamped"].astype(bool).tolist(),
        "gap_length_by_channel": span["gap_length"].astype(int).tolist(),
        "overlap_length_by_channel": span["overlap_length"].astype(int).tolist(),
        "branches": branch_info,
    }
    return [main_block, tail_block], info


def _local_extrema(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find strict local maxima/minima without SciPy."""

    if signal.size < 3:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    left = signal[:-2]
    center = signal[1:-1]
    right = signal[2:]
    maxima = np.flatnonzero((center > left) & (center >= right)) + 1
    minima = np.flatnonzero((center < left) & (center <= right)) + 1
    return maxima.astype(np.int64), minima.astype(np.int64)


def _envelope(signal: np.ndarray, extrema: np.ndarray) -> np.ndarray:
    """Build a piecewise-linear envelope through extrema and endpoints."""

    n = signal.size
    if extrema.size == 0:
        return np.full(n, float(np.mean(signal)), dtype=np.float64)
    points = np.unique(np.concatenate(([0, n - 1], extrema))).astype(np.int64)
    values = signal[points]
    grid = np.arange(n, dtype=np.float64)
    return np.interp(grid, points.astype(np.float64), values.astype(np.float64))


def _zero_crossings(signal: np.ndarray) -> int:
    if signal.size < 2:
        return 0
    values = signal[:-1] * signal[1:]
    return int(np.count_nonzero(values < 0))


def emd_decompose(
    signal: np.ndarray,
    max_imfs: int = 5,
    max_sift_iterations: int = 30,
    sift_sd_threshold: float = 0.2,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Perform a compact EMD implementation using linear envelopes.

    This implementation is deliberately deterministic and dependency-light.
    It is suitable for the current feasibility experiments; if a validated
    EMD library is introduced later, this function is the replacement point.
    """

    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size < 4:
        return [], signal.astype(np.float32)
    residue = signal.copy()
    imfs: list[np.ndarray] = []

    for _ in range(max_imfs):
        maxima, minima = _local_extrema(residue)
        if maxima.size + minima.size < 2:
            break
        h = residue.copy()
        for _ in range(max_sift_iterations):
            max_idx, min_idx = _local_extrema(h)
            if max_idx.size + min_idx.size < 2:
                break
            upper = _envelope(h, max_idx)
            lower = _envelope(h, min_idx)
            mean_envelope = 0.5 * (upper + lower)
            denominator = float(np.sum(h * h)) + 1e-12
            sd = float(np.sum((h - mean_envelope) ** 2) / denominator)
            candidate = h - mean_envelope
            cmax, cmin = _local_extrema(candidate)
            extrema_count = cmax.size + cmin.size
            zero_crossing_count = _zero_crossings(candidate)
            h = candidate
            if sd < sift_sd_threshold and abs(zero_crossing_count - extrema_count) <= 1:
                break
        imfs.append(h.astype(np.float32))
        residue = residue - h
        rmax, rmin = _local_extrema(residue)
        if rmax.size + rmin.size < 2:
            break
    return imfs, residue.astype(np.float32)


def _component_features(signal: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    energy = float(np.mean(signal * signal))
    power = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(signal.size, d=1.0)
    power_sum = float(np.sum(power)) + 1e-12
    features: dict[str, float] = {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(energy)),
        "energy": energy,
        "abs_mean": float(np.mean(np.abs(signal))),
        "peak_abs": float(np.max(np.abs(signal))),
        "zero_crossing_rate": float(_zero_crossings(signal) / max(1, signal.size - 1)),
        "spectral_centroid": float(np.sum(frequencies * power) / power_sum),
    }
    unknown = set(feature_names) - set(features)
    if unknown:
        raise ValueError(f"Unknown EMD feature names: {sorted(unknown)}")
    return np.asarray([features[name] for name in feature_names], dtype=np.float32)


def emd_feature_vector(streams: np.ndarray, config: EMDConfig) -> np.ndarray:
    """Extract a fixed-size feature vector from ``[streams, samples]``."""

    config.validate()
    streams = np.asarray(streams, dtype=np.float32)
    if streams.ndim != 2:
        raise ValueError(f"Expected [streams, samples], got {streams.shape}")

    component_count = config.max_imfs + int(config.include_residue)
    per_stream: list[np.ndarray] = []
    for stream in streams:
        imfs, residue = emd_decompose(
            stream,
            max_imfs=config.max_imfs,
            max_sift_iterations=config.max_sift_iterations,
            sift_sd_threshold=config.sift_sd_threshold,
        )
        components = list(imfs[: config.max_imfs])
        if config.include_residue:
            components.append(residue)
        while len(components) < component_count:
            components.append(np.zeros_like(stream))
        component_features = np.concatenate(
            [_component_features(component, config.feature_names) for component in components]
        )
        per_stream.append(component_features)
    matrix = np.stack(per_stream, axis=0)
    if config.stream_aggregation == "pooled":
        feature_count = len(config.feature_names)
        component_matrix = matrix.reshape(matrix.shape[0], component_count, feature_count)
        return np.mean(component_matrix, axis=(0, 1)).astype(np.float32)
    if config.stream_aggregation == "flatten":
        return matrix.reshape(-1).astype(np.float32)
    # stats mode keeps the feature dimension fixed when the number of streams
    # changes between Mean+Std, Raw50, and the single-frame forms.
    return np.concatenate(
        [np.mean(matrix, axis=0), np.std(matrix, axis=0), np.min(matrix, axis=0), np.max(matrix, axis=0)]
    ).astype(np.float32)


def sample_feature_vector(
    sample: np.ndarray,
    form: str,
    regions: Sequence[RegionSpec],
    channel_mode: int,
    mapper: DepthMapper,
    target_length: int,
    emd_config: EMDConfig,
    tukey_alpha: float = 0.3,
    dynamic_envelope_config: DynamicEnvelopeConfig | None = None,
    score_region: RegionSpec = RegionSpec(0.0, 5.0, "selection_full"),
    point_id: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preprocess one sample and concatenate EMD features from all branches."""

    if sample.ndim != 3:
        raise ValueError(f"Expected sample [50, 2, 896], got {sample.shape}")
    selected_channels = select_channels(sample, channel_mode)
    processed, selected_frames = process_frames(
        selected_channels, form, mapper=mapper, score_region=score_region
    )
    branch_features: list[np.ndarray] = []
    branch_info: list[dict[str, Any]] = []
    dynamic_info: dict[str, Any] | None = None
    if dynamic_envelope_config is not None:
        # The locator always comes from the raw selected frames, so the split
        # does not depend on which frame form is being evaluated.
        branch_signals_list, dynamic_info = prepare_dynamic_envelope_branches(
            processed,
            locator_frames=selected_channels,
            config=dynamic_envelope_config,
            target_length=target_length,
            tukey_alpha=tukey_alpha,
            point_id=point_id,
        )
        branch_specs: list[tuple[str, np.ndarray, int, int, float | None, float | None]] = []
        for branch_index, branch_signals in enumerate(branch_signals_list):
            dynamic_branch = dynamic_info["branches"][branch_index]
            branch_specs.append(
                (
                    str(dynamic_branch["name"]),
                    branch_signals,
                    int(dynamic_branch["start_index"]),
                    int(dynamic_branch["end_index_exclusive"]),
                    None,
                    None,
                )
            )
    else:
        branch_specs = []
        for region in regions:
            start, end = mapper.slice_bounds(region)
            branch_specs.append(
                (
                    region.label,
                    prepare_branch_signals(
                        processed,
                        region,
                        mapper=mapper,
                        target_length=target_length,
                        tukey_alpha=tukey_alpha,
                    ),
                    start,
                    end,
                    region.start_mm,
                    region.end_mm,
                )
            )

    for name, branch_signals, start, end, start_mm, end_mm in branch_specs:
        branch_features.append(emd_feature_vector(branch_signals, emd_config))
        branch_info.append(
            {
                "name": name,
                "start_mm": start_mm,
                "end_mm": end_mm,
                "start_index": start,
                "end_index_exclusive": end,
                "input_streams": int(branch_signals.shape[0]),
                "feature_length": int(branch_features[-1].size),
            }
        )
    info = {
        "form": canonical_form(form),
        "channel_mode": channel_mode,
        "selected_frames": None if selected_frames is None else selected_frames.tolist(),
        "region_mode": "dyn_envelope" if dynamic_envelope_config is not None else "fixed",
        "branches": branch_info,
    }
    if dynamic_info is not None:
        info["dynamic_envelope"] = dynamic_info
    return np.concatenate(branch_features).astype(np.float32), info


# The source ADC stores raw codes, and both 127 and 128 mean "zero": they
# normalize to +-0.5 / 127.5, i.e. the same physical level split across the two
# sides of the 127.5 midpoint.  Keeping that +-0.0039 jitter injects spurious
# quantization noise, which is enough to move the ``dyn_envelope`` threshold
# crossing by tens of samples (measured: 267 instead of 324 on ``N35_P2_01``).
# ``after_split_data/`` was produced from the collapsed traces, so the loader
# reproduces the collapse to keep ``--data-dir raw_data`` bit-exact.
ADC_DC_MAGNITUDE = 0.5 / 127.5
ADC_DC_ATOL = 1e-6


def replace_adc_dc_level(
    frames: np.ndarray,
    magnitude: float = ADC_DC_MAGNITUDE,
    atol: float = ADC_DC_ATOL,
) -> np.ndarray:
    """Collapse the ADC ``127``/``128`` pair onto the shared zero level.

    Samples equal to +-``magnitude`` (both codes mapping to the same physical
    level) are snapped to ``0.0``; everything else is untouched.  Returns a new
    ``float32`` array, so the stored ``X.npy`` is never mutated.
    """

    values = np.asarray(frames, dtype=np.float32)
    collapsed = values.copy()
    near_zero_code = np.isclose(collapsed, magnitude, atol=atol) | np.isclose(
        collapsed, -magnitude, atol=atol
    )
    collapsed[near_zero_code] = 0.0
    return collapsed


def load_split(data_dir: Path | str, split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load X, y, and sample metadata for one split.

    The stored frames go through :func:`replace_adc_dc_level` before being
    returned, so every consumer sees the same DC-centred traces the reference
    implementation was built on.
    """

    split_dir = Path(data_dir) / split
    x_path = split_dir / "X.npy"
    y_path = split_dir / "y.npy"
    samples_path = split_dir / "samples.json"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing X.npy/y.npy under {split_dir}")
    x = replace_adc_dc_level(np.load(x_path).astype(np.float32))
    y = np.load(y_path).astype(np.int64)
    if x.ndim != 4 or x.shape[1:] != (50, 2, 896):
        raise ValueError(f"Expected {split} X shape [N,50,2,896], got {x.shape}")
    if y.shape != (x.shape[0],):
        raise ValueError(f"Expected {split} y shape {(x.shape[0],)}, got {y.shape}")
    if samples_path.exists():
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
    else:
        samples = [{} for _ in range(x.shape[0])]
    if len(samples) != x.shape[0]:
        raise ValueError(f"{samples_path} contains {len(samples)} records, expected {x.shape[0]}")
    return x, y, samples


def build_feature_matrix(
    x: np.ndarray,
    form: str,
    regions: Sequence[RegionSpec],
    channel_mode: int,
    mapper: DepthMapper,
    target_length: int,
    emd_config: EMDConfig,
    tukey_alpha: float = 0.3,
    dynamic_envelope_config: DynamicEnvelopeConfig | None = None,
    score_region: RegionSpec = RegionSpec(0.0, 5.0, "selection_full"),
    limit: int | None = None,
    sample_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build the feature matrix for a split.

    ``sample_ids`` are optional point identifiers used only by the dynamic
    envelope locator, which keeps a small hand-checked correction table keyed
    by point id.
    """

    if limit is not None:
        x = x[:limit]
        if sample_ids is not None:
            sample_ids = list(sample_ids)[:limit]
    vectors: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    for index, sample in enumerate(x):
        point_id = None if sample_ids is None else str(sample_ids[index])
        vector, info = sample_feature_vector(
            sample,
            form=form,
            regions=regions,
            channel_mode=channel_mode,
            mapper=mapper,
            target_length=target_length,
            tukey_alpha=tukey_alpha,
            dynamic_envelope_config=dynamic_envelope_config,
            emd_config=emd_config,
            score_region=score_region,
            point_id=point_id,
        )
        vectors.append(vector)
        # Fixed-region runs retain the historical compact preview (first
        # sample only). Dynamic envelope runs need every sample's detected
        # boundary because the split is sample-dependent.
        if index == 0 or dynamic_envelope_config is not None:
            infos.append(info)
    matrix = np.stack(vectors, axis=0).astype(np.float32)
    return matrix, infos


class StandardScaler:
    """Small NumPy-only standard scaler fitted on the training split."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        values = np.asarray(values, dtype=np.float32)
        self.mean_ = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        self.scale_ = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform")
        return ((values - self.mean_) / self.scale_).astype(np.float32)

    def save(self, path: Path) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Cannot save an unfitted scaler")
        np.savez(path, mean=self.mean_, scale=self.scale_)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int | None]:
    """Compute the required binary metrics without scikit-learn."""

    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    scores = probabilities[:, 1]
    predictions = (scores >= threshold).astype(np.int64)
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))
    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    total = max(1, y_true.size)
    precision = tp / max(1, tp + fp)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2.0 * precision * sensitivity / max(1e-12, precision + sensitivity)
    auc = _roc_auc(y_true, scores)
    return {
        "accuracy": float((tp + tn) / total),
        "precision": float(precision),
        "f1_score": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auc": None if auc is None else float(auc),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = y_true == 1
    negatives = y_true == 0
    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(negatives))
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y_true[order]
    tps = np.cumsum(sorted_y == 1)
    fps = np.cumsum(sorted_y == 0)
    tpr = np.concatenate(([0.0], tps / positive_count, [1.0]))
    fpr = np.concatenate(([0.0], fps / negative_count, [1.0]))
    return float(_trapezoid(tpr, fpr))


class NumpyMLPClassifier:
    """A compact ReLU MLP classifier with Adam optimization."""

    def __init__(self, input_dim: int, config: MLPConfig) -> None:
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        dimensions = [input_dim, *config.hidden_dims, 2]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
            scale = math.sqrt(2.0 / fan_in)
            self.weights.append((self.rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))
        self.best_weights = self._copy_parameters()

    def _copy_parameters(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return ([w.copy() for w in self.weights], [b.copy() for b in self.biases])

    def _restore_parameters(self, state: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
        self.weights = [w.copy() for w in state[0]]
        self.biases = [b.copy() for b in state[1]]

    def _forward(self, x: np.ndarray, training: bool) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]]:
        activation = x.astype(np.float32)
        cache: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            previous = activation
            z = previous @ weight + bias
            if layer == len(self.weights) - 1:
                activation = z
                cache.append((previous, z, None))
                continue
            activation = np.maximum(z, 0.0)
            mask = None
            if training and self.config.dropout > 0:
                mask = (self.rng.random(activation.shape) >= self.config.dropout).astype(np.float32)
                activation = activation * mask / (1.0 - self.config.dropout)
            cache.append((previous, z, mask))
        return activation, cache

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(np.asarray(x, dtype=np.float32), training=False)
        return _softmax(logits)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Fit the MLP and optionally record test loss for visualization only.

        Test loss is never used for early stopping or parameter selection.
        """
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        x_val = np.asarray(x_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.int64)
        if (x_test is None) != (y_test is None):
            raise ValueError("x_test and y_test must be provided together")
        if x_test is not None and y_test is not None:
            x_test = np.asarray(x_test, dtype=np.float32)
            y_test = np.asarray(y_test, dtype=np.int64)
        if x_train.ndim != 2 or x_val.ndim != 2 or x_train.shape[1] != x_val.shape[1]:
            raise ValueError("Train and validation features must be 2-D with the same width")
        if x_test is not None and (x_test.ndim != 2 or x_test.shape[1] != x_train.shape[1]):
            raise ValueError("Test features must be 2-D with the same width as train features")
        m_weights = [np.zeros_like(w) for w in self.weights]
        v_weights = [np.zeros_like(w) for w in self.weights]
        m_biases = [np.zeros_like(b) for b in self.biases]
        v_biases = [np.zeros_like(b) for b in self.biases]
        history: list[dict[str, Any]] = []
        best_auc = -np.inf
        best_loss = np.inf
        stale_epochs = 0
        step = 0
        batch_size = min(self.config.batch_size, x_train.shape[0])

        for epoch in range(1, self.config.epochs + 1):
            order = self.rng.permutation(x_train.shape[0])
            train_losses: list[float] = []
            for start in range(0, x_train.shape[0], batch_size):
                batch_indices = order[start : start + batch_size]
                xb = x_train[batch_indices]
                yb = y_train[batch_indices]
                logits, cache = self._forward(xb, training=True)
                probabilities = _softmax(logits)
                loss = -np.mean(np.log(np.maximum(probabilities[np.arange(yb.size), yb], 1e-12)))
                train_losses.append(float(loss))
                delta = probabilities
                delta[np.arange(yb.size), yb] -= 1.0
                delta /= yb.size
                grad_w: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]
                grad_b: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]
                for layer in range(len(self.weights) - 1, -1, -1):
                    previous, z, mask = cache[layer]
                    grad_w[layer] = previous.T @ delta + self.config.weight_decay * self.weights[layer]
                    grad_b[layer] = np.sum(delta, axis=0)
                    if layer > 0:
                        delta = delta @ self.weights[layer].T
                        # The derivative belongs to the previous hidden
                        # layer, not to the layer whose weights were just
                        # differentiated.
                        previous_z = cache[layer - 1][1]
                        previous_mask = cache[layer - 1][2]
                        delta = delta * (previous_z > 0.0)
                        if previous_mask is not None:
                            delta = delta * previous_mask / (1.0 - self.config.dropout)
                step += 1
                beta1, beta2 = 0.9, 0.999
                for layer in range(len(self.weights)):
                    m_weights[layer] = beta1 * m_weights[layer] + (1 - beta1) * grad_w[layer]
                    v_weights[layer] = beta2 * v_weights[layer] + (1 - beta2) * (grad_w[layer] ** 2)
                    m_biases[layer] = beta1 * m_biases[layer] + (1 - beta1) * grad_b[layer]
                    v_biases[layer] = beta2 * v_biases[layer] + (1 - beta2) * (grad_b[layer] ** 2)
                    mhat_w = m_weights[layer] / (1 - beta1**step)
                    vhat_w = v_weights[layer] / (1 - beta2**step)
                    mhat_b = m_biases[layer] / (1 - beta1**step)
                    vhat_b = v_biases[layer] / (1 - beta2**step)
                    self.weights[layer] -= self.config.learning_rate * mhat_w / (np.sqrt(vhat_w) + 1e-8)
                    self.biases[layer] -= self.config.learning_rate * mhat_b / (np.sqrt(vhat_b) + 1e-8)

            train_probabilities = self.predict_proba(x_train)
            val_probabilities = self.predict_proba(x_val)
            train_metrics = binary_metrics(y_train, train_probabilities)
            val_metrics = binary_metrics(y_val, val_probabilities)
            val_loss = -float(np.mean(np.log(np.maximum(val_probabilities[np.arange(y_val.size), y_val], 1e-12))))
            test_loss = None
            if x_test is not None and y_test is not None:
                test_probabilities = self.predict_proba(x_test)
                test_loss = -float(
                    np.mean(np.log(np.maximum(test_probabilities[np.arange(y_test.size), y_test], 1e-12)))
                )
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "test_loss": test_loss,
                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
                "val_f1_score": val_metrics["f1_score"],
            }
            history.append(record)
            score = -np.inf if val_metrics["auc"] is None else float(val_metrics["auc"])
            improved = score > best_auc + self.config.min_delta
            if not improved and score == best_auc and val_loss < best_loss - self.config.min_delta:
                improved = True
            if improved:
                best_auc = score
                best_loss = val_loss
                self.best_weights = self._copy_parameters()
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.config.patience:
                break
        self._restore_parameters(self.best_weights)
        return history

    def save(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        for index, weight in enumerate(self.weights):
            arrays[f"weight_{index}"] = weight
        for index, bias in enumerate(self.biases):
            arrays[f"bias_{index}"] = bias
        np.savez(path, **arrays)


def json_ready(value: Any) -> Any:
    """Convert NumPy values and dataclasses into JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2), encoding="utf-8")

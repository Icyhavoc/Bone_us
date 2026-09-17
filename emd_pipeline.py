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

# The new ``after_split_data`` dataset uses ``validation`` while the legacy
# ``raw_data`` dataset uses ``val``.  Both names are accepted everywhere.
_AFTER_SPLIT_DIR_NAMES = {"train": "train", "val": "validation", "test": "test"}
_LEGACY_DIR_NAMES = {"train": "train", "val": "val", "test": "test"}
AFTER_SPLIT_BRANCHES = ("main", "tail")


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

    ``branch_length_mm`` is the adjustable ``a`` in the project definition.
    It is converted to sample points with the same depth-to-sample convention
    as :class:`DepthMapper`; the first branch then contains that many original
    samples starting at the detected onset of the first prominent envelope
    peak.  The second branch contains all remaining samples after that
    interval.
    """

    branch_length_mm: float = 0.75
    smooth_window: int = 21
    prominence_sigma: float = 2.0
    min_relative_height: float = 0.2
    min_peak_width: int = 12

    def validate(self) -> None:
        if self.branch_length_mm <= 0:
            raise ValueError("dynamic envelope branch_length_mm must be positive")
        if self.smooth_window <= 0:
            raise ValueError("dynamic envelope smooth_window must be positive")
        if self.prominence_sigma < 0:
            raise ValueError("dynamic envelope prominence_sigma cannot be negative")
        if not 0.0 <= self.min_relative_height <= 1.0:
            raise ValueError("dynamic envelope min_relative_height must be in [0, 1]")
        if self.min_peak_width <= 0:
            raise ValueError("dynamic envelope min_peak_width must be positive")


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


def _hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    """Compute a real-signal Hilbert envelope with NumPy's FFT primitives."""

    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        return signal.copy()
    centered = signal - np.median(signal)
    # Reflect padding prevents the FFT's implicit periodic boundary from
    # creating a false high envelope at the first or last few samples.
    pad = min(max(32, signal.size // 8), signal.size - 1)
    padded = np.pad(centered, (pad, pad), mode="reflect") if pad else centered
    spectrum = np.fft.fft(padded)
    mask = np.zeros(padded.size, dtype=np.float64)
    mask[0] = 1.0
    if padded.size % 2 == 0:
        mask[padded.size // 2] = 1.0
        mask[1 : padded.size // 2] = 2.0
    else:
        mask[1 : (padded.size + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * mask)
    return np.abs(analytic[pad : pad + signal.size]).astype(np.float64)


def _smooth_envelope(envelope: np.ndarray, window: int) -> np.ndarray:
    """Smooth an envelope while avoiding artificial zero-padding at edges."""

    envelope = np.asarray(envelope, dtype=np.float64).reshape(-1)
    if envelope.size == 0 or window <= 1:
        return envelope.copy()
    window = min(int(window), envelope.size)
    if window % 2 == 0:
        window = max(1, window - 1)
    if window == 1:
        return envelope.copy()
    radius = window // 2
    padded = np.pad(envelope, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def detect_dynamic_envelope_start(
    processed: np.ndarray,
    config: DynamicEnvelopeConfig = DynamicEnvelopeConfig(),
) -> dict[str, float | int]:
    """Find the onset of the first prominent envelope peak.

    The envelope is computed independently for every processed stream and
    physical channel, then averaged.  A robust median/MAD baseline determines
    significance.  The first local maximum above that baseline is selected;
    its onset is the point where the smoothed envelope first rises above half
    of that peak's excess over baseline.  If no significant local maximum is
    found, the global maximum is used as a deterministic fallback.
    """

    config.validate()
    if processed.ndim != 3:
        raise ValueError(f"Expected [streams, channels, samples], got {processed.shape}")
    sample_count = processed.shape[-1]
    if sample_count < 2:
        raise ValueError("Dynamic envelope detection requires at least two samples")

    flattened = processed.reshape(-1, sample_count)
    envelopes = np.stack([_hilbert_envelope(row) for row in flattened], axis=0)
    envelope = np.mean(envelopes, axis=0)
    smoothed = _smooth_envelope(envelope, config.smooth_window)
    baseline = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - baseline)))
    sigma = max(1.4826 * mad, float(np.std(smoothed)) * 0.1, 1e-8)
    global_excess = max(float(np.max(smoothed)) - baseline, 0.0)
    threshold = baseline + max(
        config.prominence_sigma * sigma,
        config.min_relative_height * global_excess,
    )
    maxima, _ = _local_extrema(smoothed)
    candidates = maxima[smoothed[maxima] >= threshold]

    peak_index: int
    if candidates.size:
        peak_index = int(candidates[0])
        selected_width = 0
        for candidate in candidates:
            peak_value = float(smoothed[candidate])
            edge_level = baseline + 0.5 * max(peak_value - baseline, sigma)
            left = int(candidate)
            right = int(candidate)
            while left > 0 and smoothed[left - 1] >= edge_level:
                left -= 1
            while right + 1 < sample_count and smoothed[right + 1] >= edge_level:
                right += 1
            width = right - left + 1
            if width >= config.min_peak_width:
                peak_index = int(candidate)
                selected_width = width
                break
        if selected_width == 0:
            candidate = peak_index
            edge_level = baseline + 0.5 * max(float(smoothed[candidate]) - baseline, sigma)
            left = candidate
            while left > 0 and smoothed[left - 1] >= edge_level:
                left -= 1
            right = candidate
            while right + 1 < sample_count and smoothed[right + 1] >= edge_level:
                right += 1
            selected_width = right - left + 1
    else:
        peak_index = int(np.argmax(smoothed))
        selected_width = 0

    peak_value = float(smoothed[peak_index])
    edge_level = baseline + 0.5 * max(peak_value - baseline, sigma)
    onset = peak_index
    while onset > 0 and smoothed[onset - 1] >= edge_level:
        onset -= 1
    return {
        "start_index": int(onset),
        "peak_index": int(peak_index),
        "peak_width": int(selected_width),
        "baseline": baseline,
        "sigma": sigma,
        "threshold": threshold,
    }


def prepare_dynamic_envelope_branches(
    processed: np.ndarray,
    config: DynamicEnvelopeConfig = DynamicEnvelopeConfig(),
    target_length: int = 512,
    tukey_alpha: float = 0.3,
    max_depth_mm: float = 5.0,
    apply_tukey: bool = True,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Create the two dynamic-envelope branches for one processed sample."""

    config.validate()
    sample_count = int(processed.shape[-1])
    detection = detect_dynamic_envelope_start(processed, config)
    if max_depth_mm <= 0:
        raise ValueError("max_depth_mm must be positive")
    requested_a = int(
        math.floor(config.branch_length_mm / max_depth_mm * sample_count + 0.5)
    )
    effective_a = min(max(1, requested_a), sample_count - 1)
    max_start = max(0, sample_count - effective_a - 1)
    start = min(int(detection["start_index"]), max_start)
    split = start + effective_a
    intervals = [(start, split), (split, sample_count)]
    branches = [
        prepare_interval_signals(
            processed,
            interval_start,
            interval_end,
            target_length=target_length,
            tukey_alpha=tukey_alpha,
            apply_tukey=apply_tukey,
        )
        for interval_start, interval_end in intervals
    ]
    branch_info = [
        {
            "name": "dynamic_peak_window",
            "start_index": int(start),
            "end_index_exclusive": int(split),
            "input_length": int(split - start),
            "feature_length": None,
        },
        {
            "name": "dynamic_after_peak_window",
            "start_index": int(split),
            "end_index_exclusive": int(sample_count),
            "input_length": int(sample_count - split),
            "feature_length": None,
        },
    ]
    info: dict[str, Any] = {
        "detected_peak_start_index": int(detection["start_index"]),
        "peak_index": int(detection["peak_index"]),
        "requested_branch_length_mm": float(config.branch_length_mm),
        "effective_branch_length_mm": float(effective_a / sample_count * max_depth_mm),
        "requested_branch_length_samples": int(requested_a),
        "effective_branch_length_samples": int(effective_a),
        "max_depth_mm": float(max_depth_mm),
        "sample_length": sample_count,
        "baseline": float(detection["baseline"]),
        "sigma": float(detection["sigma"]),
        "threshold": float(detection["threshold"]),
        "peak_width": int(detection["peak_width"]),
        "branches": branch_info,
    }
    return branches, info


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
        branch_signals_list, dynamic_info = prepare_dynamic_envelope_branches(
            processed,
            config=dynamic_envelope_config,
            target_length=target_length,
            tukey_alpha=tukey_alpha,
            max_depth_mm=mapper.max_depth_mm,
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


def load_split(data_dir: Path | str, split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load X, y, and sample metadata for one split."""

    split_dir = Path(data_dir) / split
    x_path = split_dir / "X.npy"
    y_path = split_dir / "y.npy"
    samples_path = split_dir / "samples.json"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing X.npy/y.npy under {split_dir}")
    x = np.load(x_path).astype(np.float32)
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


@dataclass(frozen=True)
class DatasetSpec:
    """Which dataset layout a run reads.

    ``legacy``
        ``raw_data``: full ``[N, 50, 2, 896]`` frames that are cut into depth
        branches and then resampled.
    ``after_split``
        ``after_split_data``: pre-cut ``main``/``tail`` branches that are
        resampled instead of depth-sliced; depth regions do not apply.
    """

    kind: str = "legacy"
    root: Path = Path("raw_data")
    branch: str = "main"

    def __post_init__(self) -> None:
        if self.kind not in {"legacy", "after_split"}:
            raise ValueError("kind must be 'legacy' or 'after_split'")
        if self.kind == "after_split" and self.branch not in AFTER_SPLIT_BRANCHES:
            raise ValueError(f"branch must be one of {AFTER_SPLIT_BRANCHES}")


def detect_dataset_kind(data_dir: Path | str) -> str:
    """Return ``after_split`` when ``data_dir`` holds branch arrays, else ``legacy``."""

    root = Path(data_dir)
    if not root.exists():
        return "legacy"
    for directory in root.iterdir():
        if directory.is_dir() and (directory / "arrays" / "main_signal.npy").exists():
            return "after_split"
    return "legacy"


def load_dataset_split(
    spec: DatasetSpec, split: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load one split for whichever dataset layout ``spec`` describes."""

    if spec.kind == "after_split":
        return load_after_split_split(spec.root, split, spec.branch)
    return load_split(spec.root, split)


def load_after_split_split(
    data_dir: Path | str, split: str, branch: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load one branch of the ``after_split_data`` dataset.

    The dataset stores two independent, fixed-length branches per point:

    * ``main_signal`` with shape ``[N, 50, 2, 170]``;
    * ``tail_signal`` with shape ``[N, 50, 2, 645]``.

    Both are raw (already normalized to ``[-1, 1]``) slices of the same
    896-sample axis, so no depth mapping is required for either branch.
    """

    if branch not in {"main", "tail"}:
        raise ValueError("branch must be 'main' or 'tail'")
    split_dir = Path(data_dir) / (_AFTER_SPLIT_DIR_NAMES.get(split, split))
    signal_path = split_dir / "arrays" / f"{branch}_signal.npy"
    y_path = split_dir / "y.npy"
    if not signal_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {branch}_signal.npy or y.npy under {split_dir}")
    x = np.load(signal_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    if x.ndim != 4 or x.shape[1:3] != (50, 2):
        raise ValueError(f"Expected {branch} signal [N,50,2,L], got {x.shape}")
    if y.shape != (x.shape[0],):
        raise ValueError(f"Expected y shape {(x.shape[0],)}, got {y.shape}")
    samples_path = split_dir / "samples.json"
    if samples_path.exists():
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
    else:
        samples = [{} for _ in range(x.shape[0])]
    absolute_index_path = split_dir / "arrays" / f"{branch}_absolute_index.npy"
    absolute_index = (
        np.load(absolute_index_path) if absolute_index_path.exists() else None
    )
    for index, record in enumerate(samples):
        record.setdefault("sample_id", record.get("point_id"))
        record.setdefault("depth_value", record.get("thickness"))
        record["branch"] = branch
        if absolute_index is not None:
            record["absolute_index_range"] = [
                int(np.min(absolute_index[index])),
                int(np.max(absolute_index[index])) + 1,
            ]
    return x, y, samples


def _select_branch_indices(absolute_index: np.ndarray, channel_mode: int) -> np.ndarray:
    """Pick the absolute-index rows matching the selected physical channels.

    Works for a single sample ``[channels, length]`` and for a whole split
    ``[samples, channels, length]``.
    """

    if absolute_index.ndim < 2 or absolute_index.shape[-2] != 2:
        raise ValueError(
            f"Expected absolute index [..., 2, length], got {absolute_index.shape}"
        )
    if channel_mode == 1:
        return absolute_index[..., 0:1, :]
    if channel_mode == 2:
        return absolute_index[..., 1:2, :]
    if channel_mode == 3:
        return absolute_index
    raise ValueError("channel_mode must be 1, 2, or 3")


def _gather_branch(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Gather ``[streams, channels, L_full]`` at ``index [channels, L_branch]``."""

    if values.ndim != 3:
        raise ValueError(f"Expected [streams, channels, samples], got {values.shape}")
    return np.take_along_axis(values, index[None, :, :], axis=2)


@dataclass
class AfterSplitPair:
    """Both pre-cut branches of one split plus the reconstructed full axis."""

    full: np.ndarray
    main: np.ndarray
    tail: np.ndarray
    main_index: np.ndarray
    tail_index: np.ndarray
    y: np.ndarray
    samples: list[dict[str, Any]]

    @property
    def count(self) -> int:
        return int(self.full.shape[0])

    def absolute_range(self, channel_mode: int) -> tuple[int, int]:
        """Smallest/largest covered absolute sample, for the frame-scoring region."""

        rows = [row for row in (self.main_index, self.tail_index)]
        selected = [_select_branch_indices(row, channel_mode) for row in rows]
        return (
            int(min(int(np.min(item)) for item in selected)),
            int(max(int(np.max(item)) for item in selected)) + 1,
        )


def load_after_split_pair(
    data_dir: Path | str, split: str, absolute_length: int = 896
) -> AfterSplitPair:
    """Load both branches and rebuild the shared original axis.

    ``main_absolute_index`` and ``tail_absolute_index`` record where each
    branch sits on the original axis, so re-inserting both branches recovers
    the original ``[N, 50, 2, 896]`` array.  The two branches overlap by a
    median of 10-16 samples and were verified to be numerically identical
    there, so the reconstruction is lossless inside ``[70, 896)``.  Samples
    before the earliest branch start stay zero; frame scoring must therefore
    use the covered range rather than the whole axis.
    """

    main, y, samples = load_after_split_split(data_dir, split, "main")
    tail, _, _ = load_after_split_split(data_dir, split, "tail")
    split_dir = Path(data_dir) / (_AFTER_SPLIT_DIR_NAMES.get(split, split))
    main_index = np.load(split_dir / "arrays" / "main_absolute_index.npy").astype(np.int64)
    tail_index = np.load(split_dir / "arrays" / "tail_absolute_index.npy").astype(np.int64)
    if main.shape[0] != tail.shape[0] or main_index.shape[0] != tail_index.shape[0]:
        raise ValueError("main and tail branches must share the same sample axis")
    full = np.zeros(
        (main.shape[0], main.shape[1], main.shape[2], absolute_length), dtype=np.float32
    )
    for index in range(main.shape[0]):
        for channel in range(main.shape[2]):
            # Fancy indexing moves the leading frame axis last, so transpose
            # the frame block before assigning into the absolute positions.
            full[index, :, channel, tail_index[index, channel]] = tail[index, :, channel].T
            full[index, :, channel, main_index[index, channel]] = main[index, :, channel].T
    return AfterSplitPair(
        full=full,
        main=main,
        tail=tail,
        main_index=main_index,
        tail_index=tail_index,
        y=y,
        samples=samples,
    )


def load_after_split_pair_split(
    data_dir: Path | str, split: str, absolute_length: int = 896
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Convenience wrapper returning ``(full_axis, y, samples)``."""

    pair = load_after_split_pair(data_dir, split, absolute_length=absolute_length)
    return pair.full, pair.y, pair.samples


def _resample_branch(x: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a branch array ``[N, 50, 2, L]`` along its last axis."""

    if x.shape[-1] == target_length:
        return np.asarray(x, dtype=np.float32)
    return resample_signals(
        x.reshape(-1, x.shape[-1]).astype(np.float32), target_length
    ).reshape(*x.shape[:-1], target_length)


def after_split_pair_feature_vector(
    full_sample: np.ndarray,
    main_index: np.ndarray,
    tail_index: np.ndarray,
    channel_mode: int,
    target_length: int,
    emd_config: EMDConfig,
    tukey_alpha: float = 0.3,
    form: str = "top3_mean",
    score_region: RegionSpec | None = None,
    mapper: DepthMapper | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Feature vector for one point using **both** pre-cut branches.

    Frame processing happens **once** on the reconstructed full axis, so both
    branches share the same processed streams (and the same selected frames
    for ``max1`` / ``top3_mean``).  Each branch is then resampled to
    ``target_length`` independently, windowed, decomposed, and the per-branch
    feature vectors are concatenated.
    """

    if full_sample.ndim != 3:
        raise ValueError(f"Expected [frames, channels, samples], got {full_sample.shape}")
    if score_region is None:
        score_region = RegionSpec(0.0, 5.0, "selection_full")
    if mapper is None:
        mapper = DepthMapper(max_depth_mm=5.0, signal_length=int(full_sample.shape[-1]))

    selected_channels = select_channels(full_sample, channel_mode)
    processed, selected_frames = process_frames(
        selected_channels, form, mapper=mapper, score_region=score_region
    )
    branch_vectors: list[np.ndarray] = []
    branch_info: list[dict[str, Any]] = []
    for branch, absolute_index in (("main", main_index), ("tail", tail_index)):
        branch_index = _select_branch_indices(absolute_index, channel_mode)
        branch_signals = _gather_branch(processed, branch_index)
        native_length = int(branch_signals.shape[-1])
        resampled = _resample_branch(branch_signals, target_length)
        windowed = resampled * tukey_window(target_length, tukey_alpha)[None, None, :]
        vector = emd_feature_vector(windowed.reshape(-1, target_length), emd_config)
        branch_vectors.append(vector)
        branch_info.append(
            {
                "name": branch,
                "native_length": native_length,
                "target_length": int(target_length),
                "resampled": native_length != int(target_length),
                "absolute_start": int(np.min(branch_index)),
                "absolute_end_exclusive": int(np.max(branch_index)) + 1,
                "input_streams": int(branch_signals.shape[0] * branch_signals.shape[1]),
                "feature_length": int(vector.size),
            }
        )
    feature = np.concatenate(branch_vectors).astype(np.float32)
    info = {
        "form": canonical_form(form),
        "channel_mode": channel_mode,
        "branches": branch_info,
        "selected_frames": None if selected_frames is None else selected_frames.tolist(),
        "score_region": [score_region.start_mm, score_region.end_mm],
        "feature_length": int(feature.size),
    }
    return feature, info


def build_after_split_pair_feature_matrix(
    pair: AfterSplitPair,
    channel_mode: int,
    target_length: int,
    emd_config: EMDConfig,
    tukey_alpha: float = 0.3,
    form: str = "top3_mean",
    score_region: RegionSpec | None = None,
    mapper: DepthMapper | None = None,
    limit: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build concatenated two-branch features for a whole split."""

    count = pair.count if limit is None else min(limit, pair.count)
    vectors: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    for index in range(count):
        vector, info = after_split_pair_feature_vector(
            pair.full[index],
            pair.main_index[index],
            pair.tail_index[index],
            channel_mode=channel_mode,
            target_length=target_length,
            emd_config=emd_config,
            tukey_alpha=tukey_alpha,
            form=form,
            score_region=score_region,
            mapper=mapper,
        )
        vectors.append(vector)
        infos.append(info)
    matrix = np.stack(vectors, axis=0).astype(np.float32)
    return matrix, infos


def build_after_split_feature_matrix(
    x: np.ndarray,
    branch: str,
    channel_mode: int,
    target_length: int,
    emd_config: EMDConfig,
    tukey_alpha: float = 0.3,
    form: str = "raw50",
    score_region: RegionSpec = RegionSpec(0.0, 5.0, "selection_full"),
    mapper: DepthMapper | None = None,
    limit: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build EMD features for every sample of one ``after_split_data`` branch.

    Frame processing and channel selection are reused unchanged.  The branch
    is then linearly resampled from its native length (170 or 645) to
    ``target_length``, windowed, and decomposed exactly like the legacy
    depth-region branches.
    """

    if limit is not None:
        x = x[:limit]
    vectors: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    for index, sample in enumerate(x):
        vector, info = after_split_feature_vector(
            sample,
            branch=branch,
            channel_mode=channel_mode,
            target_length=target_length,
            emd_config=emd_config,
            tukey_alpha=tukey_alpha,
            form=form,
            score_region=score_region,
            mapper=mapper,
        )
        vectors.append(vector)
        infos.append(info)
    matrix = np.stack(vectors, axis=0).astype(np.float32)
    return matrix, infos


def after_split_feature_vector(
    sample: np.ndarray,
    branch: str,
    channel_mode: int,
    target_length: int,
    emd_config: EMDConfig,
    tukey_alpha: float = 0.3,
    form: str = "raw50",
    score_region: RegionSpec = RegionSpec(0.0, 5.0, "selection_full"),
    mapper: DepthMapper | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Frame-process, resample one branch to ``target_length``, then run EMD."""

    if sample.ndim != 3 or sample.shape[0] < 1 or sample.shape[1] != 2:
        raise ValueError(f"Expected sample [50, 2, L], got {sample.shape}")
    mapper = mapper if mapper is not None else DepthMapper(
        max_depth_mm=5.0, signal_length=int(sample.shape[-1])
    )
    selected_channels = select_channels(sample, channel_mode)
    processed, selected_frames = process_frames(
        selected_channels, form, mapper=mapper, score_region=score_region
    )
    native_length = int(processed.shape[-1])
    resampled = _resample_branch(processed[None, ...], target_length)[0]
    windowed = resampled * tukey_window(target_length, tukey_alpha)[None, None, :]
    streams = windowed.reshape(-1, target_length)
    feature = emd_feature_vector(streams, emd_config)
    info = {
        "form": canonical_form(form),
        "channel_mode": channel_mode,
        "branch": branch,
        "native_length": native_length,
        "target_length": int(target_length),
        "resampled": native_length != int(target_length),
        "input_streams": int(streams.shape[0]),
        "feature_length": int(feature.size),
        "selected_frames": None if selected_frames is None else selected_frames.tolist(),
    }
    return feature.astype(np.float32), info


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
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build the feature matrix for a split."""

    if limit is not None:
        x = x[:limit]
    vectors: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    for index, sample in enumerate(x):
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

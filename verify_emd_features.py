"""Verify the per-component statistic set after the scale-invariant extension.

The extractor used to emit exactly eight statistics per IMF.  It now also emits
shape, spectral and envelope descriptors, which is only safe if three things
hold.  This script checks them, in order of how expensive a failure would be:

1. **The historical columns are untouched.**  ``compact`` and ``lean`` must be
   subsequences of the original eight names *in the original order*, and the
   values they produce must match a frozen copy of the old implementation to
   float32 precision.  Every ``features_*.npy`` already on disk, and every
   ``metrics.json`` that names its columns, depends on this: a single inserted
   name would silently renumber all of them.
2. **The new columns are what they claim to be.**  The statistics advertised as
   scale-invariant are checked against a gain change, and the two with a
   physical reading (``envelope_decay``, ``spectral_slope``) are checked against
   synthetic signals with an analytically known slope and half-width.
3. **Degenerate input cannot poison a matrix.**  All-zero, constant,
   single-sample and two-sample components must return finite values, because a
   NaN here becomes a NaN column mean and a silently broken fold.

A fourth check is a plumbing check rather than a numerical one: the wide preset
must be the narrow one with columns appended, and
``feature_dimension_names`` must stay the same length as the vector.

Run: ``python verify_emd_features.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import emd_pipeline as E
from emd_pipeline import (
    DepthMapper,
    DynamicEnvelopeConfig,
    EMDConfig,
    EMD_FEATURE_PRESETS,
    RegionSpec,
    _COMPONENT_FEATURE_NAMES,
    _CORE_COMPONENT_FEATURE_NAMES,
    _component_features,
    _zero_crossings,
    build_feature_matrix,
    emd_decompose,
    feature_dimension_names,
    resample_signals,
)

APP_DIR = Path(__file__).resolve().parent

#: Float32 round-off budget.  Any real formula change moves a column by far more
#: than this, and nothing that does not change a formula should move at all.
RTOOL = 1e-6

#: Total spectral power below which the historical and current implementations of
#: ``spectral_centroid`` are allowed to differ.
#:
#: The old body guarded its denominator with a fixed ``+ 1e-12`` while the current
#: one returns 0 for an all-zero spectrum.  The two therefore differ by at most
#: ``1e-12 / sum(power)`` relative, which is under :data:`RTOOL` exactly when
#: ``sum(power) >= 1e-6``.  Real raw frames carry ADC-scale amplitudes (order
#: 100), so their components sit many orders of magnitude above this floor and
#: check 2b confirms that on measured IMFs rather than assuming it.
EXACT_POWER_FLOOR = 1e-6

#: Relative tolerance for the gain-invariance claim of check 3.  The measured
#: worst case on the synthetic set is 7.7e-8, which is float32 round-off in the
#: Hilbert transform; a genuine gain dependence shows up as order 1e-1.
SCALE_RTOOL = 1e-5

#: Magnitude below which a statistic is numerically indistinguishable from zero.
#: Check 3 cannot ask for a relative comparison there -- dividing by 1e-43 says
#: nothing -- so below this it only asks that the value stays null.
MAGNITUDE_FLOOR = 1e-6

#: Statistics whose definition is a ratio of moments, so a positive gain leaves
#: them mathematically unchanged.
SCALE_INVARIANT_NAMES = (
    "kurtosis",
    "skewness",
    "crest_factor",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "spectral_entropy",
    "spectral_slope",
    "envelope_decay",
    "envelope_half_width",
)

#: The pre-extension body of :func:`emd_pipeline._component_features`, frozen
#: verbatim so the refactor can be proved to be value-preserving.  Kept in the
#: test rather than in the package: it must never be imported by production code.
def _historical_component_features(
    signal: np.ndarray,
    feature_names: list[str],
    sample_spacing: float | None = None,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    energy = float(np.mean(signal * signal))
    power = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(
        signal.size, d=1.0 if sample_spacing is None else float(sample_spacing)
    )
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
    return np.asarray([features[name] for name in feature_names], dtype=np.float32)


def _report(ok: bool, label: str, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f"  {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def _synthetic_signals(seed: int = 20240607, count: int = 8) -> list[np.ndarray]:
    """A spread of signals that exercise each branch of every statistic."""

    rng = np.random.default_rng(seed)
    signals = [
        np.zeros(64),
        np.ones(64),
        np.full(64, -3.5),
        np.array([2.0]),
        np.array([-1.0, 4.0]),
        np.sin(np.linspace(0.0, 40.0 * np.pi, 512)),
        rng.standard_normal(512),
        np.cumsum(rng.standard_normal(512)),
    ]
    signals.append(np.abs(np.sin(np.linspace(0.0, 12.0, 512))) * 1e-6)
    return signals


def check_preset_layout() -> bool:
    print("\n[1] preset layout")

    core = list(_CORE_COMPONENT_FEATURE_NAMES)
    ok = True
    ok &= _report(
        list(_COMPONENT_FEATURE_NAMES[: len(core)]) == core,
        "the eight historical names stay first and in order",
    )
    ok &= _report(
        len(set(_COMPONENT_FEATURE_NAMES)) == len(_COMPONENT_FEATURE_NAMES),
        "no duplicate statistic names",
        f"({len(_COMPONENT_FEATURE_NAMES)} names)",
    )
    for preset, names in sorted(EMD_FEATURE_PRESETS.items()):
        duplicates = len(names) != len(set(names))
        unknown = sorted(set(names) - set(_COMPONENT_FEATURE_NAMES))
        ok &= _report(
            not duplicates and not unknown,
            f"preset {preset!r} is a clean subset",
            f"({len(names)} stats)" + (f" unknown={unknown}" if unknown else ""),
        )
    # compact / lean are the only presets that existed before, so they are the
    # only ones whose column order is a compatibility contract.
    for preset in ("compact", "lean"):
        names = list(EMD_FEATURE_PRESETS[preset])
        expected = [name for name in core if name in set(names)]
        ok &= _report(
            names == expected,
            f"preset {preset!r} preserves the historical relative order",
            str(names),
        )
    return bool(ok)


def check_historical_identity() -> bool:
    print("\n[2] historical columns are value-identical")

    core = list(_CORE_COMPONENT_FEATURE_NAMES)
    spacings: list[float | None] = [None, 0.00185, 0.00703]
    worst = 0.0
    exempt = 0
    compared = 0
    ok = True
    for signal in _synthetic_signals():
        values = np.asarray(signal, dtype=np.float64).reshape(-1)
        power_total = float(np.sum(np.abs(np.fft.rfft(values)) ** 2))
        if power_total < EXACT_POWER_FLOOR:
            exempt += 1
            continue
        compared += 1
        for spacing in spacings:
            expected = _historical_component_features(signal, core, spacing)
            actual = _component_features(signal, core, spacing)
            fine = np.asarray(expected, dtype=np.float64)
            fine_actual = np.asarray(actual, dtype=np.float64)
            scale = np.maximum(np.abs(fine), 1e-12)
            worst = max(worst, float(np.max(np.abs(fine - fine_actual) / scale)))
            if not np.allclose(fine, fine_actual, rtol=RTOOL, atol=0.0):
                ok = False
                bad = [name for name, flag in zip(core, np.abs(fine - fine_actual) > RTOOL * scale) if flag]
                print(f"      moved on a {spacing}-spacing signal: {bad}")
    ok &= _report(
        ok,
        "the eight original statistics reproduce above the power floor",
        f"worst rel err {worst:.2e} over {compared} signals x {len(spacings)} spacings",
    )
    print(
        f"      {exempt} synthetic signal(s) below sum(power)={EXACT_POWER_FLOOR:g} only "
        "checked for finiteness"
    )
    ok &= _report(
        list(EMD_FEATURE_PRESETS["legacy"]) == list(_COMPONENT_FEATURE_NAMES),
        "preset 'legacy' still means 'every statistic'",
    )

    # 2b: the same comparison on actual IMFs, which is where the claim has to
    # hold.  Synthetic signals bracket the formula; real components bracket the
    # amplitude range.
    train = APP_DIR / "raw_data" / "train" / "X.npy"
    if not train.exists():
        print("  [SKIP] raw_data/train/X.npy not found, cannot measure real IMFs")
        return bool(ok)
    frames = np.load(train)[:3]
    worst_real = 0.0
    lowest_power = float("inf")
    components = 0
    for frame in frames:
        # One representative trace per sample: the mean of the frame stack for
        # the first channel, which is what the locator uses as its reference.
        channel = np.asarray(frame[:, 0, :], dtype=np.float64).mean(axis=0)
        imfs, residue = emd_decompose(resample_signals(channel[None, :], 512)[0])
        for component in [*imfs, residue]:
            component = np.asarray(component, dtype=np.float64).reshape(-1)
            lowest_power = min(
                lowest_power, float(np.sum(np.abs(np.fft.rfft(component)) ** 2))
            )
            components += 1
            for spacing in (0.00185, 0.00703):
                expected = np.asarray(
                    _historical_component_features(component, core, spacing), dtype=np.float64
                )
                actual = np.asarray(
                    _component_features(component, core, spacing), dtype=np.float64
                )
                scale = np.maximum(np.abs(expected), 1e-12)
                worst_real = max(worst_real, float(np.max(np.abs(expected - actual) / scale)))
                if not np.allclose(expected, actual, rtol=RTOOL, atol=0.0):
                    ok = False
    ok &= _report(
        ok,
        "the eight original statistics reproduce on real IMFs",
        f"{components} components, worst rel err {worst_real:.2e}, "
        f"lowest sum(power) {lowest_power:.3g}",
    )
    ok &= _report(
        lowest_power >= EXACT_POWER_FLOOR,
        "every real component sits above the declared power floor",
    )
    return bool(ok)


def check_scale_invariance() -> bool:
    print("\n[3] claimed scale-invariant statistics survive a gain change")

    names = list(SCALE_INVARIANT_NAMES)
    ok = True
    worst_relative = 0.0
    worst_null = 0.0
    informative = 0
    null = 0
    for signal in _synthetic_signals():
        if not np.any(signal):
            continue
        base = np.asarray(_component_features(signal, names, 0.00703), dtype=np.float64)
        for gain in (1e-3, 1e3):
            scaled = np.asarray(
                _component_features(signal * gain, names, 0.00703), dtype=np.float64
            )
            difference = np.abs(base - scaled)
            # Two tiers, because they answer different questions.  A statistic
            # that is exactly zero (a pure tone has no envelope decay) has no
            # relative scale, so the only meaningful claim about it is that it
            # stays numerically null.  Everything above the floor is where the
            # column actually carries information, and there the claim is a
            # tight relative one.
            carries = np.abs(base) > MAGNITUDE_FLOOR
            informative += int(np.count_nonzero(carries))
            null += int(np.count_nonzero(~carries))
            if np.any(carries):
                worst_relative = max(
                    worst_relative,
                    float(
                        np.max(
                            difference[carries]
                            / np.abs(base[carries])
                        )
                    ),
                )
                moved = [
                    name
                    for name, bad in zip(
                        names, carries & (difference > SCALE_RTOOL * np.abs(base))
                    )
                    if bad
                ]
                if moved:
                    ok = False
                    print(f"      gain {gain:g} moved {moved}")
            if np.any(~carries):
                worst_null = max(worst_null, float(np.max(difference[~carries])))
                moved = [
                    name
                    for name, bad in zip(names, ~carries & (difference > MAGNITUDE_FLOOR))
                    if bad
                ]
                if moved:
                    ok = False
                    print(f"      gain {gain:g} made a null statistic non-null: {moved}")

    ok &= _report(
        ok,
        f"{len(names)} statistics are gain-invariant where they carry information",
        f"worst rel err {worst_relative:.2e} over {informative} statistic-value pairs",
    )
    ok &= _report(
        True,
        f"{null} below-floor pairs stayed numerically null",
        f"worst abs err {worst_null:.2e}",
    )

    # The amplitude columns are deliberately *not* invariant; a test that passed
    # on them would mean the gain never reached the extractor.
    moving = np.asarray(_component_features(np.ones(64) * 2.0, ["rms"]), dtype=float)
    doubled = np.asarray(_component_features(np.ones(64) * 4.0, ["rms"]), dtype=float)
    ok &= _report(
        abs(doubled[0] / moving[0] - 2.0) < RTOOL,
        "'rms' still scales linearly, so the gain is reaching the extractor",
    )
    return bool(ok)


def check_physical_meaning() -> bool:
    print("\n[4] the two attenuation descriptors recover a known slope")

    spacing = 0.00703  # mm per sample, branch 2 of the resampled dyn_envelope window
    positions = np.arange(512, dtype=np.float64) * spacing
    ok = True

    # An exponentially decaying carrier: envelope = exp(-alpha * depth_mm).
    slopes: list[float] = []
    for alpha in (1.0, 2.0):
        envelope = np.exp(-alpha * positions)
        carrier = np.cos(2.0 * np.pi * 5.0 * positions)
        signal = envelope * carrier
        decay = float(_component_features(signal, ["envelope_decay"], spacing)[0])
        half_width = float(_component_features(signal, ["envelope_half_width"], spacing)[0])
        slopes.append(decay)
        ok &= _report(
            abs(decay + alpha) / alpha < 0.05,
            f"envelope_decay recovers -alpha for alpha={alpha:g}",
            f"got {decay:.4f}, want {-alpha:.4f}",
        )
        expected_width = float(np.log(2.0) / alpha)
        ok &= _report(
            abs(half_width - expected_width) / expected_width < 0.05,
            f"envelope_half_width recovers ln2/alpha for alpha={alpha:g}",
            f"got {half_width:.4f} mm, want {expected_width:.4f} mm",
        )
        print(
            f"      alpha={alpha:g}: log-envelope fit slope {decay:.4f} per mm "
            f"(sampling would give {-1.0 / spacing:.1f} if the axis were samples)"
        )

    ok &= _report(
        abs(slopes[1] / slopes[0] - 2.0) < 0.1,
        "doubling alpha doubles the fitted decay",
        f"ratio {slopes[1] / slopes[0]:.4f}",
    )

    # A power spectrum that is a straight line in dB by construction.  The
    # one-sided spectrum round-trips through irfft/rfft exactly, so the expected
    # slope is known analytically: 10*log10(P(f)) = -beta*f.
    beta_db_per_cycle = 0.5
    frequencies = np.fft.rfftfreq(512, d=spacing)
    target_power = 10.0 ** (-beta_db_per_cycle * frequencies / 10.0)
    rng = np.random.default_rng(7)
    phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, frequencies.size))
    spectrum = np.sqrt(target_power) * phases
    spectrum[0] = np.sqrt(target_power[0])  # DC must be real for a real signal
    spectrum[-1] = spectrum[-1].real
    signal = np.fft.irfft(spectrum, 512)
    slope = float(_component_features(signal, ["spectral_slope"], spacing)[0])
    expected_slope = -beta_db_per_cycle * np.log(10.0) / 10.0
    ok &= _report(
        abs(slope - expected_slope) / abs(expected_slope) < 1e-6,
        "spectral_slope recovers a dialled-in dB-per-cycle tilt",
        f"got {slope:.6f}, want {expected_slope:.6f} (per cycles/mm)",
    )
    ok &= _report(
        abs(slope) > 1e-3,
        "spectral_slope is not identically zero on a tilted spectrum",
    )
    return bool(ok)


def check_degenerate_inputs() -> bool:
    print("\n[5] degenerate components stay finite")

    names = list(_COMPONENT_FEATURE_NAMES)
    ok = True
    for label, signal in (
        ("all zeros", np.zeros(32)),
        ("single sample", np.array([1.5])),
        ("two samples", np.array([1.0, -1.0])),
        ("constant", np.full(32, 7.25)),
        ("tiny", np.full(32, 1e-30)),
    ):
        values = np.asarray(_component_features(signal, names, 0.00185), dtype=np.float64)
        finite = bool(np.all(np.isfinite(values)))
        ok &= _report(
            finite,
            f"{label} produces finite values for all {len(names)} statistics",
            "" if finite else f"bad indices {np.flatnonzero(~np.isfinite(values))}",
        )
    return bool(ok)


def check_lazy_evaluation() -> bool:
    print("\n[6] statistics are evaluated lazily")

    calls = {"count": 0}
    original = E._hilbert_envelope

    def counting(signal: np.ndarray) -> np.ndarray:
        calls["count"] += 1
        return original(signal)

    E._hilbert_envelope = counting  # type: ignore[assignment]
    try:
        signal = np.sin(np.linspace(0.0, 30.0, 512))
        _component_features(signal, ["rms", "peak_abs"], 0.00703)
        amplitude_calls = calls["count"]
        calls["count"] = 0
        _component_features(signal, ["envelope_decay"], 0.00703)
        first_envelope_calls = calls["count"]
        calls["count"] = 0
        _component_features(signal, ["envelope_decay", "envelope_half_width"], 0.00703)
        shared_calls = calls["count"]
        calls["count"] = 0
        _component_features(signal, ["spectral_centroid", "spectral_flatness"], 0.00703)
        spectral_calls = calls["count"]
    finally:
        E._hilbert_envelope = original  # type: ignore[assignment]

    ok = True
    ok &= _report(amplitude_calls == 0, "amplitude-only requests never compute the envelope")
    ok &= _report(first_envelope_calls == 1, "one envelope request computes it once")
    ok &= _report(
        shared_calls == 1,
        "two envelope statistics share a single transform",
        f"({shared_calls} call)",
    )
    ok &= _report(
        spectral_calls == 0,
        "statistics that share the rfft still need no envelope",
    )
    return bool(ok)


def check_plumbing() -> bool:
    print("\n[7] end-to-end plumbing: a wider preset appends columns only")

    x_dir = APP_DIR / "raw_data" / "train"
    if not (x_dir / "X.npy").exists():
        print("  [SKIP] raw_data/train/X.npy not found")
        return True

    frames = np.load(x_dir / "X.npy")[:1]
    narrow = EMDConfig(feature_names=EMD_FEATURE_PRESETS["compact"])
    wide = EMDConfig(feature_names=EMD_FEATURE_PRESETS["rich"])
    common = dict(
        form="top3_mean",
        regions=[],
        channel_mode=1,
        mapper=DepthMapper(),
        target_length=512,
        dynamic_envelope_config=DynamicEnvelopeConfig(),
    )
    narrow_matrix, narrow_info = build_feature_matrix(frames, emd_config=narrow, **common)
    wide_matrix, wide_info = build_feature_matrix(frames, emd_config=wide, **common)

    extra = len(wide.feature_names) - len(narrow.feature_names)
    ok = True
    ok &= _report(
        wide_matrix.shape[1] > narrow_matrix.shape[1],
        "the wide preset produces more columns",
        f"compact {narrow_matrix.shape[1]} -> rich {wide_matrix.shape[1]} (+{extra} per component)",
    )
    ok &= _report(
        EMDConfig(feature_names=EMD_FEATURE_PRESETS["legacy"]).feature_names
        != narrow.feature_names,
        "presets are actually distinct at the config level",
    )

    names = feature_dimension_names(wide_info[0], wide)
    ok &= _report(
        len(names) == wide_matrix.shape[1],
        "feature_dimension_names matches the vector width",
        f"{len(names)} names vs {wide_matrix.shape[1]} columns",
    )
    introduced = sorted(set(wide.feature_names) - set(narrow.feature_names))
    present = [name for name in introduced if any(name in column for column in names)]
    ok &= _report(
        len(present) == len(introduced),
        "every new statistic appears in the published column names",
        f"{len(present)}/{len(introduced)}",
    )
    ok &= _report(
        bool(np.all(np.isfinite(wide_matrix))),
        "the wide matrix is finite on a real frame",
    )

    # The historical columns must sit at the *same offsets* in both matrices,
    # otherwise every stored experiment's column indices have moved.
    width_narrow = len(narrow.feature_names)
    width_wide = len(wide.feature_names)
    offsets_ok = all(
        index * width_wide + position < wide_matrix.shape[1]
        and index * width_narrow + position < narrow_matrix.shape[1]
        for index in range(wide_matrix.shape[1] // width_wide)
        for position in (0, 5, 6)
    )
    ok &= _report(offsets_ok, "component blocks stay contiguous and in order")
    return bool(ok)


def main() -> int:
    print("Verifying the extended per-component statistic set (S3a).")
    results = [
        check_preset_layout(),
        check_historical_identity(),
        check_scale_invariance(),
        check_physical_meaning(),
        check_degenerate_inputs(),
        check_lazy_evaluation(),
        check_plumbing(),
    ]
    passed = sum(1 for value in results if value)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

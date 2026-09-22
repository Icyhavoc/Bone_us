"""Self-checks for the preprocessing options used by `compare_label_schemes.py`.

Run directly:

    python verify_scalers.py

The three scalers answer different questions and the differences are easy to
get wrong silently:

* `standard` shifts and divides by the standard deviation, so a handful of
  large samples set the scale for everybody;
* `robust` uses median and IQR/1.349, where the IQR divisor is chosen so the
  scale *equals* the standard deviation on a normal column -- that is what makes
  it a fair swap rather than a different model;
* `rank` replaces each value by its position in the training distribution and
  pushes it through the inverse normal CDF, which is monotone and therefore
  invisible to an order-only model but removes raw amplitude scale entirely.

This script also covers `_norm_ppf`, the hand-rolled inverse normal CDF, since
one can never be entirely sure that an Acklam implementation was transcribed
correctly.
"""

from __future__ import annotations

import numpy as np

from compare_label_schemes import RankScaler, RobustScaler, _norm_ppf, make_scaler

#: Exact quantiles for the standard normal, to ~15 significant digits.
KNOWN_QUANTILES = {
    0.5: 0.0,
    0.975: 1.959963984540054,
    0.025: -1.959963984540054,
    0.8413447460685429: 1.0,
    0.001: -3.090232306167813,
    0.999: 3.090232306167813,
    0.01: -2.3263478740408408,
    0.99: 2.3263478740408408,
}

SCALER_NAMES = ("standard", "robust", "rank")


def check_norm_ppf() -> None:
    print("[checks] _norm_ppf against tabulated normal quantiles")
    probabilities = np.array(list(KNOWN_QUANTILES))
    values = _norm_ppf(probabilities)
    worst = 0.0
    for probability, value in zip(probabilities, values):
        expected = KNOWN_QUANTILES[float(probability)]
        error = abs(value - expected)
        worst = max(worst, error)
        print(
            f"    ppf({probability:<20.16g}) = {value:+.15f} "
            f"(expected {expected:+.15f}, error {error:.2e})"
        )
    print(f"    worst absolute error: {worst:.3e}")
    assert worst < 1e-6, "the inverse normal CDF is not accurate enough to use"


def check_robust_matches_standard_on_gaussian() -> np.ndarray:
    print("[checks] robust scale tracks the standard deviation on Gaussian data")
    rng = np.random.default_rng(0)
    gaussian = rng.standard_normal((4000, 3))
    robust = RobustScaler().fit(gaussian)
    standard = make_scaler("standard").fit(gaussian)
    print(f"    robust   scale = {robust.scale_}")
    print(f"    standard scale = {standard.scale_}")
    assert np.allclose(robust.scale_, standard.scale_, atol=0.05), (
        "the 1.349 divisor exists precisely so these two agree on clean data"
    )
    assert np.allclose(robust.center_, 0.0, atol=0.05)
    return gaussian


def check_robust_resists_outliers(gaussian: np.ndarray) -> None:
    print("[checks] robust resists an outlier tail that inflates standard")
    tailed = gaussian.copy()
    tailed[:40, 0] += 50.0
    robust_scale = RobustScaler().fit(tailed).scale_[0]
    standard_scale = make_scaler("standard").fit(tailed).scale_[0]
    print(f"    1% outliers on column 0 -> robust {robust_scale:.4f}, standard {standard_scale:.4f}")
    assert robust_scale < 0.5 * standard_scale, "robust must not follow the outlier tail"


def check_degenerate_columns_map_to_zero() -> None:
    print("[checks] a constant column becomes exactly 0 for all three scalers")
    rng = np.random.default_rng(1)
    matrix = np.zeros((50, 4))
    matrix[:, 0] = rng.standard_normal(50)
    matrix[:, 3] = 7.0  # constant
    for name in SCALER_NAMES:
        scaler = make_scaler(name).fit(matrix)
        transformed = scaler.transform(matrix)
        assert np.all(transformed[:, 3] == 0.0), f"{name} did not zero the constant column"
        assert np.all(np.isfinite(transformed)), f"{name} produced non-finite values"
        print(f"    {name:<9s} degenerate_indices={scaler.degenerate_indices} finite=True")


def check_rank_semantics() -> None:
    print("[checks] rank is monotone, averages ties, and ignores amplitude scale")
    # Column 1 is a tie block plus one distinct value; column 2 is constant.
    train = np.array(
        [[1.0, 5.0, 2.0], [2.0, 5.0, 2.0], [3.0, 5.0, 2.0], [4.0, 5.0, 2.0], [100.0, 6.0, 2.0]]
    )
    scaler = RankScaler().fit(train)
    transformed = scaler.transform(train)
    print(f"    rank of the monotone column : {np.round(transformed[:, 0], 4)}")
    print(f"    rank of the tied column     : {np.round(transformed[:, 1], 4)}")
    assert np.all(np.diff(transformed[:, 0]) > 0), "the rank map must be strictly monotone"
    assert np.allclose(transformed[:4, 1], transformed[0, 1]), "ties must share one value"
    assert transformed[0, 1] < transformed[4, 1]
    assert np.all(transformed[:, 2] == 0.0), "a constant column must map to exactly 0"
    assert scaler.degenerate_indices == [2]

    # Held-out points outside the training range must stay finite; a 1000x
    # amplitude change must leave every training encoding bit-identical.
    held_out = scaler.transform(np.array([[0.0, 5.0, 2.0], [1e6, 5.0, 2.0]]))
    print(f"    held-out ranks              : {np.round(held_out[:, 0], 4)}")
    assert np.all(np.isfinite(held_out))
    assert held_out[0, 0] < transformed[0, 0] < transformed[-1, 0] < held_out[1, 0]

    amplified = train * np.array([1e6, 1.0, 1.0])
    amplified_encoding = RankScaler().fit(amplified).transform(amplified)
    assert np.allclose(amplified_encoding[:, 0], transformed[:, 0]), (
        "rank encodings must not depend on the amplitude of the input"
    )
    print("    1000x amplitude change leaves every rank encoding identical")


def main() -> int:
    check_norm_ppf()
    gaussian = check_robust_matches_standard_on_gaussian()
    check_robust_resists_outliers(gaussian)
    check_degenerate_columns_map_to_zero()
    check_rank_semantics()
    print()
    print("ALL SCALER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

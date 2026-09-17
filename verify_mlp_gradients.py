"""Numerical gradient check for the hybrid MLP (``NumpyMLPClassifier``).

The classifier is hand-written, so its analytic gradients are worth checking
against finite differences whenever the topology changes.  Two independent
checks run for every topology:

1. **Directional derivative** -- shift all parameters along a random unit
   direction and compare the central difference of the loss against
   ``sum(grad * direction)``.  Averaging over every entry cancels the float32
   round-off that dominates per-entry checks, so this is the sensitive test.
   A handful of random directions are used per topology and the **median** is
   reported: a direction that happens to straddle a ReLU kink makes the finite
   difference invalid, which is a property of the loss surface, not a bug.
2. **Per-entry spot check** -- sample individual weights and biases and compare
   each analytic partial derivative with its own central difference.  This
   catches a systematic slicing error, which the directional average could in
   principle mask.

Run: ``python verify_mlp_gradients.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from emd_pipeline import MLPConfig, NumpyMLPClassifier

DIRECTION_COUNT = 6
DIRECTIONAL_EPS = 1e-3
ENTRY_EPS = 1e-3
DIRECTIONAL_TOLERANCE = 5e-2
ENTRY_TOLERANCE = 5e-2
# The loss is evaluated in float32, so a central difference with step ``eps``
# cannot resolve derivatives below roughly ``float32_eps / (2 * eps)``.
# Entries whose true derivative is that small need an absolute floor, otherwise
# they are judged on round-off alone.
ENTRY_ABSOLUTE_FLOOR = 1e-4
ENTRY_PASS_RATE = 0.95

# ``(group_dims, hidden_dims)`` pairs covering the shipping configuration, the
# collapsed single-group case, and deeper / wider variants.
TOPOLOGIES: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    ((8,), (64, 32)),
    ((16, 16), (64, 32)),
    ((8, 8), (32, 16)),
    ((21, 7, 11), (16, 12, 6)),
    ((16, 16), (64,)),
    ((32, 32), (128, 64)),
)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def batch_loss(model: NumpyMLPClassifier, x: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy plus the L2 penalty the optimizer actually minimises."""
    logits, _ = model._forward(x, training=False)
    probabilities = softmax(logits).astype(np.float64)
    data_loss = -float(
        np.mean(np.log(np.maximum(probabilities[np.arange(y.size), y], 1e-12)))
    )
    penalty = 0.5 * model.config.weight_decay * sum(
        float(np.sum(w.astype(np.float64) ** 2)) for w in model.weights
    )
    return data_loss + penalty


def analytic_gradients(
    model: NumpyMLPClassifier, x: np.ndarray, y: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    logits, cache = model._forward(x, training=False)
    delta = softmax(logits)
    delta[np.arange(y.size), y] -= 1.0
    delta /= y.size
    return model._backward(cache, delta)


def sample_problem(
    group_dims: tuple[int, ...],
    hidden_dims: tuple[int, ...],
    seed: int,
) -> tuple[NumpyMLPClassifier, np.ndarray, np.ndarray, list, list, np.random.Generator]:
    """Build a fixed tiny problem plus its analytic gradients."""
    rng = np.random.default_rng(seed)
    input_dim = sum(group_dims)
    config = MLPConfig(
        hidden_dims=hidden_dims,
        dropout=0.0,
        weight_decay=0.03,
        seed=7,
    )
    model = NumpyMLPClassifier(input_dim=input_dim, config=config, group_dims=group_dims)
    x = rng.standard_normal((5, input_dim)).astype(np.float32)
    y = rng.integers(0, 2, size=5).astype(np.int64)
    grad_w, grad_b = analytic_gradients(model, x, y)
    return model, x, y, grad_w, grad_b, rng


def directional_error(
    group_dims: tuple[int, ...],
    hidden_dims: tuple[int, ...],
    seed: int,
    direction_seed: int,
) -> float:
    model, x, y, grad_w, grad_b, _ = sample_problem(group_dims, hidden_dims, seed)
    rng = np.random.default_rng(direction_seed)
    eps = DIRECTIONAL_EPS

    directions_w = [rng.standard_normal(w.shape) for w in model.weights]
    directions_b = [rng.standard_normal(b.shape) for b in model.biases]
    norm = np.sqrt(
        sum(float(np.sum(d**2)) for d in directions_w)
        + sum(float(np.sum(d**2)) for d in directions_b)
    )
    directions_w = [d / norm for d in directions_w]
    directions_b = [d / norm for d in directions_b]

    base_w = [w.copy() for w in model.weights]
    base_b = [b.copy() for b in model.biases]

    def loss_at(sign: float) -> float:
        for layer in range(len(model.weights)):
            model.weights[layer] = (
                base_w[layer] + sign * eps * directions_w[layer]
            ).astype(np.float32)
            model.biases[layer] = (
                base_b[layer] + sign * eps * directions_b[layer]
            ).astype(np.float32)
        return batch_loss(model, x, y)

    numeric = (loss_at(+1.0) - loss_at(-1.0)) / (2.0 * eps)
    for layer in range(len(model.weights)):
        model.weights[layer] = base_w[layer].copy()
        model.biases[layer] = base_b[layer].copy()

    analytic = sum(
        float(np.sum(grad_w[layer].astype(np.float64) * directions_w[layer]))
        for layer in range(len(grad_w))
    ) + sum(
        float(np.sum(grad_b[layer].astype(np.float64) * directions_b[layer]))
        for layer in range(len(grad_b))
    )
    return abs(numeric - analytic) / max(1e-9, abs(numeric), abs(analytic))


def entry_pass_rate(
    group_dims: tuple[int, ...], hidden_dims: tuple[int, ...], seed: int
) -> tuple[int, int]:
    model, x, y, grad_w, grad_b, rng = sample_problem(group_dims, hidden_dims, seed)
    eps = ENTRY_EPS
    passed = 0
    total = 0
    for layer in range(len(model.weights)):
        pairs = (
            (model.weights[layer], grad_w[layer]),
            (model.biases[layer], grad_b[layer]),
        )
        for array, gradients in pairs:
            picks = rng.choice(array.size, size=min(8, array.size), replace=False)
            for flat in picks:
                position = np.unravel_index(int(flat), array.shape)
                original = float(array[position])
                array[position] = np.float32(original + eps)
                plus = batch_loss(model, x, y)
                array[position] = np.float32(original - eps)
                minus = batch_loss(model, x, y)
                array[position] = np.float32(original)
                numeric = (plus - minus) / (2.0 * eps)
                reference = float(gradients[position])
                total += 1
                tolerance = ENTRY_ABSOLUTE_FLOOR + ENTRY_TOLERANCE * abs(reference)
                if abs(numeric - reference) <= tolerance:
                    passed += 1
    return passed, total


def check(group_dims: tuple[int, ...], hidden_dims: tuple[int, ...]) -> bool:
    model, _, _, _, _, _ = sample_problem(group_dims, hidden_dims, 0)
    errors = [
        directional_error(group_dims, hidden_dims, 0, direction_seed)
        for direction_seed in range(DIRECTION_COUNT)
    ]
    median = float(np.median(errors))
    topology = [tuple(w.shape) for w in model.weights]
    passed, total = entry_pass_rate(group_dims, hidden_dims, 0)
    rate = passed / total if total else 0.0
    ok = median < DIRECTIONAL_TOLERANCE and rate >= ENTRY_PASS_RATE
    print(
        f"groups={group_dims} hidden={hidden_dims} towers={model.tower_count} -> "
        f"{'PASS' if ok else 'FAIL'}\n"
        f"  layers={topology}\n"
        f"  directional: median={median:.3e} "
        f"min={min(errors):.3e} max={max(errors):.3e} "
        f"(median must be < {DIRECTIONAL_TOLERANCE:.0e}; per-direction outliers are ReLU kinks)\n"
        f"  per-entry: {passed}/{total}"
    )
    return ok


def main() -> int:
    results = [check(group_dims, hidden_dims) for group_dims, hidden_dims in TOPOLOGIES]
    if all(results):
        print("\nall topologies match their numerical gradients")
        return 0
    print(f"\n{results.count(False)} of {len(results)} topologies FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())

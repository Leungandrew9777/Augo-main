"""metrics.py — shared probabilistic evaluation metrics.

Used by both training-time walk-forward evaluation (train_ensemble.py) and the
post-hoc evaluation report (evaluate.py) so the two never diverge.
"""

from __future__ import annotations

import numpy as np


def brier_multiclass(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and one-hot truth."""
    n = len(y_true)
    one_hot = np.zeros((n, probs.shape[1]))
    one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean((probs - one_hot) ** 2))


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, bins: int = 10,
) -> tuple[float, list[tuple[int, float, float]]]:
    """ECE over confidence bins of the predicted outcome.

    Returns ``(ece, rows)`` where each row is ``(count, hit_rate, avg_confidence)``.
    """
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(float)
    ece = 0.0
    rows: list[tuple[int, float, float]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = float(correct[mask].mean())
        avg = float(conf[mask].mean())
        ece += (mask.sum() / len(y_true)) * abs(acc - avg)
        rows.append((int(mask.sum()), acc, avg))
    return float(ece), rows

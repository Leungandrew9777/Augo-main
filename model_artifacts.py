from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scipy.optimize import minimize


@dataclass
class WeightedSoftVotingClassifier:
    """Small pickle-safe soft-voting wrapper for manually fitted estimators."""

    estimators_: list[Any]
    weights: list[float]
    classes_: np.ndarray
    feature_names_in_: np.ndarray

    def predict_proba(self, X):
        total_weight = float(sum(self.weights))
        if total_weight <= 0:
            raise ValueError("Voting weights must sum to a positive value.")
        probs = None
        for estimator, weight in zip(self.estimators_, self.weights):
            p = estimator.predict_proba(X)
            probs = p * weight if probs is None else probs + (p * weight)
        return probs / total_weight

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class TemperatureScalingCalibrator:
    """Single-parameter multiclass temperature scaling (Guo et al., 2017).

    Fits ``T`` on out-of-fold probabilities by minimising negative log
    likelihood, then rescales log-odds by ``1/T``. A single parameter cannot
    overfit much: if the probs are already well calibrated, T stays near 1.
    """

    def __init__(self):
        self.temperature_: float = 1.0

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "TemperatureScalingCalibrator":
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y).astype(int)

        def nll(T: float) -> float:
            logits = np.log(np.clip(probs, 1e-12, None))
            z = logits / max(float(T), 1e-6)
            e = np.exp(z - z.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))

        res = minimize(nll, x0=[1.0], method="Nelder-Mead")
        self.temperature_ = float(res.x[0])
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        T = max(float(self.temperature_), 1e-6)
        logits = np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, None))
        z = logits / T
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


class StackedEnsembleClassifier:
    """Stacking ensemble: base models -> logistic meta-learner on base probs.

    The meta output is blended with the [1,1,2] soft-vote of the base models
    (``vote_weight`` = share of the soft-vote). The blend keeps the meta's
    accuracy edge while restoring some draw coverage the linear meta alone
    collapses. ``estimators_`` keeps the base models so existing runtime compat
    patches (e.g. ``_patch_model_runtime_compat``) keep working;
    ``feature_names_in_`` mirrors the base features so the app/pipeline feature
    detection is unchanged.
    """

    def __init__(
        self,
        estimators_: list[Any],
        meta_estimator_: Any,
        classes_: np.ndarray,
        feature_names_in_: np.ndarray,
        calibrator_: TemperatureScalingCalibrator | None = None,
        vote_weight: float = 0.5,
        vote_weights: list[float] | None = None,
    ):
        self.estimators_ = estimators_
        self.meta_estimator_ = meta_estimator_
        self.classes_ = np.asarray(classes_)
        self.feature_names_in_ = np.asarray(feature_names_in_, dtype=object)
        self.calibrator_ = calibrator_
        self.vote_weight = float(vote_weight)
        self.vote_weights = list(vote_weights) if vote_weights is not None else [1.0, 1.0, 2.0]

    def _meta_input(self, X):
        return np.hstack([est.predict_proba(X) for est in self.estimators_])

    def _soft_vote(self, X):
        total_weight = float(sum(self.vote_weights))
        if total_weight <= 0:
            raise ValueError("Voting weights must sum to a positive value.")
        probs = None
        for estimator, weight in zip(self.estimators_, self.vote_weights):
            p = estimator.predict_proba(X)
            probs = p * weight if probs is None else probs + (p * weight)
        return probs / total_weight

    def predict_proba(self, X):
        probs = self.meta_estimator_.predict_proba(self._meta_input(X))
        if self.vote_weight > 0:
            probs = (1.0 - self.vote_weight) * probs + self.vote_weight * self._soft_vote(X)
        if self.calibrator_ is not None:
            probs = self.calibrator_.transform(probs)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


@dataclass
class SeedAveragedClassifier:
    """Average seed-variant fits of the same estimator (variance reduction)."""

    estimators_: list[Any]
    classes_: np.ndarray

    def predict_proba(self, X):
        probs = np.mean([est.predict_proba(X) for est in self.estimators_], axis=0)
        probs = np.asarray(probs, dtype=float)
        # XGBoost returns float32; re-normalise so downstream log-loss and
        # blending see exact probabilities.
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class VectorScalingCalibrator:
    """Per-class (vector) probability calibration.

    Learns a scale and bias per class on log-probabilities, correcting
    class-specific over/under-confidence that a single temperature cannot.
    """

    def __init__(self):
        self.scale_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None

    def fit(self, probs: np.ndarray, y: np.ndarray, *, l2: float = 1e-4) -> "VectorScalingCalibrator":
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y, dtype=int)
        logp = np.log(np.clip(probs, 1e-12, None))
        n_classes = probs.shape[1]
        rows = np.arange(len(y))

        def nll(theta: np.ndarray) -> float:
            scale = theta[:n_classes]
            bias = theta[n_classes:]
            z = logp * scale + bias
            z -= z.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)
            return float(-np.mean(np.log(np.clip(p[rows, y], 1e-12, None))) + l2 * np.sum(theta ** 2))

        x0 = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])
        res = minimize(nll, x0, method="L-BFGS-B")
        self.scale_ = res.x[:n_classes]
        self.bias_ = res.x[n_classes:]
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self.scale_ is None or self.bias_ is None:
            return np.asarray(probs, dtype=float)
        logp = np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, None))
        z = logp * self.scale_ + self.bias_
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


@dataclass
class MarketBlendModel:
    """Out-of-fold stack: [model, Dixon-Coles, market] probabilities -> outcome.

    ``fit`` expects the 9-column stacked matrix; ``predict_proba`` accepts the
    three triplets separately at serving time. The per-class calibrator is fit
    on the stacker's out-of-fold outputs.
    """

    estimator_: Any = None
    calibrator_: VectorScalingCalibrator | None = None
    metadata_: dict = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=lambda: [
        f"{group}_{cls}" for group in ("model", "dc", "market")
        for cls in ("home", "draw", "away")
    ])

    def fit(self, X: np.ndarray, y: np.ndarray, *, l2: float = 1e-3) -> "MarketBlendModel":
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.estimator_ = LogisticRegression(max_iter=2000, C=1.0)
        self.estimator_.fit(X, y)
        self.calibrator_ = VectorScalingCalibrator().fit(self.estimator_.predict_proba(X), y, l2=l2)
        return self

    def predict_proba(self, model_probs, dc_probs=None, market_probs=None) -> np.ndarray:
        if dc_probs is None or market_probs is None:
            X = np.asarray(model_probs, dtype=float)
        else:
            X = np.hstack([
                np.asarray(model_probs, dtype=float),
                np.asarray(dc_probs, dtype=float),
                np.asarray(market_probs, dtype=float),
            ])
        probs = self.estimator_.predict_proba(X)
        if self.calibrator_ is not None:
            probs = self.calibrator_.transform(probs)
        return probs

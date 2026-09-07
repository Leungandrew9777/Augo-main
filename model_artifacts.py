from __future__ import annotations

from dataclasses import dataclass
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

# train_ensemble.py
import argparse
import json
import os

import pandas as pd
import numpy as np
import inspect
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, mean_poisson_deviance, confusion_matrix, mean_absolute_error
from xgboost import XGBClassifier
import joblib

from model_artifacts import StackedEnsembleClassifier, TemperatureScalingCalibrator
from features import MODEL_FEATURE_COLS

print("STEP 5: Loading rich features for ensemble training...")
df = pd.read_csv("premier_league_with_elo_best.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True, format="mixed")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

DECAY_HALF_LIFE_DAYS = 365.0


def time_decay_weights(dates: pd.Series, *, reference_date=None, half_life_days: float = DECAY_HALF_LIFE_DAYS) -> np.ndarray:
    parsed = pd.to_datetime(dates, errors="coerce")
    ref = pd.to_datetime(reference_date) if reference_date is not None else parsed.max()
    age_days = (ref - parsed).dt.days.clip(lower=0).fillna(0)
    return np.exp(-np.log(2.0) * age_days / half_life_days).to_numpy(dtype=float)


def fit_goal_model(X_fit: pd.DataFrame, y_fit: pd.Series, sample_weight):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.01, max_iter=1000)),
        ]
    )
    model.fit(X_fit, y_fit, model__sample_weight=sample_weight)
    return model


def fit_count_model(X_fit: pd.DataFrame, y_fit: pd.Series, sample_weight, kind: str = "poisson"):
    """Poisson (goals) or Tweedie power=1.5 (corners — overdispersed) count model."""
    if kind == "poisson":
        model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.01, max_iter=1000)),
        ])
    else:
        from sklearn.linear_model import TweedieRegressor
        model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", TweedieRegressor(power=1.5, alpha=0.01, max_iter=1000, link="log")),
        ])
    model.fit(X_fit, y_fit, model__sample_weight=sample_weight)
    return model


# Use only columns that actually exist in the file
feature_cols = [c for c in MODEL_FEATURE_COLS if c in df.columns]
X = df[feature_cols].fillna(df[feature_cols].median())
y = df["Result"].astype(int)  # 0=Away, 1=Draw, 2=Home
y_home_goals = df["FTHG"].astype(float)
y_away_goals = df["FTAG"].astype(float)
y_ht_home = df["HTHG"].astype(float)
y_ht_away = df["HTAG"].astype(float)
y_corner_home = df["HC"].astype(float)
y_corner_away = df["AC"].astype(float)

# Corner models also use the corner-specific rolling stats that exist in the CSV
CORNER_FEATURE_COLS = feature_cols + [c for c in [
    "home_avg_Corners", "away_avg_Corners", "diff_avg_Corners",
    "home_avg_CornersAgainst", "away_avg_CornersAgainst", "diff_avg_CornersAgainst",
] if c in df.columns]
Xc = df[CORNER_FEATURE_COLS].fillna(df[CORNER_FEATURE_COLS].median())

print(f"Training stacked ensemble on {len(X):,} matches with {len(feature_cols)} features")

# `multi_class` is deprecated from sklearn 1.5; omit it and use defaults (multiclass + lbfgs).
_sk_parts = sklearn.__version__.split(".")
_sk_major, _sk_minor = int(_sk_parts[0]), int(_sk_parts[1]) if len(_sk_parts) > 1 else 0
_mc_needed = (_sk_major, _sk_minor) < (1, 5) and "multi_class" in inspect.signature(LogisticRegression).parameters


def _lr_kwargs(C: float) -> dict:
    kw = {"max_iter": 3000, "C": C, "solver": "lbfgs"}
    if _mc_needed:
        kw["multi_class"] = "multinomial"
    return kw


# Balanced class weights: draws (~26%) are structurally under-predicted, so the
# rarer outcomes get proportionally more weight during training.
class_counts = y.value_counts().sort_index()
class_weight = float(len(y)) / (3.0 * class_counts.reindex([0, 1, 2]).astype(float).values)
print(f"Class weights (Away/Draw/Home): {np.round(class_weight, 3)}")

DEFAULT_PARAMS = {
    "xgb": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0},
    "rf": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10},
    "lr": {"C": 0.5},
}


def fit_base(X_fit: pd.DataFrame, y_fit: pd.Series, sample_weight, params: dict | None = None) -> list:
    p = params or {}
    xgb_p = p.get("xgb", DEFAULT_PARAMS["xgb"])
    rf_p = p.get("rf", DEFAULT_PARAMS["rf"])
    lr_p = p.get("lr", DEFAULT_PARAMS["lr"])
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**_lr_kwargs(lr_p["C"]))),
        ]
    )
    lr.fit(X_fit, y_fit, model__sample_weight=sample_weight)
    rf = RandomForestClassifier(
        n_estimators=rf_p["n_estimators"], max_depth=rf_p["max_depth"],
        min_samples_leaf=rf_p["min_samples_leaf"], random_state=42,
    )
    rf.fit(X_fit, y_fit, sample_weight=sample_weight)
    xgb = XGBClassifier(
        n_estimators=xgb_p["n_estimators"], learning_rate=xgb_p["learning_rate"],
        max_depth=xgb_p["max_depth"], subsample=xgb_p["subsample"],
        colsample_bytree=xgb_p["colsample_bytree"], min_child_weight=xgb_p["min_child_weight"],
        reg_lambda=xgb_p["reg_lambda"], random_state=42, eval_metric="mlogloss",
    )
    xgb.fit(X_fit, y_fit, sample_weight=sample_weight)
    return [lr, rf, xgb]


def base_meta_input(estimators: list, X_use) -> np.ndarray:
    """Stacked base-model probability matrix: 3 models x 3 classes = 9 cols."""
    return np.hstack([est.predict_proba(X_use) for est in estimators])


def fit_meta(X_meta: np.ndarray, y_meta: pd.Series) -> LogisticRegression:
    meta = LogisticRegression(**_lr_kwargs(0.5))
    meta.fit(X_meta, y_meta)
    return meta


def soft_vote_probs(estimators: list, X_use) -> np.ndarray:
    return (estimators[0].predict_proba(X_use) * 1.0
            + estimators[1].predict_proba(X_use) * 1.0
            + estimators[2].predict_proba(X_use) * 2.0) / 4.0


def _sample_params(rng: np.random.Generator) -> dict:
    return {
        "xgb": {
            "n_estimators": int(rng.choice([150, 200, 300, 400])),
            "learning_rate": float(rng.choice([0.03, 0.05, 0.08])),
            "max_depth": int(rng.choice([4, 5, 6, 8])),
            "subsample": float(rng.choice([0.7, 0.8, 0.9])),
            "colsample_bytree": float(rng.choice([0.7, 0.8, 0.9])),
            "min_child_weight": int(rng.choice([1, 3, 5])),
            "reg_lambda": float(rng.choice([0.5, 1.0, 2.0])),
        },
        "rf": {
            "n_estimators": int(rng.choice([150, 200, 300])),
            "max_depth": int(rng.choice([6, 8, 10])),
            "min_samples_leaf": int(rng.choice([5, 10, 20])),
        },
        "lr": {"C": float(rng.choice([0.1, 0.3, 0.5, 1.0]))},
    }


def walkforward_metrics(X_use, y_use, dates_use, params, n_splits: int = 3):
    """Fast walk-forward soft-vote evaluation for hyperparameter search."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses, accs = [], []
    for train_idx, test_idx in tscv.split(X_use):
        X_tr, y_tr = X_use.iloc[train_idx], y_use.iloc[train_idx]
        dates_tr = dates_use.iloc[train_idx]
        decay = time_decay_weights(dates_tr, reference_date=dates_tr.max())
        sw = decay * class_weight[y_tr.to_numpy()]
        ests = fit_base(X_tr, y_tr, sw, params)
        probs = soft_vote_probs(ests, X_use.iloc[test_idx])
        losses.append(log_loss(y_use.iloc[test_idx], probs, labels=[0, 1, 2]))
        accs.append(accuracy_score(y_use.iloc[test_idx], np.argmax(probs, axis=1)))
    return float(np.mean(losses)), float(np.mean(accs))


def brier_multiclass(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and one-hot truth."""
    n = len(y_true)
    one_hot = np.zeros((n, probs.shape[1]))
    one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean((probs - one_hot) ** 2))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, bins: int = 10) -> float:
    """ECE over confidence bins of the predicted outcome."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def print_eval_metrics(tag: str, probs: np.ndarray, y_true: np.ndarray) -> None:
    acc = float(np.mean(np.argmax(probs, axis=1) == y_true))
    ll = log_loss(y_true, probs, labels=[0, 1, 2])
    br = brier_multiclass(probs, y_true)
    ece = expected_calibration_error(probs, y_true)
    cm = confusion_matrix(y_true, np.argmax(probs, axis=1), labels=[0, 1, 2])
    recall = cm.diagonal() / cm.sum(axis=1)
    print(f"[{tag}] accuracy={acc:.4f}  log-loss={ll:.4f}  Brier={br:.4f}  ECE={ece:.4f}")
    print(f"[{tag}] per-class recall (Away/Draw/Home): {np.round(recall, 3)}")


# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tune", action="store_true", help="run hyperparameter random search first")
parser.add_argument("--trials", type=int, default=24, help="number of tuning trials")
args = parser.parse_args()

BEST_PARAMS_FILE = "best_params.json"
best_params = None
if args.tune:
    print("STEP 5b: Hyperparameter tuning (walk-forward random search) ...")
    rng = np.random.default_rng(42)
    best = None
    for i in range(args.trials):
        params = _sample_params(rng)
        loss, acc = walkforward_metrics(X, y, df["Date"], params, n_splits=3)
        print(f"   trial {i + 1}/{args.trials}: logloss={loss:.4f} acc={acc:.4f} "
              f"xgb={params['xgb']['n_estimators']}t/{params['xgb']['max_depth']}d "
              f"lr={params['xgb']['learning_rate']} rf_d={params['rf']['max_depth']} lrC={params['lr']['C']}")
        if best is None or loss < best[0]:
            best = (loss, acc, params)
    best_params = best[2]
    print(f"[OK] Best params: {best_params}  (logloss={best[0]:.4f}, acc={best[1]:.4f})")
    with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
elif os.path.exists(BEST_PARAMS_FILE):
    with open(BEST_PARAMS_FILE, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    print(f"Using saved hyperparameters from {BEST_PARAMS_FILE}: {best_params}")
else:
    best_params = DEFAULT_PARAMS
    print("Using default hyperparameters (run with --tune to search).")

# STEP 6: Walk-forward backtest (correct for sports data) + nested OOF stacking
tscv = TimeSeriesSplit(n_splits=5)
inner = TimeSeriesSplit(n_splits=3)
n = len(X)
oof_base = np.full((n, 9), np.nan)
oof_stacked = np.full((n, 3), np.nan)
oof_y = np.full(n, -1, dtype=int)

stacked_scores = []
vote_scores = []
stacked_logloss = []
home_devs = []
away_devs = []
ht_home_devs = []
ht_away_devs = []
corner_home_mae = []
corner_away_mae = []

for train_idx, test_idx in tscv.split(X):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    dates_tr = df.iloc[train_idx]["Date"]
    fold_ref = dates_tr.max()
    decay_tr = time_decay_weights(dates_tr, reference_date=fold_ref)
    sw_tr = decay_tr * class_weight[y_tr.to_numpy()]

    base = fit_base(X_tr, y_tr, sw_tr, best_params)

    # Nested out-of-fold probabilities for the meta-learner (no leakage into test)
    meta_X_tr = np.zeros((len(X_tr), 9))
    for itr, ite in inner.split(X_tr):
        dates_in = dates_tr.iloc[itr]
        ref_in = dates_in.max()
        decay_in = time_decay_weights(dates_in, reference_date=ref_in)
        sw_in = decay_in * class_weight[y_tr.iloc[itr].to_numpy()]
        base_in = fit_base(X_tr.iloc[itr], y_tr.iloc[itr], sw_in, best_params)
        meta_X_tr[ite] = base_meta_input(base_in, X_tr.iloc[ite])
    meta = fit_meta(meta_X_tr, y_tr)

    X_te = X.iloc[test_idx]
    vote_te = soft_vote_probs(base, X_te)
    vote_scores.append(accuracy_score(y.iloc[test_idx], np.argmax(vote_te, axis=1)))
    meta_te = meta.predict_proba(base_meta_input(base, X_te))
    # Blend meta with soft-vote: keeps accuracy edge, restores some draw coverage.
    stacked_te = 0.5 * meta_te + 0.5 * vote_te
    stacked_scores.append(accuracy_score(y.iloc[test_idx], np.argmax(stacked_te, axis=1)))
    stacked_logloss.append(log_loss(y.iloc[test_idx], stacked_te, labels=[0, 1, 2]))

    oof_base[test_idx] = base_meta_input(base, X_te)
    oof_stacked[test_idx] = stacked_te
    oof_y[test_idx] = y.iloc[test_idx].to_numpy()

    # Goal models per fold (unchanged)
    goal_hm = fit_goal_model(X_tr, y_home_goals.iloc[train_idx], decay_tr)
    goal_am = fit_goal_model(X_tr, y_away_goals.iloc[train_idx], decay_tr)
    pred_h = np.clip(goal_hm.predict(X.iloc[test_idx]), 0.05, None)
    pred_a = np.clip(goal_am.predict(X.iloc[test_idx]), 0.05, None)
    home_devs.append(mean_poisson_deviance(y_home_goals.iloc[test_idx], pred_h))
    away_devs.append(mean_poisson_deviance(y_away_goals.iloc[test_idx], pred_a))

    # Half-time goal models
    ht_hm = fit_count_model(X_tr, y_ht_home.iloc[train_idx], decay_tr, "poisson")
    ht_am = fit_count_model(X_tr, y_ht_away.iloc[train_idx], decay_tr, "poisson")
    ht_home_devs.append(mean_poisson_deviance(
        y_ht_home.iloc[test_idx], np.clip(ht_hm.predict(X.iloc[test_idx]), 0.05, None)))
    ht_away_devs.append(mean_poisson_deviance(
        y_ht_away.iloc[test_idx], np.clip(ht_am.predict(X.iloc[test_idx]), 0.05, None)))

    # Corner models (Tweedie, overdispersed)
    c_hm = fit_count_model(Xc.iloc[train_idx], y_corner_home.iloc[train_idx], decay_tr, "tweedie")
    c_am = fit_count_model(Xc.iloc[train_idx], y_corner_away.iloc[train_idx], decay_tr, "tweedie")
    corner_home_mae.append(mean_absolute_error(
        y_corner_home.iloc[test_idx], np.clip(c_hm.predict(Xc.iloc[test_idx]), 0.5, None)))
    corner_away_mae.append(mean_absolute_error(
        y_corner_away.iloc[test_idx], np.clip(c_am.predict(Xc.iloc[test_idx]), 0.5, None)))

print(f"[OK] STEP 6 Walk-forward CV Accuracy (stacked): {np.mean(stacked_scores):.4f}")
print(f"[OK] STEP 6 Walk-forward CV Accuracy (soft-vote [1,1,2]): {np.mean(vote_scores):.4f}")
print(f"[OK] STEP 6 Stacked CV log-loss: {np.mean(stacked_logloss):.4f}")
print(f"[OK] Goal model mean Poisson deviance: home={np.mean(home_devs):.4f}, away={np.mean(away_devs):.4f}")
print(f"[OK] HT goal model deviance: home={np.mean(ht_home_devs):.4f}, away={np.mean(ht_away_devs):.4f}")
print(f"[OK] Corner model MAE: home={np.mean(corner_home_mae):.2f}, away={np.mean(corner_away_mae):.2f}")

mask = oof_y >= 0
print("STEP 6b: Out-of-fold evaluation (stacked model, probabilistic metrics)")
print_eval_metrics("OOF", oof_stacked[mask], oof_y[mask])

# STEP 7: Final stacked model on all data + calibration + save
final_decay = time_decay_weights(df["Date"])
sw_all = final_decay * class_weight[y.to_numpy()]
base_final = fit_base(X, y, sw_all, best_params)
meta_final = fit_meta(oof_base[mask], pd.Series(oof_y[mask]))
stacked_final = StackedEnsembleClassifier(
    estimators_=base_final,
    meta_estimator_=meta_final,
    classes_=np.array([0, 1, 2]),
    feature_names_in_=np.array(list(X.columns), dtype=object),
    vote_weight=0.5,
)
calibrator = TemperatureScalingCalibrator().fit(oof_stacked[mask], oof_y[mask])
stacked_final.calibrator_ = calibrator
print(f"[OK] Calibration temperature T = {calibrator.temperature_:.4f}")

goal_model_home = fit_goal_model(X, y_home_goals, final_decay)
goal_model_away = fit_goal_model(X, y_away_goals, final_decay)
goal_model_ht_home = fit_count_model(X, y_ht_home, final_decay, "poisson")
goal_model_ht_away = fit_count_model(X, y_ht_away, final_decay, "poisson")
corner_model_home = fit_count_model(Xc, y_corner_home, final_decay, "tweedie")
corner_model_away = fit_count_model(Xc, y_corner_away, final_decay, "tweedie")
joblib.dump(stacked_final, "xgboost_premier_league_model.pkl")
joblib.dump(goal_model_home, "goal_model_home.pkl")
joblib.dump(goal_model_away, "goal_model_away.pkl")
joblib.dump(goal_model_ht_home, "goal_model_ht_home.pkl")
joblib.dump(goal_model_ht_away, "goal_model_ht_away.pkl")
joblib.dump(corner_model_home, "corner_model_home.pkl")
joblib.dump(corner_model_away, "corner_model_away.pkl")
df.to_csv("premier_league_with_elo_best.csv", index=False)

print("\nSTEP 5 + 6 + 7 COMPLETE")
print("   Stacked model saved -> xgboost_premier_league_model.pkl")
print("   Goal model home saved -> goal_model_home.pkl")
print("   Goal model away saved -> goal_model_away.pkl")
print("   HT goal models saved -> goal_model_ht_home.pkl / goal_model_ht_away.pkl")
print("   Corner models saved -> corner_model_home.pkl / corner_model_away.pkl")
print("   Data saved   -> premier_league_with_elo_best.csv")
print("   Your existing app.py and run_pipeline.py will now use the new stacked ensemble.")

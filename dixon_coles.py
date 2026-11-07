"""dixon_coles.py — time-weighted Dixon-Coles ratings from goals/xG.

A low-parameter, structurally sound complement to the gradient-boosting
classifier: per-team attack/defence strengths, home advantage and the
Dixon-Coles low-score correction, estimated with time-decay weights and L2
shrinkage (which doubles as the promoted-team prior: teams with little history
are pulled toward the league average). Outcome probabilities are derived from
the scoreline distribution rather than predicted directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from dates import parse_series

MAX_GOALS = 8
DECAY_HALF_LIFE_DAYS = 365.0


def _poisson_logpmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return k * np.log(np.clip(lam, 1e-9, None)) - lam - gammaln(k + 1.0)


def hda_from_lambdas(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
    *,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """Return class probabilities in the project's order: [Away, Draw, Home]."""
    lam = np.clip(np.asarray(lambda_home, dtype=float), 1e-6, None)
    mu = np.clip(np.asarray(lambda_away, dtype=float), 1e-6, None)
    n = len(lam)
    goals = np.arange(max_goals + 1)
    home_pmf = np.exp(_poisson_logpmf(goals[None, :], lam[:, None]))
    away_pmf = np.exp(_poisson_logpmf(goals[None, :], mu[:, None]))
    grid = home_pmf[:, :, None] * away_pmf[:, None, :]  # (n, home_goals, away_goals)

    grid[:, 0, 0] *= np.clip(1.0 - lam * mu * rho, 1e-6, None)
    grid[:, 0, 1] *= np.clip(1.0 + lam * rho, 1e-6, None)
    grid[:, 1, 0] *= np.clip(1.0 + mu * rho, 1e-6, None)
    grid[:, 1, 1] *= np.clip(1.0 - rho, 1e-6, None)
    grid /= grid.sum(axis=(1, 2), keepdims=True)

    upper = np.triu_indices(max_goals + 1, k=1)   # away win (home < away)
    lower = np.tril_indices(max_goals + 1, k=-1)  # home win (home > away)
    p_away = grid[:, upper[0], upper[1]].sum(axis=1)
    p_home = grid[:, lower[0], lower[1]].sum(axis=1)
    p_draw = np.clip(1.0 - p_away - p_home, 0.0, 1.0)
    return np.column_stack([p_away, p_draw, p_home])


@dataclass
class DixonColesModel:
    half_life_days: float = DECAY_HALF_LIFE_DAYS
    l2: float = 0.05
    use_xg: bool = True
    teams_: list[str] = field(default_factory=list)
    attack_: dict[str, float] = field(default_factory=dict)
    defense_: dict[str, float] = field(default_factory=dict)
    home_adv_: float = 0.25
    rho_: float = -0.05

    def fit(self, df: pd.DataFrame) -> "DixonColesModel":
        if "Date" not in df.columns:
            raise ValueError("DixonColesModel.fit: training frame needs a Date column")
        work = df.copy()
        work["_date"] = parse_series(work["Date"])
        work = work.dropna(subset=["_date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).sort_values("_date")
        if work.empty:
            raise ValueError("DixonColesModel.fit: no usable training rows")

        home_goals = pd.to_numeric(work["FTHG"], errors="coerce").to_numpy(dtype=float)
        away_goals = pd.to_numeric(work["FTAG"], errors="coerce").to_numpy(dtype=float)
        if self.use_xg and {"home_xg", "away_xg"}.issubset(work.columns):
            home_xg = pd.to_numeric(work["home_xg"], errors="coerce").to_numpy(dtype=float)
            away_xg = pd.to_numeric(work["away_xg"], errors="coerce").to_numpy(dtype=float)
            home_goals = np.where(np.isfinite(home_xg), home_xg, home_goals)
            away_goals = np.where(np.isfinite(away_xg), away_xg, away_goals)

        teams = sorted(set(work["HomeTeam"].astype(str)) | set(work["AwayTeam"].astype(str)))
        index = {team: i for i, team in enumerate(teams)}
        home_idx = work["HomeTeam"].astype(str).map(index).to_numpy()
        away_idx = work["AwayTeam"].astype(str).map(index).to_numpy()
        age_days = (work["_date"].max() - work["_date"]).dt.days.to_numpy(dtype=float)
        weights = np.exp(-np.log(2.0) * age_days / max(self.half_life_days, 1.0))
        constants = gammaln(home_goals + 1.0) + gammaln(away_goals + 1.0)
        n_teams = len(teams)

        def nll(theta: np.ndarray) -> float:
            attack = theta[:n_teams]
            defense = theta[n_teams:2 * n_teams]
            home_adv = theta[2 * n_teams]
            rho = theta[2 * n_teams + 1]

            lam = np.exp(np.clip(attack[home_idx] - defense[away_idx] + home_adv, -8.0, 8.0))
            mu = np.exp(np.clip(attack[away_idx] - defense[home_idx], -8.0, 8.0))
            loglik = home_goals * np.log(lam) - lam + away_goals * np.log(mu) - mu - constants

            tau = np.ones_like(lam)
            m00 = (home_goals == 0) & (away_goals == 0)
            m01 = (home_goals == 0) & (away_goals == 1)
            m10 = (home_goals == 1) & (away_goals == 0)
            m11 = (home_goals == 1) & (away_goals == 1)
            tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
            tau[m01] = 1.0 + lam[m01] * rho
            tau[m10] = 1.0 + mu[m10] * rho
            tau[m11] = 1.0 - rho
            loglik = loglik + np.log(np.clip(tau, 1e-6, None))

            penalty = self.l2 * (np.sum(attack ** 2) + np.sum(defense ** 2))
            return float(-np.sum(weights * loglik) + penalty)

        x0 = np.concatenate([np.zeros(2 * n_teams), [0.25, -0.05]])
        bounds = [(None, None)] * (2 * n_teams) + [(0.0, 1.5), (-0.6, 0.6)]
        result = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250})

        attack = result.x[:n_teams]
        defense = result.x[n_teams:2 * n_teams]
        self.teams_ = teams
        self.attack_ = {team: float(attack[i]) for team, i in index.items()}
        self.defense_ = {team: float(defense[i]) for team, i in index.items()}
        self.home_adv_ = float(result.x[2 * n_teams])
        self.rho_ = float(result.x[2 * n_teams + 1])
        return self

    @staticmethod
    def _team_columns(df: pd.DataFrame) -> tuple[str, str]:
        home = "HomeTeam" if "HomeTeam" in df.columns else "home_team"
        away = "AwayTeam" if "AwayTeam" in df.columns else "away_team"
        return home, away

    def predict_lambdas(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        home_col, away_col = self._team_columns(df)
        home = df[home_col].astype(str).map(lambda t: self.attack_.get(t, 0.0)).to_numpy(dtype=float)
        away = df[away_col].astype(str).map(lambda t: self.attack_.get(t, 0.0)).to_numpy(dtype=float)
        home_def = df[home_col].astype(str).map(lambda t: self.defense_.get(t, 0.0)).to_numpy(dtype=float)
        away_def = df[away_col].astype(str).map(lambda t: self.defense_.get(t, 0.0)).to_numpy(dtype=float)
        lam = np.exp(np.clip(home - away_def + self.home_adv_, -8.0, 8.0))
        mu = np.exp(np.clip(away - home_def, -8.0, 8.0))
        return lam, mu

    def predict_proba(self, df: pd.DataFrame, *, max_goals: int = MAX_GOALS) -> np.ndarray:
        lam, mu = self.predict_lambdas(df)
        return hda_from_lambdas(lam, mu, self.rho_, max_goals=max_goals)

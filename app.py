import reflex as rx
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import math
import datetime as dt
from typing import Any

import requests

from persistence import (
    load_archived_predictions,
    load_bankroll,
    load_results,
    load_user_picks,
    save_bankroll,
    upsert_user_picks_for_gw,
)
from bankroll import build_bankroll, build_current_suggestions, format_money
from explanations import build_explanation
from team_aliases import badge_lookup_key
from data_urls import predictions_cache_url

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Constants ─────────────────────────────────────────────────────────────────

ODDS_DISPLAY_CAP = 200.0
CACHE_FILE = os.path.join(APP_DIR, "predictions_cache.json")


def _cache_max_age_hours() -> float:
    """Max cache age before falling back to live inference.

    Default 168 h (7 days) fits a weekly ``run_pipeline`` cadence. Override with
    ``CACHE_MAX_AGE_HOURS`` or ``AUGO_CACHE_MAX_AGE_HOURS``. Use ``0`` to disable
    the stale check entirely.
    """
    for key in ("CACHE_MAX_AGE_HOURS", "AUGO_CACHE_MAX_AGE_HOURS"):
        raw = os.getenv(key, "").strip()
        if not raw:
            continue
        try:
            v = float(raw)
            if math.isfinite(v) and v >= 0:
                return v
        except ValueError:
            continue
    return 168.0


CACHE_MAX_AGE_HOURS = _cache_max_age_hours()


def _local_today_hk() -> pd.Timestamp:
    """Return today's date normalized in Hong Kong (UTC+8) time."""
    try:
        from zoneinfo import ZoneInfo

        return pd.Timestamp(dt.datetime.now(ZoneInfo("Asia/Hong_Kong")).date())
    except Exception:
        return pd.Timestamp.today().normalize()


def format_odds_display(v: float, cap: float = ODDS_DISPLAY_CAP) -> str:
    """Format odds for UI display without scientific notation."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return f"{cap:.0f}"

    if not math.isfinite(x):
        return f"{cap:.0f}"

    if x >= cap:
        return f"{cap:.0f}"
    if x >= 10:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    if x >= 1:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return f"{x:.3f}".rstrip("0").rstrip(".")

PL_TEAMS: list[str] = sorted([
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton & Hove Albion", "Chelsea", "Coventry City", "Crystal Palace",
    "Everton", "Fulham", "Hull City", "Ipswich Town", "Leeds United",
    "Liverpool", "Manchester City", "Manchester United", "Newcastle",
    "Nottingham Forest", "Sunderland", "Tottenham Hotspur",
])

TEAM_BADGES: dict[str, str] = {
    "Arsenal":                    "https://resources.premierleague.com/premierleague/badges/t3.png",
    "Aston Villa":                "https://resources.premierleague.com/premierleague/badges/t7.png",
    "Bournemouth":                "https://resources.premierleague.com/premierleague/badges/t91.png",
    "Brentford":                  "https://resources.premierleague.com/premierleague/badges/t94.png",
    "Brighton & Hove Albion":     "https://resources.premierleague.com/premierleague/badges/t36.png",
    "Burnley":                    "https://resources.premierleague.com/premierleague/badges/t90.png",
    "Chelsea":                    "https://resources.premierleague.com/premierleague/badges/t8.png",
    "Coventry City":              "https://resources.premierleague.com/premierleague25/badges-alt/9.svg",
    "Crystal Palace":             "https://resources.premierleague.com/premierleague/badges/t31.png",
    "Everton":                    "https://resources.premierleague.com/premierleague/badges/t11.png",
    "Fulham":                     "https://resources.premierleague.com/premierleague/badges/t54.png",
    "Hull City":                  "https://resources.premierleague.com/premierleague25/badges-alt/88.svg",
    "Ipswich Town":               "https://resources.premierleague.com/premierleague25/badges-alt/40.svg",
    "Leeds United":               "https://resources.premierleague.com/premierleague/badges/t2.png",
    "Liverpool":                  "https://resources.premierleague.com/premierleague/badges/t14.png",
    "Manchester City":            "https://resources.premierleague.com/premierleague/badges/t43.png",
    "Manchester United":          "https://resources.premierleague.com/premierleague/badges/t1.png",
    "Newcastle":                  "https://resources.premierleague.com/premierleague/badges/t4.png",
    "Nottingham Forest":          "https://resources.premierleague.com/premierleague/badges/t17.png",
    "Sunderland":                 "https://resources.premierleague.com/premierleague/badges/t56.png",
    "Tottenham Hotspur":          "https://resources.premierleague.com/premierleague/badges/t6.png",
    "West Ham United":            "https://resources.premierleague.com/premierleague/badges/t21.png",
    "Wolverhampton Wanderers":    "https://resources.premierleague.com/premierleague/badges/t39.png",
}

FALLBACK_BADGE = "https://resources.premierleague.com/premierleague/badges/t0.png"

FALLBACK_FIXTURES: list[dict] = [
    {"idx": 0, "date": "2026-04-11", "home_team": "Arsenal",             "away_team": "Bournemouth"},
    {"idx": 1, "date": "2026-04-11", "home_team": "Arsenal",             "away_team": "Bournemouth"},
    {"idx": 2, "date": "2026-04-11", "home_team": "Arsenal",             "away_team": "Bournemouth"},
    {"idx": 3, "date": "2026-04-11", "home_team": "Arsenal",             "away_team": "Bournemouth"},
    {"idx": 4, "date": "2026-04-12", "home_team": "Arsenal",             "away_team": "Bournemouth"},
    {"idx": 5, "date": "2026-04-12", "home_team": "Arsenal",             "away_team": "Bournemouth"},
    {"idx": 6, "date": "2026-04-12", "home_team": "Arsenal",             "away_team": "Bournemouth"},
    {"idx": 7, "date": "2026-04-12", "home_team": "Arsenal",             "away_team": "Bournemouth"},
    {"idx": 8, "date": "2026-04-12", "home_team": "Arsenal",             "away_team": "Bournemouth"},
    {"idx": 9, "date": "2026-04-14", "home_team": "Arsenal",             "away_team": "Bournemouth"},
]

from typing import TypedDict

class MatchDict(TypedDict):
    home_team: str
    away_team: str
    badge_home: str
    badge_away: str
    disp_odds_home: str
    disp_odds_draw: str
    disp_odds_away: str
    disp_prob_home: str
    disp_prob_draw: str
    disp_prob_away: str
    prob_home: float
    prob_draw: float
    prob_away: float
    fair_odds_home: float
    fair_odds_draw: float
    fair_odds_away: float
    actual: str
    user_pick: str
    model_pick: str

class GWDict(TypedDict):
    idx: int
    gw: str
    date: str
    matches: list[MatchDict]
    model_accuracy: str
    user_accuracy: str
    pnl: str
    pnl_positive: bool

class FixtureDict(TypedDict):
    idx: int
    date: str
    home_team: str
    away_team: str

class PredictionDict(TypedDict):
    match_idx: int
    home_team: str
    away_team: str
    badge_home: str
    badge_away: str
    disp_odds_home: str
    disp_odds_draw: str
    disp_odds_away: str
    disp_prob_home: str
    disp_prob_draw: str
    disp_prob_away: str
    prob_home: float
    prob_draw: float
    prob_away: float
    fair_odds_home: float
    fair_odds_draw: float
    fair_odds_away: float
    book_odds_home: float | None
    book_odds_draw: float | None
    book_odds_away: float | None
    book_prob_home: float | None
    book_prob_draw: float | None
    book_prob_away: float | None
    disp_book_odds_home: str
    disp_book_odds_draw: str
    disp_book_odds_away: str
    disp_book_prob_home: str
    disp_book_prob_draw: str
    disp_book_prob_away: str
    disp_elo_diff: str
    chart_label: str
    model_pick: str

class ChartBarDict(TypedDict):
    label: str
    home: float
    draw: float
    away: float

class EloChartDict(TypedDict):
    label: str
    elo_diff: float
    elo_positive: bool

# ── State ─────────────────────────────────────────────────────────────────────

def _norm_fixture_date_for_sig(val: Any) -> str:
    """Normalize dates so cache fixture rows match fixtures.csv (string quirks)."""
    ts = pd.to_datetime(val, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return str(val).strip() if val is not None else ""
    return str(ts.normalize().date())


def _backfill_poisson_display_fields(prediction: dict[str, Any]) -> None:
    """Derive Σλ / O2.5 / BTTS / vs-ensemble strings when missing (older caches)."""
    try:
        lh = float(prediction.get("lambda_home", 0.0))
        la = float(prediction.get("lambda_away", 0.0))
    except (TypeError, ValueError):
        lh, la = 0.0, 0.0
    poisson_ready = (lh + la) > 1e-6
    if prediction.get("disp_poisson_xg_total") in (None, "", "—"):
        if lh or la:
            prediction["disp_poisson_xg_total"] = f"{lh + la:.2f}"
    if prediction.get("disp_poisson_o25") in (None, "", "—"):
        if not poisson_ready:
            pass
        else:
            try:
                v = float(prediction.get("poisson_over_25", float("nan")))
                if math.isfinite(v):
                    prediction["disp_poisson_o25"] = f"{v * 100:.1f}%"
            except (TypeError, ValueError):
                pass
    if prediction.get("disp_poisson_btts") in (None, "", "—"):
        if not poisson_ready:
            pass
        else:
            try:
                v = float(prediction.get("poisson_btts", float("nan")))
                if math.isfinite(v):
                    prediction["disp_poisson_btts"] = f"{v * 100:.1f}%"
            except (TypeError, ValueError):
                pass
    if prediction.get("disp_poisson_vs_ensemble") in (None, "", "—"):
        try:
            e_h = float(prediction["prob_home"])
            e_d = float(prediction["prob_draw"])
            e_a = float(prediction["prob_away"])
            p_h = float(prediction["poisson_prob_home"])
            p_d = float(prediction["poisson_prob_draw"])
            p_a = float(prediction["poisson_prob_away"])
        except (TypeError, ValueError, KeyError):
            return
        if not poisson_ready or (p_h + p_d + p_a) < 0.5:
            return
        d_h = (p_h - e_h) * 100.0
        d_d = (p_d - e_d) * 100.0
        d_a = (p_a - e_a) * 100.0
        prediction["disp_poisson_vs_ensemble"] = (
            f"Poisson−ens. (pp): H {d_h:+.0f} · D {d_d:+.0f} · A {d_a:+.0f}"
        )


class State(rx.State):
    current_tab: str = "home"

    # Matchweek
    fixtures: list[dict[str, Any]] = FALLBACK_FIXTURES
    predictions: list[dict[str, Any]] = []
    gameweek_label: str = "GW32"
    prediction_source: str = ""      # "cache" | "live"
    prediction_source_note: str = "" # reason or timestamp

    # User picks for current GW (parallel to predictions list)
    user_picks: list[str] = []   # "" / "H" / "D" / "A"

    # Custom predictor
    custom_home: str = PL_TEAMS[0]
    custom_away: str = PL_TEAMS[1]
    custom_result: list[dict[str, Any]] = []

    # Insights
    safe_picks: list[dict[str, Any]] = []
    coin_flips: list[dict[str, Any]] = []
    top_pick: dict[str, Any] = {}
    underdog: dict[str, Any] = {}
    win_prob_chart: list[dict[str, Any]] = []
    elo_chart: list[dict[str, Any]] = []
    bookmaker_rows: list[dict[str, Any]] = []
    prob_divergence_rows: list[dict[str, Any]] = []
    value_edge_rows: list[dict[str, Any]] = []

    # History
    history: list[dict[str, Any]] = []
    history_selected: int = -1

    # Virtual bankroll
    bankroll_summary: dict[str, Any] = {
        "starting_bankroll": 1000.0,
        "starting_bankroll_disp": "$1000.00",
        "current_bankroll": 1000.0,
        "current_bankroll_disp": "$1000.00",
        "risk_cap": 0.05,
        "risk_cap_disp": "5.0%",
        "total_pnl": 0.0,
        "total_pnl_disp": "+$0.00",
        "pnl_positive": True,
        "roi_disp": "+0.0%",
        "record": "0-0",
        "settled_count": 0,
        "pending_count": 0,
    }
    bankroll_ledger: list[dict[str, Any]] = []
    bankroll_suggestions: list[dict[str, Any]] = []
    bankroll_starting_input: str = "1000"
    bankroll_risk_cap_input: str = "5"

    # ── Computed vars ─────────────────────────────────────────────────────────

    @rx.var
    def predictions_with_picks(self) -> list[dict[str, Any]]:
        """Merge predictions with current user picks for display."""
        result = []
        for i, p in enumerate(self.predictions):
            pick = self.user_picks[i] if i < len(self.user_picks) else ""
            result.append({**p, "user_pick": pick})
        return result

    @rx.var
    def picks_count(self) -> int:
        return sum(1 for p in self.user_picks if p != "")

    @rx.var
    def total_fixtures(self) -> int:
        return len(self.predictions)

    @rx.var
    def all_picked(self) -> bool:
        return len(self.user_picks) > 0 and all(p != "" for p in self.user_picks)

    @rx.var
    def picks_agree_count(self) -> int:
        """Count how many user picks agree with model picks."""
        count = 0
        for i, p in enumerate(self.predictions):
            if i < len(self.user_picks) and self.user_picks[i] != "":
                if self.user_picks[i] == p.get("model_pick", ""):
                    count += 1
        return count

    @rx.var
    def selected_gw_entry(self) -> dict[str, Any]:
        if self.history_selected < 0 or self.history_selected >= len(self.history):
            return {}
        return self.history[self.history_selected]

    @rx.var
    def selected_gw_matches(self) -> list[dict[str, Any]]:
        if self.history_selected < 0 or self.history_selected >= len(self.history):
            return []
        matches = self.history[self.history_selected]["matches"]
        return [dict(m, match_idx=i) for i, m in enumerate(matches)]

    @rx.var
    def prediction_source_label(self) -> str:
        if self.prediction_source == "cache":
            note = f" ({self.prediction_source_note})" if self.prediction_source_note else ""
            return f"Cached predictions{note}"
        if self.prediction_source == "live":
            return "Live inference (model run in app)"
        return "Model predictions"

    @rx.var
    def prediction_source_color(self) -> str:
        if self.prediction_source == "cache":
            return "#4CAF50"
        if self.prediction_source == "live":
            return "#FFB74D"
        return "#444"

    # ── Fixtures ──────────────────────────────────────────────────────────────

    def _reindex(self):
        for i, f in enumerate(self.fixtures):
            f["idx"] = i

    def load_fixtures_from_csv(self):
        # Use an absolute path so running from a different CWD
        # (e.g. during export/dev server) still reads the intended file.
        path = os.path.join(APP_DIR, "fixtures.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.normalize()
                df = df.dropna(subset=["date", "home_team", "away_team"])
                today = _local_today_hk()

                # Accept either "gameweek" or "matchweek" as the GW column name
                gw_col = next((c for c in ["gameweek", "matchweek"] if c in df.columns and df[c].notna().any()), None)

                if gw_col:
                    # Find the current or next gameweek:
                    # - If any matches in a GW are still today/future, that GW is active
                    # - This keeps mid-gameweek refreshes showing the full GW (including
                    #   already-played Saturday games alongside upcoming Sunday ones)
                    future = df[df["date"] >= today].sort_values("date")
                    if future.empty:
                        self.fixtures = FALLBACK_FIXTURES
                        return
                    current_gw = future.iloc[0][gw_col]
                    # Pull ALL fixtures for that GW from the full df, not just future ones
                    selected = df[df[gw_col] == current_gw].sort_values("date")
                    self.gameweek_label = f"GW{int(current_gw)}" if str(current_gw).isdigit() else f"{current_gw}"
                else:
                    # No GW column — fall back to a 4-day window around the next fixture
                    future = df[df["date"] >= today].sort_values("date")
                    if future.empty:
                        self.fixtures = FALLBACK_FIXTURES
                        return
                    start = future["date"].min()
                    end = start + pd.Timedelta(days=3)
                    selected = df[(df["date"] >= start) & (df["date"] <= end)]
                    self.gameweek_label = f"Next ({start.strftime('%b %d')})"
                rows: list[dict[str, Any]] = []
                has_b365_cols = all(c in selected.columns for c in ["B365H", "B365D", "B365A"])
                for i, row in enumerate(selected.itertuples(index=False), start=0):
                    fixture_row = {
                        "idx": i,
                        "date": str(getattr(row, "date").date()),
                        "home_team": str(getattr(row, "home_team")),
                        "away_team": str(getattr(row, "away_team")),
                    }
                    if has_b365_cols:
                        fixture_row["B365H"] = getattr(row, "B365H")
                        fixture_row["B365D"] = getattr(row, "B365D")
                        fixture_row["B365A"] = getattr(row, "B365A")
                    rows.append(fixture_row)
                if rows:
                    self.fixtures = rows
            except Exception:
                self.fixtures = FALLBACK_FIXTURES
        else:
            self.fixtures = FALLBACK_FIXTURES

    def add_fixture(self):
        self.fixtures.append({"idx": len(self.fixtures), "date": "", "home_team": "", "away_team": ""})

    def delete_fixture(self, idx: int):
        self.fixtures = [f for f in self.fixtures if f["idx"] != idx]
        self._reindex()

    def update_date(self, idx: int, value: str):
        self.fixtures[idx]["date"] = value

    def update_home_fixture(self, idx: int, value: str):
        self.fixtures[idx]["home_team"] = value

    def update_away_fixture(self, idx: int, value: str):
        self.fixtures[idx]["away_team"] = value

    # ── ML pipeline ──────────────────────────────────────────────────────────

    def _load_model_and_elo(self):
        from team_aliases import elo_lookup_key
        from features import load_model_and_elo
        model_path = os.path.join(APP_DIR, "xgboost_premier_league_model.pkl")
        elo_path = os.path.join(APP_DIR, "premier_league_with_elo_best.csv")
        model, df_elo = load_model_and_elo(model_path, elo_path)
        return model, df_elo, elo_lookup_key

    def _with_poisson_layer(self, upcoming: pd.DataFrame, df_elo: pd.DataFrame) -> pd.DataFrame:
        """Attach λ + Poisson markets using goal_model_*.pkl (same as run_pipeline)."""
        gh_path = os.path.join(APP_DIR, "goal_model_home.pkl")
        ga_path = os.path.join(APP_DIR, "goal_model_away.pkl")
        if not (os.path.isfile(gh_path) and os.path.isfile(ga_path)):
            return upcoming
        try:
            from run_pipeline import add_poisson_outputs

            gh = joblib.load(gh_path)
            ga = joblib.load(ga_path)
            return add_poisson_outputs(upcoming, gh, ga, df_elo)
        except Exception:
            return upcoming

    def _predict_upcoming(self, upcoming: pd.DataFrame, df_elo: pd.DataFrame, model) -> pd.DataFrame:
        from features import detect_model_features, ensure_model_features
        feature_cols = detect_model_features(model)
        if not feature_cols:
            raise ValueError("Cannot determine expected features from loaded model.")
        upcoming = ensure_model_features(upcoming, df_elo, feature_cols)
        probs = model.predict_proba(upcoming[feature_cols])

        # Model classes are encoded as 0=Away, 1=Draw, 2=Home.
        upcoming["prob_away"]      = probs[:, 0]
        upcoming["prob_draw"]      = probs[:, 1]
        upcoming["prob_home"]      = probs[:, 2]
        upcoming["fair_odds_home"] = 1 / upcoming["prob_home"]
        upcoming["fair_odds_draw"] = 1 / upcoming["prob_draw"]
        upcoming["fair_odds_away"] = 1 / upcoming["prob_away"]

        upcoming["disp_odds_home"] = upcoming["fair_odds_home"].map(format_odds_display)
        upcoming["disp_odds_draw"] = upcoming["fair_odds_draw"].map(format_odds_display)
        upcoming["disp_odds_away"] = upcoming["fair_odds_away"].map(format_odds_display)
        upcoming["disp_prob_home"] = upcoming["prob_home"].map(lambda v: f"{v*100:.1f}%")
        upcoming["disp_prob_draw"] = upcoming["prob_draw"].map(lambda v: f"{v*100:.1f}%")
        upcoming["disp_prob_away"] = upcoming["prob_away"].map(lambda v: f"{v*100:.1f}%")
        upcoming["disp_elo_diff"]  = upcoming["elo_diff"].map(lambda v: f"{v:+.0f}")
        upcoming["badge_home"]     = upcoming["home_team"].map(
            lambda t: TEAM_BADGES.get(badge_lookup_key(str(t)), FALLBACK_BADGE)
        )
        upcoming["badge_away"]     = upcoming["away_team"].map(
            lambda t: TEAM_BADGES.get(badge_lookup_key(str(t)), FALLBACK_BADGE)
        )
        upcoming["chart_label"]    = upcoming.apply(
            lambda r: r["home_team"][:3].upper() + " v " + r["away_team"][:3].upper(), axis=1)

        # Model's predicted outcome per match
        upcoming["model_pick"] = upcoming.apply(
            lambda r: max(
                [("H", r["prob_home"]), ("D", r["prob_draw"]), ("A", r["prob_away"])],
                key=lambda x: x[1]
            )[0], axis=1
        )

        has_book_cols = all(c in upcoming.columns for c in ("B365H", "B365D", "B365A"))
        if has_book_cols:
            book_h = pd.to_numeric(upcoming["B365H"], errors="coerce")
            book_d = pd.to_numeric(upcoming["B365D"], errors="coerce")
            book_a = pd.to_numeric(upcoming["B365A"], errors="coerce")
            valid = (book_h > 0) & (book_d > 0) & (book_a > 0)

            upcoming["book_odds_home"] = book_h.where(valid, np.nan)
            upcoming["book_odds_draw"] = book_d.where(valid, np.nan)
            upcoming["book_odds_away"] = book_a.where(valid, np.nan)

            inv_h = (1.0 / book_h).where(valid, np.nan)
            inv_d = (1.0 / book_d).where(valid, np.nan)
            inv_a = (1.0 / book_a).where(valid, np.nan)
            total = (inv_h + inv_d + inv_a).where(valid, np.nan)

            upcoming["book_prob_home"] = (inv_h / total).where(valid, np.nan)
            upcoming["book_prob_draw"] = (inv_d / total).where(valid, np.nan)
            upcoming["book_prob_away"] = (inv_a / total).where(valid, np.nan)
        else:
            upcoming["book_odds_home"] = np.nan
            upcoming["book_odds_draw"] = np.nan
            upcoming["book_odds_away"] = np.nan
            upcoming["book_prob_home"] = np.nan
            upcoming["book_prob_draw"] = np.nan
            upcoming["book_prob_away"] = np.nan

        upcoming["disp_book_odds_home"] = upcoming["book_odds_home"].map(
            lambda v: format_odds_display(v) if pd.notna(v) else ""
        )
        upcoming["disp_book_odds_draw"] = upcoming["book_odds_draw"].map(
            lambda v: format_odds_display(v) if pd.notna(v) else ""
        )
        upcoming["disp_book_odds_away"] = upcoming["book_odds_away"].map(
            lambda v: format_odds_display(v) if pd.notna(v) else ""
        )
        upcoming["disp_book_prob_home"] = upcoming["book_prob_home"].map(
            lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
        )
        upcoming["disp_book_prob_draw"] = upcoming["book_prob_draw"].map(
            lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
        )
        upcoming["disp_book_prob_away"] = upcoming["book_prob_away"].map(
            lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
        )
        return upcoming

    def _compute_insights(self, df: pd.DataFrame):
        records = df.to_dict("records")
        bookmaker_rows: list[dict[str, Any]] = []
        prob_divergence_rows: list[dict[str, Any]] = []
        value_edge_rows: list[dict[str, Any]] = []
        for r in records:
            probs = {"H": r["prob_home"], "D": r["prob_draw"], "A": r["prob_away"]}
            pred = max(probs.items(), key=lambda kv: kv[1])[0]
            r["pred"] = pred
            r["pred_prob"] = probs[pred]
            if pred == "H":
                r["pred_name"] = r["home_team"]
                r["pred_badge"] = r["badge_home"]
                r["pred_disp_prob"] = r["disp_prob_home"]
                r["pred_disp_odds"] = r["disp_odds_home"]
                r["pred_fair_odds"] = r["fair_odds_home"]
            elif pred == "A":
                r["pred_name"] = r["away_team"]
                r["pred_badge"] = r["badge_away"]
                r["pred_disp_prob"] = r["disp_prob_away"]
                r["pred_disp_odds"] = r["disp_odds_away"]
                r["pred_fair_odds"] = r["fair_odds_away"]
            else:
                r["pred_name"] = "Draw"
                r["pred_badge"] = ""
                r["pred_disp_prob"] = r["disp_prob_draw"]
                r["pred_disp_odds"] = r["disp_odds_draw"]
                r["pred_fair_odds"] = r["fair_odds_draw"]

            r["book_available"] = bool(
                pd.notna(r.get("book_prob_home"))
                and pd.notna(r.get("book_prob_draw"))
                and pd.notna(r.get("book_prob_away"))
                and pd.notna(r.get("book_odds_home"))
                and pd.notna(r.get("book_odds_draw"))
                and pd.notna(r.get("book_odds_away"))
            )

            if not r["book_available"]:
                continue

            bookmaker_rows.append({
                "label": r["chart_label"],
                "model_pick": r["model_pick"],
                "m_home": r["disp_odds_home"],
                "m_draw": r["disp_odds_draw"],
                "m_away": r["disp_odds_away"],
                "b_home": r["disp_book_odds_home"],
                "b_draw": r["disp_book_odds_draw"],
                "b_away": r["disp_book_odds_away"],
            })

            for outcome, label_o in [("H", "Home"), ("D", "Draw"), ("A", "Away")]:
                model_p = {"H": r["prob_home"], "D": r["prob_draw"], "A": r["prob_away"]}[outcome]
                book_p = {"H": r["book_prob_home"], "D": r["book_prob_draw"], "A": r["book_prob_away"]}[outcome]
                edge = (model_p - book_p) * 100.0
                prob_divergence_rows.append({
                    "label": r["chart_label"],
                    "outcome": label_o,
                    "edge_pp": round(edge, 2),
                    "edge_disp": f"{edge:+.2f} pp",
                    "model_prob_disp": {"H": r["disp_prob_home"], "D": r["disp_prob_draw"], "A": r["disp_prob_away"]}[outcome],
                    "book_prob_disp": {"H": r["disp_book_prob_home"], "D": r["disp_book_prob_draw"], "A": r["disp_book_prob_away"]}[outcome],
                })

            model_fair_map = {"H": r["fair_odds_home"], "D": r["fair_odds_draw"], "A": r["fair_odds_away"]}
            book_odds_map = {"H": r["book_odds_home"], "D": r["book_odds_draw"], "A": r["book_odds_away"]}
            for outcome in ("H", "D", "A"):
                fair_odds = model_fair_map[outcome]
                book_odds = book_odds_map[outcome]
                if pd.notna(fair_odds) and pd.notna(book_odds) and float(fair_odds) > 0:
                    edge_pct = (float(book_odds) / float(fair_odds) - 1.0) * 100.0
                    value_edge_rows.append({
                        "label": r["chart_label"],
                        "outcome": outcome,
                        "edge_pct": round(edge_pct, 2),
                        "edge_disp": f"{edge_pct:+.2f}%",
                        "model_fair_odds": format_odds_display(float(fair_odds)),
                        "book_odds": format_odds_display(float(book_odds)),
                        "edge_positive": bool(edge_pct >= 0),
                    })

        self.safe_picks = [r for r in records if r["pred_prob"] > 0.65]
        # Coin flip: no clear favourite — best outcome probability under 50%
        self.coin_flips = [
            r for r in records
            if r["pred_prob"] < 0.50
        ]
        self.top_pick = max(records, key=lambda r: r["pred_prob"], default={})
        underdogs = [r for r in records if r.get("pred") in ("H", "A")]
        self.underdog = min(underdogs, key=lambda r: r["pred_prob"], default={})
        self.win_prob_chart = [
            {"label": r["chart_label"], "home": round(r["prob_home"]*100,1),
             "draw": round(r["prob_draw"]*100,1), "away": round(r["prob_away"]*100,1)}
            for r in records
        ]
        self.elo_chart = [
            {"label": r["chart_label"], "elo_diff": round(r["elo_diff"], 0),
             "elo_positive": bool(r["elo_diff"] >= 0)}
            for r in records
        ]
        self.bookmaker_rows = bookmaker_rows
        self.prob_divergence_rows = sorted(
            prob_divergence_rows,
            key=lambda r: abs(r["edge_pp"]),
            reverse=True,
        )[:15]
        self.value_edge_rows = sorted(
            value_edge_rows,
            key=lambda r: r["edge_pct"],
            reverse=True,
        )[:15]

    def _fixture_signature(self, rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
        """Stable signature used to ensure cache matches current fixture list."""
        return sorted(
            (
                _norm_fixture_date_for_sig(r.get("date")),
                str(r.get("home_team", "")).strip(),
                str(r.get("away_team", "")).strip(),
            )
            for r in rows
        )

    def _load_predictions_cache_dict(self) -> tuple[dict[str, Any] | None, str]:
        """Read cache JSON from env, remote_data_urls.json, or predictions_cache.json."""
        url = predictions_cache_url()
        if url:
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                return json.loads(r.text), ""
            except Exception as e:
                if not os.path.exists(CACHE_FILE):
                    return None, f"remote cache failed ({e}); no local file"
        if not os.path.exists(CACHE_FILE):
            return None, "cache file missing"
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f), ""
        except Exception:
            return None, "cache unreadable"

    def _try_load_predictions_cache(self) -> tuple[bool, str]:
        cache, err = self._load_predictions_cache_dict()
        if cache is None:
            return False, err or "cache unavailable"

        preds = cache.get("predictions")
        if not isinstance(preds, list) or not preds:
            return False, "cache has no predictions"

        generated_at = pd.to_datetime(cache.get("generated_at"), errors="coerce")
        if pd.isna(generated_at):
            return False, "cache timestamp invalid"

        now = pd.Timestamp.now(tz=generated_at.tz) if generated_at.tz is not None else pd.Timestamp.now()
        age_hours = (now - generated_at).total_seconds() / 3600.0
        if CACHE_MAX_AGE_HOURS > 0 and age_hours > CACHE_MAX_AGE_HOURS:
            return False, f"cache stale ({age_hours:.1f}h old)"

        cached_gw = str(cache.get("gameweek", "")).strip()
        if cached_gw != str(self.gameweek_label).strip():
            return False, f"cache gw mismatch ({cached_gw} vs {self.gameweek_label})"

        fixture_rows = [{k: v for k, v in f.items() if k != "idx"} for f in self.fixtures]
        cache_sig = self._fixture_signature(preds)
        fixture_sig = self._fixture_signature(fixture_rows)
        if cache_sig != fixture_sig:
            return False, "cache fixtures mismatch"

        self.predictions = preds
        for i in range(len(self.predictions)):
            self.predictions[i]["match_idx"] = i
            self._ensure_poisson_defaults(self.predictions[i])
        self._restore_user_picks_for_current_gw()
        self._compute_insights(pd.DataFrame(self.predictions))
        self.prediction_source = "cache"
        self.prediction_source_note = generated_at.strftime("%Y-%m-%d %H:%M")
        self._archive_current_cache_if_missing(cache)
        return True, ""

    def _archive_current_cache_if_missing(self, cache: dict[str, Any]):
        """Write predictions_history/GW{N}.json if it doesn't yet exist."""
        gw = _gw_int_from_label(self.gameweek_label)
        if gw is None:
            return
        history_dir = os.path.join(APP_DIR, "predictions_history")
        archive_path = os.path.join(history_dir, f"GW{gw}.json")
        if os.path.exists(archive_path):
            return
        try:
            os.makedirs(history_dir, exist_ok=True)
            with open(archive_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass

    def _load_predictions_live(self):
        model, df_elo, _elo_key = self._load_model_and_elo()
        raw = [{k: v for k, v in f.items() if k != "idx"} for f in self.fixtures]
        upcoming = pd.DataFrame(raw)
        upcoming["date"] = pd.to_datetime(upcoming["date"])
        from features import compute_current_elo
        upcoming = compute_current_elo(upcoming, df_elo, _elo_key)
        upcoming = self._predict_upcoming(upcoming, df_elo, model)
        upcoming = self._with_poisson_layer(upcoming, df_elo)
        self.predictions = upcoming.to_dict("records")
        for i in range(len(self.predictions)):
            self.predictions[i]["match_idx"] = i
            self._ensure_poisson_defaults(self.predictions[i])
        self._restore_user_picks_for_current_gw()
        self._compute_insights(upcoming)

    @staticmethod
    def _ensure_poisson_defaults(prediction: dict[str, Any]) -> None:
        prediction.setdefault("lambda_home", 0.0)
        prediction.setdefault("lambda_away", 0.0)
        prediction.setdefault("disp_lambda_home", "—")
        prediction.setdefault("disp_lambda_away", "—")
        prediction.setdefault("poisson_prob_home", 0.0)
        prediction.setdefault("poisson_prob_draw", 0.0)
        prediction.setdefault("poisson_prob_away", 0.0)
        prediction.setdefault("disp_poisson_prob_home", "—")
        prediction.setdefault("disp_poisson_prob_draw", "—")
        prediction.setdefault("disp_poisson_prob_away", "—")
        prediction.setdefault("poisson_over_25", 0.0)
        prediction.setdefault("poisson_btts", 0.0)
        prediction.setdefault("disp_poisson_xg_total", "—")
        prediction.setdefault("disp_poisson_o25", "—")
        prediction.setdefault("disp_poisson_btts", "—")
        prediction.setdefault("disp_poisson_vs_ensemble", "—")
        prediction.setdefault("poisson_correct_scores", [])
        prediction.setdefault("disp_poisson_correct_scores", "—")
        _backfill_poisson_display_fields(prediction)
        prediction["explanation"] = build_explanation(prediction)
        prediction.setdefault("explanation_summary", prediction["explanation"].get("driver_summary", ""))

    def _restore_user_picks_for_current_gw(self):
        """Populate self.user_picks from user_picks.json based on the current GW."""
        n = len(self.predictions)
        picks: list[str] = [""] * n
        gw = _gw_int_from_label(self.gameweek_label)
        if gw is not None:
            saved = load_user_picks().get(gw, {})
            for i, p in enumerate(self.predictions):
                m_idx = int(p.get("match_idx", i))
                pick = saved.get(m_idx, "")
                if pick in ("H", "D", "A"):
                    picks[i] = pick
        self.user_picks = picks

    def _rebuild_bankroll(self):
        settings = load_bankroll()
        archives = load_archived_predictions()
        results = load_results()
        summary = build_bankroll(archives, results, settings)
        settings["current_bankroll"] = summary["current_bankroll"]
        settings["ledger"] = summary["ledger"]
        try:
            save_bankroll(settings)
        except Exception:
            pass

        self.bankroll_summary = {k: v for k, v in summary.items() if k not in ("ledger", "latest_ledger", "suggestions")}
        self.bankroll_ledger = summary["latest_ledger"]
        self.bankroll_suggestions = build_current_suggestions(
            self.predictions,
            float(summary["current_bankroll"]),
            float(summary["risk_cap"]),
        )
        self.bankroll_starting_input = str(summary["starting_bankroll"])
        self.bankroll_risk_cap_input = str(round(float(summary["risk_cap"]) * 100, 2))

    def update_bankroll_starting(self, value: str):
        self.bankroll_starting_input = value

    def update_bankroll_risk_cap(self, value: str):
        self.bankroll_risk_cap_input = value

    def save_bankroll_settings(self):
        try:
            starting = max(float(self.bankroll_starting_input), 1.0)
            risk_cap_pct = max(0.0, min(100.0, float(self.bankroll_risk_cap_input)))
        except ValueError:
            return rx.toast.error("Bankroll settings must be numbers.")
        settings = load_bankroll()
        settings["starting_bankroll"] = starting
        settings["risk_cap"] = risk_cap_pct / 100.0
        settings["current_bankroll"] = starting
        settings["ledger"] = []
        save_bankroll(settings)
        self._rebuild_bankroll()
        return rx.toast.success("Bankroll settings saved.")

    def load_predictions(self):
        self.load_fixtures_from_csv()
        self.prediction_source = ""
        self.prediction_source_note = ""

        cache_ok, cache_reason = self._try_load_predictions_cache()
        try:
            self._rebuild_history()
        except Exception:
            pass
        try:
            self._rebuild_bankroll()
        except Exception:
            pass

        if cache_ok:
            return rx.toast.success("Loaded cached predictions.")

        try:
            self._load_predictions_live()
            self.prediction_source = "live"
            self.prediction_source_note = cache_reason
            self._rebuild_bankroll()
            reason = cache_reason if cache_reason else "cache unavailable"
            return rx.toast.warning(f"Using live inference: {reason}")
        except Exception as e:
            return rx.toast.error(f"Prediction error: {e}")

    # ── Custom predictor ──────────────────────────────────────────────────────

    def run_custom_prediction(self):
        if self.custom_home == self.custom_away:
            return rx.toast.error("Pick two different teams.")
        try:
            model, df_elo, _elo_key = self._load_model_and_elo()
            df = pd.DataFrame([{
                "date": str(_local_today_hk().date()),
                "home_team": self.custom_home,
                "away_team": self.custom_away,
            }])
            df["date"] = pd.to_datetime(df["date"])
            from features import compute_current_elo
            df = compute_current_elo(df, df_elo, _elo_key)
            df = self._predict_upcoming(df, df_elo, model)
            df = self._with_poisson_layer(df, df_elo)
            self.custom_result = df.to_dict("records")
        except Exception as e:
            return rx.toast.error(f"Prediction error: {e}")

    # ── User picks ────────────────────────────────────────────────────────────

    def set_user_pick(self, match_idx: int, pick: str):
        """Set the user's pick for a specific match."""
        picks = list(self.user_picks)
        while len(picks) <= match_idx:
            picks.append("")
        picks[match_idx] = pick
        self.user_picks = picks

    def lock_in_picks(self):
        """Persist user picks to disk for the current GW, then refresh history."""
        if not self.predictions:
            return rx.toast.error("No predictions loaded.")
        if not self.all_picked:
            return rx.toast.error("Pick an outcome for every match first.")

        gw = _gw_int_from_label(self.gameweek_label)
        if gw is None:
            return rx.toast.error("Could not determine current gameweek number.")

        picks_by_idx: dict[int, str] = {}
        for i, p in enumerate(self.predictions):
            pick = self.user_picks[i] if i < len(self.user_picks) else ""
            if pick in ("H", "D", "A"):
                picks_by_idx[int(p.get("match_idx", i))] = pick

        try:
            upsert_user_picks_for_gw(gw, picks_by_idx)
        except Exception as e:
            return rx.toast.error(f"Failed to save picks: {e}")

        self._rebuild_history()
        agree = self.picks_agree_count
        total = self.total_fixtures
        return rx.toast.success(
            f"{self.gameweek_label} locked in! You agreed with model on {agree}/{total} matches."
        )

    # ── History ───────────────────────────────────────────────────────────────

    def select_history(self, idx: int):
        self.history_selected = idx

    def back_to_history_list(self):
        self.history_selected = -1

    def set_actual_result(self, gw_idx: int, match_idx: int, result: str):
        """Deprecated: actual results now come from results.csv automatically."""
        return None

    def _rebuild_history(self):
        """Rebuild self.history from disk (archives + results.csv + user_picks.json)."""
        archives = load_archived_predictions()
        results = load_results()
        all_user_picks = load_user_picks()

        entries: list[dict[str, Any]] = []
        for gw in sorted(archives.keys(), reverse=True):
            cache = archives[gw]
            preds = cache.get("predictions", []) if isinstance(cache, dict) else []
            if not preds:
                continue
            picks_for_gw = all_user_picks.get(gw, {})

            matches: list[dict[str, Any]] = []
            for i, p in enumerate(preds):
                if not isinstance(p, dict):
                    continue
                home = str(p.get("home_team", ""))
                away = str(p.get("away_team", ""))
                key = (gw, home, away)
                actual = ""
                home_goals = ""
                away_goals = ""
                if key in results:
                    actual = results[key]["actual"]
                    home_goals = str(results[key]["home_goals"])
                    away_goals = str(results[key]["away_goals"])
                m_idx = int(p.get("match_idx", i))
                date_str = str(p.get("date", "")) if p.get("date") is not None else ""
                if date_str:
                    date_str = date_str.split("T")[0].split(" ")[0]
                matches.append({
                    "date":           date_str,
                    "home_team":      home,
                    "away_team":      away,
                    "badge_home":     p.get("badge_home", FALLBACK_BADGE),
                    "badge_away":     p.get("badge_away", FALLBACK_BADGE),
                    "disp_odds_home": p.get("disp_odds_home", ""),
                    "disp_odds_draw": p.get("disp_odds_draw", ""),
                    "disp_odds_away": p.get("disp_odds_away", ""),
                    "disp_prob_home": p.get("disp_prob_home", ""),
                    "disp_prob_draw": p.get("disp_prob_draw", ""),
                    "disp_prob_away": p.get("disp_prob_away", ""),
                    "prob_home":      float(p.get("prob_home", 0.0) or 0.0),
                    "prob_draw":      float(p.get("prob_draw", 0.0) or 0.0),
                    "prob_away":      float(p.get("prob_away", 0.0) or 0.0),
                    "fair_odds_home": float(p.get("fair_odds_home", 0.0) or 0.0),
                    "fair_odds_draw": float(p.get("fair_odds_draw", 0.0) or 0.0),
                    "fair_odds_away": float(p.get("fair_odds_away", 0.0) or 0.0),
                    "actual":         actual,
                    "home_goals":     home_goals,
                    "away_goals":     away_goals,
                    "user_pick":      picks_for_gw.get(m_idx, ""),
                    "model_pick":     str(p.get("model_pick", "")),
                    "match_idx":      m_idx,
                })

            entry = self._summarize_history_entry(gw, matches, cache)
            entry["idx"] = len(entries)
            entries.append(entry)

        self.history = entries
        if self.history_selected >= len(self.history):
            self.history_selected = -1

    @staticmethod
    def _summarize_history_entry(
        gw: int,
        matches: list[dict[str, Any]],
        cache: dict[str, Any],
    ) -> dict[str, Any]:
        completed = [m for m in matches if m["actual"] in ("H", "D", "A")]
        gw_label = str(cache.get("gameweek") or f"GW{gw}")
        first_date = matches[0].get("date", "") if matches else ""

        n = len(completed)
        if n == 0:
            return {
                "gw":             gw_label,
                "gw_num":         gw,
                "date":           first_date,
                "matches":        matches,
                "model_accuracy": "",
                "user_accuracy":  "",
                "pnl":            "",
                "pnl_positive":   False,
            }

        model_correct = 0
        user_correct = 0
        pnl = 0.0

        for m in completed:
            actual = m["actual"]
            model_pick = m.get("model_pick", "")
            user_pick = m.get("user_pick", "")
            if model_pick == actual:
                model_correct += 1
            if user_pick == actual:
                user_correct += 1
            odds_map = {
                "H": m["fair_odds_home"],
                "D": m["fair_odds_draw"],
                "A": m["fair_odds_away"],
            }
            if model_pick and model_pick == actual:
                pnl += odds_map.get(model_pick, 1.0) - 1.0
            elif model_pick:
                pnl -= 1.0

        model_acc = model_correct / n * 100
        user_acc = user_correct / n * 100
        return {
            "gw":             gw_label,
            "gw_num":         gw,
            "date":           first_date,
            "matches":        matches,
            "model_accuracy": f"{model_acc:.0f}%  ({model_correct}/{n})",
            "user_accuracy":  f"{user_acc:.0f}%  ({user_correct}/{n})",
            "pnl":            format_money(pnl, signed=True),
            "pnl_positive":   bool(pnl >= 0),
        }


def _gw_int_from_label(label: str) -> int | None:
    if not label:
        return None
    m = re.search(r"\d+", str(label))
    return int(m.group(0)) if m else None


# ── Shared UI helpers ─────────────────────────────────────────────────────────

def badge_img(url, size: str = "28px") -> rx.Component:
    return rx.image(src=url, width=size, height=size,
                    style={"object_fit": "contain", "flex_shrink": "0"})


def odds_col(label: str, odds, prob, color: str) -> rx.Component:
    return rx.vstack(
        rx.text(label, color="white",  font_size="0.65em", letter_spacing="0.1em", font_weight="600"),
        rx.text(odds,  color="white", font_size="1.5em",  font_weight="700",       line_height="1"),
        rx.text(prob,  color="white", font_size="0.74em"),
        spacing="1",
        align="center",
        width="33%",
    )


def poisson_summary(p: dict) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text("POISSON", color="#555", font_size="0.58em", letter_spacing="0.12em", font_weight="700"),
            rx.spacer(),
            rx.text("λ ", color="#555", font_size="0.62em"),
            rx.text(p["disp_lambda_home"], color="#4CAF50", font_size="0.66em", font_weight="700"),
            rx.text("-", color="#333", font_size="0.66em"),
            rx.text(p["disp_lambda_away"], color="#F44336", font_size="0.66em", font_weight="700"),
            width="100%",
            align="center",
            spacing="1",
        ),
        rx.hstack(
            rx.text("H ", rx.text.span(p["disp_poisson_prob_home"], font_weight="700"), color="#4CAF50", font_size="0.62em"),
            rx.text("D ", rx.text.span(p["disp_poisson_prob_draw"], font_weight="700"), color="#FFC107", font_size="0.62em"),
            rx.text("A ", rx.text.span(p["disp_poisson_prob_away"], font_weight="700"), color="#F44336", font_size="0.62em"),
            spacing="3",
            width="100%",
        ),
        rx.hstack(
            rx.text("Σλ", color="#555", font_size="0.58em", font_weight="600"),
            rx.text(p["disp_poisson_xg_total"], color="#ccc", font_size="0.62em", font_weight="700"),
            rx.text("·", color="#333", font_size="0.62em", padding_x="4px"),
            rx.text("O2.5", color="#555", font_size="0.58em", font_weight="600"),
            rx.text(p["disp_poisson_o25"], color="#9CCC65", font_size="0.62em", font_weight="700"),
            rx.text("·", color="#333", font_size="0.62em", padding_x="4px"),
            rx.text("BTTS", color="#555", font_size="0.58em", font_weight="600"),
            rx.text(p["disp_poisson_btts"], color="#81D4FA", font_size="0.62em", font_weight="700"),
            width="100%",
            align="center",
            spacing="1",
        ),
        rx.text(
            p["disp_poisson_vs_ensemble"],
            color="#777",
            font_size="0.6em",
            width="100%",
            line_height="1.35",
        ),
        spacing="2",
        width="100%",
        padding="8px 10px",
        background_color="#0f0f0f",
        border="1px solid #1b1b1b",
        border_radius="7px",
    )


def explanation_summary(p: dict) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text("WHY", color="#555", font_size="0.58em", letter_spacing="0.12em", font_weight="700"),
            rx.spacer(),
            rx.text("Pick ", color="#555", font_size="0.62em"),
            rx.text(p["model_pick"], color="#ddd", font_size="0.66em", font_weight="700"),
            width="100%",
            align="center",
            spacing="1",
        ),
        rx.text(
            p["explanation_summary"],
            color="#aaa",
            font_size="0.66em",
            line_height="1.35",
            width="100%",
        ),
        spacing="2",
        width="100%",
        padding="8px 10px",
        background_color="#101010",
        border="1px solid #1b1b1b",
        border_radius="7px",
    )


def match_card(p: dict) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                badge_img(p["badge_home"]),
                rx.text(p["home_team"], font_weight="600", font_size="0.82em",
                        color="#d0d0d0", flex="1", text_align="right"),
                rx.text("v", color="#2a2a2a", font_size="0.7em", padding_x="8px"),
                rx.text(p["away_team"], font_weight="600", font_size="0.82em",
                        color="#d0d0d0", flex="1"),
                badge_img(p["badge_away"]),
                width="100%",
                align="center",
            ),
            rx.box(height="1px", width="100%", background_color="#1e1e1e"),
            rx.hstack(
                odds_col("H", p["disp_odds_home"], p["disp_prob_home"], "#4CAF50"),
                odds_col("D", p["disp_odds_draw"], p["disp_prob_draw"], "#FFC107"),
                odds_col("A", p["disp_odds_away"], p["disp_prob_away"], "#F44336"),
                width="100%",
                justify="between",
            ),
            poisson_summary(p),
            explanation_summary(p),
            spacing="3",
            align="center",
            width="100%",
        ),
        width="100%",
        padding="13px 14px",
        background_color="#141414",
        border_radius="8px",
        border="1px solid #1e1e1e",
    )


def section_label(text: str) -> rx.Component:
    return rx.text(text, color="#555", font_size="0.68em",
                   letter_spacing="0.12em", font_weight="700",
                   padding_bottom="4px")


# ── Home tab ──────────────────────────────────────────────────────────────────

def home_tab() -> rx.Component:
    return rx.vstack(
        rx.cond(
            State.predictions.length() > 0,
            rx.vstack(
                rx.hstack(
                    section_label(State.gameweek_label),
                    rx.spacer(),
                    rx.text(State.prediction_source_label, color=State.prediction_source_color, font_size="0.7em"),
                    width="100%",
                    align="center",
                    padding_bottom="6px",
                ),
                rx.foreach(State.predictions.to(list[dict[str, Any]]), match_card),
                width="100%",
                spacing="2",
            ),
            rx.box(
                rx.text("Loading predictions…", color="#444", font_size="0.88em"),
                text_align="center", padding_y="48px", width="100%",
            ),
        ),
        width="100%",
        spacing="2",
    )


# ── Predictor tab ─────────────────────────────────────────────────────────────

def pick_outcome_btn(label: str, pick_dict: dict, outcome: str, color: str) -> rx.Component:
    """A single H / D / A pick button for a fixture."""
    is_selected = pick_dict["user_pick"] == outcome
    is_model    = pick_dict["model_pick"] == outcome
    return rx.box(
        rx.vstack(
            rx.text(label, font_size="0.75em", font_weight="700",
                    color=rx.cond(is_selected, "#0E1117", "#888"),
                    line_height="1"),
            rx.cond(
                is_model,
                rx.text("◆", font_size="0.5em",
                        color=rx.cond(is_selected, "#0E1117", color),
                        line_height="1"),
                rx.box(height="8px"),
            ),
            spacing="1",
            align="center",
        ),
        on_click=State.set_user_pick(pick_dict["match_idx"], outcome),
        background_color=rx.cond(is_selected, color, "#1a1a1a"),
        border=rx.cond(is_model,
                       rx.cond(is_selected, f"2px solid {color}", f"2px solid {color}"),
                       "2px solid #2a2a2a"),
        border_radius="6px",
        padding="6px 0",
        width="58px",
        text_align="center",
        cursor="pointer",
        transition="all 0.15s ease",
    )


def pick_card(p: dict) -> rx.Component:
    """Single fixture card with user pick buttons and model indicator."""
    return rx.box(
        rx.vstack(
            # Teams row
            rx.hstack(
                badge_img(p["badge_home"], "22px"),
                rx.text(p["home_team"], font_size="0.78em", font_weight="600",
                        color="#d0d0d0", flex="1", text_align="right", no_of_lines=1),
                rx.text("v", color="#2a2a2a", font_size="0.65em", padding_x="6px"),
                rx.text(p["away_team"], font_size="0.78em", font_weight="600",
                        color="#d0d0d0", flex="1", no_of_lines=1),
                badge_img(p["badge_away"], "22px"),
                width="100%",
                align="center",
            ),
            rx.box(height="1px", width="100%", background_color="#1e1e1e"),
            # Pick buttons row
            rx.hstack(
                pick_outcome_btn("H", p, "H", "#4CAF50"),
                pick_outcome_btn("D", p, "D", "#FFC107"),
                pick_outcome_btn("A", p, "A", "#F44336"),
                rx.spacer(),
                # Probability hint
                rx.vstack(
                    rx.text(p["disp_prob_home"], color="#4CAF50", font_size="0.65em"),
                    rx.text(p["disp_prob_draw"], color="#FFC107", font_size="0.65em"),
                    rx.text(p["disp_prob_away"], color="#F44336", font_size="0.65em"),
                    spacing="0",
                    align="end",
                ),
                width="100%",
                align="center",
                spacing="2",
            ),
            # Legend hint
            rx.text("◆ = model pick", color="#333", font_size="0.6em", align_self="end"),
            spacing="2",
            width="100%",
        ),
        width="100%",
        padding="12px 14px",
        background_color="#141414",
        border_radius="8px",
        border="1px solid #1e1e1e",
    )


def predictor_tab() -> rx.Component:
    return rx.vstack(
        rx.cond(
            State.predictions.length() == 0,
            rx.box(
                rx.text("Predictions not loaded yet.", color="#444", font_size="0.85em"),
                text_align="center", padding_y="48px", width="100%",
            ),
            rx.vstack(
                # Header with GW label + picks counter
                rx.hstack(
                    rx.vstack(
                        rx.text(State.gameweek_label, color="white",
                                font_size="1em", font_weight="700"),
                        rx.text("Your Picks vs Model", color="#555",
                                font_size="0.68em", letter_spacing="0.08em"),
                        spacing="0",
                        align="start",
                    ),
                    rx.spacer(),
                    # Picks progress pill
                    rx.box(
                        rx.hstack(
                            rx.text(State.picks_count.to_string(), color="white",
                                    font_size="0.9em", font_weight="700"),
                            rx.text("/" + State.total_fixtures.to_string(),
                                    color="#555", font_size="0.9em"),
                            spacing="0",
                        ),
                        padding="4px 10px",
                        background_color="#1a1a1a",
                        border="1px solid #2a2a2a",
                        border_radius="20px",
                    ),
                    width="100%",
                    align="center",
                    padding_bottom="8px",
                ),

                # Match pick cards
                rx.foreach(
                    State.predictions_with_picks.to(list[dict[str, Any]]),
                    pick_card,
                ),

                # Agreement summary (shown once picks start)
                rx.cond(
                    State.picks_count > 0,
                    rx.box(
                        rx.hstack(
                            rx.icon("handshake", size=14, color="#888"),
                            rx.text(
                                "You agree with model on ",
                                rx.text.span(State.picks_agree_count.to_string(),
                                             color="white", font_weight="700"),
                                rx.text.span(" / " + State.picks_count.to_string()),
                                " picks so far",
                                color="#555",
                                font_size="0.72em",
                            ),
                            spacing="2",
                            align="center",
                        ),
                        padding="8px 12px",
                        background_color="#141414",
                        border_radius="6px",
                        border="1px solid #1e1e1e",
                        width="100%",
                    ),
                    rx.box(),
                ),

                # Lock In button
                rx.box(
                    rx.hstack(
                        rx.icon("lock", size=15, color=rx.cond(State.all_picked, "white", "#444")),
                        rx.text(
                            "Lock In Picks & Save",
                            color=rx.cond(State.all_picked, "white", "#444"),
                            font_size="0.88em", font_weight="600",
                        ),
                        spacing="2",
                        align="center",
                        justify="center",
                    ),
                    on_click=State.lock_in_picks,
                    width="100%",
                    padding_y="12px",
                    background_color=rx.cond(State.all_picked, "#FF4B4B", "#1a1a1a"),
                    border="1px solid",
                    border_color=rx.cond(State.all_picked, "#FF4B4B", "#2a2a2a"),
                    border_radius="8px",
                    text_align="center",
                    cursor=rx.cond(State.all_picked, "pointer", "default"),
                    transition="all 0.2s ease",
                ),

                width="100%",
                spacing="2",
            ),
        ),
        width="100%",
        spacing="2",
    )


# ── Insights tab ──────────────────────────────────────────────────────────────

def insight_card(title: str, body: rx.Component) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(title, color="#888", font_size="0.68em",
                    letter_spacing="0.1em", font_weight="700"),
            body,
            spacing="2",
            align="start",
            width="100%",
        ),
        width="100%",
        padding="12px 14px",
        background_color="#141414",
        border_radius="8px",
        border="1px solid #1e1e1e",
    )


def safe_pick_row(p: dict) -> rx.Component:
    return rx.hstack(
        badge_img(p["badge_home"], "20px"),
        rx.text(p["home_team"], color="#d0d0d0", font_size="0.8em", flex="1",
                text_align="right", no_of_lines=1, style={"min_width": "0"}),
        rx.text("v", color="#333", font_size="0.7em"),
        rx.text(p["away_team"], color="#d0d0d0", font_size="0.8em", flex="1",
                text_align="left", no_of_lines=1, style={"min_width": "0"}),
        badge_img(p["badge_away"], "20px"),
        rx.spacer(),
        rx.text(p["pred_name"], color="#aaa", font_size="0.72em", font_weight="600"),
        rx.text(p["pred_disp_prob"], color="#4CAF50", font_size="0.8em", font_weight="600"),
        width="100%",
        align="center",
        spacing="2",
    )


def coin_flip_row(p: dict) -> rx.Component:
    return rx.hstack(
        badge_img(p["badge_home"], "20px"),
        rx.text(p["home_team"], color="#d0d0d0", font_size="0.8em", flex="1"),
        rx.text("v", color="#333", font_size="0.7em"),
        rx.text(p["away_team"], color="#d0d0d0", font_size="0.8em", flex="1"),
        badge_img(p["badge_away"], "20px"),
        width="100%",
        align="center",
        spacing="2",
    )


def _prob_bar(label: str, value, color: str) -> rx.Component:
    return rx.hstack(
        rx.text(label, color="#888", font_size="0.6em", width="18px", flex_shrink="0",
                text_align="right"),
        rx.box(
            rx.text(value.to_string() + "%", color="white", font_size="0.55em",
                    padding_x="4px", white_space="nowrap"),
            background_color=color,
            border_radius="3px",
            height="16px",
            min_width="28px",
            width=value.to_string() + "%",
            display="flex",
            align_items="center",
        ),
        spacing="2",
        width="100%",
        align="center",
    )


def chart_bar_row(item: dict) -> rx.Component:
    return rx.vstack(
        rx.text(item["label"], color="#ccc", font_size="0.7em", font_weight="600",
                letter_spacing="0.04em"),
        rx.vstack(
            _prob_bar("H", item["home"], "#4CAF50"),
            _prob_bar("D", item["draw"], "#FFC107"),
            _prob_bar("A", item["away"], "#F44336"),
            spacing="1",
            width="100%",
        ),
        spacing="1",
        width="100%",
        padding_bottom="6px",
    )


def elo_bar_row(item: dict) -> rx.Component:
    return rx.hstack(
        rx.text(item["label"], color="#555", font_size="0.65em",
                width="70px", flex_shrink="0"),
        rx.box(
            rx.text(item["elo_diff"], color="white", font_size="0.6em", padding_x="4px"),
            background_color=rx.cond(item["elo_positive"], "#4CAF50", "#F44336"),
            border_radius="3px",
            height="14px",
            min_width="28px",
            display="flex",
            align_items="center",
        ),
        width="100%",
        align="center",
        spacing="2",
    )


def _odds_pair(label: str, model_val, book_val) -> rx.Component:
    """Single H/D/A column showing model vs book odds stacked."""
    return rx.vstack(
        rx.text(label, color="#666", font_size="0.6em", font_weight="600"),
        rx.text(model_val, color="#ccc", font_size="0.68em"),
        rx.text(book_val, color="#6fb6ff", font_size="0.68em"),
        spacing="0",
        align="center",
        width="60px",
    )


def bookmaker_odds_row(item: dict) -> rx.Component:
    return rx.hstack(
        rx.text(item["label"], color="#bbb", font_size="0.72em", width="72px", flex_shrink="0"),
        rx.text(item["model_pick"], color="#FF9800", font_size="0.68em", width="20px"),
        _odds_pair("H", item["m_home"], item["b_home"]),
        _odds_pair("D", item["m_draw"], item["b_draw"]),
        _odds_pair("A", item["m_away"], item["b_away"]),
        width="100%",
        align="center",
        spacing="2",
    )


def prob_divergence_row(item: dict) -> rx.Component:
    return rx.hstack(
        rx.text(item["label"], color="#bbb", font_size="0.72em", width="72px", flex_shrink="0"),
        rx.text(item["outcome"], color="#888", font_size="0.68em", width="38px"),
        rx.text(item["edge_disp"],
                color=rx.cond(item["edge_pp"].to(float) >= 0, "#4CAF50", "#F44336"),
                font_size="0.70em", font_weight="600", width="60px"),
        rx.text("M ", rx.text.span(item["model_prob_disp"]), color="#888", font_size="0.66em", width="62px"),
        rx.text("B ", rx.text.span(item["book_prob_disp"]), color="#6fb6ff", font_size="0.66em"),
        width="100%",
        align="center",
        spacing="2",
    )


def value_edge_row(item: dict) -> rx.Component:
    return rx.hstack(
        rx.text(item["label"], color="#bbb", font_size="0.72em", width="72px", flex_shrink="0"),
        rx.text(item["outcome"], color="#FF9800", font_size="0.68em", width="20px"),
        rx.text(item["edge_disp"],
                color=rx.cond(item["edge_positive"], "#4CAF50", "#F44336"),
                font_size="0.70em", font_weight="600", width="60px"),
        rx.text("M ", rx.text.span(item["model_fair_odds"]), color="#888", font_size="0.66em", width="56px"),
        rx.text("B ", rx.text.span(item["book_odds"]), color="#6fb6ff", font_size="0.66em"),
        width="100%",
        align="center",
        spacing="2",
    )


def insights_tab() -> rx.Component:
    return rx.vstack(
        rx.cond(
            State.predictions.length() == 0,
            rx.box(
                rx.text("Run predictions on the Home tab first.", color="#444", font_size="0.85em"),
                text_align="center", padding_y="48px", width="100%",
            ),
            rx.vstack(
                insight_card(
                    "SAFE PICKS  >65%",
                    rx.cond(
                        State.safe_picks.length() > 0,
                        rx.vstack(rx.foreach(State.safe_picks.to(list[dict[str, Any]]), safe_pick_row),
                                  spacing="2", width="100%"),
                        rx.text("No match clears 65% this week.", color="#555", font_size="0.8em"),
                    ),
                ),
                insight_card(
                    "COIN FLIPS  (no clear favourite  <50%)",
                    rx.cond(
                        State.coin_flips.length() > 0,
                        rx.vstack(rx.foreach(State.coin_flips.to(list[dict[str, Any]]), coin_flip_row),
                                  spacing="2", width="100%"),
                        rx.text("No true coin-flip matches this week.", color="#555", font_size="0.8em"),
                    ),
                ),
                insight_card(
                    "WIN PROBABILITIES  ( H / D / A )",
                    rx.vstack(
                        rx.foreach(State.win_prob_chart.to(list[dict[str, Any]]), chart_bar_row),
                        spacing="3", width="100%",
                    ),
                ),
                insight_card(
                    "BOOKMAKER VS MODEL ODDS  ( H / D / A )",
                    rx.cond(
                        State.bookmaker_rows.length() > 0,
                        rx.vstack(
                            rx.hstack(
                                rx.text("", width="72px"),
                                rx.text("", width="20px"),
                                rx.text("Model", color="#888", font_size="0.58em", width="60px", text_align="center"),
                                rx.text("", width="0px"),
                                rx.text("", width="60px"),
                                rx.text("Book", color="#6fb6ff", font_size="0.58em", width="60px", text_align="center"),
                                width="100%", spacing="2",
                            ),
                            rx.foreach(State.bookmaker_rows.to(list[dict[str, Any]]), bookmaker_odds_row),
                            spacing="2",
                            width="100%",
                        ),
                        rx.text(
                            "No bookmaker odds found in cached predictions for this gameweek.",
                            color="#555",
                            font_size="0.8em",
                        ),
                    ),
                ),
                insight_card(
                    "PROBABILITY DIVERGENCE  (model - book, per outcome)",
                    rx.cond(
                        State.prob_divergence_rows.length() > 0,
                        rx.vstack(
                            rx.foreach(State.prob_divergence_rows.to(list[dict[str, Any]]), prob_divergence_row),
                            spacing="2",
                            width="100%",
                        ),
                        rx.text(
                            "No probability divergence available without bookmaker probabilities.",
                            color="#555",
                            font_size="0.8em",
                        ),
                    ),
                ),
                insight_card(
                    "FAIR-ODDS EDGE  (book / model fair odds - 1, per outcome)",
                    rx.cond(
                        State.value_edge_rows.length() > 0,
                        rx.vstack(
                            rx.foreach(State.value_edge_rows.to(list[dict[str, Any]]), value_edge_row),
                            spacing="2",
                            width="100%",
                        ),
                        rx.text(
                            "No fair-odds edge available without bookmaker odds.",
                            color="#555",
                            font_size="0.8em",
                        ),
                    ),
                ),
                insight_card(
                    "ELO ADVANTAGE  (home positive = home favoured)",
                    rx.vstack(
                        rx.foreach(State.elo_chart.to(list[dict[str, Any]]), elo_bar_row),
                        spacing="3", width="100%",
                    ),
                ),
                width="100%",
                spacing="3",
            ),
        ),
        width="100%",
        spacing="2",
    )


# ── History tab ───────────────────────────────────────────────────────────────

def pick_chip(label: str, color: str, is_correct: bool, has_actual: bool) -> rx.Component:
    """Small chip showing a pick with correct/incorrect indicator."""
    return rx.hstack(
        rx.text(label, font_size="0.72em", font_weight="700",
                color=rx.cond(has_actual,
                              rx.cond(is_correct, "#0E1117", "white"),
                              "white")),
        rx.cond(
            has_actual,
            rx.cond(
                is_correct,
                rx.icon("check", size=10, color="#0E1117"),
                rx.icon("x", size=10, color="white"),
            ),
            rx.box(),
        ),
        spacing="1",
        align="center",
        padding="3px 8px",
        background_color=rx.cond(has_actual,
                                  rx.cond(is_correct, color, "#3a1a1a"),
                                  "#1e1e1e"),
        border="1px solid",
        border_color=rx.cond(has_actual,
                              rx.cond(is_correct, color, "#F44336"),
                              color),
        border_radius="20px",
    )


def actual_chip(actual: str, home_goals: str, away_goals: str) -> rx.Component:
    """Read-only result chip; shows H/D/A + score, or 'Pending' when no result yet."""
    has_actual = actual != ""
    score = rx.cond(
        (home_goals != "") & (away_goals != ""),
        rx.hstack(
            rx.text(home_goals, color="#bbb", font_size="0.72em", font_weight="700"),
            rx.text("-", color="#666", font_size="0.72em", font_weight="700"),
            rx.text(away_goals, color="#bbb", font_size="0.72em", font_weight="700"),
            spacing="1",
            align="center",
        ),
        rx.box(),
    )
    return rx.cond(
        has_actual,
        rx.hstack(
            rx.text(actual, color="white", font_size="0.72em", font_weight="700"),
            score,
            spacing="2",
            align="center",
            padding="4px 10px",
            background_color="#1a1a1a",
            border="1px solid #2a2a2a",
            border_radius="6px",
        ),
        rx.text("Pending", color="#555", font_size="0.7em",
                padding="4px 10px",
                background_color="#1a1a1a",
                border="1px dashed #2a2a2a",
                border_radius="6px"),
    )


def history_match_detail_row(m: dict) -> rx.Component:
    """Detailed row in expanded GW view: user pick, model pick, actual (read-only)."""
    has_actual = m["actual"] != ""
    user_correct  = m["user_pick"]  == m["actual"]
    model_correct = m["model_pick"] == m["actual"]

    return rx.box(
        rx.vstack(
            # Teams
            rx.hstack(
                badge_img(m["badge_home"], "18px"),
                rx.text(m["home_team"], color="#bbb", font_size="0.75em",
                        flex="1", text_align="right", no_of_lines=1),
                rx.text("v", color="#333", font_size="0.65em"),
                rx.text(m["away_team"], color="#bbb", font_size="0.75em",
                        flex="1", no_of_lines=1),
                badge_img(m["badge_away"], "18px"),
                width="100%",
                align="center",
                spacing="2",
            ),
            # Picks + actual
            rx.hstack(
                # User pick chip
                rx.vstack(
                    rx.text("YOU", color="#555", font_size="0.55em", letter_spacing="0.1em"),
                    pick_chip(m["user_pick"], "#6C63FF", user_correct & has_actual, has_actual),
                    spacing="1", align="center",
                ),
                rx.text("vs", color="#2a2a2a", font_size="0.65em"),
                # Model pick chip
                rx.vstack(
                    rx.text("MODEL", color="#555", font_size="0.55em", letter_spacing="0.1em"),
                    pick_chip(m["model_pick"], "#FF9800", model_correct & has_actual, has_actual),
                    spacing="1", align="center",
                ),
                rx.spacer(),
                # Actual result (read-only, from results.csv)
                rx.vstack(
                    rx.text("RESULT", color="#555", font_size="0.55em", letter_spacing="0.1em"),
                    actual_chip(m["actual"], m["home_goals"], m["away_goals"]),
                    spacing="1",
                    align="end",
                ),
                width="100%",
                align="end",
                spacing="2",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        padding="10px 12px",
        background_color="#0f0f0f",
        border_radius="6px",
        border="1px solid #1a1a1a",
    )


def history_gw_card(entry: dict) -> rx.Component:
    """Summary card for a saved GW in the history list."""
    has_results = entry["model_accuracy"] != ""
    return rx.box(
        rx.vstack(
            # GW + date row
            rx.hstack(
                rx.text(entry["gw"], color="white", font_size="0.85em", font_weight="700"),
                rx.text(entry["date"], color="#444", font_size="0.72em"),
                rx.spacer(),
                rx.icon("chevron-right", size=14, color="#333"),
                width="100%",
                align="center",
            ),
            # Accuracy chips row
            rx.cond(
                has_results,
                rx.hstack(
                    rx.hstack(
                        rx.box(width="8px", height="8px", border_radius="50%",
                               background_color="#6C63FF"),
                        rx.text("You: ", color="#555", font_size="0.7em"),
                        rx.text(entry["user_accuracy"], color="#6C63FF",
                                font_size="0.72em", font_weight="600"),
                        spacing="1", align="center",
                    ),
                    rx.text("|", color="#222", font_size="0.7em"),
                    rx.hstack(
                        rx.box(width="8px", height="8px", border_radius="50%",
                               background_color="#FF9800"),
                        rx.text("Model: ", color="#555", font_size="0.7em"),
                        rx.text(entry["model_accuracy"], color="#FF9800",
                                font_size="0.72em", font_weight="600"),
                        spacing="1", align="center",
                    ),
                    rx.spacer(),
                    rx.text(entry["pnl"],
                            color=rx.cond(entry["pnl_positive"], "#4CAF50", "#F44336"),
                            font_size="0.7em", font_weight="600"),
                    width="100%",
                    align="center",
                    spacing="2",
                ),
                rx.text("Enter results to see accuracy", color="#333", font_size="0.68em"),
            ),
            spacing="2",
            width="100%",
        ),
        on_click=State.select_history(entry["idx"]),
        width="100%",
        padding="12px 14px",
        background_color="#141414",
        border_radius="8px",
        border="1px solid #1e1e1e",
        cursor="pointer",
    )


def history_detail_view() -> rx.Component:
    """Expanded view for a selected GW."""
    entry = State.selected_gw_entry
    return rx.vstack(
        # Back button + header
        rx.hstack(
            rx.box(
                rx.hstack(
                    rx.icon("chevron-left", size=14, color="#888"),
                    rx.text("Back", color="#888", font_size="0.75em"),
                    spacing="1", align="center",
                ),
                on_click=State.back_to_history_list,
                cursor="pointer",
            ),
            rx.spacer(),
            rx.vstack(
                rx.text(entry["gw"], color="white", font_size="0.9em", font_weight="700"),
                rx.text(entry["date"], color="#444", font_size="0.68em"),
                spacing="0", align="end",
            ),
            width="100%",
            align="center",
            padding_bottom="4px",
        ),
        rx.box(height="1px", width="100%", background_color="#1e1e1e"),

        # Score summary (if any results entered)
        rx.cond(
            entry["model_accuracy"] != "",
            rx.hstack(
                # User score box
                rx.box(
                    rx.vstack(
                        rx.text("YOU", color="#6C63FF", font_size="0.6em",
                                letter_spacing="0.1em", font_weight="700"),
                        rx.text(entry["user_accuracy"], color="white",
                                font_size="0.82em", font_weight="700"),
                        spacing="1", align="center",
                    ),
                    flex="1",
                    padding="10px 8px",
                    background_color="#12101f",
                    border="1px solid #2a2560",
                    border_radius="8px",
                    text_align="center",
                ),
                # vs divider
                rx.text("vs", color="#2a2a2a", font_size="0.75em"),
                # Model score box
                rx.box(
                    rx.vstack(
                        rx.text("MODEL", color="#FF9800", font_size="0.6em",
                                letter_spacing="0.1em", font_weight="700"),
                        rx.text(entry["model_accuracy"], color="white",
                                font_size="0.82em", font_weight="700"),
                        spacing="1", align="center",
                    ),
                    flex="1",
                    padding="10px 8px",
                    background_color="#1a1000",
                    border="1px solid #3a2800",
                    border_radius="8px",
                    text_align="center",
                ),
                width="100%",
                align="center",
                spacing="2",
            ),
            rx.box(
                rx.text("Awaiting results. Update results.csv after the matches play.",
                        color="#333", font_size="0.72em"),
                text_align="center", padding_y="6px", width="100%",
            ),
        ),

        # Match rows
        rx.vstack(
            rx.foreach(
                State.selected_gw_matches.to(list[dict[str, Any]]),
                history_match_detail_row,
            ),
            width="100%",
            spacing="2",
        ),
        width="100%",
        spacing="3",
    )


def history_tab() -> rx.Component:
    return rx.vstack(
        rx.cond(
            State.history_selected >= 0,
            # Detail view for selected GW
            history_detail_view(),
            # List view
            rx.cond(
                State.history.length() == 0,
                rx.box(
                    rx.vstack(
                        rx.icon("clock", size=32, color="#222"),
                        rx.text("No history yet.", color="#444", font_size="0.85em"),
                        rx.text("Run python run_pipeline.py for a gameweek to start populating history.",
                                color="#333", font_size="0.75em", text_align="center"),
                        spacing="2",
                        align="center",
                    ),
                    text_align="center", padding_y="48px", width="100%",
                ),
                rx.vstack(
                    section_label("GAMEWEEK HISTORY"),
                    rx.foreach(State.history.to(list[dict[str, Any]]), history_gw_card),
                    width="100%",
                    spacing="2",
                ),
            ),
        ),
        width="100%",
        spacing="2",
    )


# ── Bankroll tab ──────────────────────────────────────────────────────────────

def bankroll_stat(label: str, value, color: str = "white") -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(label, color="#555", font_size="0.6em", letter_spacing="0.1em", font_weight="700"),
            rx.text(value, color=color, font_size="0.95em", font_weight="700"),
            spacing="1",
            align="center",
        ),
        flex="1",
        padding="10px 8px",
        background_color="#141414",
        border="1px solid #1e1e1e",
        border_radius="8px",
    )


def bankroll_bet_row(bet: dict) -> rx.Component:
    status_color = rx.cond(
        bet["status"] == "win",
        "#4CAF50",
        rx.cond(bet["status"] == "loss", "#F44336", "#888"),
    )
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(bet["gw"], color="#555", font_size="0.65em", font_weight="700"),
                rx.hstack(
                    rx.text(bet["home_team"], color="#ddd", font_size="0.72em", no_of_lines=1),
                    rx.text("vs", color="#444", font_size="0.65em"),
                    rx.text(bet["away_team"], color="#ddd", font_size="0.72em", no_of_lines=1),
                    flex="1",
                    spacing="1",
                    align="center",
                ),
                rx.text(bet["status"], color=status_color, font_size="0.65em", font_weight="700"),
                width="100%",
                align="center",
                spacing="2",
            ),
            rx.hstack(
                rx.text("Bet ", rx.text.span(bet["bet_outcome"], font_weight="700"), color="#aaa", font_size="0.66em"),
                rx.text("Odds ", rx.text.span(bet["odds"].to_string(), font_weight="700"), color="#777", font_size="0.66em"),
                rx.text("Stake ", rx.text.span(bet["stake_disp"], font_weight="700"), color="#777", font_size="0.66em"),
                rx.spacer(),
                rx.text(bet["pnl_disp"], color=status_color, font_size="0.68em", font_weight="700"),
                width="100%",
                align="center",
            ),
            rx.text("Edge ", rx.text.span(bet["edge_disp"]), " · Kelly ", rx.text.span(bet["kelly_disp"]), color="#555", font_size="0.62em"),
            spacing="1",
            width="100%",
        ),
        padding="10px 12px",
        background_color="#101010",
        border="1px solid #1b1b1b",
        border_radius="7px",
        width="100%",
    )


def bankroll_suggestion_row(bet: dict) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(bet["match"], color="#ddd", font_size="0.72em", flex="1", no_of_lines=1),
                rx.text(bet["bet_outcome"], color="#4CAF50", font_size="0.7em", font_weight="700"),
                width="100%",
                align="center",
            ),
            rx.text(
                "Edge ",
                rx.text.span(bet["edge_disp"]),
                " · Stake ",
                rx.text.span(bet["stake_disp"]),
                " · Kelly ",
                rx.text.span(bet["kelly_disp"]),
                color="#666",
                font_size="0.62em",
            ),
            spacing="1",
            width="100%",
        ),
        padding="9px 11px",
        background_color="#101010",
        border="1px solid #1b1b1b",
        border_radius="7px",
        width="100%",
    )


def bankroll_tab() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            bankroll_stat("BANKROLL", State.bankroll_summary["current_bankroll_disp"], "white"),
            bankroll_stat("P/L", State.bankroll_summary["total_pnl_disp"], rx.cond(State.bankroll_summary["pnl_positive"], "#4CAF50", "#F44336")),
            bankroll_stat("ROI", State.bankroll_summary["roi_disp"], "#FFB74D"),
            width="100%",
            spacing="2",
        ),
        rx.hstack(
            bankroll_stat("RECORD", State.bankroll_summary["record"], "#aaa"),
            bankroll_stat("PENDING", State.bankroll_summary["pending_count"].to_string(), "#888"),
            bankroll_stat("RISK CAP", State.bankroll_summary["risk_cap_disp"], "#6C63FF"),
            width="100%",
            spacing="2",
        ),
        rx.box(
            rx.vstack(
                section_label("BANKROLL SETTINGS"),
                rx.hstack(
                    rx.vstack(
                        rx.text("Starting balance used to size stakes.", color="#777", font_size="0.68em"),
                        rx.input(
                            value=State.bankroll_starting_input,
                            on_change=State.update_bankroll_starting,
                            placeholder="Starting bankroll",
                            width="100%",
                            background_color="#101010",
                            color="white",
                            border="1px solid #242424",
                        ),
                        width="50%",
                        spacing="1",
                    ),
                    rx.vstack(
                        rx.text("Max % of bankroll to stake per bet (safety limit).", color="#777", font_size="0.68em"),
                        rx.input(
                            value=State.bankroll_risk_cap_input,
                            on_change=State.update_bankroll_risk_cap,
                            placeholder="Risk cap %",
                            width="100%",
                            background_color="#101010",
                            color="white",
                            border="1px solid #242424",
                        ),
                        width="50%",
                        spacing="1",
                    ),
                    width="100%",
                    spacing="2",
                ),
                rx.button("Save Settings", on_click=State.save_bankroll_settings, width="100%", background_color="#1a1a1a", color="white"),
                rx.text("Saving resets bankroll and clears the ledger.", color="#555", font_size="0.64em", text_align="center", width="100%"),
                spacing="2",
                width="100%",
            ),
            width="100%",
            padding="12px",
            background_color="#141414",
            border="1px solid #1e1e1e",
            border_radius="8px",
        ),
        section_label("CURRENT VALUE BETS"),
        rx.cond(
            State.bankroll_suggestions.length() == 0,
            rx.text("No positive-edge bookmaker bets available for the current cache.", color="#444", font_size="0.75em"),
            rx.vstack(
                rx.foreach(State.bankroll_suggestions.to(list[dict[str, Any]]), bankroll_suggestion_row),
                width="100%",
                spacing="2",
            ),
        ),
        section_label("LEDGER"),
        rx.cond(
            State.bankroll_ledger.length() == 0,
            rx.text("No bankroll bets yet. Run the pipeline with bookmaker odds to populate suggestions.", color="#444", font_size="0.75em"),
            rx.vstack(
                rx.foreach(State.bankroll_ledger.to(list[dict[str, Any]]), bankroll_bet_row),
                width="100%",
                spacing="2",
            ),
        ),
        width="100%",
        spacing="3",
    )


# ── Navbar ────────────────────────────────────────────────────────────────────

NAV_ITEMS = [
    ("home",      "Home",      "home"),
    ("predictor", "Predictor", "crosshair"),
    ("insights",  "Insights",  "bar-chart-2"),
    ("history",   "History",   "clock"),
    ("bankroll",  "Bankroll",  "wallet"),
]


def nav_btn(tab: str, label: str, icon: str) -> rx.Component:
    active = State.current_tab == tab
    return rx.box(
        rx.vstack(
            rx.icon(icon, size=17, color=rx.cond(active, "white", "#444")),
            rx.text(label, font_size="0.6em", letter_spacing="0.03em",
                    font_weight=rx.cond(active, "600", "400"),
                    color=rx.cond(active, "white", "#444")),
            spacing="1",
            align="center",
        ),
        on_click=State.set_current_tab(tab),
        width="20%",
        height="56px",
        display="flex",
        align_items="center",
        justify_content="center",
        background_color=rx.cond(active, "#1a0000", "#0E1117"),
        border_top=rx.cond(active, "2px solid #FF4B4B", "2px solid transparent"),
        cursor="pointer",
    )


def navbar() -> rx.Component:
    return rx.hstack(
        *[nav_btn(tab, label, icon) for tab, label, icon in NAV_ITEMS],
        width="100%",
        spacing="0",
        position="fixed",
        bottom="0",
        left="0",
        right="0",
        z_index="1000",
        background_color="#0E1117",
        border_top="1px solid #1a1a1a",
    )


# ── Page ──────────────────────────────────────────────────────────────────────

def index() -> rx.Component:
    content = rx.cond(
        State.current_tab == "home",
        home_tab(),
        rx.cond(
            State.current_tab == "predictor",
            predictor_tab(),
            rx.cond(
                State.current_tab == "insights",
                insights_tab(),
                rx.cond(
                    State.current_tab == "history",
                    history_tab(),
                    rx.cond(
                        State.current_tab == "bankroll",
                        bankroll_tab(),
                        rx.box(),
                    ),
                ),
            ),
        ),
    )

    return rx.box(
        rx.vstack(
            rx.box(
                rx.heading("Augo", size="6", color="white", font_weight="700"),
                rx.text("XGBoost + ELO · 2025/26", color="#333",
                        font_size="0.7em", letter_spacing="0.06em"),
                padding_x="16px",
                padding_top="18px",
                padding_bottom="10px",
            ),
            rx.box(height="1px", width="100%", background_color="#1a1a1a"),
            rx.box(
                content,
                width="100%",
                padding_x="16px",
                padding_bottom="80px",
                padding_top="12px",
            ),
            spacing="0",
            width="100%",
            min_height="100vh",
            background_color="#0E1117",
        ),
        navbar(),
        width="100%",
        background_color="#0E1117",
    )


app = rx.App(
    style={
        "background_color": "#0E1117",
        "color": "white",
        "font_family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    }
)
app.add_page(index, route="/", on_load=State.load_predictions)
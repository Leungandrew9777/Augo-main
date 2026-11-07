#!/usr/bin/env python3
"""data_quality.py — hard data-quality gate for the Augo pipeline.

Sections
--------
``--data``       fixtures.csv / results.csv: required columns, parseable dates,
                 duplicates, round (gameweek) consistency, result↔fixture GW
                 mapping.
``--history``    premier_league_historical_clean.csv: season coverage, valid
                 results, duplicate matches.
``--artifacts``  ELO CSV / team_state.json / model: freshness, team coverage,
                 and an end-to-end prediction smoke test.

With no flags all sections run. Exits non-zero on errors so the scheduler and
CI abort loudly instead of degrading silently. Warnings are printed but do not
fail the run (e.g. off-season with no upcoming fixtures).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import dates
from rounds import cluster_dated
from team_aliases import elo_lookup_key, fixture_lookup_key

APP_DIR = os.path.dirname(os.path.abspath(__file__))

FIXTURES_FILE = os.getenv("AUGO_FIXTURES_FILE") or os.path.join(APP_DIR, "fixtures.csv")
RESULTS_FILE = os.getenv("AUGO_RESULTS_FILE") or os.path.join(APP_DIR, "results.csv")
CLEAN_FILE = os.getenv("AUGO_CLEAN_FILE") or os.path.join(APP_DIR, "premier_league_historical_clean.csv")
ELO_FILE = os.getenv("AUGO_ELO_FILE") or os.path.join(APP_DIR, "premier_league_with_elo_best.csv")
STATE_FILE = os.getenv("AUGO_STATE_FILE") or os.path.join(APP_DIR, "team_state.json")
MODEL_FILE = os.getenv("AUGO_MODEL_FILE") or os.path.join(APP_DIR, "xgboost_premier_league_model.pkl")
MARKET_BLEND_FILE = os.getenv("AUGO_MARKET_BLEND_FILE") or os.path.join(APP_DIR, "market_blend.pkl")
DIXON_COLES_FILE = os.getenv("AUGO_DIXON_COLES_FILE") or os.path.join(APP_DIR, "dixon_coles_model.pkl")
EURO_RESULTS_FILE = os.getenv("AUGO_EURO_RESULTS_FILE") or os.path.join(APP_DIR, "european_results.csv")
EURO_FEATURES_FILE = os.getenv("AUGO_EURO_FEATURES_FILE") or os.path.join(APP_DIR, "european_features.csv")
EXTERNAL_ELO_FILE = os.getenv("AUGO_EXTERNAL_ELO_FILE") or os.path.join(APP_DIR, "external_elo.csv")

HISTORY_SEASONS = [
    "2627", "2526", "2425", "2324", "2223", "2122", "2021",
    "1920", "1819", "1718", "1617", "1516", "1415", "1314",
]
HISTORY_MIN_SEASON_ROWS = 300
MAX_ROUND_FIXTURES = 10
MAX_ROUND_SPAN_DAYS = 4
MAX_GOALS = 30

Issues = tuple[list[str], list[str]]  # (errors, warnings)


def _read_csv(path: str, required: list[str], label: str) -> tuple[pd.DataFrame | None, list[str]]:
    if not os.path.exists(path):
        return None, [f"{label} not found: {path}"]
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return None, [f"{label} unreadable: {exc}"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, [f"{label} missing column(s): {', '.join(missing)}"]
    return df, []


# ── fixtures + results ────────────────────────────────────────────────────────

def validate_fixtures(df: pd.DataFrame, *, today=None) -> Issues:
    errors: list[str] = []
    warnings: list[str] = []
    today = today or dates.today_uk()

    work = df.copy()
    work["_iso"] = [dates.iso_date(v) for v in work["date"]]
    bad_dates = int(work["_iso"].isna().sum())
    if bad_dates:
        errors.append(f"fixtures.csv: {bad_dates} row(s) with unparseable dates")
    work["_gw"] = pd.to_numeric(work["gameweek"], errors="coerce")
    bad_gw = int(work["_gw"].isna().sum())
    if bad_gw:
        errors.append(f"fixtures.csv: {bad_gw} row(s) with non-numeric gameweek")
    work = work.dropna(subset=["_iso", "_gw"]).copy()
    work["_gw"] = work["_gw"].astype(int)

    dupes = work[work.duplicated(subset=["_iso", "home_team", "away_team"], keep=False)]
    if not dupes.empty:
        sample = dupes[["_iso", "home_team", "away_team"]].head(3).to_dict("records")
        errors.append(f"fixtures.csv: {len(dupes)} duplicate fixture(s), e.g. {sample}")

    clusters = cluster_dated(work.to_dict("records"), date_key="_iso")
    seen_gw: dict[int, str] = {}
    gw_order: list[int] = []
    for cluster in clusters:
        cluster_dates = [d for d, _ in cluster]
        rows = [r for _, r in cluster]
        gws = sorted({int(r["_gw"]) for r in rows})
        label = f"{cluster_dates[0]}..{cluster_dates[-1]}"
        if len(gws) != 1:
            errors.append(f"fixtures.csv: round {label} mixes gameweeks {gws}")
            continue
        gw = gws[0]
        gw_order.append(gw)
        if gw in seen_gw:
            errors.append(f"fixtures.csv: gameweek {gw} spans multiple date blocks "
                          f"({seen_gw[gw]} and {label})")
        seen_gw.setdefault(gw, label)

        if len(rows) > MAX_ROUND_FIXTURES:
            errors.append(f"fixtures.csv: GW{gw} has {len(rows)} fixtures "
                          f"(a PL round has at most {MAX_ROUND_FIXTURES})")
        span = (cluster_dates[-1] - cluster_dates[0]).days
        if span > MAX_ROUND_SPAN_DAYS:
            errors.append(f"fixtures.csv: GW{gw} spans {span} days ({label}) — "
                          "rounds merged?")
        teams = [
            fixture_lookup_key(str(r[name]))
            for r in rows for name in ("home_team", "away_team")
        ]
        doubled = sorted({t for t in teams if teams.count(t) > 1})
        if doubled:
            errors.append(f"fixtures.csv: GW{gw} has team(s) twice: {doubled} "
                          "(merged rounds?)")

    if gw_order != sorted(gw_order):
        errors.append(f"fixtures.csv: gameweeks are not monotonic with dates: {gw_order}")

    if not any(d >= today for d, _ in (item for cluster in clusters for item in cluster)):
        warnings.append("fixtures.csv: no upcoming fixtures (off-season or stale sync)")
    return errors, warnings


def validate_results(df_results: pd.DataFrame, df_fixtures: pd.DataFrame, *, today=None) -> Issues:
    errors: list[str] = []
    warnings: list[str] = []
    today = today or dates.today_uk()

    res = df_results.copy()
    res["_iso"] = [dates.iso_date(v) for v in res["date"]]
    bad_dates = int(res["_iso"].isna().sum())
    if bad_dates:
        errors.append(f"results.csv: {bad_dates} row(s) with unparseable dates")
    res["_gw"] = pd.to_numeric(res["gameweek"], errors="coerce")
    bad_gw = int(res["_gw"].isna().sum())
    if bad_gw:
        errors.append(f"results.csv: {bad_gw} row(s) with non-numeric gameweek")
    for col in ("home_goals", "away_goals"):
        res[col] = pd.to_numeric(res[col], errors="coerce")
        bad = int(res[col].isna().sum())
        if bad:
            errors.append(f"results.csv: {bad} row(s) with non-numeric {col}")
        out_of_range = res[(res[col] < 0) | (res[col] > MAX_GOALS)]
        if not out_of_range.empty:
            errors.append(f"results.csv: {len(out_of_range)} row(s) with implausible {col}")
    res = res.dropna(subset=["_iso", "_gw"])

    dupes = res[res.duplicated(subset=["_iso", "home_team", "away_team"], keep=False)]
    if not dupes.empty:
        sample = dupes[["_iso", "home_team", "away_team"]].head(3).to_dict("records")
        errors.append(f"results.csv: {len(dupes)} duplicate result(s), e.g. {sample}")

    fx = df_fixtures.copy()
    fx["_iso"] = [dates.iso_date(v) for v in fx["date"]]
    fx["_gw"] = pd.to_numeric(fx["gameweek"], errors="coerce")
    fx = fx.dropna(subset=["_iso", "_gw"]).copy()
    fx["_gw"] = fx["_gw"].astype(int)
    fx["_h"] = fx["home_team"].map(lambda v: fixture_lookup_key(str(v)))
    fx["_a"] = fx["away_team"].map(lambda v: fixture_lookup_key(str(v)))
    by_date = {
        (r["_iso"], r["_h"], r["_a"]): int(r["_gw"]) for _, r in fx.iterrows()
    }
    by_gw: dict[tuple[int, str, str], list[str]] = {}
    for _, r in fx.iterrows():
        by_gw.setdefault((int(r["_gw"]), r["_h"], r["_a"]), []).append(r["_iso"])
    max_fixture_gw = int(fx["_gw"].max()) if not fx.empty else 0

    unmatched: list[str] = []
    for _, r in res.iterrows():
        h = fixture_lookup_key(str(r["home_team"]))
        a = fixture_lookup_key(str(r["away_team"]))
        gw_result = int(r["_gw"])
        gw_fixture = by_date.get((r["_iso"], h, a))
        if gw_fixture is None:
            candidates = by_gw.get((gw_result, h, a), [])
            played = pd.to_datetime(r["_iso"], errors="coerce")
            close = []
            for candidate in candidates:
                stored = pd.to_datetime(candidate, errors="coerce")
                if pd.notna(played) and pd.notna(stored) and abs((played - stored).days) <= 14:
                    close.append(candidate)
            if not close:
                unmatched.append(f"{r['home_team']} v {r['away_team']} ({r['date']})")
                continue
            continue  # matched by GW + pairing with an acceptable date drift
        if gw_fixture != gw_result:
            errors.append(
                f"results.csv: {r['home_team']} v {r['away_team']} ({r['date']}) is "
                f"GW{gw_result} here but GW{gw_fixture} in fixtures.csv"
            )

    if unmatched:
        warnings.append(
            f"results.csv: {len(unmatched)} result(s) not found in fixtures.csv "
            f"(older matches are expected): {', '.join(unmatched[:3])}"
        )

    future_results = res[res["_iso"] > str(today)]
    if not future_results.empty:
        warnings.append(f"results.csv: {len(future_results)} result(s) dated after today")

    return errors, warnings


# ── historical clean data ─────────────────────────────────────────────────────

def validate_clean_history(
    df: pd.DataFrame,
    seasons: list[str] | None = None,
    *,
    min_season_rows: int = HISTORY_MIN_SEASON_ROWS,
) -> Issues:
    errors: list[str] = []
    warnings: list[str] = []
    seasons = seasons or HISTORY_SEASONS

    work = df.copy()
    work["_date"] = dates.parse_series(work["Date"])
    bad_dates = int(work["_date"].isna().sum())
    if bad_dates:
        errors.append(f"clean history: {bad_dates} row(s) with unparseable dates")
    work = work.dropna(subset=["_date"])

    bad_ftr = work[~work["FTR"].astype(str).isin(["H", "D", "A"])]
    if not bad_ftr.empty:
        errors.append(f"clean history: {len(bad_ftr)} row(s) with invalid FTR")

    dupes = work[work.duplicated(subset=["_date", "HomeTeam", "AwayTeam"], keep=False)]
    if not dupes.empty:
        sample = dupes[["_date", "HomeTeam", "AwayTeam"]].head(3).to_dict("records")
        errors.append(f"clean history: {len(dupes)} duplicate match(es), e.g. {sample}")

    work["_season"] = work["Season"].astype(str)
    counts = work["_season"].value_counts().to_dict()
    for index, season in enumerate(seasons):
        count = int(counts.get(str(season), 0))
        if index == 0:
            if count == 0:
                warnings.append(f"clean history: current season {season} has no matches yet")
        elif count < min_season_rows:
            errors.append(
                f"clean history: season {season} has only {count} matches "
                f"(expected >= {min_season_rows}); date parsing or download failed"
            )
    extra = sorted(set(counts) - {str(s) for s in seasons})
    if extra:
        warnings.append(f"clean history: unexpected season codes {extra}")
    return errors, warnings


# ── artifacts + end-to-end smoke test ────────────────────────────────────────

def _state_last_match_date(state: dict):
    raw = state.get("last_match_date")
    return dates.iso_date(raw) if raw else None


def validate_artifacts(*, today=None) -> Issues:
    errors: list[str] = []
    warnings: list[str] = []
    today = today or dates.today_uk()

    fixtures, fx_errors = _read_csv(FIXTURES_FILE, ["gameweek", "date", "home_team", "away_team"], "fixtures.csv")
    errors.extend(fx_errors)
    elo, elo_errors = _read_csv(ELO_FILE, ["Date", "HomeTeam", "AwayTeam"], "ELO file")
    errors.extend(elo_errors)

    state = None
    if not os.path.exists(STATE_FILE):
        errors.append(f"team_state.json missing: {STATE_FILE} — run feature_engineering.py")
    else:
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            errors.append(f"team_state.json unreadable: {exc}")
    if isinstance(state, dict):
        ratings = state.get("ratings")
        teams = state.get("teams")
        if not isinstance(ratings, dict) or not ratings:
            errors.append("team_state.json: missing/empty ratings")
        if not isinstance(teams, dict) or not teams:
            errors.append("team_state.json: missing/empty team features")
        if not isinstance(state.get("window"), (int, float)):
            errors.append("team_state.json: window missing")
        if not isinstance(state.get("home_advantage"), (int, float)):
            errors.append("team_state.json: home_advantage missing")

    if elo is not None and isinstance(state, dict):
        elo_max = dates.parse_series(elo["Date"]).max()
        state_last = _state_last_match_date(state)
        if pd.notna(elo_max) and state_last and state_last != str(elo_max.date()):
            errors.append(
                f"team_state.json is out of sync with the ELO file "
                f"(state last match {state_last}, ELO last match {elo_max.date()}) — "
                "run feature_engineering.py"
            )
        results, _ = _read_csv(RESULTS_FILE, ["date", "gameweek", "home_team", "away_team"], "results.csv")
        if results is not None:
            res_dates = dates.parse_series(results["date"]).dropna()
            if not res_dates.empty and state_last and state_last < str(res_dates.max().date()):
                # football-data.co.uk can lag a day or two behind results.csv, so
                # this is a warning rather than a hard failure.
                warnings.append(
                    f"team_state.json is older than the latest result "
                    f"({state_last} vs {res_dates.max().date()}) — "
                    "the next feature_engineering.py run will refresh it"
                )

    if fixtures is not None and isinstance(state, dict):
        ratings = state.get("ratings") or {}
        fixture_teams = set(
            fixture_lookup_key(str(v))
            for col in ("home_team", "away_team")
            for v in fixtures[col].dropna()
        )
        missing = sorted(t for t in fixture_teams if elo_lookup_key(t) not in ratings and t not in ratings)
        if missing:
            errors.append(f"team_state.json: no rating for team(s) {missing}")

    for path, label, hint in [
        (MARKET_BLEND_FILE, "market blend", "run train_ensemble.py"),
        (DIXON_COLES_FILE, "Dixon-Coles model", "run train_ensemble.py"),
    ]:
        if not os.path.exists(path):
            errors.append(f"{label} missing: {path} — {hint}")

    if os.path.exists(MODEL_FILE) and os.path.exists(ELO_FILE) and isinstance(state, dict) and fixtures is not None:
        try:
            info = _smoke_predict(fixtures, today)
            if info:
                print(f"   {info}")
        except Exception as exc:
            errors.append(f"prediction smoke test failed: {exc}")
    elif not os.path.exists(MODEL_FILE):
        errors.append(f"model missing: {MODEL_FILE}")

    return errors, warnings


def validate_european_data(*, today=None) -> Issues:
    """Validate the P2 European results/features + optional external Elo file."""
    errors: list[str] = []
    warnings: list[str] = []
    today = today or dates.today_uk()

    results, res_errors = _read_csv(
        EURO_RESULTS_FILE,
        ["date", "competition", "season", "home_team", "away_team", "home_goals", "away_goals"],
        "european results",
    )
    if results is None:
        warnings.extend(res_errors + ["run european_data.py to build it"])
    else:
        work = results.copy()
        work["_date"] = dates.parse_series(work["date"])
        if work["_date"].isna().any():
            errors.append(f"european results: {int(work['_date'].isna().sum())} unparseable dates")
        work = work.dropna(subset=["_date"])
        dupes = work[work.duplicated(subset=["_date", "home_team", "away_team"], keep=False)]
        if not dupes.empty:
            errors.append(f"european results: {len(dupes)} duplicate match(es)")
        for col in ("home_goals", "away_goals"):
            goals = pd.to_numeric(work[col], errors="coerce")
            if goals.isna().any() or ((goals < 0) | (goals > 15)).any():
                errors.append(f"european results: invalid {col}")
        if not work["competition"].astype(str).str.contains("EPL").any():
            errors.append("european results: EPL competition missing")
        if (work["_date"].max().date() if not work.empty else today) < today - pd.Timedelta(days=200):
            warnings.append("european results look stale — rerun european_data.py")

    features, feat_errors = _read_csv(
        EURO_FEATURES_FILE,
        ["date", "home_team", "away_team", "euroelo_diff"],
        "european features",
    )
    if features is None:
        warnings.extend(feat_errors + ["run european_data.py to build it"])
    else:
        if len(features) < 1000:
            warnings.append(f"european features: only {len(features)} rows")
        coverage = float(features["euroelo_diff"].notna().mean()) if len(features) else 0.0
        if coverage < 0.9:
            warnings.append(f"european features: euroelo_diff coverage {coverage:.1%}")
        for col in ("euroelo_home", "euroelo_away"):
            values = pd.to_numeric(features[col], errors="coerce").dropna()
            if not values.empty and ((values < 500) | (values > 3000)).any():
                errors.append(f"european features: implausible {col}")
        diffs = pd.to_numeric(features["euroelo_diff"], errors="coerce").dropna()
        if not diffs.empty and float(diffs.abs().max()) > 1500:
            errors.append("european features: implausible euroelo_diff")

    if os.path.exists(EXTERNAL_ELO_FILE):
        external, ext_errors = _read_csv(EXTERNAL_ELO_FILE, ["team", "elo"], "external Elo")
        if external is None:
            errors.extend(ext_errors)
        else:
            elo = pd.to_numeric(external["elo"], errors="coerce")
            if elo.isna().any() or ((elo < 1000) | (elo > 2600)).any():
                errors.append("external Elo: invalid rating(s)")
            if "date" in external.columns:
                bad_dates = external["date"].map(dates.iso_date).isna().sum()
                if bad_dates:
                    errors.append(f"external Elo: {int(bad_dates)} invalid date(s)")

    return errors, warnings


def _smoke_predict(fixtures: pd.DataFrame, today) -> str:
    from features import (  # local imports keep the gate startup light
        MODEL_FEATURE_COLS,
        compute_current_elo,
        detect_model_features,
        ensure_model_features,
        load_model_and_elo,
    )

    model, df_elo = load_model_and_elo(MODEL_FILE, ELO_FILE)
    fx = fixtures.copy()
    fx["_date"] = dates.parse_series(fx["date"])
    fx = fx.dropna(subset=["_date", "home_team", "away_team"])
    upcoming_rows = fx[fx["_date"] >= pd.Timestamp(today)]
    if upcoming_rows.empty:
        return "prediction smoke test skipped (no upcoming fixtures)"

    gw = int(pd.to_numeric(upcoming_rows["gameweek"], errors="coerce").dropna().iloc[0])
    selected = fx[pd.to_numeric(fx["gameweek"], errors="coerce") == gw].sort_values("_date")
    upcoming = pd.DataFrame({
        "date": selected["_date"].dt.strftime("%Y-%m-%d"),
        "home_team": selected["home_team"].astype(str),
        "away_team": selected["away_team"].astype(str),
    }).reset_index(drop=True)
    upcoming = compute_current_elo(upcoming, df_elo)
    feature_cols = detect_model_features(model) or MODEL_FEATURE_COLS
    upcoming = ensure_model_features(upcoming, df_elo, feature_cols)

    frame = upcoming[feature_cols]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"non-numeric model feature(s): {non_numeric}")
    if frame.isna().any().any():
        cols = [c for c in feature_cols if frame[c].isna().any()]
        raise ValueError(f"missing model feature value(s): {cols}")
    probs = np.asarray(model.predict_proba(frame), dtype=float)
    if probs.shape != (len(frame), 3):
        raise ValueError(f"unexpected prediction shape {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("non-finite probabilities")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probabilities do not sum to 1")

    from inference import load_dixon_coles, load_market_blend, predict_matches

    blend = load_market_blend()
    if blend is not None and load_dixon_coles() is not None:
        odds_frame = upcoming.copy()
        odds_frame["B365H"] = 2.0
        odds_frame["B365D"] = 3.5
        odds_frame["B365A"] = 4.0
        out = predict_matches(odds_frame, model, df_elo)
        if not bool(out["used_market"].all()):
            raise ValueError("market blend was not applied despite odds being present")
        blended = out[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
        if not np.allclose(blended.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("blended probabilities do not sum to 1")

    return (
        f"prediction smoke test OK (GW{gw}, {len(frame)} fixtures, "
        f"market blend={'yes' if blend is not None else 'no'})"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _report(name: str, errors: list[str], warnings: list[str]) -> None:
    status = "FAILED" if errors else "OK"
    print(f"[{status}] {name}")
    for message in warnings:
        print(f"   warning: {message}")
    for message in errors:
        print(f"   error:   {message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Augo data-quality gate")
    parser.add_argument("--data", action="store_true", help="validate fixtures.csv + results.csv")
    parser.add_argument("--history", action="store_true", help="validate the cleaned match history")
    parser.add_argument("--artifacts", action="store_true", help="validate ELO/team-state/model + smoke test")
    parser.add_argument("--europe", action="store_true", help="validate European results/features/external Elo")
    args = parser.parse_args(argv)

    run_all = not (args.data or args.history or args.artifacts or args.europe)
    failed = False

    if run_all or args.data:
        fixtures, errors = _read_csv(FIXTURES_FILE, ["gameweek", "date", "home_team", "away_team"], "fixtures.csv")
        warnings: list[str] = []
        if fixtures is not None:
            e, w = validate_fixtures(fixtures)
            errors, warnings = errors + e, warnings + w
        results, res_errors = _read_csv(RESULTS_FILE, ["date", "gameweek", "home_team", "away_team", "home_goals", "away_goals"], "results.csv")
        errors.extend(res_errors)
        if fixtures is not None and results is not None:
            e, w = validate_results(results, fixtures)
            errors, warnings = errors + e, warnings + w
        _report("fixtures/results", errors, warnings)
        failed |= bool(errors)

    if run_all or args.history:
        clean, errors = _read_csv(CLEAN_FILE, ["Date", "HomeTeam", "AwayTeam", "FTR", "Season"], "clean history")
        warnings = []
        if clean is not None:
            e, w = validate_clean_history(clean)
            errors, warnings = errors + e, warnings + w
        _report("historical data", errors, warnings)
        failed |= bool(errors)

    if run_all or args.artifacts:
        errors, warnings = validate_artifacts()
        _report("artifacts", errors, warnings)
        failed |= bool(errors)

    if run_all or args.europe:
        errors, warnings = validate_european_data()
        _report("european data", errors, warnings)
        failed |= bool(errors)

    print("data quality gate: FAILED" if failed else "data quality gate: passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

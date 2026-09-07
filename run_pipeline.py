#!/usr/bin/env python3
"""
run_pipeline.py  —  Augo prediction pipeline
Run ONCE before each gameweek. Output: predictions_cache.json
Usage:
    python run_pipeline.py              # auto-detects next gameweek
    python run_pipeline.py --gw 33      # force a specific matchweek
"""
import argparse, json, math, os, re, sys
from datetime import datetime
import joblib, pandas as pd
import requests
from dotenv import load_dotenv
from team_aliases import badge_lookup_key
from features import (
    MODEL_FEATURE_COLS,
    compute_current_elo,
    detect_model_features,
    ensure_model_features,
    normalize_elo_columns,
    patch_model_runtime_compat,
)

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths so the pipeline works regardless of CWD.
CACHE_FILE     = os.path.join(APP_DIR, "predictions_cache.json")
FIXTURES_FILE  = os.path.join(APP_DIR, "fixtures.csv")
ELO_FILE       = os.path.join(APP_DIR, "premier_league_with_elo_best.csv")
MODEL_FILE     = os.path.join(APP_DIR, "xgboost_premier_league_model.pkl")
GOAL_HOME_FILE = os.path.join(APP_DIR, "goal_model_home.pkl")
GOAL_AWAY_FILE = os.path.join(APP_DIR, "goal_model_away.pkl")
HISTORY_DIR    = os.path.join(APP_DIR, "predictions_history")


TEAM_BADGES: dict[str, str] = {
    "Arsenal":                 "https://resources.premierleague.com/premierleague/badges/t3.png",
    "Aston Villa":             "https://resources.premierleague.com/premierleague/badges/t7.png",
    "Bournemouth":             "https://resources.premierleague.com/premierleague/badges/t91.png",
    "Brentford":               "https://resources.premierleague.com/premierleague/badges/t94.png",
    "Brighton & Hove Albion":  "https://resources.premierleague.com/premierleague/badges/t36.png",
    "Burnley":                 "https://resources.premierleague.com/premierleague/badges/t90.png",
    "Chelsea":                 "https://resources.premierleague.com/premierleague/badges/t8.png",
    "Crystal Palace":          "https://resources.premierleague.com/premierleague/badges/t31.png",
    "Everton":                 "https://resources.premierleague.com/premierleague/badges/t11.png",
    "Fulham":                  "https://resources.premierleague.com/premierleague/badges/t54.png",
    "Leeds United":            "https://resources.premierleague.com/premierleague/badges/t2.png",
    "Liverpool":               "https://resources.premierleague.com/premierleague/badges/t14.png",
    "Manchester City":         "https://resources.premierleague.com/premierleague/badges/t43.png",
    "Manchester United":       "https://resources.premierleague.com/premierleague/badges/t1.png",
    "Newcastle":               "https://resources.premierleague.com/premierleague/badges/t4.png",
    "Nottingham Forest":       "https://resources.premierleague.com/premierleague/badges/t17.png",
    "Sunderland":              "https://resources.premierleague.com/premierleague/badges/t56.png",
    "Tottenham Hotspur":       "https://resources.premierleague.com/premierleague/badges/t6.png",
    "West Ham United":         "https://resources.premierleague.com/premierleague/badges/t21.png",
    "Wolverhampton Wanderers": "https://resources.premierleague.com/premierleague/badges/t39.png",
}
FALLBACK_BADGE = "https://resources.premierleague.com/premierleague/badges/t0.png"


def run_model(upcoming: pd.DataFrame, model, df_elo: pd.DataFrame) -> pd.DataFrame:
    feature_cols_for_model = detect_model_features(model) or MODEL_FEATURE_COLS
    upcoming = ensure_model_features(upcoming, df_elo, feature_cols_for_model)
    probs = model.predict_proba(upcoming[feature_cols_for_model])
    # Model classes are encoded as 0=Away, 1=Draw, 2=Home.
    upcoming["prob_away"] = probs[:, 0]
    upcoming["prob_draw"] = probs[:, 1]
    upcoming["prob_home"] = probs[:, 2]
    upcoming["fair_odds_home"] = 1 / upcoming["prob_home"]
    upcoming["fair_odds_draw"] = 1 / upcoming["prob_draw"]
    upcoming["fair_odds_away"] = 1 / upcoming["prob_away"]
    upcoming["disp_odds_home"] = upcoming["fair_odds_home"].map(lambda v: f"{v:.3g}")
    upcoming["disp_odds_draw"] = upcoming["fair_odds_draw"].map(lambda v: f"{v:.3g}")
    upcoming["disp_odds_away"] = upcoming["fair_odds_away"].map(lambda v: f"{v:.3g}")
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
        lambda r: r["home_team"][:3].upper() + " v " + r["away_team"][:3].upper(), axis=1
    )
    upcoming["model_pick"] = upcoming.apply(
        lambda r: max(
            [("H", r["prob_home"]), ("D", r["prob_draw"]), ("A", r["prob_away"])],
            key=lambda x: x[1],
        )[0], axis=1,
    )

    # Optional bookmaker odds (from fixtures.csv if provided).
    has_book_cols = all(c in upcoming.columns for c in ("B365H", "B365D", "B365A"))
    if has_book_cols:
        book_h = pd.to_numeric(upcoming["B365H"], errors="coerce")
        book_d = pd.to_numeric(upcoming["B365D"], errors="coerce")
        book_a = pd.to_numeric(upcoming["B365A"], errors="coerce")
        valid = (book_h > 0) & (book_d > 0) & (book_a > 0)

        upcoming["book_odds_home"] = book_h.where(valid, pd.NA)
        upcoming["book_odds_draw"] = book_d.where(valid, pd.NA)
        upcoming["book_odds_away"] = book_a.where(valid, pd.NA)

        inv_h = (1.0 / book_h).where(valid, pd.NA)
        inv_d = (1.0 / book_d).where(valid, pd.NA)
        inv_a = (1.0 / book_a).where(valid, pd.NA)
        total = (inv_h + inv_d + inv_a).where(valid, pd.NA)

        upcoming["book_prob_home"] = (inv_h / total).where(valid, pd.NA)
        upcoming["book_prob_draw"] = (inv_d / total).where(valid, pd.NA)
        upcoming["book_prob_away"] = (inv_a / total).where(valid, pd.NA)

        upcoming["disp_book_odds_home"] = upcoming["book_odds_home"].map(lambda v: f"{v:.3g}" if pd.notna(v) else "")
        upcoming["disp_book_odds_draw"] = upcoming["book_odds_draw"].map(lambda v: f"{v:.3g}" if pd.notna(v) else "")
        upcoming["disp_book_odds_away"] = upcoming["book_odds_away"].map(lambda v: f"{v:.3g}" if pd.notna(v) else "")
        upcoming["disp_book_prob_home"] = upcoming["book_prob_home"].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "")
        upcoming["disp_book_prob_draw"] = upcoming["book_prob_draw"].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "")
        upcoming["disp_book_prob_away"] = upcoming["book_prob_away"].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "")
    else:
        for col in (
            "book_odds_home", "book_odds_draw", "book_odds_away",
            "book_prob_home", "book_prob_draw", "book_prob_away",
        ):
            upcoming[col] = pd.NA
        for col in (
            "disp_book_odds_home", "disp_book_odds_draw", "disp_book_odds_away",
            "disp_book_prob_home", "disp_book_prob_draw", "disp_book_prob_away",
        ):
            upcoming[col] = ""
    return upcoming


def _goal_model_features(goal_model) -> list[str]:
    if hasattr(goal_model, "feature_names_in_"):
        return [str(c) for c in list(getattr(goal_model, "feature_names_in_", []))]
    if hasattr(goal_model, "named_steps"):
        for step in goal_model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return [str(c) for c in list(getattr(step, "feature_names_in_", []))]
    return []


def poisson_markets(lambda_home: float, lambda_away: float, *, max_goals: int = 6, top_k: int = 5, rho: float = -0.12) -> dict:
    """Poisson scoreline markets with a Dixon-Coles low-score correction.

    ``rho`` (< 0) boosts the likelihood of 0-0 / 1-0 / 0-1 / 1-1 scorelines,
    which independent Poisson under-estimates (and which are the draw-heavy
    results). See Dixon & Coles (1997).
    """
    lambda_home = max(float(lambda_home), 0.05)
    lambda_away = max(float(lambda_away), 0.05)

    home_probs = [
        math.exp(-lambda_home) * (lambda_home ** goals) / math.factorial(goals)
        for goals in range(max_goals + 1)
    ]
    away_probs = [
        math.exp(-lambda_away) * (lambda_away ** goals) / math.factorial(goals)
        for goals in range(max_goals + 1)
    ]

    def tau(h: int, a: int) -> float:
        if h == 0 and a == 0:
            return max(1.0 - lambda_home * lambda_away * rho, 0.01)
        if h == 0 and a == 1:
            return max(1.0 + lambda_home * rho, 0.01)
        if h == 1 and a == 0:
            return max(1.0 + lambda_away * rho, 0.01)
        if h == 1 and a == 1:
            return max(1.0 - rho, 0.01)
        return 1.0

    score_probs: list[dict] = []
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    p_over_25 = 0.0
    p_btts = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = home_probs[h] * away_probs[a] * tau(h, a)
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
            if h + a > 2.5:
                p_over_25 += p
            if h > 0 and a > 0:
                p_btts += p
            score_probs.append({"score": f"{h}-{a}", "p": p})

    # Normalize H/D/A to absorb the tiny tail beyond max_goals.
    total = p_home + p_draw + p_away
    if total > 0:
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    top_scores = sorted(score_probs, key=lambda x: x["p"], reverse=True)[:top_k]
    return {
        "poisson_prob_home": p_home,
        "poisson_prob_draw": p_draw,
        "poisson_prob_away": p_away,
        "poisson_over_25": p_over_25,
        "poisson_btts": p_btts,
        "poisson_correct_scores": [
            {
                "score": s["score"],
                "p": round(float(s["p"]), 6),
                "disp_p": f"{float(s['p']) * 100:.1f}%",
            }
            for s in top_scores
        ],
    }


def _disp_poisson_vs_ensemble_row(r: pd.Series) -> str:
    """Poisson minus ensemble outcome probabilities, in percentage points."""
    try:
        e_h, e_d, e_a = float(r["prob_home"]), float(r["prob_draw"]), float(r["prob_away"])
        p_h, p_d, p_a = (
            float(r["poisson_prob_home"]),
            float(r["poisson_prob_draw"]),
            float(r["poisson_prob_away"]),
        )
    except (TypeError, ValueError, KeyError):
        return "—"
    d_h = (p_h - e_h) * 100.0
    d_d = (p_d - e_d) * 100.0
    d_a = (p_a - e_a) * 100.0
    return f"Poisson−ens. (pp): H {d_h:+.0f} · D {d_d:+.0f} · A {d_a:+.0f}"


def add_poisson_outputs(upcoming: pd.DataFrame, home_goal_model, away_goal_model, df_elo: pd.DataFrame) -> pd.DataFrame:
    home_features = _goal_model_features(home_goal_model)
    away_features = _goal_model_features(away_goal_model)
    if not home_features or not away_features:
        return upcoming

    all_features = sorted(set(home_features) | set(away_features))
    upcoming = ensure_model_features(upcoming, df_elo, all_features)
    lambda_home = home_goal_model.predict(upcoming[home_features])
    lambda_away = away_goal_model.predict(upcoming[away_features])

    upcoming["lambda_home"] = pd.Series(lambda_home).clip(lower=0.05)
    upcoming["lambda_away"] = pd.Series(lambda_away).clip(lower=0.05)
    markets = upcoming.apply(
        lambda r: poisson_markets(float(r["lambda_home"]), float(r["lambda_away"])),
        axis=1,
    )
    market_df = pd.DataFrame(list(markets))
    for col in market_df.columns:
        upcoming[col] = market_df[col]

    upcoming["disp_lambda_home"] = upcoming["lambda_home"].map(lambda v: f"{v:.2f}")
    upcoming["disp_lambda_away"] = upcoming["lambda_away"].map(lambda v: f"{v:.2f}")
    upcoming["disp_poisson_prob_home"] = upcoming["poisson_prob_home"].map(lambda v: f"{v*100:.1f}%")
    upcoming["disp_poisson_prob_draw"] = upcoming["poisson_prob_draw"].map(lambda v: f"{v*100:.1f}%")
    upcoming["disp_poisson_prob_away"] = upcoming["poisson_prob_away"].map(lambda v: f"{v*100:.1f}%")
    tot_lambda = upcoming["lambda_home"] + upcoming["lambda_away"]
    upcoming["disp_poisson_xg_total"] = tot_lambda.map(lambda v: f"{float(v):.2f}")
    upcoming["disp_poisson_o25"] = upcoming["poisson_over_25"].map(lambda v: f"{float(v) * 100:.1f}%")
    upcoming["disp_poisson_btts"] = upcoming["poisson_btts"].map(lambda v: f"{float(v) * 100:.1f}%")
    upcoming["disp_poisson_vs_ensemble"] = upcoming.apply(_disp_poisson_vs_ensemble_row, axis=1)
    return upcoming


ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"


def fetch_odds_from_api(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch h2h odds from The Odds API (single call) and inject B365H/D/A columns.

    If the API key is missing or the call fails, the dataframe is returned
    unchanged (no B365 columns) so the downstream code falls back gracefully.
    """
    from team_aliases import fixture_lookup_key

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        print("⚠️  ODDS_API_KEY not set — skipping live bookmaker odds.")
        return fixtures_df

    params = {
        "apiKey": api_key,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }

    # Narrow the time window to ±2 days around the fixture dates
    if "date" in fixtures_df.columns and not fixtures_df.empty:
        earliest = fixtures_df["date"].min() - pd.Timedelta(days=1)
        latest = fixtures_df["date"].max() + pd.Timedelta(days=2)
        params["commenceTimeFrom"] = earliest.strftime("%Y-%m-%dT00:00:00Z")
        params["commenceTimeTo"] = latest.strftime("%Y-%m-%dT23:59:59Z")

    try:
        print("Fetching bookmaker odds from The Odds API …")
        resp = requests.get(ODDS_API_URL, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        print(f"   API quota: {used} used, {remaining} remaining")
        resp.raise_for_status()
        events = resp.json()
    except Exception as exc:
        print(f"⚠️  Odds API request failed: {exc}")
        return fixtures_df

    if not events:
        print("   No upcoming EPL events returned by API.")
        return fixtures_df

    # Build lookup: (normalised_home, normalised_away) -> (avg_H, avg_D, avg_A)
    odds_lookup: dict[tuple[str, str], tuple[float, float, float]] = {}
    for ev in events:
        home = fixture_lookup_key(ev.get("home_team", ""))
        away = fixture_lookup_key(ev.get("away_team", ""))
        h_prices, d_prices, a_prices = [], [], []

        for bm in ev.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                price_map: dict[str, float] = {}
                for outcome in mkt.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    if price is None:
                        continue
                    if name == "Draw":
                        price_map["draw"] = float(price)
                    elif fixture_lookup_key(name) == home:
                        price_map["home"] = float(price)
                    elif fixture_lookup_key(name) == away:
                        price_map["away"] = float(price)
                if "home" in price_map:
                    h_prices.append(price_map["home"])
                if "draw" in price_map:
                    d_prices.append(price_map["draw"])
                if "away" in price_map:
                    a_prices.append(price_map["away"])

        if h_prices and d_prices and a_prices:
            odds_lookup[(home, away)] = (
                sum(h_prices) / len(h_prices),
                sum(d_prices) / len(d_prices),
                sum(a_prices) / len(a_prices),
            )

    # Match fixture rows to API events and populate B365H/D/A
    b365h, b365d, b365a = [], [], []
    matched = 0
    for _, row in fixtures_df.iterrows():
        key = (str(row["home_team"]).strip(), str(row["away_team"]).strip())
        if key in odds_lookup:
            h, d, a = odds_lookup[key]
            b365h.append(h)
            b365d.append(d)
            b365a.append(a)
            matched += 1
        else:
            b365h.append(None)
            b365d.append(None)
            b365a.append(None)

    fixtures_df = fixtures_df.copy()
    fixtures_df["B365H"] = b365h
    fixtures_df["B365D"] = b365d
    fixtures_df["B365A"] = b365a
    print(f"   Matched odds for {matched}/{len(fixtures_df)} fixtures.")
    return fixtures_df


def launch_ui():
    """Start the Reflex dev server in the background after a successful run."""
    import shutil
    import subprocess

    exe = shutil.which("reflex")
    if not exe:
        print("⚠️  reflex not found on PATH — run `reflex run` manually.")
        return
    print("   Launching Reflex UI (reflex run) …")
    subprocess.Popen([exe, "run"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, default=None)
    parser.add_argument("--launch", action="store_true",
                        help="auto-launch the Reflex UI after a successful run")
    args = parser.parse_args()

    for path, hint in [
        (MODEL_FILE,    "Run FeatureEng.py first."),
        (GOAL_HOME_FILE, "Run train_ensemble.py first."),
        (GOAL_AWAY_FILE, "Run train_ensemble.py first."),
        (ELO_FILE,      "Run ELO.py first."),
        (FIXTURES_FILE, "fixtures.csv must have: matchweek, date, home_team, away_team"),
    ]:
        if not os.path.exists(path):
            sys.exit(f"❌  {path} not found — {hint}")

    print("Loading model and ELO data …")
    model  = joblib.load(MODEL_FILE)
    goal_model_home = joblib.load(GOAL_HOME_FILE)
    goal_model_away = joblib.load(GOAL_AWAY_FILE)
    patch_model_runtime_compat(model)
    df_elo = pd.read_csv(ELO_FILE)
    df_elo = normalize_elo_columns(df_elo)
    df_elo["date"] = pd.to_datetime(df_elo["date"])

    df_fix = pd.read_csv(FIXTURES_FILE)
    df_fix["date"] = pd.to_datetime(df_fix["date"], dayfirst=True, errors="coerce").dt.normalize()
    df_fix = df_fix.dropna(subset=["date", "home_team", "away_team"])

    try:
        from zoneinfo import ZoneInfo

        today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Hong_Kong")).date())
    except Exception:
        today = pd.Timestamp.today().normalize()
    future = df_fix[df_fix["date"] >= today].sort_values("date")

    # Detect gameweek column (your CSV uses "matchweek")
    gw_col = next((c for c in ("matchweek", "gameweek") if c in df_fix.columns), None)

    if args.gw is not None:
        if gw_col is None:
            sys.exit("❌  --gw specified but no matchweek/gameweek column found.")
        selected = df_fix[df_fix[gw_col] == args.gw].sort_values("date")
        if selected.empty:
            sys.exit(f"❌  No fixtures found for matchweek {args.gw}.")
        gw_label = f"GW{args.gw}"
    elif gw_col and df_fix[gw_col].notna().any():
        if future.empty:
            # Season finished (or fixtures.csv only contains past). Default to last known GW.
            next_gw = int(df_fix[gw_col].dropna().astype(int).max())
        else:
            next_gw = int(future.iloc[0][gw_col])

        # Optional prompt: allow choosing ANY gameweek (past/current/future).
        # (Keeps --gw for scripted/non-interactive usage.)
        chosen_gw = next_gw
        if sys.stdin.isatty():
            try:
                available = (
                    df_fix[gw_col]
                    .dropna()
                    .astype(int)
                    .sort_values()
                    .unique()
                    .tolist()
                )
            except Exception:
                available = []

            if available:
                print(f"Auto-selected gameweek: GW{int(next_gw)}")
                print("Available gameweeks:", ", ".join(f"GW{g}" for g in available[:20]) +
                      (" …" if len(available) > 20 else ""))
                raw = input("Enter any GW number to simulate (or press Enter to keep selected): ").strip()
                if raw:
                    try:
                        gw_int = int(raw)
                        if gw_int not in available:
                            print(f"⚠️  GW{gw_int} not found in fixtures.csv; using GW{int(next_gw)}.")
                        else:
                            chosen_gw = gw_int
                    except ValueError:
                        print(f"⚠️  Invalid input; using GW{int(next_gw)}.")

        # Important: select from ALL fixtures so we show the full gameweek,
        # not just matches on/after today.
        selected = df_fix[df_fix[gw_col] == chosen_gw].sort_values("date")
        gw_label = f"GW{int(chosen_gw)}"
    else:
        start    = future["date"].min()
        selected = future[future["date"] <= start + pd.Timedelta(days=3)]
        gw_label = f"Next ({start.strftime('%b %d')})"

    print(f"Gameweek  : {gw_label}  ({len(selected)} fixtures)")

    known = set(pd.concat([df_elo["home_team"], df_elo["away_team"]]).unique())
    for _, row in selected.iterrows():
        for team in (row["home_team"], row["away_team"]):
            if team not in known:
                print(f"⚠️   Unknown team '{team}' — ELO defaults to 1500")

    # Fetch live bookmaker odds (single API call) and inject B365H/D/A columns
    selected = fetch_odds_from_api(selected)

    selected_cols = ["date", "home_team", "away_team"]
    for col in ("B365H", "B365D", "B365A"):
        if col in selected.columns:
            selected_cols.append(col)
    upcoming = selected[selected_cols].copy().reset_index(drop=True)
    upcoming = compute_current_elo(upcoming, df_elo)
    # Optional: adjust ELO for injuries/suspensions before scoring (see lineups.py)
    from lineups import apply_availability_if_enabled
    upcoming = apply_availability_if_enabled(upcoming)
    upcoming = run_model(upcoming, model, df_elo)
    upcoming = add_poisson_outputs(upcoming, goal_model_home, goal_model_away, df_elo)

    records: list[dict] = []
    for i, row in upcoming.iterrows():
        rec = {}
        for k, v in row.items():
            if isinstance(v, (list, dict)):
                rec[k] = v
            elif hasattr(v, "isoformat"):
                rec[k] = str(v.date())
            elif pd.isna(v):
                rec[k] = None
            elif hasattr(v, "item"):
                rec[k] = v.item()
            else:
                rec[k] = v
        rec["match_idx"] = int(i)
        records.append(rec)

    cache = {
        "generated_at": datetime.now().isoformat(),
        "gameweek":     gw_label,
        "predictions":  records,
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    gw_match = re.search(r"\d+", str(gw_label))
    if gw_match:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        archive_path = os.path.join(HISTORY_DIR, f"GW{gw_match.group(0)}.json")
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        print(f"   Archived snapshot → {archive_path}")

    print(f"\n{'─'*72}")
    print(f"{'HOME':<26}  {'AWAY':<26}  H%    D%    A%   Pick")
    print(f"{'─'*72}")
    for r in records:
        print(f"{r['home_team']:<26}  {r['away_team']:<26}  "
              f"{r['prob_home']*100:4.1f}  {r['prob_draw']*100:4.1f}  "
              f"{r['prob_away']*100:4.1f}  [{r['model_pick']}]")
    print(f"{'─'*72}")
    print(f"\n✅  Predictions cached → {CACHE_FILE}")
    print(f"   Now launch:  reflex run")

    if args.launch:
        launch_ui()


if __name__ == "__main__":
    main()
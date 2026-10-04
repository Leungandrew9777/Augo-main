#!/usr/bin/env python3
"""
run_pipeline.py  —  Augo prediction pipeline
Run ONCE before each gameweek. Output: predictions_cache.json
Usage:
    python run_pipeline.py              # auto-detects next gameweek
    python run_pipeline.py --gw 33      # force a specific matchweek

Model/feature/market logic lives in ``inference.py`` so the CLI and the app
share one serving path.
"""
import argparse, json, os, sys
from datetime import datetime
import joblib, pandas as pd
import requests
from dotenv import load_dotenv
from persistence import archive_predictions
from features import (
    compute_current_elo,
    normalize_elo_columns,
    patch_model_runtime_compat,
)
from inference import (
    add_corner_outputs,
    add_ht_outputs,
    add_poisson_outputs,
    predict_matches,
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
HT_HOME_FILE   = os.path.join(APP_DIR, "goal_model_ht_home.pkl")
HT_AWAY_FILE   = os.path.join(APP_DIR, "goal_model_ht_away.pkl")
CORNER_HOME_FILE = os.path.join(APP_DIR, "corner_model_home.pkl")
CORNER_AWAY_FILE = os.path.join(APP_DIR, "corner_model_away.pkl")


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
        (MODEL_FILE,    "Run train_ensemble.py first."),
        (GOAL_HOME_FILE, "Run train_ensemble.py first."),
        (GOAL_AWAY_FILE, "Run train_ensemble.py first."),
        (ELO_FILE,      "Run feature_engineering.py first."),
        (FIXTURES_FILE, "fixtures.csv must have: matchweek, date, home_team, away_team"),
    ]:
        if not os.path.exists(path):
            sys.exit(f"❌  {path} not found — {hint}")

    print("Loading model and ELO data …")
    model  = joblib.load(MODEL_FILE)
    goal_model_home = joblib.load(GOAL_HOME_FILE)
    goal_model_away = joblib.load(GOAL_AWAY_FILE)
    ht_model_home = joblib.load(HT_HOME_FILE) if os.path.exists(HT_HOME_FILE) else None
    ht_model_away = joblib.load(HT_AWAY_FILE) if os.path.exists(HT_AWAY_FILE) else None
    corner_model_home = joblib.load(CORNER_HOME_FILE) if os.path.exists(CORNER_HOME_FILE) else None
    corner_model_away = joblib.load(CORNER_AWAY_FILE) if os.path.exists(CORNER_AWAY_FILE) else None
    patch_model_runtime_compat(model)
    df_elo = pd.read_csv(ELO_FILE)
    df_elo = normalize_elo_columns(df_elo)
    df_elo["date"] = pd.to_datetime(df_elo["date"])

    df_fix = pd.read_csv(FIXTURES_FILE)
    df_fix["date"] = pd.to_datetime(df_fix["date"], dayfirst=True, errors="coerce").dt.normalize()
    df_fix = df_fix.dropna(subset=["date", "home_team", "away_team"])

    # Fixture dates are UK calendar dates → compare against Europe/London
    # date. Asia/Hong_Kong is 7-8h ahead and flips GW a day early (e.g.
    # 2026-09-14 17:40 UTC is still Sep 14 in London but already Sep 15 in HK).
    try:
        from zoneinfo import ZoneInfo

        today = pd.Timestamp(datetime.now(ZoneInfo("Europe/London")).date())
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
        # (Keeps --gw for scripted/non-interactive usage.) Guarded against EOF so
        # scheduled/headless runs (Task Scheduler) never crash on input().
        chosen_gw = next_gw
        if sys.stdin.isatty() and not os.environ.get("AUGO_NON_INTERACTIVE"):
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
                try:
                    raw = input("Enter any GW number to simulate (or press Enter to keep selected): ").strip()
                except (EOFError, KeyboardInterrupt):
                    raw = ""   # no interactive input -> keep auto-selected GW
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
    upcoming = predict_matches(upcoming, model, df_elo)
    upcoming = add_poisson_outputs(upcoming, goal_model_home, goal_model_away, df_elo)
    if ht_model_home is not None and ht_model_away is not None:
        upcoming = add_ht_outputs(upcoming, ht_model_home, ht_model_away, df_elo)
    if corner_model_home is not None and corner_model_away is not None:
        upcoming = add_corner_outputs(upcoming, corner_model_home, corner_model_away, df_elo)

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

    archived = archive_predictions(cache, gw_label)
    if archived is not None:
        archive_path, created = archived
        # Existing archives are kept: they may hold pre-match bookmaker odds
        # (unavailable once matches start) and drive grading/wallets.
        if created:
            print(f"   Archived snapshot → {archive_path}")
        else:
            print(f"   Archive {archive_path} already exists — kept (cache only updated).")

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

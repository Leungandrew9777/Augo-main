# Augo

Augo is a Premier League match-outcome predictor: a soft-voting ensemble
(Logistic Regression + Random Forest + XGBoost) trained on Football-Data
historical results and ELO ratings, served through a Reflex web UI that lets
you make picks against the model and grades them automatically as real results
land.

## The streamlined rundown

```
ONE-TIME (or after retraining):
  python data_pipeline.py        -> premier_league_historical_clean.csv
  python feature_engineering.py  -> premier_league_with_elo_best.csv
  python train_ensemble.py       -> xgboost_premier_league_model.pkl

WEEKLY (each matchday):
  # 1. update fixtures.csv for the upcoming GW (matchweek, date, home_team, away_team)
  python run_pipeline.py [--gw N]
  #   -> predictions_cache.json
  #   -> predictions_history/GW{N}.json   (snapshot, archived)

  # 2. launch the web UI
  reflex run
```

## File map

### Code

| File | Role |
| --- | --- |
| `data_pipeline.py` | Step 1: download + clean Football-Data CSVs into `premier_league_historical_clean.csv`. |
| `feature_engineering.py` | Step 2: rolling stats, xG proxies, ELO, head-to-head -> `premier_league_with_elo_best.csv`. |
| `train_ensemble.py` | Step 3: fit the soft-voting ensemble -> `xgboost_premier_league_model.pkl`. |
| `run_pipeline.py` | Weekly: load model + ELO, score upcoming fixtures, fetch live odds, write predictions cache + history snapshot. |
| `app.py` | Reflex UI (Predictor, Insights, Custom predictor, History). |
| `team_aliases.py` | Maps team-name variants between fixtures, ELO history, results, and the odds API. |
| `persistence.py` | Disk-backed history layer: archived predictions, `results.csv`, `user_picks.json`. |
| `rxconfig.py`, `Augo/` | Reflex plumbing. |

### Data

| File | Role |
| --- | --- |
| `fixtures.csv` | Source of truth for upcoming gameweek fixtures. Edit weekly. |
| `results.csv` | Ground-truth scores (`gameweek, home_team, away_team, home_goals, away_goals`). Drives auto-grading on the History tab. |
| `premier_league_historical_clean.csv` | Output of `data_pipeline.py`. |
| `premier_league_with_elo_best.csv` | Output of `feature_engineering.py`. |
| `xgboost_premier_league_model.pkl` | Output of `train_ensemble.py`. |
| `predictions_cache.json` | Latest predictions written by `run_pipeline.py`; consumed by `app.py`. |
| `predictions_history/GW{N}.json` | Per-gameweek snapshot. Power the History tab. |
| `user_picks.json` | Persisted user picks per gameweek (`{gw: {match_idx: "H"/"D"/"A"}}`). |

### Config / docs

`requirements.txt`, `.env`, `.gitignore`, `README.md`.

## Required env vars

Place these in `.env` (loaded automatically by `python-dotenv`):

```
ODDS_API_KEY=your_the_odds_api_key
```

`ODDS_API_KEY` is optional; without it `run_pipeline.py` skips the live
bookmaker-odds fetch and the cache will not include `book_*` fields.

## How history persistence works

The History tab is rebuilt from disk every time `app.py` boots, so picks and
past gameweeks survive restarts:

1. `run_pipeline.py` writes `predictions_cache.json` **and** archives a copy at
   `predictions_history/GW{N}.json` for every run.
2. When you make picks and click **Lock in picks**, `app.py` writes them to
   `user_picks.json` keyed by gameweek + match index.
3. `persistence.load_results()` reads `results.csv`, normalises team names via
   `team_aliases.fixture_lookup_key`, and produces a `(gw, home, away) ->
   actual` map. `app.py._rebuild_history()` joins these three sources and
   computes per-GW user accuracy, model accuracy, and PnL automatically.
4. Gameweeks with no results yet show as **Pending** until rows for that GW
   appear in `results.csv`.

To bring an old gameweek "back to life", drop a copy of its predictions into
`predictions_history/GW{N}.json` and add the score rows to `results.csv` —
the History tab will pick it up on next load.

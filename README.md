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

WEEKLY (each matchday — automated by scheduled.ps1 / GitHub Actions):
  # 1. refresh fixtures.csv + results.csv
  python sync.py all
  # 2. re-download history, then rebuild rolling form, ELO, H2H features
  python data_pipeline.py
  python feature_engineering.py
  # 3. score the upcoming GW
  python run_pipeline.py [--gw N]
  #   -> predictions_cache.json
  #   -> predictions_history/GW{N}.json   (snapshot, archived)

  # 4. launch the web UI
  reflex run

MONTHLY (manual retrain — deliberately NOT part of the automation):
  python data_pipeline.py         # re-download all seasons (includes finished GWs)
  python feature_engineering.py   # rebuild rolling form + ELO features
  python train_ensemble.py        # retrain ensemble + goal/HT/corner models
  python run_pipeline.py [--gw N] # re-score the upcoming GW with the new models
```

## Retraining (manual, roughly monthly)

`run_pipeline.py` only *scores* fixtures with the current model. Retraining is
deliberately kept out of the weekly automation — run it yourself when you
decide (roughly once a month), ideally once a gameweek's results are final:

```bash
# 1. (optional) refresh fixtures.csv / results.csv first
python sync.py all

# 2. re-download + clean all seasons (now includes the finished GW)
python data_pipeline.py
#    -> premier_league_historical_clean.csv (with Understat xG)

# 3. rebuild rolling form, strength-of-schedule ELO, H2H features
python feature_engineering.py
#    -> premier_league_with_elo_best.csv

# 4. retrain the stacked ensemble + goal/HT/corner models
python train_ensemble.py            # reuses saved best_params.json if present
python train_ensemble.py --tune     # optional: walk-forward random search first
#    -> xgboost_premier_league_model.pkl, goal_model_{home,away}.pkl,
#       goal_model_ht_{home,away}.pkl, corner_model_{home,away}.pkl

# 5. re-score the upcoming GW so the cache/UI uses the new models
python run_pipeline.py [--gw N]
```

Notes:

- `data_pipeline.py` pulls every season listed at the bottom of the file
  (currently 2627 → 1314) from football-data.co.uk. If the current season's CSV
  is not published yet (early-season 503), run
  `python bootstrap_season.py [--season 2627]` to append the finished matches
  from Understat (goals + real xG; missing shots/corners/odds are median-filled)
  before step 3.
- `train_ensemble.py` validates with time-ordered walk-forward CV
  (`TimeSeriesSplit`), so no future matches leak into training, and prints
  accuracy / log-loss / Brier / ECE. `--tune` writes the winning parameters to
  `best_params.json` for subsequent runs.
- Retrain only after the GW's results are final. Existing
  `predictions_history/GW{N}.json` snapshots are never overwritten, so history,
  grading, and bankroll stay intact; re-running `run_pipeline.py` only updates
  `predictions_cache.json`.
- The scheduled weekly job (`scheduled.ps1 -Mode weekly`) and the GitHub
  Actions workflow refresh fixtures/results, rebuild features and re-score the
  next GW, but never retrain — run the steps above manually (roughly monthly)
  when you want the model updated. Publishing to the deployed app still happens
  via `python publish.py --push`.

## File map

### Code

| File | Role |
| --- | --- |
| `data_pipeline.py` | Step 1: download + clean Football-Data CSVs into `premier_league_historical_clean.csv`. |
| `feature_engineering.py` | Step 2: rolling stats, xG proxies, ELO, head-to-head -> `premier_league_with_elo_best.csv`. |
| `train_ensemble.py` | Step 3: fit the soft-voting ensemble -> `xgboost_premier_league_model.pkl`. |
| `bootstrap_season.py` | Fallback for step 1 when football-data.co.uk hasn't published the current season yet: appends finished matches (goals + real xG) from Understat. |
| `run_pipeline.py` | Weekly: load model + ELO, score upcoming fixtures, fetch live odds, write predictions cache + history snapshot. |
| `inference.py` | Shared serving path (ensemble probabilities + Poisson/HT/corner markets) used by both `run_pipeline.py` and `app.py`. |
| `app.py` | Reflex UI (Predictor, Insights, Custom predictor, History). |
| `league.py` | Premier League team list + badge URLs shared by the pipeline and UI. |
| `metrics.py` | Shared Brier / expected-calibration-error metrics for training and evaluation. |
| `team_aliases.py` | Maps team-name variants between fixtures, ELO history, results, and the odds API. |
| `display.py` | Shared odds formatting so `run_pipeline.py` and `app.py` render identical display strings. |
| `bankroll.py` | Virtual-wallet strategy registry (`WALLETS`), ledger builders and portfolio/PnL summaries. |
| `persistence.py` | Disk-backed history layer: archived predictions, `results.csv`, `user_picks.json`. |
| `rxconfig.py`, `.web/` | Reflex plumbing. |

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

Optional:

- `AUGO_LOCAL_DATA=1` — prefer local `results.csv` / `predictions_cache.json`
  over the published remote URLs (the scheduler sets this automatically). By
  default the deployed app is remote-first so it always reads the latest
  `data_repo` push.
- `CACHE_MAX_AGE_HOURS=168` — max cache age before the app falls back to live
  inference (`0` disables the stale check).

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

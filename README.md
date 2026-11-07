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
  python feature_engineering.py  -> premier_league_with_elo_best.csv + team_state.json
  python train_ensemble.py       -> xgboost_premier_league_model.pkl,
                                    market_blend.pkl, dixon_coles_model.pkl

WEEKLY (each matchday — automated by scheduled.ps1 / GitHub Actions):
  # 1. refresh fixtures.csv + results.csv, then check them
  python sync.py all
  python data_quality.py --data
  # 2. re-download history; rebuild rolling form, ELO, H2H features and the
  #    serving snapshot (team_state.json), then check the artifacts
  python data_pipeline.py
  python european_data.py          # top-5 + UEFA results, cross-league Elo (P2)
  python feature_engineering.py
  python data_quality.py --history --artifacts --europe
  # 3. score the upcoming GW
  python run_pipeline.py [--gw N]
  #   -> predictions_cache.json
  #   -> predictions_history/GW{N}.json   (snapshot, archived)

  # 4. launch the web UI
  reflex run

MONTHLY (manual retrain — deliberately NOT part of the automation):
  python data_pipeline.py         # re-download all seasons (includes finished GWs)
  python feature_engineering.py   # rebuild rolling form + ELO features + team_state.json
  python train_ensemble.py        # retrain ensemble + market blend + Dixon-Coles (+ gate)
  python backtest.py --json       # rolling-origin validation vs market/base-rate
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

# 2b. refresh the top-5/UEFA results + cross-league Elo (candidate features)
python european_data.py

# 3. rebuild rolling form, strength-of-schedule ELO, H2H features + serving state
python feature_engineering.py
#    -> premier_league_with_elo_best.csv
#    -> team_state.json   (post-last-match ELO + rolling features used at inference)

# 4. retrain the stacked ensemble + market blend + Dixon-Coles + goal/HT/corner
python train_ensemble.py            # reuses saved best_params.json if present
python train_ensemble.py --tune     # optional: walk-forward random search first
#    -> xgboost_premier_league_model.pkl, market_blend.pkl, dixon_coles_model.pkl,
#       goal_model_{home,away}.pkl, goal_model_ht_{home,away}.pkl,
#       corner_model_{home,away}.pkl
#    The retrain refuses to save if the market-relative OOF gate fails
#    (blend worse than market-only by more than GATE_TOLERANCE); --force overrides.

# 5. full rolling-origin validation (last 5 seasons + current; market/base-rate
#    baselines, nested tuning available with --tune, bootstrap CIs)
python backtest.py --seasons 5 --seeds 42,7,2026 --json

# 6. validate the rebuilt artifacts, then re-score the upcoming GW
python data_quality.py --history --artifacts
python run_pipeline.py [--gw N]
```

Notes:

- `data_pipeline.py` pulls every season in `data_quality.HISTORY_SEASONS`
  (currently 2627 → 1314) from football-data.co.uk and refuses to write the
  cleaned file if any season is missing or truncated. If the current season's CSV
  is not published yet (early-season 503), run
  `python bootstrap_season.py [--season 2627]` to append the finished matches
  from Understat (goals + real xG; missing shots/corners/odds are median-filled)
  before step 3.
- `train_ensemble.py` validates with time-ordered walk-forward CV
  (`TimeSeriesSplit`), so no future matches leak into training, and prints
  accuracy / log-loss / Brier / ECE. `--tune` writes the winning parameters to
  `best_params.json` for subsequent runs.
- `data_quality.py` is a hard gate: it validates fixture/result consistency
  (rounds, duplicates, result↔fixture gameweeks), historical season coverage,
  artifact freshness and runs an end-to-end prediction smoke test. The
  scheduled jobs abort on errors and alert via Telegram.
- `python -m pytest` runs the test suite (`requirements-dev.txt`); CI runs it
  on Python 3.14 with the pinned dependencies before the data-quality gate.
- Serving probabilities are anchored to the market: the base classifier is
  blended out-of-fold with the de-vigged B365 prices and the Dixon-Coles
  scoreline model, then per-class (vector) calibrated. If odds are missing for
  a fixture the base probabilities are used for that row (`used_market` flag in
  the cache).
- `backtest.py` trains each of the last N seasons on earlier seasons only and
  reports base, Dixon-Coles, market-only, base-rate and market-blend log-loss /
  Brier / accuracy / ECE on the same rows, with bootstrap CIs for the
  blend-minus-market difference. `--tune` adds nested hyperparameter search.
- `train_ensemble.py` gates saving on the market-relative out-of-fold log-loss
  (not on CV accuracy). The historical feature CSV is read-only during
  training/backtests; only model artifacts and the new `backtest_report.json`
  are written.
- Retrain only after the GW's results are final. Existing
  `predictions_history/GW{N}.json` snapshots are never overwritten, so history,
  grading, and bankroll stay intact; re-running `run_pipeline.py` only updates
  `predictions_cache.json`.
- The scheduled weekly job (`scheduled.ps1 -Mode weekly`) and the GitHub
  Actions workflow refresh fixtures/results, rebuild features and re-score the
  next GW, but never retrain — run the steps above manually (roughly monthly)
  when you want the model updated. Publishing to the deployed app still happens
  via `python publish.py --push`.

## European data & candidate signals (P2)

`european_data.py` builds a free, reproducible cross-league dataset:

- **Top-5 domestic** (openfootball football.json): EPL, La Liga, Bundesliga,
  Serie A, Ligue 1, 2013/14 -> current.
- **UEFA competitions** (openfootball): UCL 2013/14+, UEL/UECL 2021/22-2024/25
  where published (2026/27 UEFA data is not published yet).
- Outputs: `european_results.csv`, `european_elo_history.csv`,
  `european_elo_current.csv`, and `european_features.csv` (cross-league Elo +
  UEFA rest features for every PL match).

ClubElo's CSV API is currently down/behind auth, so the cross-league Elo is
computed here. You can still supply ClubElo ratings manually (dated history is
best; an undated file is treated as a current snapshot):

```bash
python european_data.py --import-elo path/to/clubelo.csv   # columns: team,elo[,date]
# -> external_elo.csv (validated, canonical team names)
```

Notes:

- `FootballELO.margin_multiplier` now updates ratings after draws (the old
  formula returned 0 for a draw, silently freezing ELO).
- **Feature policy:** the P2 candidates are merged into
  `premier_league_with_elo_best.csv` but are **not** in `MODEL_FEATURE_COLS`.
  The rolling-origin ablation (5 seasons + current, market-anchored blend)
  showed no material improvement: base log-loss 0.9953 (baseline) vs 0.9978
  (+euroelo) and 0.9950 (+UEFA rest); the blend stayed at 0.966 (vs market
  0.963). They remain available for future work but do not enter the model
  until they beat the baseline.
- The raw download cache lives in `data/external/raw/` (gitignored); the
  derived CSVs above are tracked.

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
| `dates.py` | Shared date parsing for the mixed `DD/MM/YYYY` + `DD/MM/YY` Football-Data files. |
| `rounds.py` | Date-based gameweek clustering/renumbering, shared by sync and validation. |
| `market.py` | De-vigged bookmaker probabilities and the out-of-fold market/DC blend. |
| `dixon_coles.py` | Time-weighted Dixon-Coles ratings (goals/xG) -> scoreline-based H/D/A probabilities. |
| `european_data.py` | Top-5 + UEFA results, cross-league Elo, UEFA rest features, external ClubElo import. |
| `backtest.py` | Season rolling-origin backtest with market/base-rate baselines, bootstrap CIs, per-class reliability. |
| `data_quality.py` | Hard data-quality gate: fixtures/results, history coverage, artifact freshness, prediction smoke test. |
| `tests/` | Pytest suite: rounds, mixed-date parsing, data quality, team state, end-to-end serving. |
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
| `premier_league_with_elo_best.csv` | Output of `feature_engineering.py` (historical training features). |
| `team_state.json` | Output of `feature_engineering.py`: post-last-match ELO ratings and per-team/venue rolling features served at inference time. |
| `xgboost_premier_league_model.pkl` | Output of `train_ensemble.py` (market-blind base ensemble). |
| `market_blend.pkl` | Output of `train_ensemble.py`: out-of-fold `[model, Dixon-Coles, market]` stacker + vector calibrator. |
| `dixon_coles_model.pkl` | Output of `train_ensemble.py`: time-weighted attack/defence ratings used by the blend. |
| `backtest_report.json` | Output of `backtest.py --json` (rolling-origin metrics/baselines/CIs). |
| `european_results.csv` | P2 top-5 + UEFA match results (canonical team keys). |
| `european_elo_history.csv` | Per-team cross-league Elo after each match (as-of queries). |
| `european_elo_current.csv` | Latest cross-league Elo per team. |
| `european_features.csv` | Candidate P2 features for PL matches (cross-league Elo + UEFA rest). |
| `external_elo.csv` | Optional canonicalised ClubElo ratings imported by `european_data.py --import-elo`. |
| `predictions_cache.json` | Latest predictions written by `run_pipeline.py`; consumed by `app.py`. |
| `predictions_history/GW{N}.json` | Per-gameweek snapshot. Power the History tab. |
| `user_picks.json` | Persisted user picks per gameweek (`{gw: {match_idx: "H"/"D"/"A"}}`). |

### Config / docs

`requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.env`, `.gitignore`, `.github/workflows/`, `README.md`.

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

import json

import pandas as pd
import pytest

import european_data as ed


def test_canonical_team_maps_both_name_variants():
    assert ed.canonical_team("Aston Villa FC") == "Aston Villa"
    assert ed.canonical_team("Aston Villa") == "Aston Villa"
    assert ed.canonical_team("Manchester United FC") == "Man United"
    assert ed.canonical_team("Nottingham Forest FC") == "Nott'm Forest"
    assert ed.canonical_team("FC Bayern München") == "Bayern München"
    assert ed.canonical_team("Paris Saint-Germain FC") == "Paris Saint-Germain"


def test_parse_domestic_json_handles_dict_and_list_scores():
    payload = json.dumps({"matches": [
        {"date": "2025-08-15", "team1": "Liverpool FC", "team2": "AFC Bournemouth", "score": {"ft": [2, 1]}},
        {"date": "2025-08-16", "team1": "Aston Villa", "team2": "Newcastle United FC", "score": [0, 0]},
        {"date": "2025-08-17", "team1": "Chelsea FC", "team2": "Arsenal FC", "score": None},
    ]})
    rows = ed.parse_domestic_json(payload, "2025-26", "EPL")
    assert len(rows) == 2
    assert rows[0]["home_team"] == "Liverpool"
    assert rows[0]["away_team"] == "Bournemouth"
    assert (rows[0]["home_goals"], rows[0]["away_goals"]) == (2, 1)
    assert rows[1]["home_team"] == "Aston Villa"
    assert (rows[1]["home_goals"], rows[1]["away_goals"]) == (0, 0)


def test_parse_uefa_txt_dates_aet_and_unplayed():
    text = """
= UEFA Champions League 2025/26

# Date       Tue Sep 16 2025 - Sat May 30 2026 (256d)
# Matches    4

▪ League, Matchday 1
  Tue Sep 16 2025
    18:45  Athletic Club (ESP) v Arsenal FC (ENG)  0-2 (0-1)
           PSV (NED) v Royale Union Saint-Gilloise (BEL)  1-3 (0-2)
  Tue Sep 30
    21:00  Arsenal FC (ENG) v PSV (NED)  1-1 a.e.t. (0-0)
  Wed Oct 1
    21:00  Real Madrid CF (ESP) v Club Brugge KV (BEL)  2-4 pen. 1-0 a.e.t. (1-0, 1-0)
  Mon Jan 5 2026
    21:00  Arsenal FC (ENG) v Real Madrid CF (ESP)
"""
    rows = ed.parse_uefa_txt(text, "2025-26", "UCL")
    assert [r["date"] for r in rows] == ["2025-09-16", "2025-09-16", "2025-09-30", "2025-10-01"]
    assert rows[0]["home_team"] == "Athletic Club"
    assert rows[0]["away_team"] == "Arsenal"
    # Extra-time score is used; the penalty shootout is ignored.
    assert (rows[2]["home_goals"], rows[2]["away_goals"]) == (1, 1)
    assert (rows[3]["home_goals"], rows[3]["away_goals"]) == (1, 0)
    assert rows[3]["home_team"] == "Real Madrid"


def _mini_results():
    return pd.DataFrame([
        {"date": "2025-01-01", "competition": "EPL", "season": "2024-25",
         "home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 0},
        {"date": "2025-01-08", "competition": "EPL", "season": "2024-25",
         "home_team": "B", "away_team": "A", "home_goals": 1, "away_goals": 1},
    ])


def test_elo_history_and_asof_is_strictly_before():
    history, current = ed.build_elo(_mini_results())
    lookup = ed._asof_lookup(history)
    assert lookup("A", "2025-01-01")[0] is None  # before the first match
    after_first = lookup("A", "2025-01-02")[0]
    after_second = lookup("A", "2025-01-09")[0]
    assert after_first is not None and after_first > 1500
    assert after_second is not None and after_second != after_first
    assert current.set_index("team").loc["A", "games"] == 2


def test_rest_features_midweek_flags():
    uefa_days = {"A": ["2025-09-16", "2025-09-30"]}
    since, until, mid, nxt = ed._rest_features("A", "2025-09-20", uefa_days)
    assert since == 4
    assert until == 10
    assert mid == 1
    assert nxt == 0


def test_import_external_elo_canonicalises(tmp_path, monkeypatch):
    source = tmp_path / "clubelo.csv"
    source.write_text(
        "team,date,elo\nMan City,2026-11-01,1910\nArsenal,2026-11-01,1950\n",
        encoding="utf-8",
    )
    output = tmp_path / "external_elo.csv"
    monkeypatch.setattr(ed, "EXTERNAL_ELO_FILE", str(output))

    df = ed.import_external_elo(str(source))
    assert set(df["team"]) == {"Man City", "Arsenal"}
    assert df["date"].tolist() == ["2026-11-01", "2026-11-01"]
    assert output.exists()


def test_import_external_elo_rejects_bad_schema(tmp_path, monkeypatch):
    source = tmp_path / "bad.csv"
    source.write_text("club,rating\nArsenal,1900\n", encoding="utf-8")
    monkeypatch.setattr(ed, "EXTERNAL_ELO_FILE", str(tmp_path / "out.csv"))
    with pytest.raises(SystemExit):
        ed.import_external_elo(str(source))

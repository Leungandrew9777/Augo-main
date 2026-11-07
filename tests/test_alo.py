import pandas as pd
import pytest

import alo


def test_alo_key_maps_pl_names():
    assert alo.alo_key("Forest") == "Nott'm Forest"
    assert alo.alo_key("Man City") == "Man City"
    assert alo.alo_key("Bayern München") == "Bayern München"


def _export(path, rows):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8-sig")
    return path


def test_import_alo_parses_commas_and_schema(tmp_path):
    source = _export(tmp_path / "ClubAloRankings.csv", [
        "Club Ranking,Club Name,Club ALO,Country",
        '1,Arsenal,"2,046",England',
        '6,Paris SG,"1,988",France',
        '46,Forest,"1,789",England',
    ])
    output = tmp_path / "alo_rankings.csv"
    df = alo.import_alo(str(source), out=str(output))

    assert output.exists()
    assert set(df["team"]) == {"Arsenal", "Paris SG", "Nott'm Forest"}
    arsenal = df[df["team"] == "Arsenal"].iloc[0]
    assert float(arsenal["alo"]) == 2046
    assert arsenal["country"] == "England"
    assert float(df[df["team"] == "Paris SG"].iloc[0]["alo"]) == 1988


def test_import_alo_drops_invalid_rows(tmp_path):
    source = _export(tmp_path / "bad.csv", [
        "Club Ranking,Club Name,Club ALO,Country",
        "1,Arsenal,2046,England",
        "2,Broken,not-a-number,England",
        "3,TooLow,500,England",
    ])
    df = alo.import_alo(str(source), out=str(tmp_path / "out.csv"))
    assert set(df["team"]) == {"Arsenal"}


def test_import_alo_rejects_bad_schema(tmp_path):
    source = _export(tmp_path / "bad.csv", ["foo,bar", "Arsenal,1900"])
    with pytest.raises(SystemExit):
        alo.import_alo(str(source), out=str(tmp_path / "out.csv"))


def test_attach_alo_adds_reference_columns():
    ratings = {"Arsenal": 2046.0, "Nott'm Forest": 1789.0}
    upcoming = pd.DataFrame([{
        "home_team": "Arsenal",
        "away_team": "Nott'm Forest",
    }])
    out = alo.attach_alo(upcoming, ratings)
    assert out.loc[0, "alo_home"] == pytest.approx(2046.0)
    assert out.loc[0, "alo_away"] == pytest.approx(1789.0)
    assert out.loc[0, "alo_diff"] == pytest.approx(2046.0 - 1789.0)


def test_load_alo_round_trip(tmp_path):
    path = tmp_path / "alo.csv"
    pd.DataFrame({"team": ["Arsenal", "Man City"], "alo": [2046.0, 2013.0]}).to_csv(path, index=False)
    ratings = alo.load_alo(str(path))
    assert ratings == {"Arsenal": 2046.0, "Man City": 2013.0}

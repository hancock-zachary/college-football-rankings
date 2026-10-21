import pandas as pd
import pytest
from typer.testing import CliRunner

from cfb_bias.cli import app
from cfb_bias.db import read_sql
from cfb_bias.ratings.weekly import (
    conference_strength,
    strength_of_schedule,
    weekly_ratings,
)


def _game(gid, home, away, hp=30, ap=20, week=1, stype="regular", season=2024,
          cfp=False, hc="fbs", ac="fbs"):
    return {
        "game_id": gid, "season": season, "week": week, "season_type": stype,
        "completed": True, "neutral_site": False,
        "home_team": home, "home_classification": hc, "home_points": hp,
        "away_team": away, "away_classification": ac, "away_points": ap,
        "is_cfp": cfp,
    }


def _frame(rows):
    df = pd.DataFrame(rows)
    df["home_points"] = df["home_points"].astype("Int64")
    df["away_points"] = df["away_points"].astype("Int64")
    return df


TEAMS = ["A", "B", "C", "D", "FCS"]


def test_sos_uses_opponent_ratings():
    games = pd.DataFrame({
        "team_a": ["A", "A"], "team_b": ["B", "C"], "weight": [1.0, 1.0],
    })
    sos = strength_of_schedule(games, pd.Series({"A": 0.5, "B": 0.7, "C": 0.3}))
    assert sos["A"] == pytest.approx(0.5)
    assert sos["B"] == pytest.approx(0.5)  # B's only opponent is A
    assert sos["C"] == pytest.approx(0.5)


def test_sos_is_weight_averaged_and_nan_without_games():
    games = pd.DataFrame({
        "team_a": ["A", "A"], "team_b": ["B", "C"], "weight": [3.0, 1.0],
    })
    r = pd.Series({"A": 0.5, "B": 0.8, "C": 0.4, "D": 0.5})
    sos = strength_of_schedule(games, r)
    assert sos["A"] == pytest.approx((3 * 0.8 + 0.4) / 4)
    assert pd.isna(sos["D"])


def test_conference_median_resists_outlier():
    r = pd.Series({"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.2})
    conf = pd.Series({"a": "X", "b": "X", "c": "X", "d": "X", "e": "Y"})
    out = conference_strength(r, conf).set_index("conference")
    assert out.loc["X", "median_rating"] == pytest.approx(0.5)
    assert out.loc["X", "n_teams"] == 4
    assert out.loc["Y", "n_teams"] == 1
    assert list(conference_strength(r, conf).columns) == [
        "conference", "median_rating", "n_teams"]


def test_weekly_cutoff_is_monotonic_in_games():
    games = _frame([
        _game(1, "A", "B", week=1),
        _game(2, "C", "D", week=2),
    ])
    out = weekly_ratings(games, TEAMS, 2024, n_boot=5)
    assert "FCS" not in set(out["team"])
    assert set(out["week"]) == {1, 2, 99}
    w1 = out[(out.week == 1) & (out.method == "colley")].set_index("team")
    assert w1.loc["C", "rating"] == pytest.approx(0.5)
    assert w1.loc["D", "rating"] == pytest.approx(0.5)
    assert pd.isna(w1.loc["C", "sos"])
    assert w1.loc["A", "rating"] > 0.5
    w2 = out[(out.week == 2) & (out.method == "colley")].set_index("team")
    assert w2.loc["C", "rating"] > 0.5
    assert len(w1) == 4  # every FBS team gets a row, idle ones included
    assert set(out["method"]) == {"colley", "massey"}


def test_final_week_99_includes_cfp_only_with_weight():
    reg = [_game(1, "A", "B", week=1), _game(2, "B", "C", week=2),
           _game(3, "C", "D", week=3)]
    bowl = _game(4, "D", "A", week=1, stype="postseason", hp=40, ap=3)
    cfp = _game(5, "D", "A", week=2, stype="postseason", hp=40, ap=3, cfp=True)

    def final(rows):
        out = weekly_ratings(_frame(rows), TEAMS, 2024, n_boot=5)
        last = out[(out.week == 3)].set_index(["method", "team"])["rating"]
        w99 = out[(out.week == 99)].set_index(["method", "team"])["rating"]
        return last, w99

    last, w99 = final(reg + [bowl])
    pd.testing.assert_series_equal(last, w99)
    last, w99 = final(reg + [cfp])
    assert not (last - w99).abs().lt(1e-9).all()
    assert w99[("colley", "D")] > last[("colley", "D")]


@pytest.mark.db
def test_cli_ratings_end_to_end(pg_engine, monkeypatch):
    from sqlalchemy import text

    from cfb_bias.db import upsert

    rows = _frame([
        _game(9001, "A", "B", week=1), _game(9002, "C", "D", week=1),
        _game(9003, "B", "C", week=2), _game(9004, "D", "A", week=2),
    ])
    rows = rows.assign(season=2099)
    with pg_engine.begin() as conn:
        conn.execute(text("DELETE FROM ratings_weekly"))
    upsert(pg_engine, "games", rows, ["game_id"])
    ts = pd.DataFrame({
        "season": 2099, "team": ["A", "B", "C", "D"],
        "conference": ["X", "X", "Y", "Y"], "classification": "fbs",
    })
    upsert(pg_engine, "team_seasons", ts, ["season", "team"])

    monkeypatch.setenv(
        "DATABASE_URL", pg_engine.url.render_as_string(hide_password=False))
    runner = CliRunner()
    args = ["ratings", "--season", "2099", "--bootstrap", "20"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output

    def counts():
        return read_sql(pg_engine,
                        "SELECT method, count(*) n FROM ratings_weekly "
                        "WHERE season = 2099 GROUP BY method ORDER BY method")

    first = counts()
    assert list(first["method"]) == ["colley", "massey"]
    assert (first["n"] == 12).all()  # 4 teams x (week 1, week 2, week 99)
    assert runner.invoke(app, args).exit_code == 0
    assert counts().equals(first)

    empty = runner.invoke(app, ["ratings", "--season", "1900"])
    assert empty.exit_code == 1

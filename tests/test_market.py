import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from typer.testing import CliRunner

from cfb_bias.analysis.market import consensus_spread, market_ratings
from cfb_bias.cli import app
from cfb_bias.db import read_sql, upsert


def _games(rows):
    """rows: (game_id, week, home, away[, season_type])."""
    return pd.DataFrame({
        "game_id": [r[0] for r in rows],
        "week": [r[1] for r in rows],
        "season_type": [r[4] if len(r) > 4 else "regular" for r in rows],
        "home_team": [r[2] for r in rows],
        "away_team": [r[3] for r in rows],
        "home_classification": "fbs",
        "away_classification": "fbs",
        "neutral_site": False,
    })


def test_consensus_spread_median_ignores_null():
    lines = pd.DataFrame({
        "game_id": [1, 1, 1, 2, 2, 3],
        "provider": ["a", "b", "c", "a", "b", "a"],
        "spread": [-3.0, -7.0, None, None, None, -10.0],
    })
    out = consensus_spread(lines)
    assert out.loc[1] == -5.0
    assert out.loc[3] == -10.0
    assert 2 not in out.index
    assert out.index.name == "game_id"


def test_market_rating_orders_favourites():
    # B is favoured away at C, so the fitted home edge does not absorb the gaps.
    games = _games([(1, 1, "A", "B"), (2, 2, "C", "B")])
    spreads = pd.Series({1: -14.0, 2: 14.0})
    out = market_ratings(games, spreads, ["A", "B", "C"], poll_week=3).set_index("team")
    assert out.loc["A", "market_rating"] > out.loc["B", "market_rating"] > out.loc["C", "market_rating"]
    assert (out["n_games"] == pd.Series({"A": 1, "B": 2, "C": 1})).all()


def test_window_excludes_old_games():
    games = _games([(1, 1, "A", "B"), (2, 6, "D", "C")])
    spreads = pd.Series({1: -7.0, 2: 7.0})
    out = market_ratings(games, spreads, ["A", "B", "C", "D"], poll_week=6).set_index("team")
    assert out.loc["A", "n_games"] == 0 and np.isnan(out.loc["A", "market_rating"])
    assert out.loc["C", "n_games"] == 1 and not np.isnan(out.loc["C", "market_rating"])


def test_team_without_lines_is_nan():
    games = _games([(1, 2, "A", "B"), (2, 2, "C", "D")])
    spreads = pd.Series({1: -7.0})
    out = market_ratings(games, spreads, ["A", "B", "C", "D"], poll_week=3).set_index("team")
    assert out.loc["C", "n_games"] == 0 and np.isnan(out.loc["C", "market_rating"])
    assert out.loc["A", "n_games"] == 1 and not np.isnan(out.loc["A", "market_rating"])


def test_fcs_pooled_excluded_and_postseason_ignored():
    games = _games([(1, 2, "A", "X"), (2, 2, "A", "B", "postseason")])
    games.loc[0, "away_classification"] = "fcs"
    spreads = pd.Series({1: -21.0, 2: 30.0})
    out = market_ratings(games, spreads, ["A", "B"], poll_week=3).set_index("team")
    assert "FCS" not in out.index
    assert out.loc["A", "n_games"] == 1 and out.loc["A", "market_rating"] > 0
    assert out.loc["B", "n_games"] == 0


@pytest.mark.db
def test_cli_market_writes_poll_weeks(pg_engine, monkeypatch):
    S = 2411
    tables = ("games", "poll_ranks", "team_seasons", "betting_lines", "market_weekly")

    def clean():
        with pg_engine.begin() as conn:
            for t in tables:
                conn.execute(text(f"DELETE FROM {t} WHERE season = :s"), {"s": S})
    clean()
    upsert(pg_engine, "team_seasons", pd.DataFrame({
        "season": [S] * 3, "team": ["MktA", "MktB", "MktC"],
        "conference": ["x"] * 3, "classification": ["fbs"] * 3}), ["season", "team"])
    upsert(pg_engine, "games", pd.DataFrame({
        "game_id": [9200001, 9200002], "season": [S, S], "week": [1, 2],
        "season_type": ["regular"] * 2, "completed": [False, False],
        "neutral_site": [False, False],
        "home_team": ["MktA", "MktC"], "away_team": ["MktB", "MktB"],
        "home_classification": ["fbs"] * 2, "away_classification": ["fbs"] * 2}), ["game_id"])
    upsert(pg_engine, "betting_lines", pd.DataFrame({
        "game_id": [9200001, 9200002], "season": [S, S], "week": [1, 2],
        "season_type": ["regular"] * 2, "provider": ["p", "p"],
        "home_team": ["MktA", "MktC"], "away_team": ["MktB", "MktB"],
        "spread": [-14.0, 14.0]}), ["game_id", "provider"])
    upsert(pg_engine, "poll_ranks", pd.DataFrame({
        "season": [S, S, S], "season_type": ["regular", "regular", "postseason"],
        "week": [3, 4, 1], "poll": ["AP Top 25"] * 3, "is_final": [False] * 3,
        "rank": [1, 1, 1], "team": ["MktA"] * 3}), ["season", "season_type", "week", "poll", "team"])
    monkeypatch.setattr("cfb_bias.cli.get_engine", lambda url: pg_engine)
    try:
        res = CliRunner().invoke(app, ["market", "--from", str(S), "--to", str(S)])
        assert res.exit_code == 0, res.output
        df = read_sql(pg_engine, "SELECT * FROM market_weekly WHERE season = :s", {"s": S})
        assert sorted(df["week"].unique()) == [3, 4]
        w3 = df[df["week"] == 3].set_index("team")
        assert w3.loc["MktA", "market_rating"] > w3.loc["MktB", "market_rating"] > w3.loc["MktC", "market_rating"]
    finally:
        clean()

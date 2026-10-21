import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from cfb_bias.analysis.features import (
    asof_ratings_week,
    build_features,
    poll_scores,
    resume_through,
)
from cfb_bias.db import upsert


def test_asof_join_handles_bye_and_preseason():
    got = asof_ratings_week([1, 3, 5, 6], [1, 2, 4])
    assert got == {1: None, 3: 2, 5: 4, 6: 4}


def test_cfp_uses_rank_score_not_points():
    polls = pd.DataFrame({
        "week": [10, 10], "poll": ["Playoff Committee Rankings"] * 2,
        "team": ["A", "B"], "rank": [1, 25], "points": [0, 0],
    })
    s = poll_scores(polls, "Playoff Committee Rankings", use_points=False)
    got = s.set_index("team")["score"]
    assert got["A"] == pytest.approx(1.0)
    assert got["B"] == pytest.approx(1 / 25)


def test_ap_points_share():
    polls = pd.DataFrame({
        "week": [2, 2, 3], "poll": ["AP Top 25"] * 3,
        "team": ["A", "B", "A"], "rank": [1, 2, 1], "points": [1500, 750, 1400],
    })
    s = poll_scores(polls, "AP Top 25", use_points=True)
    got = s.set_index(["poll_week", "team"])["score"]
    assert got[(2, "A")] == pytest.approx(1.0)
    assert got[(2, "B")] == pytest.approx(0.5)
    assert got[(3, "A")] == pytest.approx(1.0)


S = 2501
FBS = ["FtA", "FtB", "FtC"]


@pytest.fixture
def seeded(pg_engine):
    def clean():
        with pg_engine.begin() as conn:
            for t in ("games", "poll_ranks", "team_seasons", "ratings_weekly",
                      "market_weekly", "game_media"):
                conn.execute(text(f"DELETE FROM {t} WHERE season = :s"), {"s": S})
    clean()
    upsert(pg_engine, "team_seasons", pd.DataFrame({
        "season": [S] * 4, "team": FBS + ["FtFcs"],
        "conference": ["X"] * 4, "classification": ["fbs"] * 3 + ["fcs"],
    }), ["season", "team"])
    # FtA: W vs FtB (wk1), W vs FtC (wk2), L vs FtFcs (wk3, FCS = bad loss), W wk4 (future)
    rows = [
        (9200001, 1, "FtA", "FtB", 'fbs', 'fbs', 30, 10),
        (9200002, 2, "FtC", "FtA", 'fbs', 'fbs', 3, 20),
        (9200003, 3, "FtFcs", "FtA", 'fcs', 'fbs', 24, 21),
        (9200004, 4, "FtA", "FtB", 'fbs', 'fbs', 50, 0),
    ]
    games = pd.DataFrame({
        "game_id": [r[0] for r in rows], "season": S, "week": [r[1] for r in rows],
        "season_type": "regular", "completed": True,
        "home_team": [r[2] for r in rows], "away_team": [r[3] for r in rows],
        "home_classification": [r[4] for r in rows],
        "away_classification": [r[5] for r in rows],
        "home_points": [r[6] for r in rows], "away_points": [r[7] for r in rows],
    })
    upsert(pg_engine, "games", games, ["game_id"])
    upsert(pg_engine, "game_media", pd.DataFrame({
        "game_id": [9200001, 9200002, 9200003], "season": S, "week": [1, 2, 3],
        "season_type": "regular", "home_team": "x", "away_team": "y",
        "media_type": "tv", "outlet": ["ESPN", "ESPN+", "NBC"],
    }), ["game_id", "outlet"])
    polls = pd.DataFrame({
        "season": S, "season_type": "regular",
        "week": [1, 1, 2, 4, 4, 4],
        "poll": ["AP Top 25", "AP Top 25", "AP Top 25", "AP Top 25", "AP Top 25",
                 "Coaches Poll"],
        "rank": [1, 2, 1, 1, 2, 1],
        "team": ["FtA", "FtB", "FtB", "FtB", "FtC", "FtC"],
        "points": [1500, 750, 1000, 1000, 500, 100],
    })
    upsert(pg_engine, "poll_ranks", polls, ["season", "season_type", "week", "poll", "team"])
    # ratings at weeks 1,2,4 (bye at 3); colley ranks: week 2 -> FtC best, FtB, FtA
    rw = []
    for wk, cols in {1: [0.6, 0.4, 0.5], 2: [0.3, 0.4, 0.9], 4: [0.9, 0.1, 0.5]}.items():
        for team, c in zip(FBS, cols):
            rw.append((S, wk, team, "colley", c, c - .1, c + .1, 0.5))
            rw.append((S, wk, team, "massey", c * 10, 0, 0, 1.5))
    upsert(pg_engine, "ratings_weekly", pd.DataFrame(
        rw, columns=["season", "week", "team", "method", "rating", "lo", "hi", "sos"]),
        ["season", "week", "team", "method"])
    upsert(pg_engine, "market_weekly", pd.DataFrame({
        "season": S, "week": [4], "team": ["FtA"], "market_rating": [7.5], "n_games": [3],
    }), ["season", "week", "team"])
    yield pg_engine
    clean()


def test_build_features_counts_wins_through_prior_week(seeded):
    f = build_features(seeded, S).set_index(["poll_week", "team"])
    assert sorted(set(f.index.get_level_values(0))) == [1, 2, 4]
    assert len(f) == 9  # 3 FBS teams x 3 AP poll weeks; FCS excluded
    pre = f.loc[(1, "FtA")]
    assert pd.isna(pre["ratings_week"])
    assert pre["games_played"] == 0 and pre["wins"] == 0
    assert pre["colley"] == 0.5 and pre["massey"] == 0 and pd.isna(pre["colley_sos"])
    assert pd.isna(pre["won_last"]) and np.isnan(pre["national_tv_share"])
    assert pre["ap_score"] == pytest.approx(1.0)
    assert pd.isna(pre["coaches_rank_score"]) and pd.isna(pre["cfp_rank_score"])
    # poll week 4 -> games through week 3, ratings_week 2 (bye in ratings at 3)
    a = f.loc[(4, "FtA")]
    assert a["ratings_week"] == 2
    assert (a["wins"], a["losses"], a["games_played"]) == (2, 1, 3)
    assert a["won_last"] == False
    assert a["colley"] == pytest.approx(0.3)
    assert a["market_rating"] == pytest.approx(7.5)
    # ESPN and NBC count; ESPN+ does not -> 2 of 3 games
    assert a["national_tv_share"] == pytest.approx(2 / 3)
    assert a["ap_score"] == 0  # unranked in week 4
    assert a["coaches_rank_score"] == 0  # coaches poll exists wk4, FtA unranked
    c = f.loc[(4, "FtC")]
    assert c["coaches_rank_score"] == pytest.approx(1.0)
    assert np.isnan(f.loc[(4, "FtB"), "market_rating"])
    # week 2 poll: games through week 1 only
    a2 = f.loc[(2, "FtA")]
    assert (a2["wins"], a2["games_played"], a2["ratings_week"]) == (1, 1, 1)
    assert a2["won_last"] == True


def test_quality_wins_and_bad_losses(seeded):
    f = build_features(seeded, S).set_index(["poll_week", "team"])
    # ratings_week 2: colley ranks FtC=1, FtB=2, FtA=3 (all <= 25, none > 60)
    a = f.loc[(4, "FtA")]
    assert a["quality_wins"] == 2     # beat FtB and FtC, both top-25
    assert a["bad_losses"] == 1       # lost to non-FBS FtFcs
    b = f.loc[(4, "FtB")]
    assert (b["quality_wins"], b["bad_losses"]) == (0, 0)  # lost to top-25 FtA


def _results(team, outcomes):
    """outcomes: list of (opp, opp_fbs, won) -> long-form team_results rows."""
    return pd.DataFrame({
        "team": team, "opp": [o[0] for o in outcomes], "opp_fbs": [o[1] for o in outcomes],
        "week": range(1, len(outcomes) + 1), "won": [o[2] for o in outcomes],
        "lost": [not o[2] for o in outcomes], "game_id": range(len(outcomes)),
        "national": False,
    })


def test_quality_win_and_bad_loss_rank_boundaries():
    # hand-built ranks as rank(method="min") yields them: TieA/TieB share rank 25
    colley_rank = pd.Series({"R25": 25.0, "R26": 26.0, "TieA": 25.0, "TieB": 25.0,
                             "R60": 60.0, "R61": 61.0})
    wins = _results("W", [("R25", True, True), ("R26", True, True), ("NoRank", True, True),
                          ("TieA", True, True), ("TieB", True, True)])
    losses = _results("L", [("R60", True, False), ("R61", True, False),
                            ("NoRank", True, False), ("FcsTeam", False, False)])
    out = resume_through(pd.concat([wins, losses], ignore_index=True), 99, colley_rank)
    # quality: rank 25 and both tied-at-25 teams; not rank 26 or an FBS team with NaN rank
    assert out.loc["W", "quality_wins"] == 3
    assert out.loc["W", "bad_losses"] == 0
    # bad: rank 61 and the FCS loss; not rank 60 or an FBS team with NaN rank
    assert out.loc["L", "bad_losses"] == 2
    assert out.loc["L", "quality_wins"] == 0
    # each boundary on its own
    for opp, rank_expected in (("R25", 1), ("R26", 0), ("NoRank", 0), ("TieB", 1)):
        one = resume_through(_results("W", [(opp, True, True)]), 99, colley_rank)
        assert one.loc["W", "quality_wins"] == rank_expected, opp
    for opp, fbs, bad_expected in (("R60", True, 0), ("R61", True, 1), ("NoRank", True, 0),
                                   ("FcsTeam", False, 1)):
        one = resume_through(_results("L", [(opp, fbs, False)]), 99, colley_rank)
        assert one.loc["L", "bad_losses"] == bad_expected, opp


def test_build_features_without_ratings_raises(seeded):
    with seeded.begin() as conn:
        conn.execute(text("DELETE FROM ratings_weekly WHERE season = :s"), {"s": S})
    with pytest.raises(ValueError, match=rf"season {S}.*cfb ratings --season {S}"):
        build_features(seeded, S)


def test_build_features_preseason_only_without_ratings_ok(seeded):
    with seeded.begin() as conn:
        conn.execute(text("DELETE FROM ratings_weekly WHERE season = :s"), {"s": S})
        conn.execute(text("DELETE FROM poll_ranks WHERE season = :s AND week > 1"), {"s": S})
    f = build_features(seeded, S)
    assert set(f["poll_week"]) == {1}
    assert (f["colley"] == 0.5).all()

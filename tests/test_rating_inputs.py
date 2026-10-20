import pandas as pd
import pytest

from cfb_bias.ratings.inputs import game_weight, rating_games


@pytest.mark.parametrize(
    "season,stype,cfp,w",
    [
        (2019, "regular", False, 1.0),
        (2019, "postseason", False, 0.5),
        (2019, "postseason", True, 0.5),
        (2020, "postseason", False, 0.0),
        (2024, "postseason", True, 0.25),
        (2024, "regular", False, 1.0),
    ],
)
def test_game_weight(season, stype, cfp, w):
    assert game_weight(season, stype, cfp) == w


def _game(gid, home, away, hp=30, ap=20, week=1, stype="regular", season=2023,
          completed=True, neutral=False, hc="fbs", ac="fbs", cfp=False):
    return {
        "game_id": gid, "season": season, "week": week, "season_type": stype,
        "completed": completed, "neutral_site": neutral,
        "home_team": home, "home_classification": hc, "home_points": hp,
        "away_team": away, "away_classification": ac, "away_points": ap,
        "is_cfp": cfp,
    }


def _frame(rows):
    df = pd.DataFrame(rows)
    df["home_points"] = df["home_points"].astype("Int64")
    df["away_points"] = df["away_points"].astype("Int64")
    return df


def test_basic_shape_and_dtypes():
    out = rating_games(_frame([_game(1, "A", "B", neutral=True)]), 2023, None, True)
    assert list(out.columns) == [
        "game_id", "week", "team_a", "team_b", "points_a", "points_b",
        "a_won", "neutral_site", "a_home", "weight"]
    r = out.iloc[0]
    assert (r.team_a, r.team_b, r.points_a, r.points_b) == ("A", "B", 30, 20)
    assert r.a_won and r.neutral_site and not r.a_home and r.weight == 1.0
    assert out["a_won"].dtype == bool and out["a_home"].dtype == bool
    assert out["points_a"].dtype.kind == "i" and out["weight"].dtype == float


def test_home_flag_and_loss():
    out = rating_games(_frame([_game(1, "A", "B", hp=10, ap=20)]), 2023, None, True)
    assert out.iloc[0].a_home and not out.iloc[0].a_won


def test_incomplete_games_excluded():
    df = _frame([_game(1, "A", "B"), _game(2, "C", "D", completed=False),
                 _game(3, "E", "F", hp=None, ap=None)])
    assert list(rating_games(df, 2023, None, True).game_id) == [1]


def test_fcs_opponents_pooled():
    df = _frame([_game(1, "A", "Furman", ac="fcs"), _game(2, "Wofford", "B", hc="fcs")])
    out = rating_games(df, 2023, None, True)
    assert out.team_b[0] == "FCS" and out.team_a[1] == "FCS"
    assert set(out.team_a) | set(out.team_b) == {"A", "FCS", "B"}


def test_both_non_fbs_dropped():
    df = _frame([_game(1, "X", "Y", hc="fcs", ac="div2"), _game(2, "A", "B")])
    assert list(rating_games(df, 2023, None, True).game_id) == [2]


def test_ties_dropped():
    df = _frame([_game(1, "A", "B", hp=7, ap=7), _game(2, "C", "D")])
    assert list(rating_games(df, 2023, None, True).game_id) == [2]


def test_2021_non_cfp_bowl_dropped():
    df = _frame([_game(1, "A", "B", stype="postseason", season=2021),
                 _game(2, "C", "D", stype="postseason", season=2021, cfp=True)])
    out = rating_games(df, 2021, None, True)
    assert list(out.game_id) == [2] and out.weight[0] == 0.25


def test_include_postseason_flag():
    df = _frame([_game(1, "A", "B", season=2019, stype="postseason"),
                 _game(2, "C", "D", season=2019)])
    assert list(rating_games(df, 2019, None, False).game_id) == [2]
    out = rating_games(df, 2019, None, True)
    assert sorted(out.game_id) == [1, 2]
    assert out.set_index("game_id").weight[1] == 0.5


def test_through_week_cutoff():
    df = _frame([_game(1, "A", "B", week=4), _game(2, "C", "D", week=5),
                 _game(3, "E", "F", week=15, stype="postseason", season=2019)])
    out = rating_games(df, 2023, 4, True)
    assert list(out.game_id) == [1]
    df["season"] = 2019
    out = rating_games(df, 2019, 4, True)
    assert sorted(out.game_id) == [1, 3]  # postseason unaffected by through_week


def test_season_filter_and_reset_index():
    df = _frame([_game(1, "A", "B", season=2022), _game(2, "C", "D"), _game(3, "E", "F")])
    out = rating_games(df, 2023, None, True)
    assert list(out.index) == [0, 1] and list(out.game_id) == [2, 3]


def test_nullable_boolean_inputs():
    df = _frame([
        _game(1, "A", "B"),
        _game(2, "C", "D"),
        _game(3, "E", "F", season=2021, stype="postseason"),
        _game(4, "G", "H"),
    ])
    for c in ("completed", "neutral_site", "is_cfp"):
        df[c] = df[c].astype("boolean")
    df.loc[1, "completed"] = pd.NA
    df.loc[2, "is_cfp"] = pd.NA
    df.loc[3, "neutral_site"] = pd.NA
    out = rating_games(df, 2023, None, True)
    assert list(out.game_id) == [1, 4]
    assert out.a_home.tolist() == [True, True]
    assert out.neutral_site.tolist() == [False, False]
    assert list(rating_games(df, 2021, None, True).game_id) == []

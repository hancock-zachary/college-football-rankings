import pandas as pd
import pytest

from cfb_bias.analysis.stability import choose_start_week, rating_stability

TEAMS = [f"T{i}" for i in range(10)]
# rating order per week for each team (index = team); week 99 is the truth 0..9
WEEK_ORDER = {
    1: [5, 2, 9, 0, 7, 1, 8, 3, 6, 4],  # near-random
    2: [3, 9, 1, 6, 0, 8, 2, 7, 4, 5],  # near-random
    3: [4, 0, 6, 1, 8, 2, 9, 3, 7, 5],  # still noisy
    4: [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],  # nearly sorted
    99: list(range(10)),
}


def _ratings(seasons=(2020, 2021), method="colley", weeks=None):
    rows = []
    for season in seasons:
        for week, vals in (weeks or WEEK_ORDER).items():
            for team, v in zip(TEAMS, vals):
                rows.append((season, week, team, method, float(v), 0.0, 0.0, 0.0))
    return pd.DataFrame(
        rows, columns=["season", "week", "team", "method", "rating", "lo", "hi", "sos"]
    )


def test_rating_stability_shape_and_values():
    out = rating_stability(_ratings())
    assert list(out.columns) == ["ratings_week", "mean_spearman", "n_seasons"]
    assert out["ratings_week"].tolist() == [1, 2, 3, 4]  # 99 excluded
    assert (out["n_seasons"] == 2).all()
    w4 = out.set_index("ratings_week").loc[4, "mean_spearman"]
    assert w4 == pytest.approx(1 - 6 * 2 / (10 * 99))
    assert out.set_index("ratings_week").loc[1, "mean_spearman"] < 0.75


def test_ignores_non_colley_and_skips_seasons_without_week_99():
    colley = _ratings(seasons=(2020,))
    massey = _ratings(seasons=(2020,), method="massey", weeks={1: WEEK_ORDER[99], 99: WEEK_ORDER[99]})
    no99 = _ratings(seasons=(2019,), weeks={1: WEEK_ORDER[1], 4: WEEK_ORDER[4]})
    out = rating_stability(pd.concat([colley, massey, no99]))
    assert out["ratings_week"].tolist() == [1, 2, 3, 4]
    assert (out["n_seasons"] == 1).all()


def test_only_teams_in_both_week_and_99():
    df = _ratings(seasons=(2020,))
    # drop a team from week 4 only; correlation is computed on the other 9
    df = df[~((df["week"] == 4) & (df["team"] == "T9"))]
    out = rating_stability(df).set_index("ratings_week")
    assert out.loc[4, "mean_spearman"] == pytest.approx(1.0)


def test_choose_start_week_first_stable_plus_one():
    stab = rating_stability(_ratings())
    assert choose_start_week(stab, threshold=0.75) == 5


def test_choose_start_week_handles_gaps():
    stab = pd.DataFrame(
        {"ratings_week": [1, 3, 5], "mean_spearman": [0.2, 0.8, 0.9], "n_seasons": [3, 3, 3]}
    )
    assert choose_start_week(stab) == 4


def test_never_stable_raises():
    stab = pd.DataFrame(
        {"ratings_week": [1, 2], "mean_spearman": [0.1, 0.5], "n_seasons": [3, 3]}
    )
    with pytest.raises(ValueError):
        choose_start_week(stab)

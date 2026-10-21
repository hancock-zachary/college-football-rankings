import pandas as pd
import pytest

from cfb_bias.ratings.bootstrap import bootstrap_ratings


def mean_rate(games, teams):
    return pd.Series(float(len(games)), index=pd.Index(teams, name="team"))


def test_n_must_be_positive():
    with pytest.raises(ValueError):
        bootstrap_ratings(mean_rate, pd.DataFrame({"x": [1]}), ["A"], n=0)


def test_sorted_and_reset_index():
    out = bootstrap_ratings(mean_rate, pd.DataFrame({"x": [1, 2]}), ["B", "A"], n=5)
    assert list(out["team"]) == ["A", "B"]
    assert list(out.index) == [0, 1]


def test_absent_team_draws_are_nan_not_prior():
    import numpy as np

    from cfb_bias.ratings.bootstrap import bootstrap_weighted

    a = np.array([0]); b = np.array([1]); w = np.array([1.0])
    # Team 2 never plays: every draw NaN -> lo/hi NaN, full-sample rating kept at 0.5.
    out = bootstrap_weighted(
        lambda wt: np.array([0.7, 0.3, 0.5]), a, b, w, ["A", "B", "C"], n=10
    ).set_index("team")
    assert out.loc["C", "rating"] == 0.5
    assert np.isnan(out.loc["C", "lo"]) and np.isnan(out.loc["C", "hi"])
    assert out.loc["A", "lo"] == pytest.approx(0.7)


def test_colley_bootstrap_ignores_absent_draws():
    import numpy as np

    from cfb_bias.ratings.colley import colley_bootstrap

    games = pd.DataFrame({
        "team_a": ["A", "A", "C"], "team_b": ["B", "B", "D"],
        "a_won": [True, True, True], "weight": [1.0] * 3,
    })
    out = colley_bootstrap(games, list("ABCD"), n=200, seed=1).set_index("team")
    # C/D are in one game of three, so absent from ~30% of draws; interval excludes
    # the 0.5 prior those draws would otherwise contribute.
    assert out.loc["C", "lo"] > 0.5
    assert out.loc["D", "hi"] < 0.5


def test_bootstrap_weighted_limits_blas_threads():
    import numpy as np
    from threadpoolctl import threadpool_info

    from cfb_bias.ratings.bootstrap import bootstrap_weighted

    seen = []

    def fit(wt):
        seen.append([lib["num_threads"] for lib in threadpool_info()
                     if lib["user_api"] == "blas"])
        return np.array([0.6, 0.4])

    bootstrap_weighted(fit, np.array([0]), np.array([1]), np.array([1.0]),
                       ["A", "B"], n=3)
    assert seen and all(t == 1 for counts in seen for t in counts)

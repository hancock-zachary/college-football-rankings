import pandas as pd
import pytest

from cfb_bias.ratings.colley import colley, colley_bootstrap


def g(*rows):  # rows of (a, b, a_won, weight)
    return pd.DataFrame(rows, columns=["team_a", "team_b", "a_won", "weight"])


def test_unweighted_mean_is_half():
    r = colley(g(("A", "B", True, 1), ("B", "C", True, 1), ("C", "D", False, 1)), list("ABCD"))
    assert r.mean() == pytest.approx(0.5)


def test_team_with_no_games_is_neutral():
    r = colley(g(("A", "B", True, 1)), ["A", "B", "C"])
    assert r["C"] == pytest.approx(0.5)


def test_transitive_order():
    r = colley(g(("A", "B", True, 1), ("B", "C", True, 1)), list("ABC"))
    assert r["A"] > r["B"] > r["C"]


def test_margin_is_ignored():
    base = g(("A", "B", True, 1), ("B", "C", True, 1))
    one = base.assign(points_a=[10, 50], points_b=[9, 0])
    two = base.assign(points_a=[70, 3], points_b=[0, 2])
    pd.testing.assert_series_equal(colley(one, list("ABC")), colley(two, list("ABC")))


def test_weight_zero_equals_absent_game():
    with_zero = g(("A", "B", True, 1), ("B", "C", True, 0), ("C", "A", True, 1))
    without = g(("A", "B", True, 1), ("C", "A", True, 1))
    pd.testing.assert_series_equal(colley(with_zero, list("ABC")), colley(without, list("ABC")))


def test_weighted_formula_exact():
    # A beats B with weight 0.5: C = [[2.5,-.5],[-.5,2.5]], b = [1.25, 0.75]
    r = colley(g(("A", "B", True, 0.5)), ["A", "B"])
    assert r["A"] == pytest.approx(7 / 12)
    assert r["B"] == pytest.approx(5 / 12)


def test_empty_games_all_half():
    r = colley(g(), ["A", "B"])
    assert (r == 0.5).all()


def test_unknown_team_raises():
    with pytest.raises(ValueError, match="Z"):
        colley(g(("A", "Z", True, 1)), ["A", "B"])


def test_bootstrap_interval_contains_rating_and_is_deterministic():
    games = g(("A", "B", True, 1), ("B", "C", True, 1), ("C", "D", False, 1),
              ("A", "C", True, 1), ("B", "D", False, 1), ("A", "D", True, 1))
    out = colley_bootstrap(games, list("ABCD"), n=200, seed=3)
    assert list(out.columns) == ["team", "rating", "lo", "hi"]
    assert list(out["team"]) == list("ABCD")
    assert (out["lo"] <= out["hi"]).all()
    assert (out["lo"] <= out["rating"] + 1e-9).all() and (out["rating"] <= out["hi"] + 1e-9).all()
    pd.testing.assert_frame_equal(out, colley_bootstrap(games, list("ABCD"), n=200, seed=3))
    full = colley(games, list("ABCD"))
    assert out["rating"].to_numpy() == pytest.approx(full.to_numpy())

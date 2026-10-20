import numpy as np
import pandas as pd
import pytest

from cfb_bias.ratings.massey import massey, massey_bootstrap


def g(*rows):  # rows of (a, b, points_a, points_b, a_home, weight)
    return pd.DataFrame(
        rows, columns=["team_a", "team_b", "points_a", "points_b", "a_home", "weight"]
    )


def chain():
    return g(("A", "B", 24, 10, False, 1.0), ("B", "C", 24, 10, False, 1.0))


def test_margin_order():
    r = massey(chain(), list("ABC"))
    assert r["A"] > r["B"] > r["C"]
    assert r["A"] - r["C"] > r["A"] - r["B"]


def test_margin_cap():
    big = g(("A", "B", 70, 0, False, 1.0), ("B", "C", 24, 10, False, 1.0))
    capped = g(("A", "B", 28, 0, False, 1.0), ("B", "C", 24, 10, False, 1.0))
    pd.testing.assert_series_equal(massey(big, list("ABC")), massey(capped, list("ABC")))


def test_centred():
    assert massey(chain(), list("ABC")).mean() == pytest.approx(0.0, abs=1e-12)


def test_team_with_no_games_is_zero():
    r = massey(chain(), list("ABCD"))
    assert r["D"] == 0.0
    assert r.mean() == pytest.approx(0.0, abs=1e-12)
    assert r[["A", "B", "C"]].mean() == pytest.approx(0.0, abs=1e-12)


def test_unknown_team_raises():
    with pytest.raises(ValueError):
        massey(chain(), ["A", "B"])


def test_weight_zero_equals_absent_game():
    with_zero = pd.concat([chain(), g(("C", "D", 40, 0, False, 0.0))], ignore_index=True)
    a = massey(with_zero, list("ABCD"))
    b = massey(chain(), list("ABCD"))
    pd.testing.assert_series_equal(a, b)


def test_home_field_is_absorbed_not_credited():
    # A wins only at home, B wins only at home: h soaks up the home edge.
    games = g(("A", "B", 20, 10, True, 1.0), ("B", "A", 20, 10, True, 1.0))
    r = massey(games.assign(team_a=["A", "B"], team_b=["B", "A"]), ["A", "B"])
    assert r["A"] == pytest.approx(r["B"], abs=1e-9)


def test_exact_two_team_solution():
    # one game, margin 10, ridge 1, no home: minimise (10-(ra-rb))^2 + ra^2 + rb^2
    # => ra = 10/3, rb = -10/3 (ridge-shrunk)
    r = massey(g(("A", "B", 20, 10, False, 1.0)), ["A", "B"])
    assert r["A"] == pytest.approx(10 / 3)
    assert r["B"] == pytest.approx(-10 / 3)


def test_bootstrap_shape_and_determinism():
    games = pd.concat([chain()] * 5, ignore_index=True)
    one = massey_bootstrap(games, list("CAB"), n=50, seed=1)
    two = massey_bootstrap(games, list("CAB"), n=50, seed=1)
    pd.testing.assert_frame_equal(one, two)
    assert list(one.columns) == ["team", "rating", "lo", "hi"]
    assert list(one["team"]) == ["A", "B", "C"]
    assert (one["lo"] <= one["hi"]).all()
    full = massey(games, list("CAB"))
    assert np.allclose(one["rating"], full[one["team"]].to_numpy())


def test_bootstrap_forwards_kwargs():
    games = pd.concat([chain()] * 5, ignore_index=True)
    a = massey_bootstrap(games, list("ABC"), n=10, ridge=1.0)["rating"]
    b = massey_bootstrap(games, list("ABC"), n=10, ridge=100.0)["rating"]
    assert not np.allclose(a, b)

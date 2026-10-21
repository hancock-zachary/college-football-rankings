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
    # A hosts B and C and wins each by 10; B-C is a neutral-site tie. With the home
    # coefficient, part of A's edge is credited to home field, so A rates lower than
    # when the same games are neutral.
    def games(home):
        return g(
            ("A", "B", 20, 10, home, 1.0),
            ("A", "C", 20, 10, home, 1.0),
            ("B", "C", 17, 17, False, 1.0),
        )

    home = massey(games(True), list("ABC"))
    neutral = massey(games(False), list("ABC"))
    assert home["A"] < neutral["A"] - 1.0


def test_home_coefficient_is_not_shrunk_by_ridge():
    # A hosts B and C and wins each by 10; B-C is a neutral tie. With a large ridge an
    # unpenalised h (=10) explains the margins entirely, so ratings stay ~0; a missing
    # or penalised h would leave A clearly positive.
    games = g(
        ("A", "B", 20, 10, True, 1.0),
        ("A", "C", 20, 10, True, 1.0),
        ("B", "C", 17, 17, False, 1.0),
    )
    r = massey(games, list("ABC"), ridge=100.0)
    assert r.abs().max() < 1e-6


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


def _dense_reference(games, teams, cap=28, ridge=1.0):
    """The original dense-lstsq formulation, kept to check the normal-equation fit."""
    index = {t: i for i, t in enumerate(teams)}
    k, m = len(teams), len(games)
    a = games["team_a"].map(index).to_numpy()
    b = games["team_b"].map(index).to_numpy()
    w = games["weight"].to_numpy(dtype=float)
    margin = np.clip((games["points_a"] - games["points_b"]).to_numpy(float), -cap, cap)
    X = np.zeros((m, k + 1))
    X[np.arange(m), a] = 1.0
    X[np.arange(m), b] = -1.0
    X[:, k] = games["a_home"].to_numpy(float)
    sw = np.sqrt(w)
    pen = np.zeros((k, k + 1))
    pen[:, :k] = np.sqrt(ridge) * np.eye(k)
    coef, *_ = np.linalg.lstsq(
        np.vstack([X * sw[:, None], pen]), np.concatenate([margin * sw, np.zeros(k)]),
        rcond=None)
    r = coef[:k]
    played = np.zeros(k, dtype=bool)
    played[a[w > 0]] = True
    played[b[w > 0]] = True
    return np.where(played, r - r[played].mean(), 0.0)


def test_fast_massey_matches_dense_reference():
    rng = np.random.default_rng(7)
    teams = [f"T{i}" for i in range(20)]
    m = 80
    a = rng.integers(0, 20, m)
    b = (a + rng.integers(1, 20, m)) % 20
    games = pd.DataFrame({
        "team_a": [teams[i] for i in a], "team_b": [teams[i] for i in b],
        "points_a": rng.integers(0, 60, m), "points_b": rng.integers(0, 60, m),
        "weight": rng.choice([0.0, 0.25, 0.5, 1.0], m),
        "a_home": rng.random(m) < 0.7,
    })
    for ridge in (1.0, 5.0):
        np.testing.assert_allclose(
            massey(games, teams, ridge=ridge).to_numpy(),
            _dense_reference(games, teams, ridge=ridge), atol=1e-8)

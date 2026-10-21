import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor

from cfb_bias.analysis import dml
from cfb_bias.analysis.dml import DmlResult, partial_out

THETA = 0.3


@pytest.fixture(autouse=True)
def fast_learner(monkeypatch):
    """Fewer boosting rounds keep the suite fast; production uses the default."""
    monkeypatch.setattr(
        dml, "make_learner",
        lambda seed: HistGradientBoostingRegressor(max_iter=40, random_state=seed),
    )


def synthetic(n_seasons: int, n_teams: int, theta: float, seed: int) -> pd.DataFrame:
    """y = sin(x1) + x2**2 + theta*z(b) + noise, with b confounded by x1."""
    rng = np.random.default_rng(seed)
    n = n_seasons * n_teams
    season = np.repeat(np.arange(2014, 2014 + n_seasons), n_teams)
    team = np.tile([f"T{i:03d}" for i in range(n_teams)], n_seasons)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-1, 1, n)
    b = x1 + rng.normal(0, 0.7, n)
    zb = (b - b.mean()) / b.std()
    y = np.sin(x1) + x2**2 + theta * zb + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"season": season, "team": team, "x1": x1, "x2": x2, "b": b, "y": y}
    )


def _coef(r: DmlResult, var: str) -> pd.Series:
    return r.coefficients.set_index("variable").loc[var]


def test_recovers_known_effect():
    df = synthetic(n_seasons=8, n_teams=60, theta=THETA, seed=1)
    r = partial_out(df, "y", ["x1", "x2"], ["b"])
    row = _coef(r, "b")
    assert row.ci_lo < THETA < row.ci_hi
    assert list(r.coefficients.columns) == ["variable", "coef", "se", "ci_lo", "ci_hi", "p_value"]
    assert r.n_obs == len(df)
    assert r.n_teams == 60
    assert r.n_dropped == 0
    assert r.y_resid.index.equals(df.index)


def test_null_effect_not_significant():
    df = synthetic(n_seasons=8, n_teams=60, theta=0.0, seed=2)
    r = partial_out(df, "y", ["x1", "x2"], ["b"])
    assert _coef(r, "b").p_value > 0.05


def test_rows_with_missing_brand_dropped_and_counted():
    df = synthetic(n_seasons=6, n_teams=40, theta=THETA, seed=3)
    df.loc[[0, 5, 17], "b"] = np.nan
    df.loc[[9], "y"] = np.nan
    df.loc[[11, 12], "x1"] = np.nan  # NaN in X is allowed: rows kept
    r = partial_out(df, "y", ["x1", "x2"], ["b"])
    assert r.n_dropped == 4
    assert r.n_obs == len(df) - 4
    assert not r.y_resid.index.isin([0, 5, 9, 17]).any()
    assert {11, 12} <= set(r.y_resid.index)
    assert r.y_resid.notna().all()


def test_constant_brand_column_dropped_with_warning():
    df = synthetic(n_seasons=6, n_teams=40, theta=THETA, seed=4)
    df["flat"] = 1.0
    df["enrollment"] = np.nan
    with pytest.warns(UserWarning) as rec:
        r = partial_out(df, "y", ["x1", "x2"], ["b", "flat", "enrollment"])
    messages = " ".join(str(w.message) for w in rec)
    assert "flat" in messages and "enrollment" in messages
    assert list(r.coefficients["variable"]) == ["b"]
    assert r.n_dropped == 0  # all-NaN column removed before the row drop


def test_no_usable_brand_columns_raises():
    df = synthetic(n_seasons=4, n_teams=20, theta=THETA, seed=5)
    df["flat"] = 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="no usable brand columns"):
            partial_out(df, "y", ["x1", "x2"], ["flat"])


def test_single_group_raises():
    df = synthetic(n_seasons=1, n_teams=40, theta=THETA, seed=6)
    with pytest.raises(ValueError, match="at least 2 distinct"):
        partial_out(df, "y", ["x1", "x2"], ["b"])


def test_naive_regression_is_biased_but_dml_is_not():
    df = synthetic(n_seasons=8, n_teams=60, theta=THETA, seed=7)
    zb = (df["b"] - df["b"].mean()) / df["b"].std(ddof=0)
    naive = sm.OLS(df["y"], sm.add_constant(zb)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["team"])[0]}
    )
    lo, hi = naive.conf_int(0.05).iloc[1]
    assert not (lo < THETA < hi)
    row = _coef(partial_out(df, "y", ["x1", "x2"], ["b"]), "b")
    assert row.ci_lo < THETA < row.ci_hi


def test_constant_fractional_brand_column_dropped_with_warning():
    df = synthetic(n_seasons=6, n_teams=40, theta=THETA, seed=8)
    df["flat"] = 0.1  # float std of a constant 0.1 column is ~1e-17, not 0
    with pytest.warns(UserWarning, match="flat"):
        r = partial_out(df, "y", ["x1", "x2"], ["b", "flat"])
    assert list(r.coefficients["variable"]) == ["b"]


def test_two_brand_columns_estimated_jointly():
    df = synthetic(n_seasons=8, n_teams=60, theta=THETA, seed=9)
    df = df.rename(columns={"b": "b1"})
    df["b2"] = np.random.default_rng(99).normal(size=len(df))  # independent, theta = 0
    r = partial_out(df, "y", ["x1", "x2"], ["b1", "b2"])
    assert list(r.coefficients["variable"]) == ["b1", "b2"]
    b1, b2 = _coef(r, "b1"), _coef(r, "b2")
    assert b1.ci_lo < THETA < b1.ci_hi
    assert b2.p_value > 0.05


def test_rows_with_missing_cluster_dropped_and_counted():
    df = synthetic(n_seasons=6, n_teams=40, theta=THETA, seed=10)
    df["team"] = df["team"].astype(object)
    df.loc[[3, 4], "team"] = None
    df.loc[[7], "b"] = np.nan
    r = partial_out(df, "y", ["x1", "x2"], ["b"])
    assert r.n_dropped == 3
    assert not r.y_resid.index.isin([3, 4, 7]).any()
    assert r.n_teams == 40

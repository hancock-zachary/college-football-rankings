import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cfb_bias.analysis.team_effects import team_bias

COLS = ["team", "n_seasons", "n_weeks", "mean_resid", "se", "ci_lo", "ci_hi",
        "p_value", "q_value", "significant"]


def _panel(effects, n_seasons, weeks=10, sd=0.05, seed=0):
    """effects: team -> list of per-season shifts (or scalar)."""
    rng = np.random.default_rng(seed)
    rows = []
    for team, eff in effects.items():
        for s in range(n_seasons):
            shift = eff[s] if isinstance(eff, (list, tuple)) else eff
            for _ in range(weeks):
                rows.append((team, 2014 + s, shift + rng.normal(0, sd)))
    df = pd.DataFrame(rows, columns=["team", "season", "r"])
    return df["r"], df["team"], df["season"]


def test_persistent_team_flagged():
    effects = {"A": 0.1, **{f"T{i}": 0.0 for i in range(20)}}
    out = team_bias(*_panel(effects, 6)).set_index("team")
    assert list(out.columns) == COLS[1:]
    assert out.loc["A", "significant"]
    assert out.loc["A", "mean_resid"] > 0.05
    assert out.loc["A", "n_seasons"] == 6 and out.loc["A", "n_weeks"] == 60
    assert out.loc["A", "ci_lo"] > 0


def test_one_big_season_not_persistent():
    effects = {"B": [0.5, 0, 0, 0, 0, 0], **{f"T{i}": 0.0 for i in range(20)}}
    out = team_bias(*_panel(effects, 6)).set_index("team")
    assert not out.loc["B", "significant"]


def test_min_seasons_excludes():
    resid, teams, seasons = _panel({"A": 0.1, "B": 0.1}, 4)
    keep = ~((teams == "B") & (seasons >= 2016))
    out = team_bias(resid[keep], teams[keep], seasons[keep], min_seasons=3)
    assert list(out["team"]) == ["A"]
    out2 = team_bias(resid[keep], teams[keep], seasons[keep], min_seasons=2)
    assert set(out2["team"]) == {"A", "B"}


def test_sorted_and_empty():
    out = team_bias(*_panel({"A": 0.1, "B": -0.1, "C": 0.0}, 5))
    assert list(out["mean_resid"]) == sorted(out["mean_resid"], reverse=True)
    empty = team_bias(pd.Series(dtype=float), pd.Series(dtype=object), pd.Series(dtype=int))
    assert list(empty.columns) == COLS and empty.empty
    few = team_bias(*_panel({"A": 0.1}, 2))
    assert list(few.columns) == COLS and few.empty


def test_zero_variance_team_excluded_from_family():
    resid, teams, seasons = _panel({"A": 0.0, "Z": 0.0, "T": 0.0}, 5)
    resid = resid.copy()
    resid[teams == "Z"] = 0.2  # identical season means
    out = team_bias(resid, teams, seasons).set_index("team")
    assert np.isnan(out.loc["Z", "p_value"]) and np.isnan(out.loc["Z", "q_value"])
    assert not out.loc["Z", "significant"]
    assert out.loc[["A", "T"], "q_value"].notna().all()


def test_bh_controls_false_positives():
    effects = {f"T{i}": 0.0 for i in range(130)}
    out = team_bias(*_panel(effects, 8, weeks=10, sd=0.1, seed=42))
    assert len(out) == 130
    assert out["significant"].sum() <= 6


def test_eligible_limits_output_and_bh_family():
    effects = {"A": 0.1, "Floor": -0.02, **{f"T{i}": 0.0 for i in range(10)}}
    resid, teams, seasons = _panel(effects, 6, sd=0.01)
    full = team_bias(resid, teams, seasons).set_index("team")
    assert full.loc["Floor", "significant"]  # tiny one-signed floor effect is "significant"
    eligible = set(effects) - {"Floor"}
    out = team_bias(resid, teams, seasons, eligible=eligible).set_index("team")
    assert set(out.index) == eligible
    # BH family excludes the ineligible team: q-values are recomputed over 11 tests, not 12
    expected_q = multipletests(out["p_value"], method="fdr_bh")[1]
    np.testing.assert_allclose(out["q_value"], expected_q)
    assert out.loc["A", "significant"]
    # None keeps every team
    assert set(team_bias(resid, teams, seasons, eligible=None)["team"]) == set(effects)

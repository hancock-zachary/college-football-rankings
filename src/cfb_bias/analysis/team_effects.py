"""Per-team persistent bias: do a team's poll standing residuals stay one-signed across seasons?"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

COLUMNS = ["team", "n_seasons", "n_weeks", "mean_resid", "se", "ci_lo", "ci_hi",
           "p_value", "q_value", "significant"]


def team_bias(resid: pd.Series, teams: pd.Series, seasons: pd.Series,
              min_seasons: int = 3, q: float = 0.05,
              eligible: set[str] | None = None) -> pd.DataFrame:
    """Per-team one-sample t-test of team-season mean residuals against 0, BH-adjusted.

    `resid`, `teams`, `seasons` are aligned. Unit of analysis is the team-season
    mean; teams with fewer than `min_seasons` seasons are dropped. Teams whose
    season means are all identical (zero variance) have no defined t statistic:
    their p/q are NaN, they are never significant, and they are excluded from the
    BH family. If `eligible` is given, only those teams are tested and reported
    (the runner passes teams ranked at least once, so never-ranked teams' floor
    residuals stay out of the family). Sorted by mean_resid descending.
    """
    df = pd.DataFrame({"r": resid.to_numpy(), "team": teams.to_numpy(),
                       "season": seasons.to_numpy()}).dropna()
    if eligible is not None:
        df = df[df["team"].isin(eligible)]
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    n_weeks = df.groupby("team").size()
    means = df.groupby(["team", "season"])["r"].mean()
    rows = []
    for team, m in means.groupby(level="team"):
        x = m.to_numpy()
        n = len(x)
        if n < min_seasons:
            continue
        mean = x.mean()
        sd = x.std(ddof=1)
        se = sd / np.sqrt(n)
        if sd > 0:
            p = stats.ttest_1samp(x, 0.0).pvalue
            half = stats.t.ppf(0.975, n - 1) * se
        else:
            p, half = np.nan, 0.0
        rows.append((team, n, int(n_weeks[team]), mean, se, mean - half, mean + half, p))
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.DataFrame(rows, columns=COLUMNS[:8])
    out["q_value"] = np.nan
    ok = out["p_value"].notna()
    if ok.any():
        out.loc[ok, "q_value"] = multipletests(out.loc[ok, "p_value"], method="fdr_bh")[1]
    out["significant"] = (out["q_value"] < q).fillna(False).astype(bool)
    return out.sort_values("mean_resid", ascending=False, ignore_index=True)[COLUMNS]

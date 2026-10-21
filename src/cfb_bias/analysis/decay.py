"""Preseason persistence curve.

For each poll week, OLS of the poll standing on résumé-so-far features plus the
preseason AP score, with team-clustered SEs. The preseason coefficient traces
how quickly the preseason prior fades; `half_life` reads off when it halves.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

REGRESSORS = ["colley", "colley_sos", "wins", "losses", "preseason_ap_score"]
PRESEASON = "preseason_ap_score"
MIN_ROWS = 10
OUT_COLUMNS = ["poll", "poll_week", "coef", "se", "ci_lo", "ci_hi", "n_obs"]


def preseason_decay(df: pd.DataFrame, y: str, poll: str) -> pd.DataFrame:
    """Per-poll-week preseason coefficient. Rows with NaNs are dropped; weeks with
    < 10 rows or < 2 teams, or whose fit yields non-finite estimates, are skipped.

    A singular (collinear) design is not skipped: statsmodels fits it with a
    pseudo-inverse, and the preseason coefficient stays identified as long as the
    preseason score is not itself collinear with the other regressors.
    """
    rows = []
    for week, g in df.groupby("poll_week", sort=True):
        g = g.dropna(subset=[y, *REGRESSORS])
        if len(g) < MIN_ROWS or g["team"].nunique() < 2:
            continue
        X = sm.add_constant(g[REGRESSORS].astype(float), has_constant="add")
        codes = pd.factorize(g["team"])[0]
        try:
            res = sm.OLS(g[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": codes})
            coef, se = float(res.params[PRESEASON]), float(res.bse[PRESEASON])
            lo, hi = (float(v) for v in res.conf_int(0.05).loc[PRESEASON])
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.isfinite([coef, se, lo, hi]).all():
            continue
        rows.append((poll, int(week), coef, se, lo, hi, len(g)))
    return pd.DataFrame(rows, columns=OUT_COLUMNS)


def half_life(decay: pd.DataFrame) -> int | None:
    """First later poll week whose coefficient is <= half the earliest week's; None if
    never, or if the reference coefficient is not positive."""
    if decay.empty:
        return None
    d = decay.sort_values("poll_week")
    ref = float(d["coef"].iloc[0])
    if ref <= 0:
        return None
    hit = d.iloc[1:][d.iloc[1:]["coef"] <= ref / 2]
    return int(hit["poll_week"].iloc[0]) if not hit.empty else None

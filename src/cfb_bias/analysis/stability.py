"""Start week for the main bias test, chosen from Colley rating stability.

Early-season ratings rest on few games. We measure how well the Colley ranking
at each regular ratings week already matches the final (week 99) ranking, and
start the poll test at the first poll week whose underlying ratings are stable.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FINAL_WEEK = 99


def rating_stability(ratings: pd.DataFrame) -> pd.DataFrame:
    """Mean per-season Spearman between Colley at each week and Colley at week 99.

    Returns columns ratings_week, mean_spearman, n_seasons. Week 99 is the
    comparison target, not a candidate. Seasons without week 99 are skipped;
    only teams present in both the week and week 99 are correlated.
    """
    colley = ratings[ratings["method"] == "colley"]
    per_week: dict[int, list[float]] = {}
    for _, season_df in colley.groupby("season"):
        final = season_df[season_df["week"] == FINAL_WEEK].set_index("team")["rating"]
        if final.empty:
            continue
        for week, wk in season_df[season_df["week"] != FINAL_WEEK].groupby("week"):
            wk_r = wk.set_index("team")["rating"]
            common = wk_r.index.intersection(final.index)
            if len(common) < 3:
                continue
            rho = spearmanr(wk_r.loc[common], final.loc[common]).statistic
            if np.isfinite(rho):
                per_week.setdefault(int(week), []).append(float(rho))
    rows = [(w, float(np.mean(v)), len(v)) for w, v in sorted(per_week.items())]
    out = pd.DataFrame(rows, columns=["ratings_week", "mean_spearman", "n_seasons"])
    return out.astype({"ratings_week": "int64", "n_seasons": "int64"})


def choose_start_week(stability: pd.DataFrame, threshold: float = 0.75) -> int:
    """First poll week whose underlying ratings week reaches the threshold.

    Poll week w reflects games through week w-1, so the answer is
    ratings_week + 1 of the earliest stable ratings week.
    """
    stable = stability[stability["mean_spearman"] >= threshold]
    if stable.empty:
        raise ValueError(f"rating stability never reaches {threshold}")
    return int(stable["ratings_week"].min()) + 1

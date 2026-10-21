"""Cross-fitted partialling-out (double ML, partially linear model).

Poll standing y is residualised on résumé features X, each brand variable b
(z-scored) is residualised on X the same way, and ỹ is regressed on all b̃
jointly with team-clustered standard errors. Coefficients are "per 1 SD of b,
net of résumé".
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut

COEF_COLUMNS = ["variable", "coef", "se", "ci_lo", "ci_hi", "p_value"]


@dataclass
class DmlResult:
    coefficients: pd.DataFrame
    y_resid: pd.Series
    n_obs: int
    n_teams: int
    n_dropped: int
    joint_wald: dict | None = None  # {"stat", "df", "p_value"}: all brand coefficients = 0


def make_learner(seed: int) -> HistGradientBoostingRegressor:
    """Nuisance learner for E[y|X] and E[b|X].

    early_stopping is pinned off: sklearn's "auto" turns it on above 10,000 rows,
    which would give large and small analyses different learners.
    """
    return HistGradientBoostingRegressor(early_stopping=False, random_state=seed)


def _as_float(frame: pd.DataFrame | pd.Series) -> np.ndarray:
    return frame.to_numpy(dtype=float, na_value=np.nan)


def _cross_fit(X: np.ndarray, target: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold residual target - E[target|X] with leave-one-group-out folds."""
    pred = np.empty_like(target)
    for train, test in LeaveOneGroupOut().split(X, target, groups):
        model = make_learner(seed).fit(X[train], target[train])
        pred[test] = model.predict(X[test])
    return target - pred


def partial_out(
    df: pd.DataFrame,
    y: str,
    x_cols: list[str],
    b_cols: list[str],
    group_col: str = "season",
    cluster_col: str = "team",
    seed: int = 0,
) -> DmlResult:
    """Effect of each brand variable on y after résumé features X are accounted for."""
    b_cols = list(b_cols)
    empty = [b for b in b_cols if df[b].isna().all()]
    for b in empty:
        warnings.warn(f"brand column {b!r} is entirely missing; dropped", stacklevel=2)
    b_cols = [b for b in b_cols if b not in empty]

    keep = df[[y, *b_cols, cluster_col]].notna().all(axis=1)
    data = df.loc[keep]
    n_dropped = int((~keep).sum())

    constant = [b for b in b_cols if data[b].nunique(dropna=True) <= 1]
    for b in constant:
        warnings.warn(f"brand column {b!r} is constant over the used rows; dropped", stacklevel=2)
    b_cols = [b for b in b_cols if b not in constant]
    if not b_cols:
        raise ValueError("no usable brand columns remain")

    groups = data[group_col].to_numpy()
    if pd.unique(groups).size < 2:
        raise ValueError(f"cross-fitting needs at least 2 distinct {group_col!r} values")

    X = _as_float(data[x_cols])
    B = _as_float(data[b_cols])
    B = (B - B.mean(axis=0)) / B.std(axis=0)

    y_tilde = _cross_fit(X, _as_float(data[y]), groups, seed)
    b_tilde = np.column_stack([_cross_fit(X, B[:, j], groups, seed) for j in range(B.shape[1])])

    clusters = pd.factorize(data[cluster_col])[0]
    fit = sm.OLS(y_tilde, b_tilde).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    ci = np.asarray(fit.conf_int(0.05))
    wald = fit.wald_test(np.eye(len(b_cols)), scalar=True)  # chi2, cluster-robust cov
    joint_wald = {
        "stat": float(wald.statistic),
        "df": int(wald.df_denom),
        "p_value": float(wald.pvalue),
    }
    coefficients = pd.DataFrame(
        {
            "variable": b_cols,
            "coef": np.asarray(fit.params),
            "se": np.asarray(fit.bse),
            "ci_lo": ci[:, 0],
            "ci_hi": ci[:, 1],
            "p_value": np.asarray(fit.pvalues),
        },
        columns=COEF_COLUMNS,
    )
    return DmlResult(
        coefficients=coefficients,
        y_resid=pd.Series(y_tilde, index=data.index, name=f"{y}_resid"),
        n_obs=len(data),
        n_teams=int(pd.unique(clusters).size),
        n_dropped=n_dropped,
        joint_wald=joint_wald,
    )

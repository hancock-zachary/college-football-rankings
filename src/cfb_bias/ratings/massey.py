"""Massey-style margin rating (secondary): does the conclusion change if margin counts?"""

import numpy as np
import pandas as pd

from cfb_bias.ratings.bootstrap import bootstrap_ratings


def massey(
    games: pd.DataFrame, teams: list[str], cap: int = 28, ridge: float = 1.0
) -> pd.Series:
    """Ridge-regularised weighted least-squares margin ratings, centred to mean 0.

    Fits clip(points_a - points_b, -cap, cap) = r_a - r_b + h * a_home with a single
    unpenalised home-field coefficient h (not returned). Ratings are centred over teams
    with positive-weight games; teams without any are exactly 0.
    """
    index = {t: i for i, t in enumerate(teams)}
    for col in ("team_a", "team_b"):
        unknown = sorted(set(games[col]) - index.keys())
        if unknown:
            raise ValueError(f"games reference teams not in `teams`: {unknown}")
    k = len(teams)
    m = len(games)
    a = games["team_a"].map(index).to_numpy(dtype=np.intp)
    b = games["team_b"].map(index).to_numpy(dtype=np.intp)
    w = games["weight"].to_numpy(dtype=float)
    margin = np.clip(
        (games["points_a"] - games["points_b"]).to_numpy(dtype=float), -cap, cap
    )
    home = games["a_home"].to_numpy(dtype=float)

    X = np.zeros((m, k + 1))
    rows = np.arange(m)
    X[rows, a] = 1.0
    X[rows, b] = -1.0
    X[:, k] = home
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = margin * sw

    penalty = np.zeros((k, k + 1))
    penalty[:, :k] = np.sqrt(ridge) * np.eye(k)
    coef, *_ = np.linalg.lstsq(
        np.vstack([Xw, penalty]), np.concatenate([yw, np.zeros(k)]), rcond=None
    )
    r = coef[:k]

    played = np.zeros(k, dtype=bool)
    live = w > 0
    played[a[live]] = True
    played[b[live]] = True
    r = np.where(played, r - r[played].mean(), 0.0) if played.any() else np.zeros(k)
    return pd.Series(r, index=pd.Index(teams, name="team"))


def massey_bootstrap(
    games: pd.DataFrame, teams: list[str], n: int = 500, seed: int = 0, **kw
) -> pd.DataFrame:
    """Massey ratings with 5th/95th percentile bounds from resampling games."""
    return bootstrap_ratings(
        lambda g, t: massey(g, t, **kw), games, teams, n=n, seed=seed
    )

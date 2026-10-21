"""Massey-style margin rating (secondary): does the conclusion change if margin counts?"""

import numpy as np
import pandas as pd

from cfb_bias.ratings.bootstrap import bootstrap_weighted


def _encode(games: pd.DataFrame, teams: list[str], cap: int):
    index = {t: i for i, t in enumerate(teams)}
    for col in ("team_a", "team_b"):
        unknown = sorted(set(games[col]) - index.keys())
        if unknown:
            raise ValueError(f"games reference teams not in `teams`: {unknown}")
    a = games["team_a"].map(index).to_numpy(dtype=np.intp)
    b = games["team_b"].map(index).to_numpy(dtype=np.intp)
    w = games["weight"].to_numpy(dtype=float)
    margin = np.clip(
        (games["points_a"] - games["points_b"]).to_numpy(dtype=float), -cap, cap
    )
    home = games["a_home"].to_numpy(dtype=float)
    return a, b, w, margin, home


def _fit(a, b, w, margin, home, k: int, ridge: float) -> np.ndarray:
    """Solve the weighted ridge normal equations; h is the last (unpenalised) unknown."""
    K = k + 1
    wh = w * home
    # X'WX: teams block, team/home border, home/home corner.
    G = (np.bincount(a * K + a, w, K * K) + np.bincount(b * K + b, w, K * K)).astype(float)
    G -= np.bincount(a * K + b, w, K * K) + np.bincount(b * K + a, w, K * K)
    G += np.bincount(a * K + k, wh, K * K) + np.bincount(k * K + a, wh, K * K)
    G -= np.bincount(b * K + k, wh, K * K) + np.bincount(k * K + b, wh, K * K)
    G = G.reshape(K, K).astype(float)
    G[k, k] += float(np.dot(wh, home))
    rhs = np.zeros(K)
    wm = w * margin
    rhs[:k] = np.bincount(a, wm, k) - np.bincount(b, wm, k)
    rhs[k] = float(np.dot(wh, margin))
    team_diag = np.arange(k)
    G[team_diag, team_diag] += ridge
    try:
        if ridge <= 0:
            raise np.linalg.LinAlgError
        coef = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(G, rhs, rcond=None)[0]
    r = coef[:k]
    played = (np.bincount(a, w, k) + np.bincount(b, w, k)) > 0
    return np.where(played, r - r[played].mean(), 0.0) if played.any() else np.zeros(k)


def massey(
    games: pd.DataFrame, teams: list[str], cap: int = 28, ridge: float = 1.0
) -> pd.Series:
    """Ridge-regularised weighted least-squares margin ratings, centred to mean 0.

    Fits clip(points_a - points_b, -cap, cap) = r_a - r_b + h * a_home with a single
    unpenalised home-field coefficient h (not returned), via the normal equations.
    Ratings are centred over teams with positive-weight games; teams without any are 0.
    """
    a, b, w, margin, home = _encode(games, teams, cap)
    r = _fit(a, b, w, margin, home, len(teams), ridge)
    return pd.Series(r, index=pd.Index(teams, name="team"))


def massey_bootstrap(
    games: pd.DataFrame, teams: list[str], n: int = 500, seed: int = 0,
    cap: int = 28, ridge: float = 1.0,
) -> pd.DataFrame:
    """Massey ratings with 5th/95th percentile bounds from resampling games."""
    a, b, w, margin, home = _encode(games, teams, cap)
    k = len(teams)
    return bootstrap_weighted(
        lambda wt: _fit(a, b, wt, margin, home, k, ridge), a, b, w, teams, n=n, seed=seed
    )

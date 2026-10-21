"""Weighted Colley rating: win/loss only, margin of victory is ignored."""

import numpy as np
import pandas as pd

from cfb_bias.ratings.bootstrap import bootstrap_weighted


def _encode(games: pd.DataFrame, teams: list[str]):
    index = {t: i for i, t in enumerate(teams)}
    for col in ("team_a", "team_b"):
        unknown = sorted(set(games[col]) - index.keys())
        if unknown:
            raise ValueError(f"games reference teams not in `teams`: {unknown}")
    a = games["team_a"].map(index).to_numpy(dtype=np.intp)
    b = games["team_b"].map(index).to_numpy(dtype=np.intp)
    won = games["a_won"].to_numpy(dtype=bool)
    w = games["weight"].to_numpy(dtype=float)
    return a, b, won, w


def _solve(a, b, won, w, n_teams: int) -> np.ndarray:
    k = n_teams
    diag = 2.0 + np.bincount(a, w, k) + np.bincount(b, w, k)
    off = np.bincount(a * k + b, w, k * k) + np.bincount(b * k + a, w, k * k)
    C = -off.reshape(k, k).astype(float)
    C[np.diag_indices(k)] += diag
    signed = np.where(won, w, -w) / 2.0
    rhs = 1.0 + np.bincount(a, signed, k) - np.bincount(b, signed, k)
    return np.linalg.solve(C, rhs)


def colley(games: pd.DataFrame, teams: list[str]) -> pd.Series:
    """Weighted Colley ratings indexed by team. Needs team_a, team_b, a_won, weight."""
    a, b, won, w = _encode(games, teams)
    return pd.Series(_solve(a, b, won, w, len(teams)), index=pd.Index(teams, name="team"))


def colley_bootstrap(
    games: pd.DataFrame, teams: list[str], n: int = 500, seed: int = 0
) -> pd.DataFrame:
    """Colley ratings with 5th/95th percentile bounds from resampling games."""
    a, b, won, w = _encode(games, teams)
    k = len(teams)
    return bootstrap_weighted(
        lambda wt: _solve(a, b, won, wt, k), a, b, w, teams, n=n, seed=seed
    )

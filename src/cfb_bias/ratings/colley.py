"""Weighted Colley rating: win/loss only, margin of victory is ignored."""

import numpy as np
import pandas as pd

from cfb_bias.ratings.bootstrap import bootstrap_ratings


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
    C = 2.0 * np.eye(n_teams)
    np.add.at(C, (a, a), w)
    np.add.at(C, (b, b), w)
    np.add.at(C, (a, b), -w)
    np.add.at(C, (b, a), -w)
    signed = np.where(won, w, -w) / 2.0
    rhs = np.ones(n_teams)
    np.add.at(rhs, a, signed)
    np.add.at(rhs, b, -signed)
    return np.linalg.solve(C, rhs)


def colley(games: pd.DataFrame, teams: list[str]) -> pd.Series:
    """Weighted Colley ratings indexed by team. Needs team_a, team_b, a_won, weight."""
    a, b, won, w = _encode(games, teams)
    return pd.Series(_solve(a, b, won, w, len(teams)), index=pd.Index(teams, name="team"))


def colley_bootstrap(
    games: pd.DataFrame, teams: list[str], n: int = 500, seed: int = 0
) -> pd.DataFrame:
    """Colley ratings with 5th/95th percentile bounds from resampling games."""
    return bootstrap_ratings(colley, games, teams, n=n, seed=seed)

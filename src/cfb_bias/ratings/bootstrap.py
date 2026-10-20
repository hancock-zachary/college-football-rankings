"""Shared game-resampling bootstrap for rating functions."""

from collections.abc import Callable

import numpy as np
import pandas as pd


def bootstrap_ratings(
    rate_fn: Callable[[pd.DataFrame, list[str]], pd.Series],
    games: pd.DataFrame,
    teams: list[str],
    n: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Ratings with 5th/95th percentile bounds from resampling games with replacement.

    `rating` is `rate_fn` on the full sample; `lo`/`hi` come from `n` resamples.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rating = rate_fn(games, teams).reindex(teams).to_numpy(dtype=float)
    m = len(games)
    if m == 0:
        lo = hi = rating
    else:
        rng = np.random.default_rng(seed)
        draws = np.empty((n, len(teams)))
        for s in range(n):
            idx = rng.integers(0, m, size=m)
            sample = games.iloc[idx]
            draws[s] = rate_fn(sample, teams).reindex(teams).to_numpy(dtype=float)
        lo, hi = np.percentile(draws, [5, 95], axis=0)
    out = pd.DataFrame({"team": teams, "rating": rating, "lo": lo, "hi": hi})
    return out.sort_values("team").reset_index(drop=True)

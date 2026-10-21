"""Shared game-resampling bootstrap for rating functions."""

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def bootstrap_weighted(
    fit: Callable[[np.ndarray], np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    teams: list[str],
    n: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Game-resampling bootstrap on encoded arrays via count weights.

    `fit(weights)` returns one rating per team. Resampling game rows with replacement
    is identical to fitting with weights `w * counts`, where `counts` is how often each
    game was drawn. A team with zero total resampled weight gets NaN for that draw
    (it was absent from the resample, not neutral); `lo`/`hi` ignore NaN draws and are
    NaN if a team is absent from every draw. `rating` is the full-sample fit.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    k = len(teams)
    # Hundreds of small (~130x130) solves: multithreaded BLAS is ~300x slower here.
    with threadpool_limits(limits=1, user_api="blas"):
        rating = np.asarray(fit(w), dtype=float)
        m = len(w)
        if m == 0:
            lo = hi = rating
        else:
            rng = np.random.default_rng(seed)
            draws = np.empty((n, k))
            for s in range(n):
                counts = np.bincount(rng.integers(0, m, size=m), minlength=m)
                wc = w * counts
                r = np.asarray(fit(wc), dtype=float)
                present = np.bincount(a, wc, k) + np.bincount(b, wc, k) > 0
                draws[s] = np.where(present, r, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> NaN
                lo, hi = np.nanpercentile(draws, [5, 95], axis=0)
    out = pd.DataFrame({"team": teams, "rating": rating, "lo": lo, "hi": hi})
    return out.sort_values("team").reset_index(drop=True)

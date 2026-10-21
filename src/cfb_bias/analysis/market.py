"""Market rating: team strength implied by consensus betting spreads (perceived quality)."""

import pandas as pd

from cfb_bias.ratings.inputs import POOLED_NON_FBS
from cfb_bias.ratings.massey import massey

WINDOW_WEEKS = 4  # game weeks poll_week-3 .. poll_week
MARKET_CAP = 100  # expected margins are not clipped in practice
MARKET_RIDGE = 1.0


def consensus_spread(lines: pd.DataFrame) -> pd.Series:
    """Median non-null home-perspective spread per game (negative = home favoured)."""
    spread = lines["spread"].astype(float)
    return spread.groupby(lines["game_id"]).median().dropna().rename("spread")


def market_ratings(games: pd.DataFrame, spreads: pd.Series, teams: list[str],
                   poll_week: int) -> pd.DataFrame:
    """Massey fit to expected margins (-spread) over a 4-week window ending at poll_week.

    Non-FBS opponents are pooled as "FCS" (never output); teams with no lined game in
    the window get NaN rating and n_games 0. Output covers `teams` (FBS) only.
    """
    lo = max(1, poll_week - (WINDOW_WEEKS - 1))
    g = games[(games["season_type"] == "regular") & games["week"].between(lo, poll_week)]
    g = g[g["game_id"].isin(spreads.index)]
    home_fbs = g["home_classification"] == "fbs"
    away_fbs = g["away_classification"] == "fbs"
    keep = home_fbs | away_fbs
    g, home_fbs, away_fbs = g[keep], home_fbs[keep], away_fbs[keep]

    neutral = g["neutral_site"].astype("boolean").fillna(False).astype(bool)
    frame = pd.DataFrame({
        "team_a": g["home_team"].where(home_fbs, POOLED_NON_FBS),
        "team_b": g["away_team"].where(away_fbs, POOLED_NON_FBS),
        "points_a": -g["game_id"].map(spreads).astype(float),
        "points_b": 0.0,
        "a_home": ~neutral,
        "weight": 1.0,
    })
    fit_teams = sorted(set(teams) | set(frame["team_a"]) | set(frame["team_b"]) | {POOLED_NON_FBS})
    rating = massey(frame, fit_teams, cap=MARKET_CAP, ridge=MARKET_RIDGE)
    counts = pd.concat([frame["team_a"], frame["team_b"]]).value_counts()

    out = pd.DataFrame({"team": sorted(set(teams) - {POOLED_NON_FBS})})
    out["n_games"] = out["team"].map(counts).fillna(0).astype(int)
    out["market_rating"] = out["team"].map(rating).where(out["n_games"] > 0)
    return out[["team", "market_rating", "n_games"]]

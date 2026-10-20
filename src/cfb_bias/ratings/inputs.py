"""Rating inputs: per-game weights and FBS filtering. Pure functions, poll-free."""

import pandas as pd

POOLED_NON_FBS = "FCS"
_COLUMNS = ["game_id", "week", "team_a", "team_b", "points_a", "points_b",
            "a_won", "neutral_site", "a_home", "weight"]


def _truthy(col: pd.Series) -> pd.Series:
    """Plain bool Series; null (NA/None/NaN) counts as False."""
    return col.astype("boolean").fillna(False).astype(bool)


def game_weight(season: int, season_type: str, is_cfp: bool) -> float:
    """Regular season (incl. conference title games) 1.0; postseason per era."""
    if season_type != "postseason":
        return 1.0
    if season <= 2019:
        return 0.5
    return 0.25 if is_cfp else 0.0


def rating_games(games: pd.DataFrame, season: int, through_week: int | None,
                 include_postseason: bool) -> pd.DataFrame:
    """Weighted, FBS-pooled game list for one season, ready for Colley/Massey."""
    g = games[(games["season"] == season) & _truthy(games["completed"])]
    g = g[g["home_points"].notna() & g["away_points"].notna()]

    is_post = g["season_type"] == "postseason"
    if not include_postseason:
        g = g[~is_post]
        is_post = is_post[g.index]
    if through_week is not None:
        g = g[is_post | (g["week"] <= through_week)]

    home_fbs = g["home_classification"] == "fbs"
    away_fbs = g["away_classification"] == "fbs"
    g = g[home_fbs | away_fbs]
    home_fbs, away_fbs = home_fbs[g.index], away_fbs[g.index]

    points_a = g["home_points"].astype("int64")
    points_b = g["away_points"].astype("int64")
    weight = pd.Series(
        [game_weight(season, st, cfp) for st, cfp in zip(g["season_type"], _truthy(g["is_cfp"]))],
        index=g.index, dtype=float)

    neutral = _truthy(g["neutral_site"])
    out = pd.DataFrame({
        "game_id": g["game_id"],
        "week": g["week"],
        "team_a": g["home_team"].where(home_fbs, POOLED_NON_FBS),
        "team_b": g["away_team"].where(away_fbs, POOLED_NON_FBS),
        "points_a": points_a,
        "points_b": points_b,
        "a_won": (points_a > points_b).astype(bool),
        "neutral_site": neutral,
        "a_home": ~neutral,
        "weight": weight,
    })
    out = out[(points_a != points_b) & (weight > 0)]
    return out[_COLUMNS].reset_index(drop=True)

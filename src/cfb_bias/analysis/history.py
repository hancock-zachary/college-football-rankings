"""Historical-success brand variables: all-time win % and AP national titles."""

import pandas as pd

AP_POLL = "AP Top 25"
_SEASON_TYPE_ORDER = {"regular": 0, "postseason": 1}


def alltime_win_pct(games: pd.DataFrame, season: int, first_season: int = 1936) -> pd.Series:
    """Win % (ties = 1/2) over completed games in [first_season, season-1], by team."""
    g = games[
        games["season"].between(first_season, season - 1)
        & games["completed"].fillna(False).astype(bool)
        & games["home_points"].notna()
        & games["away_points"].notna()
    ]
    hp = g["home_points"].astype(float)
    ap = g["away_points"].astype(float)
    home_score = (hp > ap).astype(float) + 0.5 * (hp == ap)
    long = pd.concat(
        [
            pd.DataFrame({"team": g["home_team"], "score": home_score}),
            pd.DataFrame({"team": g["away_team"], "score": 1.0 - home_score}),
        ]
    )
    return long.groupby("team")["score"].mean().rename("alltime_win_pct")


def ap_titles(polls: pd.DataFrame, season: int, first_season: int = 1936) -> pd.Series:
    """Seasons in [first_season, season-1] each team was #1 in that season's final AP poll."""
    p = polls[
        (polls["poll"] == AP_POLL) & polls["season"].between(first_season, season - 1)
    ].copy()
    p["_type"] = p["season_type"].map(_SEASON_TYPE_ORDER)
    final = p.sort_values(["season", "_type", "week"]).groupby("season")[["_type", "week"]].tail(1)
    last = p.loc[final.index].set_index("season")[["_type", "week"]]
    p = p.merge(last.rename(columns={"_type": "_ft", "week": "_fw"}), on="season")
    champs = p[(p["_type"] == p["_ft"]) & (p["week"] == p["_fw"]) & (p["rank"] == 1)]
    return champs.groupby("team").size().rename("ap_titles")

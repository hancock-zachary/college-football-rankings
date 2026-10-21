"""Weekly poll-free ratings, strength of schedule and conference strength."""

import numpy as np
import pandas as pd

from cfb_bias.ratings.colley import colley_bootstrap
from cfb_bias.ratings.inputs import POOLED_NON_FBS, rating_games
from cfb_bias.ratings.massey import massey_bootstrap

FINAL_WEEK = 99
_METHODS = {"colley": colley_bootstrap, "massey": massey_bootstrap}


def strength_of_schedule(games: pd.DataFrame, ratings: pd.Series) -> pd.Series:
    """Weight-averaged rating of opponents faced; NaN for teams with no games.

    `games` needs team_a, team_b, weight. The FCS pseudo-team counts at its own rating.
    """
    both = pd.concat([
        pd.DataFrame({"team": games["team_a"], "opp": games["team_b"]}),
        pd.DataFrame({"team": games["team_b"], "opp": games["team_a"]}),
    ], ignore_index=True)
    w = np.concatenate([games["weight"].to_numpy(float)] * 2)
    both["w"] = w
    both["wr"] = w * both["opp"].map(ratings).to_numpy(float)
    agg = both.groupby("team")[["wr", "w"]].sum()
    sos = (agg["wr"] / agg["w"]).where(agg["w"] > 0)
    return sos.reindex(ratings.index).rename("sos")


def conference_strength(ratings: pd.Series, team_conf: pd.Series) -> pd.DataFrame:
    """Median team rating and team count per conference."""
    df = pd.DataFrame({"rating": ratings, "conference": team_conf}).dropna(
        subset=["rating", "conference"])
    out = df.groupby("conference")["rating"].agg(median_rating="median", n_teams="size")
    return out.reset_index()


def weekly_ratings(games: pd.DataFrame, teams: list[str], season: int,
                   n_boot: int = 500) -> pd.DataFrame:
    """Ratings through each regular-season week plus week 99 (with weighted postseason)."""
    g = games[(games["season"] == season) & (games["season_type"] != "postseason")]
    g = g[g["completed"].astype("boolean").fillna(False).astype(bool)]
    weeks = sorted(int(w) for w in g["week"].dropna().unique())
    cutoffs = [(w, w, False) for w in weeks] + [(FINAL_WEEK, None, True)]

    frames = []
    for week, through, post in cutoffs:
        rg = rating_games(games, season, through, post)
        for method, fn in _METHODS.items():
            res = fn(rg, teams, n=n_boot, seed=0).set_index("team")
            res["sos"] = strength_of_schedule(rg, res["rating"])
            res = res.drop(index=POOLED_NON_FBS, errors="ignore").reset_index()
            res.insert(0, "season", season)
            res.insert(1, "week", week)
            res.insert(3, "method", method)
            frames.append(res)
    cols = ["season", "week", "team", "method", "rating", "lo", "hi", "sos"]
    return pd.concat(frames, ignore_index=True)[cols]

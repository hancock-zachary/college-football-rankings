"""Team-week features: poll targets, résumé features (poll- and margin-free), exposure."""

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from cfb_bias.db import read_sql

AP, COACHES, CFP = "AP Top 25", "Coaches Poll", "Playoff Committee Rankings"
NATIONAL_OUTLETS = ("ABC", "CBS", "NBC", "FOX", "ESPN")
QUALITY_RANK, BAD_RANK = 25, 60
PRESEASON_COLLEY, PRESEASON_MASSEY = 0.5, 0.0
FEATURE_COLUMNS = [
    "season", "poll_week", "team", "ratings_week", "ap_score", "ap_rank_score",
    "coaches_rank_score", "cfp_rank_score", "wins", "losses", "games_played", "colley",
    "colley_sos", "massey", "massey_sos", "quality_wins", "bad_losses", "won_last",
    "market_rating", "national_tv_share",
]


def asof_ratings_week(poll_weeks, ratings_weeks) -> dict[int, int | None]:
    """Latest ratings week <= poll_week - 1 for each poll week (None if none exists)."""
    available = sorted(ratings_weeks)
    return {
        pw: max((w for w in available if w <= pw - 1), default=None) for pw in poll_weeks
    }


def poll_scores(polls: pd.DataFrame, poll_name: str, use_points: bool) -> pd.DataFrame:
    """Score of every ranked team per poll week: points / max points, or (26-rank)/25."""
    p = polls[polls["poll"] == poll_name]
    if use_points:
        pts = p["points"].astype(float)
        top = pts.groupby(p["week"]).transform("max").where(lambda s: s > 0)
        score = pts / top
    else:
        score = (26 - p["rank"].astype(float)) / 25
    return pd.DataFrame(
        {"poll_week": p["week"], "team": p["team"], "score": score}
    ).reset_index(drop=True)


def add_poll_score(frame: pd.DataFrame, polls: pd.DataFrame, poll_name: str,
                   use_points: bool, column: str, null_if_absent: bool) -> pd.DataFrame:
    """Left-join poll scores; unranked -> 0, or NaN for weeks the poll was not held."""
    s = poll_scores(polls, poll_name, use_points).rename(columns={"score": column})
    out = frame.merge(s, on=["poll_week", "team"], how="left")
    held = out["poll_week"].isin(set(polls.loc[polls["poll"] == poll_name, "week"]))
    out[column] = out[column].fillna(0.0)
    if null_if_absent:
        out[column] = out[column].where(held, np.nan)  # poll not held that week -> NULL
    return out


def team_results(games: pd.DataFrame, national_ids: set) -> pd.DataFrame:
    """Long form (one row per team per completed regular-season game with scores)."""
    g = games[(games["season_type"] == "regular") & games["completed"].astype("boolean").fillna(False)
              & games["home_points"].notna() & games["away_points"].notna()]
    home = pd.DataFrame({
        "team": g["home_team"], "opp": g["away_team"], "opp_fbs": g["away_classification"] == "fbs",
        "week": g["week"], "won": g["home_points"] > g["away_points"],
        "lost": g["home_points"] < g["away_points"], "game_id": g["game_id"],
    })
    away = pd.DataFrame({
        "team": g["away_team"], "opp": g["home_team"], "opp_fbs": g["home_classification"] == "fbs",
        "week": g["week"], "won": g["away_points"] > g["home_points"],
        "lost": g["away_points"] < g["home_points"], "game_id": g["game_id"],
    })
    out = pd.concat([home, away], ignore_index=True)
    out["national"] = out["game_id"].isin(national_ids)
    return out.sort_values(["team", "week"], kind="stable").reset_index(drop=True)


def resume_through(results: pd.DataFrame, through_week: int,
                   colley_rank: pd.Series) -> pd.DataFrame:
    """Per-team résumé counts over games with week <= through_week (index = team)."""
    r = results[results["week"] <= through_week].copy()
    opp_rank = r["opp"].map(colley_rank)
    r["quality"] = r["won"] & r["opp_fbs"] & (opp_rank <= QUALITY_RANK)
    r["bad"] = r["lost"] & (~r["opp_fbs"] | (opp_rank > BAD_RANK))
    by = r.groupby("team")
    return pd.DataFrame({
        "wins": by["won"].sum(), "losses": by["lost"].sum(), "games_played": by.size(),
        "quality_wins": by["quality"].sum(), "bad_losses": by["bad"].sum(),
        "won_last": by["won"].last(), "national_tv_share": by["national"].mean(),
    })


def build_features(engine: Engine, season: int) -> pd.DataFrame:
    """`team_week_features` rows: every FBS team x each regular-season AP poll week."""
    params = {"season": season}
    teams = read_sql(
        engine,
        "SELECT team FROM team_seasons WHERE season = :season AND classification = 'fbs'",
        params,
    )["team"]
    polls = read_sql(
        engine,
        'SELECT week, poll, team, "rank", points FROM poll_ranks '
        "WHERE season = :season AND season_type = 'regular'",
        params,
    )
    poll_weeks = sorted(set(polls.loc[polls["poll"] == AP, "week"]))
    games = read_sql(engine, "SELECT * FROM games WHERE season = :season", params)
    ratings = read_sql(
        engine,
        "SELECT week, team, method, rating, sos FROM ratings_weekly WHERE season = :season",
        params,
    )
    market = read_sql(
        engine,
        "SELECT week AS poll_week, team, market_rating FROM market_weekly "
        "WHERE season = :season",
        params,
    )
    media = read_sql(
        engine,
        "SELECT DISTINCT game_id FROM game_media WHERE season = :season AND outlet = ANY(:o)",
        {**params, "o": list(NATIONAL_OUTLETS)},
    )
    if ratings.empty and any(pw > 1 for pw in poll_weeks):
        raise ValueError(
            f"season {season} has AP poll weeks after the preseason but no ratings_weekly "
            f"rows; run `cfb ratings --season {season}` first"
        )
    results = team_results(games, set(media["game_id"]))
    asof = asof_ratings_week(poll_weeks, set(ratings["week"]))
    fbs = set(teams)

    frames = []
    for pw in poll_weeks:
        rw = asof[pw]
        base = pd.DataFrame({"team": sorted(fbs)}).set_index("team")
        if rw is None:
            colley_rank = pd.Series(dtype=float)
            for col, val in (("colley", PRESEASON_COLLEY if pw <= 1 else np.nan),
                             ("massey", PRESEASON_MASSEY if pw <= 1 else np.nan)):
                base[col] = val
            base["colley_sos"] = base["massey_sos"] = np.nan
        else:
            at = ratings[ratings["week"] == rw]
            by_method = {m: at[at["method"] == m].set_index("team") for m in ("colley", "massey")}
            for m, d in by_method.items():
                base[m] = d["rating"].reindex(base.index)
                base[f"{m}_sos"] = d["sos"].reindex(base.index)
            fbs_colley = by_method["colley"]["rating"].reindex(base.index).dropna()
            colley_rank = fbs_colley.rank(ascending=False, method="min")
        res = resume_through(results, pw - 1, colley_rank).reindex(base.index)
        base = base.join(res)
        base["ratings_week"] = rw
        base["poll_week"] = pw
        frames.append(base.reset_index())
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=FEATURE_COLUMNS)
    for col in ("wins", "losses", "games_played", "quality_wins", "bad_losses"):
        out[col] = out[col].fillna(0).astype(int)
    out["won_last"] = out["won_last"].astype("boolean")
    out["ratings_week"] = out["ratings_week"].astype("Int64")
    out["season"] = season
    out = add_poll_score(out, polls, AP, True, "ap_score", False)
    out = add_poll_score(out, polls, AP, False, "ap_rank_score", False)
    out = add_poll_score(out, polls, COACHES, False, "coaches_rank_score", True)
    out = add_poll_score(out, polls, CFP, False, "cfp_rank_score", True)
    out = out.merge(market, on=["poll_week", "team"], how="left")
    return out[FEATURE_COLUMNS]

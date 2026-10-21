"""Team brand / perceived-quality variables (one row per FBS team-season)."""

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_bias.analysis.history import AP_POLL, alltime_win_pct, ap_titles
from cfb_bias.db import read_sql

FIRST_SEASON = 1936
CONFERENCE_COLUMNS = {
    "sec": "SEC",
    "big_ten": "Big Ten",
    "big_12": "Big 12",
    "acc": "ACC",
    "pac_12": "Pac-12",
}
BRAND_COLUMNS = [
    "season", "team", "conference", "alltime_win_pct", "ap_titles", "prior_colley",
    "preseason_ap_score", "recruiting_4yr", "talent", "enrollment",
]


def conference_dummies(conf: pd.Series) -> pd.DataFrame:
    """0/1 columns for the five power conferences; everyone else is the baseline."""
    return pd.DataFrame(
        {col: (conf == name).astype(int) for col, name in CONFERENCE_COLUMNS.items()},
        index=conf.index,
    )


def build_team_brand(engine: Engine, season: int) -> pd.DataFrame:
    """`team_brand` rows for every FBS team in team_seasons for `season`."""
    teams = read_sql(
        engine,
        "SELECT team, conference FROM team_seasons "
        "WHERE season = :season AND classification = 'fbs'",
        {"season": season},
    )
    out = teams.copy()
    out.insert(0, "season", season)
    t = out["team"]

    games = read_sql(
        engine,
        "SELECT season, completed, home_team, away_team, home_points, away_points "
        "FROM games WHERE completed AND season BETWEEN :first AND :last",
        {"first": FIRST_SEASON, "last": season - 1},
    )
    out["alltime_win_pct"] = t.map(alltime_win_pct(games, season, FIRST_SEASON)).astype(float)

    polls = read_sql(
        engine,
        'SELECT season, season_type, week, poll, team, "rank" FROM poll_ranks '
        "WHERE poll = :poll AND season BETWEEN :first AND :last",
        {"poll": AP_POLL, "first": FIRST_SEASON, "last": season - 1},
    )
    out["ap_titles"] = t.map(ap_titles(polls, season, FIRST_SEASON)).fillna(0).astype(int)

    colley = read_sql(
        engine,
        "SELECT team, rating FROM ratings_weekly "
        "WHERE season = :prev AND week = 99 AND method = 'colley'",
        {"prev": season - 1},
    ).set_index("team")["rating"]
    out["prior_colley"] = t.map(colley).astype(float)

    pre = read_sql(
        engine,
        "SELECT team, points FROM poll_ranks WHERE season = :season AND poll = :poll "
        "AND season_type = 'regular' AND week = 1",
        {"season": season, "poll": AP_POLL},
    )
    top = pre["points"].max() if not pre.empty else float("nan")
    share = (pre.set_index("team")["points"] / top) if top and top > 0 else pd.Series(dtype=float)
    out["preseason_ap_score"] = t.map(share).fillna(0.0).astype(float)

    rec = read_sql(
        engine,
        "SELECT team, AVG(points) AS points FROM team_recruiting "
        "WHERE season BETWEEN :lo AND :season GROUP BY team",
        {"lo": season - 3, "season": season},
    ).set_index("team")["points"]
    out["recruiting_4yr"] = t.map(rec).astype(float)

    talent = read_sql(
        engine, "SELECT team, talent FROM team_talent WHERE season = :season", {"season": season}
    ).set_index("team")["talent"]
    out["talent"] = t.map(talent).astype(float)

    enroll = read_sql(
        engine, "SELECT team, undergrad_enrollment FROM school_enrollment"
    ).set_index("team")["undergrad_enrollment"]
    out["enrollment"] = t.map(enroll).astype(float)

    return out[BRAND_COLUMNS].sort_values("team").reset_index(drop=True)

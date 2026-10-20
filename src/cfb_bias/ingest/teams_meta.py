"""Ingest team-level metadata: recruiting, talent, coaches, venues."""

import pandas as pd
from sqlalchemy import Engine

from cfb_bias.cfbd import CfbdClient
from cfb_bias.db import upsert

RECRUITING_COLUMNS = ["season", "team", "rank", "points"]
TALENT_COLUMNS = ["season", "team", "talent"]
COACH_COLUMNS = [
    "season", "team", "coach", "games", "wins", "losses", "preseason_rank", "postseason_rank",
]
VENUE_COLUMNS = [
    "venue_id", "name", "city", "state", "capacity", "latitude", "longitude",
    "dome", "grass", "construction_year",
]
# column -> CFBD field
_RECRUITING_FIELDS = {"season": "year", "team": "team", "rank": "rank", "points": "points"}
_TALENT_FIELDS = {"season": "year", "team": "team", "talent": "talent"}
_COACH_SEASON_FIELDS = {
    "season": "year", "team": "school", "games": "games", "wins": "wins", "losses": "losses",
    "preseason_rank": "preseasonRank", "postseason_rank": "postseasonRank",
}
_VENUE_FIELDS = {
    "venue_id": "id", "name": "name", "city": "city", "state": "state",
    "capacity": "capacity", "latitude": "latitude", "longitude": "longitude",
    "dome": "dome", "grass": "grass", "construction_year": "constructionYear",
}


def _frame(rows: list[dict], columns: list[str], ints=(), floats=(), bools=()) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    for col in ints:
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    for col in floats:
        df[col] = pd.to_numeric(df[col]).astype("Float64")
    for col in bools:
        df[col] = df[col].astype("boolean")
    return df


def recruiting_to_frame(records: list[dict]) -> pd.DataFrame:
    rows = [{c: r.get(s) for c, s in _RECRUITING_FIELDS.items()} for r in records]
    return _frame(rows, RECRUITING_COLUMNS, ints=["season", "rank"], floats=["points"])


def talent_to_frame(records: list[dict]) -> pd.DataFrame:
    rows = [{c: r.get(s) for c, s in _TALENT_FIELDS.items()} for r in records]
    return _frame(rows, TALENT_COLUMNS, ints=["season"], floats=["talent"])


def coaches_to_frame(records: list[dict]) -> pd.DataFrame:
    """One row per (season, team, coach). Ties are not stored; a coach with several stints
    at the same team in a season gets one row with summed games/wins/losses (ranks: first)."""
    rows = [
        {"coach": f"{c.get('firstName')} {c.get('lastName')}"}
        | {col: s.get(src) for col, src in _COACH_SEASON_FIELDS.items()}
        for c in records
        for s in c.get("seasons") or []
    ]
    int_cols = ["season", "games", "wins", "losses", "preseason_rank", "postseason_rank"]
    df = _frame(rows, COACH_COLUMNS, ints=int_cols)
    grouped = df.groupby(["season", "team", "coach"], sort=False)
    out = grouped.agg(
        games=("games", lambda s: s.sum(min_count=1)),
        wins=("wins", lambda s: s.sum(min_count=1)),
        losses=("losses", lambda s: s.sum(min_count=1)),
        preseason_rank=("preseason_rank", "first"),
        postseason_rank=("postseason_rank", "first"),
    ).reset_index()
    return out[COACH_COLUMNS].astype({c: "Int64" for c in int_cols})


def venues_to_frame(records: list[dict]) -> pd.DataFrame:
    rows = [{c: r.get(s) for c, s in _VENUE_FIELDS.items()} for r in records]
    return _frame(
        rows, VENUE_COLUMNS,
        ints=["venue_id", "capacity", "construction_year"],
        floats=["latitude", "longitude"], bools=["dome", "grass"],
    )


def pull_recruiting(client: CfbdClient, engine: Engine, season: int, refresh: bool = False) -> int:
    df = recruiting_to_frame(client.get("/recruiting/teams", {"year": season}, refresh=refresh))
    return upsert(engine, "team_recruiting", df, ["season", "team"])


def pull_talent(client: CfbdClient, engine: Engine, season: int, refresh: bool = False) -> int:
    df = talent_to_frame(client.get("/talent", {"year": season}, refresh=refresh))
    return upsert(engine, "team_talent", df, ["season", "team"])


def pull_coaches(client: CfbdClient, engine: Engine, season: int, refresh: bool = False) -> int:
    df = coaches_to_frame(client.get("/coaches", {"year": season}, refresh=refresh))
    return upsert(engine, "coach_seasons", df, ["season", "team", "coach"])


def pull_venues(client: CfbdClient, engine: Engine, refresh: bool = False) -> int:
    """Venues are not per-season; fetches the full list."""
    df = venues_to_frame(client.get("/venues", refresh=refresh))
    return upsert(engine, "venues", df, ["venue_id"])

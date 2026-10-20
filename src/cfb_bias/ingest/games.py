"""Ingest CFBD games and FBS team-seasons into Postgres."""

import re

import pandas as pd
from sqlalchemy import Engine

from cfb_bias.cfbd import CfbdClient, CfbdError
from cfb_bias.db import upsert

GAME_COLUMNS = [
    "game_id", "season", "week", "season_type", "start_date", "completed",
    "neutral_site", "conference_game", "home_team", "home_conference",
    "home_classification", "home_points", "away_team", "away_conference",
    "away_classification", "away_points", "venue_id", "notes", "is_cfp",
]
TEAM_COLUMNS = ["season", "team", "conference", "classification"]

# game column -> CFBD field
_GAME_FIELDS = {
    "game_id": "id", "season": "season", "week": "week", "season_type": "seasonType",
    "start_date": "startDate", "completed": "completed", "neutral_site": "neutralSite",
    "conference_game": "conferenceGame", "home_team": "homeTeam",
    "home_conference": "homeConference", "home_classification": "homeClassification",
    "home_points": "homePoints", "away_team": "awayTeam",
    "away_conference": "awayConference", "away_classification": "awayClassification",
    "away_points": "awayPoints", "venue_id": "venueId", "notes": "notes",
}
_INT_COLS = ["game_id", "season", "week", "home_points", "away_points", "venue_id"]
_BOOL_COLS = ["completed", "neutral_site", "conference_game"]
_CFP_NOTES = re.compile(r"college football playoff|\bcfp\b", re.IGNORECASE)


def _is_cfp(record: dict) -> bool:
    return bool(record.get("playoff")) or bool(_CFP_NOTES.search(record.get("notes") or ""))


def games_to_frame(records: list[dict]) -> pd.DataFrame:
    rows = [
        {col: rec.get(src) for col, src in _GAME_FIELDS.items()} | {"is_cfp": _is_cfp(rec)}
        for rec in records
    ]
    df = pd.DataFrame(rows, columns=GAME_COLUMNS)
    for col in _INT_COLS:
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    for col in _BOOL_COLS + ["is_cfp"]:
        df[col] = df[col].astype("boolean")
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True)
    return df


def teams_to_frame(records: list[dict], season: int) -> pd.DataFrame:
    rows = [
        {
            "season": season,
            "team": r.get("school"),
            "conference": r.get("conference"),
            "classification": r.get("classification"),
        }
        for r in records
    ]
    df = pd.DataFrame(rows, columns=TEAM_COLUMNS)
    df["season"] = df["season"].astype("Int64")
    return df


def _fetch_games(client: CfbdClient, season: int, refresh: bool) -> list[dict]:
    try:
        return client.get("/games", {"year": season, "seasonType": "both"}, refresh=refresh)
    except CfbdError:
        regular = client.get("/games", {"year": season, "seasonType": "regular"}, refresh=refresh)
        post = client.get("/games", {"year": season, "seasonType": "postseason"}, refresh=refresh)
        return regular + post


def pull_games(client: CfbdClient, engine: Engine, season: int, refresh: bool = False) -> int:
    """Fetch and upsert games and FBS team-seasons. Returns total rows written."""
    games = games_to_frame(_fetch_games(client, season, refresh))
    teams = teams_to_frame(client.get("/teams/fbs", {"year": season}, refresh=refresh), season)
    return upsert(engine, "games", games, ["game_id"]) + upsert(
        engine, "team_seasons", teams, ["season", "team"]
    )

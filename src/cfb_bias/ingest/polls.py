"""Ingest CFBD poll rankings into Postgres."""

import pandas as pd
from sqlalchemy import Engine

from cfb_bias.cfbd import CfbdClient
from cfb_bias.db import upsert
from cfb_bias.ingest.fetch import get_season_types

POLL_COLUMNS = [
    "season", "season_type", "week", "poll", "is_final", "rank", "team",
    "conference", "first_place_votes", "points",
]
# column -> field on each entry of PollWeek.polls[].ranks
_RANK_FIELDS = {
    "rank": "rank", "team": "school", "conference": "conference",
    "first_place_votes": "firstPlaceVotes", "points": "points",
}
_INT_COLS = ["season", "week", "rank", "first_place_votes"]


def polls_to_frame(records: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "season": week.get("season"), "season_type": week.get("seasonType"),
            "week": week.get("week"), "poll": poll.get("poll"), "is_final": poll.get("isFinal"),
        }
        | {col: rank.get(src) for col, src in _RANK_FIELDS.items()}
        for week in records
        for poll in week.get("polls") or []
        for rank in poll.get("ranks") or []
    ]
    df = pd.DataFrame(rows, columns=POLL_COLUMNS)
    for col in _INT_COLS:
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    df["is_final"] = df["is_final"].astype("boolean")
    df["points"] = pd.to_numeric(df["points"]).astype("Float64")
    return df


def pull_polls(client: CfbdClient, engine: Engine, season: int, refresh: bool = False) -> int:
    df = polls_to_frame(get_season_types(client, "/rankings", season, refresh))
    return upsert(engine, "poll_ranks", df, ["season", "season_type", "week", "poll", "team"])

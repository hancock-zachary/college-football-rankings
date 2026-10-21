"""Ingest CFBD per-game TV/media assignments into Postgres."""

import pandas as pd
from sqlalchemy import Engine

from cfb_bias.cfbd import CfbdClient
from cfb_bias.db import upsert
from cfb_bias.ingest.fetch import get_season_types

MEDIA_COLUMNS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team",
    "media_type", "outlet",
]
# column -> field on each /games/media record
_FIELDS = {
    "game_id": "id", "season": "season", "week": "week", "season_type": "seasonType",
    "home_team": "homeTeam", "away_team": "awayTeam", "media_type": "mediaType",
    "outlet": "outlet",
}
_INT_COLS = ["game_id", "season", "week"]


def media_to_frame(records: list[dict]) -> pd.DataFrame:
    rows = [{col: rec.get(src) for col, src in _FIELDS.items()} for rec in records]
    df = pd.DataFrame(rows, columns=MEDIA_COLUMNS)
    for col in _INT_COLS:
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    df = df.dropna(subset=["game_id", "outlet"])
    return df.drop_duplicates(["game_id", "outlet"]).reset_index(drop=True)


def pull_media(client: CfbdClient, engine: Engine, season: int, refresh: bool = False) -> int:
    df = media_to_frame(get_season_types(client, "/games/media", season, refresh))
    return upsert(engine, "game_media", df, ["game_id", "outlet"])

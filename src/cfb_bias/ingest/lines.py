"""Ingest CFBD betting lines (one row per game and provider) into Postgres."""

import pandas as pd
from sqlalchemy import Engine

from cfb_bias.cfbd import CfbdClient
from cfb_bias.db import upsert
from cfb_bias.ingest.fetch import get_season_types

LINE_COLUMNS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team", "provider",
    "spread", "spread_open", "over_under", "home_moneyline", "away_moneyline",
]
# column -> field on the BettingGame record
_GAME_FIELDS = {
    "game_id": "id", "season": "season", "week": "week", "season_type": "seasonType",
    "home_team": "homeTeam", "away_team": "awayTeam",
}
# column -> field on each entry of BettingGame.lines
_LINE_FIELDS = {
    "provider": "provider", "spread": "spread", "spread_open": "spreadOpen",
    "over_under": "overUnder", "home_moneyline": "homeMoneyline",
    "away_moneyline": "awayMoneyline",
}
_INT_COLS = ["game_id", "season", "week"]
_FLOAT_COLS = ["spread", "spread_open", "over_under", "home_moneyline", "away_moneyline"]


def lines_to_frame(records: list[dict]) -> pd.DataFrame:
    """Spread is stored as CFBD reports it (home team's perspective)."""
    rows = [
        {col: game.get(src) for col, src in _GAME_FIELDS.items()}
        | {col: line.get(src) for col, src in _LINE_FIELDS.items()}
        for game in records
        for line in game.get("lines") or []
    ]
    df = pd.DataFrame(rows, columns=LINE_COLUMNS)
    for col in _INT_COLS:
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    for col in _FLOAT_COLS:
        df[col] = pd.to_numeric(df[col]).astype("Float64")
    return df.drop_duplicates(["game_id", "provider"], keep="last").reset_index(drop=True)


def pull_lines(client: CfbdClient, engine: Engine, season: int, refresh: bool = False) -> int:
    df = lines_to_frame(get_season_types(client, "/lines", season, refresh))
    return upsert(engine, "betting_lines", df, ["game_id", "provider"])

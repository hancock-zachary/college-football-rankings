import json
from pathlib import Path

import pytest

from cfb_bias.db import init_schema, read_sql
from cfb_bias.ingest.media import MEDIA_COLUMNS, media_to_frame
from cfb_bias.ingest.runner import parse_only

FIXTURES = Path(__file__).parent / "fixtures"


def records():
    return json.loads((FIXTURES / "media_2023_sample.json").read_text(encoding="utf-8"))


def test_media_to_frame_columns():
    df = media_to_frame(records())
    assert list(df.columns) == MEDIA_COLUMNS == [
        "game_id", "season", "week", "season_type", "home_team", "away_team",
        "media_type", "outlet",
    ]
    assert df.loc[df.outlet == "FOX"].iloc[0].game_id == 401525555


def test_media_drops_null_outlet():
    df = media_to_frame(records())
    assert df.outlet.notna().all()
    assert 401525556 not in set(df.game_id)


def test_media_dedupes_game_outlet():
    df = media_to_frame(records())
    assert not df.duplicated(["game_id", "outlet"]).any()
    assert len(df) == 3  # ESPN (deduped), ESPN+, FOX


def test_media_empty():
    assert list(media_to_frame([]).columns) == MEDIA_COLUMNS


def test_parse_only_accepts_media():
    assert parse_only("media") == ["media"]


@pytest.mark.db
def test_new_tables_exist(pg_engine):
    init_schema(pg_engine)
    for table in [
        "game_media", "team_brand", "market_weekly", "team_week_features",
        "school_enrollment", "analysis_params", "bias_coefficients", "team_bias",
        "preseason_decay",
    ]:
        assert len(read_sql(pg_engine, f"SELECT * FROM {table}")) == 0

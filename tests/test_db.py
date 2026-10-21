import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from cfb_bias.db import init_schema, read_sql, upsert

pytestmark = pytest.mark.db


@pytest.fixture
def clean(pg_engine):
    with pg_engine.begin() as conn:
        conn.execute(text("TRUNCATE team_recruiting, venues, games"))
    return pg_engine


def test_upsert_is_idempotent(clean):
    df = pd.DataFrame({"season": [2023], "team": ["Ohio State"], "rank": [4], "points": [300.1]})
    upsert(clean, "team_recruiting", df, ["season", "team"])
    df.loc[0, "rank"] = 3
    upsert(clean, "team_recruiting", df, ["season", "team"])
    out = read_sql(clean, "select * from team_recruiting")
    assert len(out) == 1 and out.loc[0, "rank"] == 3


def test_init_schema_twice_is_safe(pg_engine):
    init_schema(pg_engine)
    init_schema(pg_engine)


def test_upsert_empty_dataframe_returns_zero(clean):
    empty = pd.DataFrame({"season": [], "team": [], "rank": [], "points": []})
    assert upsert(clean, "team_recruiting", empty, ["season", "team"]) == 0
    assert len(read_sql(clean, "select * from team_recruiting")) == 0


def test_upsert_writes_nan_and_nat_as_null(clean):
    df = pd.DataFrame(
        {
            "game_id": [1],
            "season": [2023],
            "start_date": [pd.NaT],
            "home_points": [np.nan],
            "home_team": [None],
            "completed": [True],
        }
    )
    assert upsert(clean, "games", df, ["game_id"]) == 1
    out = read_sql(clean, "select * from games where game_id = 1")
    assert out.loc[0, "start_date"] is pd.NaT or pd.isna(out.loc[0, "start_date"])
    assert pd.isna(out.loc[0, "home_points"]) and out.loc[0, "home_team"] is None


def test_upsert_returns_row_count_and_params_in_read_sql(clean):
    df = pd.DataFrame({"venue_id": [1, 2], "name": ["A", "B"], "capacity": [100, 200]})
    assert upsert(clean, "venues", df, ["venue_id"]) == 2
    out = read_sql(clean, "select name from venues where venue_id = :v", {"v": 2})
    assert out["name"].tolist() == ["B"]


def test_upsert_joins_caller_transaction(clean):
    df = pd.DataFrame({"season": [2023], "team": ["Rollback U"], "rank": [1], "points": [1.0]})
    with pytest.raises(RuntimeError), clean.begin() as conn:
        assert upsert(clean, "team_recruiting", df, ["season", "team"], conn=conn) == 1
        raise RuntimeError("abort")
    assert len(read_sql(clean, "select * from team_recruiting")) == 0

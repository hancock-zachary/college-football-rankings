import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from cfb_bias.cfbd import CfbdClient
from cfb_bias.db import read_sql
from cfb_bias.ingest.games import (
    GAME_COLUMNS,
    games_to_frame,
    pull_games,
    teams_to_frame,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_NEUTRAL_ID = 401520171
FIXTURE_INCOMPLETE_ID = 401540001
FIXTURE_CFP_ID = 401525260
FIXTURE_BOWL_ID = 401549999
FIXTURE_FCS_ID = 401525477


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_games_to_frame_maps_fields():
    df = games_to_frame(load_fixture("games_2023_sample.json"))
    assert list(df.columns) == GAME_COLUMNS
    row = df.set_index("game_id").loc[FIXTURE_NEUTRAL_ID]
    assert row.neutral_site and row.season_type == "regular"


def test_classification_comes_from_game_record():
    recs = [{"id": 1, "season": 2023, "homeTeam": "Sam Houston", "homeClassification": "fbs"},
            {"id": 2, "season": 2022, "homeTeam": "Sam Houston", "homeClassification": "fcs"}]
    df = games_to_frame(recs).set_index("game_id")
    assert df.loc[1, "home_classification"] == "fbs" and df.loc[2, "home_classification"] == "fcs"


def test_fcs_opponent_kept():
    df = games_to_frame(load_fixture("games_2023_sample.json")).set_index("game_id")
    assert df.loc[FIXTURE_FCS_ID, "away_classification"] == "fcs"


def test_cfp_flag_from_notes():
    df = games_to_frame(load_fixture("games_2023_sample.json")).set_index("game_id")
    assert df.loc[FIXTURE_CFP_ID, "is_cfp"]
    assert not df.loc[FIXTURE_NEUTRAL_ID, "is_cfp"]
    assert not df.loc[FIXTURE_BOWL_ID, "is_cfp"]


def test_cfp_flag_from_playoff_field_and_cfp_token():
    df = games_to_frame([
        {"id": 1, "playoff": True},
        {"id": 2, "notes": "CFP National Championship"},
        {"id": 3, "notes": "cfpx bowl"},
    ]).set_index("game_id")
    assert df.loc[1, "is_cfp"] and df.loc[2, "is_cfp"] and not df.loc[3, "is_cfp"]


def test_incomplete_game_kept_with_null_points():
    df = games_to_frame(load_fixture("games_2023_sample.json")).set_index("game_id")
    assert not df.loc[FIXTURE_INCOMPLETE_ID, "completed"]
    assert pd.isna(df.loc[FIXTURE_INCOMPLETE_ID, "home_points"])


def test_missing_keys_become_null_and_empty_input_ok():
    df = games_to_frame([{"id": 9}])
    assert list(df.columns) == GAME_COLUMNS
    assert pd.isna(df.loc[0, "week"]) and pd.isna(df.loc[0, "neutral_site"])
    assert games_to_frame([]).empty


def test_teams_to_frame():
    df = teams_to_frame(load_fixture("teams_fbs_2023_sample.json"), 2023)
    assert list(df.columns) == ["season", "team", "conference", "classification"]
    assert set(df.season) == {2023} and "Michigan" in set(df.team)


def _client(tmp_path, handler):
    return CfbdClient("k", tmp_path, httpx.MockTransport(handler), min_interval=0)


def _handler(calls, reject_both=False):
    games = load_fixture("games_2023_sample.json")
    teams = load_fixture("teams_fbs_2023_sample.json")

    def handler(req):
        calls.append(dict(req.url.params) | {"path": req.url.path})
        if req.url.path == "/teams/fbs":
            return httpx.Response(200, json=teams)
        st = req.url.params["seasonType"]
        if st == "both":
            return httpx.Response(400, json={}) if reject_both else httpx.Response(200, json=games)
        return httpx.Response(200, json=[g for g in games if g["seasonType"] == st])

    return handler


def _count(engine, sql):
    return int(read_sql(engine, sql).n[0])


@pytest.mark.db
def test_pull_games_writes_and_is_idempotent(tmp_path, pg_engine):
    client = _client(tmp_path, _handler([]))
    pull_games(client, pg_engine, 2023)
    n_games = _count(pg_engine, "SELECT count(*) n FROM games")
    n_teams = _count(pg_engine, "SELECT count(*) n FROM team_seasons WHERE season = 2023")
    assert n_games == 7 and n_teams == 4
    pull_games(client, pg_engine, 2023, refresh=True)
    assert _count(pg_engine, "SELECT count(*) n FROM games") == n_games
    assert _count(pg_engine, "SELECT count(*) n FROM team_seasons WHERE season = 2023") == n_teams
    row = read_sql(
        pg_engine, "SELECT * FROM games WHERE game_id = :i", {"i": FIXTURE_INCOMPLETE_ID}
    ).iloc[0]
    assert pd.isna(row.home_points) and not row.completed
    cfp = read_sql(pg_engine, "SELECT is_cfp FROM games WHERE game_id = :i", {"i": FIXTURE_CFP_ID})
    assert bool(cfp.is_cfp[0])


@pytest.mark.db
def test_pull_games_falls_back_when_both_rejected(tmp_path, pg_engine):
    calls = []
    pull_games(_client(tmp_path, _handler(calls, reject_both=True)), pg_engine, 2023)
    types = [c["seasonType"] for c in calls if c["path"] == "/games"]
    assert types == ["both", "regular", "postseason"]
    assert _count(pg_engine, "SELECT count(*) n FROM games") == 7

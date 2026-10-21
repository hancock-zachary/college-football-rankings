import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
from typer.testing import CliRunner

from cfb_bias.cfbd import CfbdClient, CfbdError
from cfb_bias.cli import app
from cfb_bias.db import read_sql
from cfb_bias.ingest.fetch import get_season_types
from cfb_bias.ingest.lines import lines_to_frame
from cfb_bias.ingest.polls import polls_to_frame
from cfb_bias.ingest.runner import ALL_SOURCES, parse_only, pull_all
from cfb_bias.ingest.teams_meta import (
    coaches_to_frame,
    recruiting_to_frame,
    talent_to_frame,
    venues_to_frame,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_polls_flatten():
    df = polls_to_frame(load_fixture("rankings_2023_sample.json"))
    assert list(df.columns) == [
        "season", "season_type", "week", "poll", "is_final", "rank", "team",
        "conference", "first_place_votes", "points",
    ]
    ap = df[(df.poll == "AP Top 25") & (df.week == 1) & (df.season_type == "regular")]
    assert len(ap) == 25 and ap["rank"].min() == 1 and ap["rank"].max() == 25
    assert ap.set_index("rank").loc[1, "team"] == "Georgia"
    final = df[df.poll == "Playoff Committee Rankings"].iloc[0]
    assert final.is_final and pd.isna(final.points) and pd.isna(final.first_place_votes)
    assert not df.duplicated(["season", "season_type", "week", "poll", "team"]).any()
    assert polls_to_frame([]).empty


def test_polls_drop_rows_with_null_keys(pg_engine):
    from cfb_bias.ingest.polls import clean_polls

    recs = [{"season": 2023, "seasonType": "regular", "week": 1, "polls": [
        {"poll": "AP Top 25", "ranks": [
            {"rank": 1, "school": "A"}, {"rank": 2, "school": None}]},
        {"poll": None, "ranks": [{"rank": 1, "school": "B"}]}]},
        {"season": 2023, "seasonType": "regular", "week": None, "polls": [
            {"poll": "AP Top 25", "ranks": [{"rank": 1, "school": "C"}]}]}]
    df = clean_polls(polls_to_frame(recs))
    assert list(df["team"]) == ["A"]


def test_lines_one_row_per_provider():
    df = lines_to_frame(load_fixture("lines_2023_sample.json"))
    assert list(df.columns) == [
        "game_id", "season", "week", "season_type", "home_team", "away_team", "provider",
        "spread", "spread_open", "over_under", "home_moneyline", "away_moneyline",
    ]
    assert not df.duplicated(["game_id", "provider"]).any()
    assert len(df) == 5  # game with empty lines contributes nothing
    row = df.set_index(["game_id", "provider"]).loc[(401520281, "DraftKings")]
    assert row.spread == -31.5 and row.spread_open == -30.5 and row.over_under == 54.5
    assert row.home_moneyline == -12000 and row.away_team == "East Carolina"
    nulls = df.set_index(["game_id", "provider"]).loc[(401520281, "ESPN Bet")]
    assert pd.isna(nulls.spread) and pd.isna(nulls.home_moneyline)
    assert lines_to_frame([]).empty


def test_recruiting_and_talent():
    rec = recruiting_to_frame(load_fixture("recruiting_2023_sample.json"))
    assert list(rec.columns) == ["season", "team", "rank", "points"]
    assert rec.set_index("team").loc["Alabama", "rank"] == 1
    tal = talent_to_frame(load_fixture("talent_2023_sample.json"))
    assert list(tal.columns) == ["season", "team", "talent"]
    assert tal.set_index("team").loc["Georgia", "talent"] == 985.4
    assert recruiting_to_frame([]).empty and talent_to_frame([]).empty


def test_coaches_one_row_per_season():
    df = coaches_to_frame(load_fixture("coaches_2023_sample.json"))
    assert list(df.columns) == [
        "season", "team", "coach", "games", "wins", "losses", "preseason_rank", "postseason_rank",
    ]
    assert not df.duplicated(["season", "team", "coach"]).any()
    h = df.set_index("coach").loc["Jim Harbaugh"]
    assert h.team == "Michigan" and h.wins == 15 and h.postseason_rank == 1


def test_coaches_two_stints_same_team_summed():
    df = coaches_to_frame(load_fixture("coaches_2023_sample.json")).set_index("coach")
    m = df.loc["Sherrone Moore"]
    assert (m.games, m.wins, m.losses) == (13, 7, 6)
    assert pd.isna(m.preseason_rank)


def test_venues():
    df = venues_to_frame(load_fixture("venues_sample.json"))
    assert list(df.columns) == [
        "venue_id", "name", "city", "state", "capacity", "latitude", "longitude",
        "dome", "grass", "construction_year",
    ]
    row = df.set_index("venue_id").loc[3057]
    assert row["name"] == "Michigan Stadium" and row.capacity == 107601
    assert row.grass and not row.dome and row.construction_year == 1927
    assert pd.isna(df.set_index("venue_id").loc[9999, "dome"])


def _client(tmp_path, handler):
    return CfbdClient("k", tmp_path, httpx.MockTransport(handler), min_interval=0)


def test_season_types_fallback_on_non_auth_error(tmp_path):
    seen = []

    def handler(req):
        st = req.url.params["seasonType"]
        seen.append(st)
        return httpx.Response(400, json={}) if st == "both" else httpx.Response(200, json=[{"t": st}])

    out = get_season_types(_client(tmp_path, handler), "/x", 2023, False)
    assert seen == ["both", "regular", "postseason"]
    assert out == [{"t": "regular"}, {"t": "postseason"}]


@pytest.mark.parametrize("status", [401, 403, 429])
def test_season_types_no_fallback_on_auth_or_quota(tmp_path, status):
    seen = []

    def handler(req):
        seen.append(req.url.params["seasonType"])
        return httpx.Response(status, json={})

    with pytest.raises(CfbdError):
        get_season_types(_client(tmp_path, handler), "/x", 2023, False)
    assert seen == ["both"]


def test_season_types_no_fallback_when_key_missing(tmp_path):
    client = CfbdClient(None, tmp_path, httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(CfbdError, match="CFBD_API_KEY"):
        get_season_types(client, "/x", 2023, False)


def test_parse_only():
    assert parse_only(None) == list(ALL_SOURCES)
    assert parse_only("lines, games") == ["games", "lines"]
    with pytest.raises(ValueError, match="bogus"):
        parse_only("games,bogus")


def test_cli_rejects_unknown_only():
    result = CliRunner().invoke(app, ["pull", "--season", "2023", "--only", "games,bogus"])
    assert result.exit_code == 1 and "bogus" in result.output


def _full_handler():
    fixtures = {
        "/games": "games_2023_sample.json",
        "/teams/fbs": "teams_fbs_2023_sample.json",
        "/rankings": "rankings_2023_sample.json",
        "/lines": "lines_2023_sample.json",
        "/recruiting/teams": "recruiting_2023_sample.json",
        "/talent": "talent_2023_sample.json",
        "/coaches": "coaches_2023_sample.json",
        "/venues": "venues_sample.json",
        "/games/media": "media_2023_sample.json",
    }

    def handler(req):
        return httpx.Response(200, json=load_fixture(fixtures[req.url.path]))

    return handler


TABLES = ["games", "team_seasons", "poll_ranks", "betting_lines", "team_recruiting",
          "team_talent", "coach_seasons", "venues", "game_media"]


@pytest.mark.db
def test_pull_all_fills_every_table_and_is_idempotent(tmp_path, pg_engine):
    client = _client(tmp_path, _full_handler())
    pull_all(client, pg_engine, 2023, list(ALL_SOURCES))
    counts = {t: int(read_sql(pg_engine, f"SELECT count(*) n FROM {t}").n[0]) for t in TABLES}
    assert all(n > 0 for n in counts.values()), counts
    assert counts["coach_seasons"] == 4 and counts["betting_lines"] == 5
    assert counts["game_media"] == 3
    pull_all(client, pg_engine, 2023, list(ALL_SOURCES), refresh=True)
    again = {t: int(read_sql(pg_engine, f"SELECT count(*) n FROM {t}").n[0]) for t in TABLES}
    assert again == counts

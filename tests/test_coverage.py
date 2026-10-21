import pandas as pd
import pytest
from sqlalchemy import text

from cfb_bias.coverage import coverage_table, write_coverage_md
from cfb_bias.db import upsert

AP, COACHES, CFP = "AP Top 25", "Coaches Poll", "Playoff Committee Rankings"


def _game(gid, season, completed=True, is_cfp=False, home="fbs", away="fbs"):
    return {"game_id": gid, "season": season, "week": 1, "season_type": "regular",
            "completed": completed, "is_cfp": is_cfp,
            "home_classification": home, "away_classification": away}


def _poll(season, week, poll, season_type="regular", team="A"):
    return {"season": season, "season_type": season_type, "week": week, "poll": poll,
            "is_final": False, "rank": 1, "team": team}


def _clean(engine):
    with engine.begin() as conn:
        for t in ("games", "poll_ranks", "betting_lines", "team_recruiting",
                  "team_talent", "team_seasons"):
            conn.execute(text(f"DELETE FROM {t} WHERE season IN (2097, 2098, 2099)"))


@pytest.fixture
def seeded(pg_engine):
    _clean(pg_engine)
    games = [
        _game(970001, 2097), _game(970002, 2097), _game(970003, 2097),
        _game(970004, 2097, is_cfp=True), _game(970005, 2097, completed=False),
        _game(970006, 2097, home="fcs", away="fcs"),
        _game(970007, 2097, home="fcs", away="fbs"),
        _game(980001, 2098), _game(980002, 2098), _game(980003, 2098),
    ]
    upsert(pg_engine, "games", pd.DataFrame(games), ["game_id"])
    lines = [(970001, 2097, "p1"), (970001, 2097, "p2"), (970004, 2097, "p1"),
             (970005, 2097, "p1"), (970006, 2097, "p1"), (980001, 2098, "p1")]
    upsert(pg_engine, "betting_lines",
           pd.DataFrame(lines, columns=["game_id", "season", "provider"]),
           ["game_id", "provider"])
    polls = [_poll(2097, w, AP) for w in (0, 1, 2)]
    polls += [_poll(2097, 1, AP, "postseason"), _poll(2097, 0, COACHES),
              _poll(2097, 1, COACHES), _poll(2097, 14, CFP), _poll(2097, 15, CFP)]
    polls += [_poll(2098, w, AP) for w in (2, 3)]
    upsert(pg_engine, "poll_ranks", pd.DataFrame(polls),
           ["season", "season_type", "week", "poll", "team"])
    ts = [(2097, "A", "fbs"), (2097, "B", "fbs"), (2097, "C", "fcs"),
          (2098, "A", "fbs")]
    upsert(pg_engine, "team_seasons",
           pd.DataFrame(ts, columns=["season", "team", "classification"]),
           ["season", "team"])
    upsert(pg_engine, "team_recruiting",
           pd.DataFrame({"season": [2097, 2097, 2098], "team": ["A", "B", "A"]}),
           ["season", "team"])
    upsert(pg_engine, "team_talent",
           pd.DataFrame({"season": [2098], "team": ["A"]}), ["season", "team"])
    yield pg_engine
    _clean(pg_engine)  # other tests assert absolute table counts


def test_coverage_table_values(seeded):
    df = coverage_table(seeded, range(2097, 2100)).set_index("season")
    assert list(df.index) == [2097, 2098, 2099]
    a, b, c = df.loc[2097], df.loc[2098], df.loc[2099]
    assert (a.fbs_teams, a.games_completed, a.games_with_lines) == (2, 5, 2)
    assert a.pct_games_with_lines == 40.0
    assert a.ap_poll_weeks == 4  # regular 0,1,2 + postseason 1
    assert bool(a.ap_has_preseason) is True
    assert (a.coaches_poll_weeks, a.cfp_poll_weeks, a.cfp_games_flagged) == (2, 2, 1)
    assert (a.recruiting_teams, a.talent_teams) == (2, 0)
    assert (b.games_completed, b.games_with_lines, b.pct_games_with_lines) == (3, 1, 33.3)
    assert b.ap_poll_weeks == 2
    assert bool(b.ap_has_preseason) is False
    assert (b.coaches_poll_weeks, b.cfp_games_flagged, b.talent_teams) == (0, 0, 1)
    assert c.drop("pct_games_with_lines").astype(float).sum() == 0
    assert c.pct_games_with_lines == 0
    assert bool(c.ap_has_preseason) is False


def test_write_coverage_md(seeded, tmp_path):
    df = coverage_table(seeded, range(2097, 2100))
    out = tmp_path / "sub" / "cov.md"
    write_coverage_md(df, out)
    text_ = out.read_text(encoding="utf-8")
    assert "| season |" in text_ and "| 2097 |" in text_
    assert "AP Top 25" in text_ and "Coaches Poll" in text_
    # first CFBD week label per season, per poll
    assert "2097: 0" in text_ and "2098: 2" in text_

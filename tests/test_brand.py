import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from typer.testing import CliRunner

from cfb_bias.analysis.brand import build_team_brand, conference_dummies
from cfb_bias.cli import app
from cfb_bias.db import read_sql, upsert

pytestmark = pytest.mark.db

S = 2401  # distinctive season
TEAMS = ["BrandU", "BrandTech"]


@pytest.fixture
def seeded(pg_engine):
    def clean():
        with pg_engine.begin() as conn:
            for t in ("games", "poll_ranks", "team_seasons", "team_recruiting",
                      "team_talent", "ratings_weekly", "team_brand"):
                conn.execute(text(f"DELETE FROM {t} WHERE season BETWEEN 2395 AND 2402"))
            conn.execute(text("DELETE FROM school_enrollment WHERE team = ANY(:t)"),
                         {"t": TEAMS + ["BrandNoData"]})
    clean()
    ts = pd.DataFrame({
        "season": [S - 1, S, S, S, S],
        "team": ["BrandU", "BrandU", "BrandTech", "BrandNoData", "BrandFcs"],
        "conference": ["Big 12", "SEC", "ACC", "Sun Belt", "Big Sky"],
        "classification": ["fbs", "fbs", "fbs", "fbs", "fcs"],
    })
    upsert(pg_engine, "team_seasons", ts, ["season", "team"])
    games = pd.DataFrame({
        "game_id": [9100001, 9100002, 9100003],
        "season": [2399, 2399, 2400],
        "week": [1, 2, 1], "season_type": ["regular"] * 3,
        "completed": [True] * 3,
        "home_team": ["BrandU", "BrandTech", "BrandU"],
        "away_team": ["BrandTech", "BrandU", "BrandTech"],
        "home_points": [10, 3, 7], "away_points": [3, 10, 7],
    })
    upsert(pg_engine, "games", games, ["game_id"])
    polls = pd.DataFrame({
        "season": [2399, S, S, S, S, S],
        "season_type": ["postseason", "regular", "regular", "regular", "regular", "postseason"],
        "week": [1, 1, 1, 1, 2, 1],
        "poll": ["AP Top 25", "AP Top 25", "AP Top 25", "Coaches Poll", "AP Top 25",
                 "AP Top 25"],
        "rank": [1, 1, 2, 1, 1, 1],
        "team": ["BrandU", "BrandU", "Other1", "BrandTech", "BrandTech", "BrandTech"],
        "points": [1500, 1500, 750, 99, 5, 5],
    })
    upsert(pg_engine, "poll_ranks", polls, ["season", "season_type", "week", "poll", "team"])
    rec = pd.DataFrame({
        "season": [S - 4, S - 3, S - 2, S, S + 1],
        "team": ["BrandU", "BrandU", "BrandU", "BrandU", "BrandU"],
        "rank": [1] * 5, "points": [900.0, 100.0, 200.0, 300.0, 5000.0],
    })
    upsert(pg_engine, "team_recruiting", rec, ["season", "team"])
    upsert(pg_engine, "team_talent",
           pd.DataFrame({"season": [S], "team": ["BrandU"], "talent": [850.5]}),
           ["season", "team"])
    rw = pd.DataFrame({
        "season": [S - 1, S - 1, S - 1], "week": [99, 99, 5],
        "team": ["BrandU", "BrandU", "BrandU"],
        "method": ["colley", "massey", "colley"], "rating": [0.8, 9.0, 0.1],
    })
    upsert(pg_engine, "ratings_weekly", rw, ["season", "week", "team", "method"])
    yield pg_engine
    clean()


def test_brand_columns_for_fbs_teams(seeded):
    b = build_team_brand(seeded, S).set_index("team")
    assert set(b.index) == {"BrandU", "BrandTech", "BrandNoData"}  # FBS only
    u = b.loc["BrandU"]
    assert u["conference"] == "SEC"
    assert u["ap_titles"] == 1
    assert u["prior_colley"] == pytest.approx(0.8)
    assert u["preseason_ap_score"] == pytest.approx(1.0)
    assert u["recruiting_4yr"] == pytest.approx(200.0)
    assert u["talent"] == pytest.approx(850.5)
    assert np.isnan(u["enrollment"])
    t = b.loc["BrandTech"]
    assert np.isnan(t["prior_colley"]) and np.isnan(t["talent"])
    assert np.isnan(t["recruiting_4yr"])


def test_alltime_and_titles_values(seeded):
    b = build_team_brand(seeded, S).set_index("team")
    # BrandU: win, win, tie over 3 games = 2.5/3; BrandTech: 0.5/3
    assert b.loc["BrandU", "alltime_win_pct"] == pytest.approx(2.5 / 3)
    assert b.loc["BrandTech", "alltime_win_pct"] == pytest.approx(0.5 / 3)
    assert b.loc["BrandU", "ap_titles"] == 1  # 2399 final AP #1 (its postseason poll)
    assert b.loc["BrandTech", "ap_titles"] == 0
    assert np.isnan(b.loc["BrandNoData", "alltime_win_pct"])
    assert b.loc["BrandNoData", "ap_titles"] == 0


def test_preseason_unranked_is_zero(seeded):
    b = build_team_brand(seeded, S).set_index("team")
    assert b.loc["BrandTech", "preseason_ap_score"] == 0.0
    assert b.loc["BrandNoData", "preseason_ap_score"] == 0.0


def test_recruiting_4yr_window(seeded):
    b = build_team_brand(seeded, S).set_index("team")
    # classes S-3..S = 100, 200, 300 (S-1 missing); S-4 and S+1 excluded
    assert b.loc["BrandU", "recruiting_4yr"] == pytest.approx(200.0)


def test_conference_from_that_season(seeded):
    prev = build_team_brand(seeded, S - 1).set_index("team")
    cur = build_team_brand(seeded, S).set_index("team")
    assert prev.loc["BrandU", "conference"] == "Big 12"
    d_prev = conference_dummies(prev["conference"])
    assert d_prev.loc["BrandU", "big_12"] == 1 and d_prev.loc["BrandU", "sec"] == 0
    d = conference_dummies(cur["conference"])
    assert d.loc["BrandU", "sec"] == 1 and d.loc["BrandU", "big_12"] == 0


def test_enrollment_used_when_present(seeded):
    upsert(seeded, "school_enrollment",
           pd.DataFrame({"team": ["BrandU"], "unitid": [1], "institution": ["BrandU Univ"],
                         "undergrad_enrollment": [40000.0], "source_year": [2022]}), ["team"])
    b = build_team_brand(seeded, S).set_index("team")
    assert b.loc["BrandU", "enrollment"] == 40000.0
    assert np.isnan(b.loc["BrandTech", "enrollment"])


def test_conference_dummies_exact_names():
    d = conference_dummies(pd.Series(["SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "MAC"]))
    assert list(d.columns) == ["sec", "big_ten", "big_12", "acc", "pac_12"]
    assert d.sum(axis=1).tolist() == [1, 1, 1, 1, 1, 0]


def test_cli_brand_upserts_each_season(seeded, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", seeded.url.render_as_string(hide_password=False))
    monkeypatch.setenv("CFBD_API_KEY", "x")
    r = CliRunner().invoke(app, ["brand", "--from", str(S - 1), "--to", str(S)])
    assert r.exit_code == 0, r.output
    assert f"Season {S}: wrote 3 team_brand rows" in r.output
    assert f"Season {S - 1}: wrote 1 team_brand rows" in r.output
    out = read_sql(seeded, "SELECT * FROM team_brand WHERE season = :s", {"s": S})
    assert len(out) == 3

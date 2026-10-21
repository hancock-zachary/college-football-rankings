"""Data coverage audit: per-season counts of what the pipeline has ingested."""

from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from cfb_bias.db import read_sql

AP_POLL = "AP Top 25"
COACHES_POLL = "Coaches Poll"
CFP_POLL = "Playoff Committee Rankings"

COLUMNS = [
    "season", "fbs_teams", "games_completed", "games_with_lines", "pct_games_with_lines",
    "ap_poll_weeks", "ap_has_preseason", "coaches_poll_weeks", "cfp_poll_weeks",
    "cfp_games_flagged", "recruiting_teams", "talent_teams",
]
_PARAMS_SQL = "season BETWEEN :lo AND :hi"
_HAS_FBS = "(home_classification = 'fbs' OR away_classification = 'fbs')"


def _counts(engine: Engine, sql: str, params: dict) -> pd.Series:
    """Run a 'SELECT season, n ...' query into a season-indexed Series."""
    df = read_sql(engine, sql, params)
    return df.set_index("season")["n"] if not df.empty else pd.Series(dtype="int64")


def coverage_table(engine: Engine, seasons: range) -> pd.DataFrame:
    """One row per season in `seasons` (zeros where there is no data).

    `df.attrs["first_week"]` maps poll name -> {season: minimum CFBD week}.
    """
    p = {"lo": seasons.start, "hi": seasons.stop - 1}
    w = _PARAMS_SQL
    fbs = _counts(engine, f"SELECT season, count(*) AS n FROM team_seasons "
                          f"WHERE {w} AND classification = 'fbs' GROUP BY season", p)
    completed = _counts(engine, f"SELECT season, count(*) AS n FROM games "
                                f"WHERE {w} AND completed AND {_HAS_FBS} GROUP BY season", p)
    with_lines = _counts(
        engine,
        "SELECT g.season, count(*) AS n FROM games g WHERE g.season BETWEEN :lo AND :hi "
        "AND g.completed AND (g.home_classification = 'fbs' OR g.away_classification = 'fbs') "
        "AND EXISTS (SELECT 1 FROM betting_lines b WHERE b.game_id = g.game_id) "
        "GROUP BY g.season", p)
    cfp_games = _counts(engine, f"SELECT season, count(*) AS n FROM games "
                                f"WHERE {w} AND is_cfp GROUP BY season", p)
    recruiting = _counts(engine, f"SELECT season, count(*) AS n FROM team_recruiting "
                                 f"WHERE {w} GROUP BY season", p)
    talent = _counts(engine, f"SELECT season, count(*) AS n FROM team_talent "
                             f"WHERE {w} GROUP BY season", p)
    polls = read_sql(
        engine,
        "SELECT season, poll, season_type, week FROM poll_ranks "
        f"WHERE {w} GROUP BY season, poll, season_type, week", p)

    def weeks(poll: str) -> pd.Series:
        return polls[polls["poll"] == poll].groupby("season").size()

    first_week = {
        poll: polls[polls["poll"] == poll].groupby("season")["week"].min().astype(int).to_dict()
        for poll in (AP_POLL, COACHES_POLL, CFP_POLL)
    }
    ap = polls[polls["poll"] == AP_POLL]
    preseason = {
        int(s): bool(
            (g["week"].min() <= 1)
            and ((g["season_type"] == "regular") & (g["week"] == g["week"].min())).any()
        )
        for s, g in ap.groupby("season")
    }

    idx = pd.Index(list(seasons), name="season")
    df = pd.DataFrame(index=idx)
    df["fbs_teams"] = fbs
    df["games_completed"] = completed
    df["games_with_lines"] = with_lines
    df["ap_poll_weeks"] = weeks(AP_POLL)
    df["coaches_poll_weeks"] = weeks(COACHES_POLL)
    df["cfp_poll_weeks"] = weeks(CFP_POLL)
    df["cfp_games_flagged"] = cfp_games
    df["recruiting_teams"] = recruiting
    df["talent_teams"] = talent
    df = df.fillna(0).astype(int)
    df["pct_games_with_lines"] = (
        (100 * df["games_with_lines"] / df["games_completed"].where(df["games_completed"] > 0))
        .fillna(0).round(1)
    )
    df["ap_has_preseason"] = [preseason.get(s, False) for s in idx]
    df = df.reset_index()[COLUMNS]
    df.attrs["first_week"] = first_week
    return df


def write_coverage_md(df: pd.DataFrame, path: Path) -> None:
    """Write the coverage table plus per-poll first-week-label notes as markdown."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(" --- " for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in rec) + " |"
            for rec in df.itertuples(index=False, name=None)]
    lines = ["# Data coverage", "", header, sep, *rows, "",
             "## First week label per poll", "",
             "Minimum CFBD `week` value holding each poll, per season "
             "(shows how the preseason poll is labelled).", ""]
    for poll, by_season in df.attrs.get("first_week", {}).items():
        detail = ", ".join(f"{s}: {wk}" for s, wk in sorted(by_season.items())) or "no data"
        lines.append(f"- **{poll}** — {detail}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

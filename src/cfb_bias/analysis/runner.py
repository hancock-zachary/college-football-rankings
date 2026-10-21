"""Analysis runner: the six bias analyses, preseason decay, and run parameters into Postgres."""

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from cfb_bias.analysis.brand import conference_dummies
from cfb_bias.analysis.decay import half_life, preseason_decay
from cfb_bias.analysis.dml import partial_out
from cfb_bias.analysis.stability import choose_start_week, rating_stability
from cfb_bias.analysis.team_effects import team_bias
from cfb_bias.db import read_sql, upsert

STABILITY_THRESHOLD = 0.75
TALENT_FIRST_SEASON = 2015
DECAY_FIRST_WEEK = 2

RESUME_X = ["wins", "losses", "games_played", "colley", "colley_sos", "quality_wins",
            "bad_losses", "won_last"]
MARGIN_X = ["massey", "massey_sos"]
BRAND_B = ["alltime_win_pct", "ap_titles", "prior_colley", "preseason_ap_score",
           "recruiting_4yr", "market_rating", "national_tv_share", "sec", "big_ten",
           "big_12", "acc", "pac_12", "enrollment"]


@dataclass(frozen=True)
class Analysis:
    y: str
    x: list[str]
    b: list[str]
    first_season: int | None = None


ANALYSES = {
    "main": Analysis("ap_score", RESUME_X, BRAND_B),
    "margin_sensitivity": Analysis("ap_score", RESUME_X + MARGIN_X, BRAND_B),
    "talent_2015plus": Analysis("ap_score", RESUME_X, BRAND_B + ["talent"], TALENT_FIRST_SEASON),
    "ap_rank": Analysis("ap_rank_score", RESUME_X, BRAND_B),
    "coaches_rank": Analysis("coaches_rank_score", RESUME_X, BRAND_B),
    "cfp_rank": Analysis("cfp_rank_score", RESUME_X, BRAND_B),
}
DECAY_POLLS = {"AP Top 25": "ap_score", "Coaches Poll": "coaches_rank_score"}


def load_frame(engine: Engine, seasons: range) -> pd.DataFrame:
    """team_week_features joined to team_brand (+ conference dummies) for the seasons."""
    df = read_sql(
        engine,
        "SELECT f.*, b.conference, b.alltime_win_pct, b.ap_titles, b.prior_colley, "
        "b.preseason_ap_score, b.recruiting_4yr, b.talent, b.enrollment "
        "FROM team_week_features f LEFT JOIN team_brand b "
        "ON b.season = f.season AND b.team = f.team "
        "WHERE f.season BETWEEN :lo AND :hi ORDER BY f.season, f.poll_week, f.team",
        {"lo": seasons.start, "hi": seasons.stop - 1},
    )
    df = df.join(conference_dummies(df["conference"]))
    df["won_last"] = df["won_last"].astype("boolean").astype("Float64").astype(float)
    return df


def compute_start_week(engine: Engine, seasons: range,
                       threshold: float = STABILITY_THRESHOLD) -> int:
    ratings = read_sql(
        engine,
        "SELECT season, week, team, method, rating FROM ratings_weekly "
        "WHERE method = 'colley' AND season BETWEEN :lo AND :hi",
        {"lo": seasons.start, "hi": seasons.stop - 1},
    )
    return choose_start_week(rating_stability(ratings), threshold=threshold)


KEYS = {
    "bias_coefficients": ["analysis", "variable"],
    "team_bias": ["analysis", "team"],
    "preseason_decay": ["poll", "poll_week"],
    "analysis_params": ["key"],
}


def run_one(name: str, spec: Analysis, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One analysis, no writes: (bias_coefficients rows, team_bias rows)."""
    rows = df[df[spec.y].notna()]
    if spec.first_season is not None:
        rows = rows[rows["season"] >= spec.first_season]
    result = partial_out(rows, spec.y, spec.x, spec.b)
    coefs = result.coefficients.assign(
        analysis=name, n_obs=result.n_obs, n_teams=result.n_teams, n_dropped=result.n_dropped
    )
    idx = result.y_resid.index
    teams = team_bias(result.y_resid, rows.loc[idx, "team"], rows.loc[idx, "season"])
    teams.insert(0, "analysis", name)
    return coefs, teams


def _stack(frames) -> pd.DataFrame:
    """Concatenate, skipping empty frames (an empty result writes nothing)."""
    kept = [f for f in frames if not f.empty]
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()


def _fmt(v: int | None) -> str:
    return "none" if v is None else str(v)


def run_analyses(engine: Engine, seasons: range, start_week: int | None = None) -> dict[str, int]:
    """All six bias analyses plus preseason decay; returns rows written per table.

    Every result is computed before anything is written, and all deletes and writes
    happen in one transaction: a failed run leaves the previous results intact.
    """
    threshold = "override"
    if start_week is None:
        start_week = compute_start_week(engine, seasons)
        threshold = str(STABILITY_THRESHOLD)
    df = load_frame(engine, seasons)

    window = df[df["poll_week"] >= start_week]
    results = {name: run_one(name, spec, window) for name, spec in ANALYSES.items()}
    decay_rows = df[df["poll_week"] >= DECAY_FIRST_WEEK]
    decays = {poll: preseason_decay(decay_rows, y, poll) for poll, y in DECAY_POLLS.items()}

    params = pd.DataFrame({
        "key": ["start_week", "stability_threshold", "season_from", "season_to",
                "half_life_ap", "half_life_coaches"],
        "value": [str(start_week), threshold, str(seasons.start), str(seasons.stop - 1),
                  _fmt(half_life(decays["AP Top 25"])), _fmt(half_life(decays["Coaches Poll"]))],
    })
    out = {
        "bias_coefficients": _stack(c for c, _ in results.values()),
        "team_bias": _stack(t for _, t in results.values()),
        "preseason_decay": _stack(decays.values()),
        "analysis_params": params,
    }
    with engine.begin() as conn:
        for table in ("bias_coefficients", "team_bias"):
            conn.execute(text(f"DELETE FROM {table} WHERE analysis = ANY(:a)"),
                         {"a": list(ANALYSES)})
        conn.execute(text("DELETE FROM preseason_decay WHERE poll = ANY(:p)"),
                     {"p": list(DECAY_POLLS)})
        return {table: upsert(engine, table, frame, KEYS[table], conn=conn)
                for table, frame in out.items()}

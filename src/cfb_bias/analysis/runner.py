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


def _replace(engine: Engine, table: str, where: str, params: dict) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table} WHERE {where}"), params)


def run_one(engine: Engine, name: str, spec: Analysis, df: pd.DataFrame) -> tuple[int, int]:
    """Run one analysis and replace its rows; returns (bias_coefficients, team_bias) rows."""
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
    for table in ("bias_coefficients", "team_bias"):
        _replace(engine, table, "analysis = :a", {"a": name})
    n_coef = upsert(engine, "bias_coefficients", coefs, ["analysis", "variable"])
    n_team = upsert(engine, "team_bias", teams, ["analysis", "team"])
    return n_coef, n_team


def run_analyses(engine: Engine, seasons: range, start_week: int | None = None) -> dict[str, int]:
    """All six bias analyses plus preseason decay; returns rows written per table."""
    threshold = "override"
    if start_week is None:
        start_week = compute_start_week(engine, seasons)
        threshold = str(STABILITY_THRESHOLD)
    df = load_frame(engine, seasons)

    written = {"bias_coefficients": 0, "team_bias": 0, "preseason_decay": 0,
               "analysis_params": 0}
    window = df[df["poll_week"] >= start_week]
    for name, spec in ANALYSES.items():
        n_coef, n_team = run_one(engine, name, spec, window)
        written["bias_coefficients"] += n_coef
        written["team_bias"] += n_team

    half_lives = {}
    decay_rows = df[df["poll_week"] >= DECAY_FIRST_WEEK]
    for poll, y in DECAY_POLLS.items():
        decay = preseason_decay(decay_rows, y, poll)
        _replace(engine, "preseason_decay", "poll = :p", {"p": poll})
        written["preseason_decay"] += upsert(engine, "preseason_decay", decay,
                                             ["poll", "poll_week"])
        half_lives[poll] = half_life(decay)

    def fmt(v: int | None) -> str:
        return "none" if v is None else str(v)

    params = pd.DataFrame({
        "key": ["start_week", "stability_threshold", "half_life_ap", "half_life_coaches"],
        "value": [str(start_week), threshold, fmt(half_lives["AP Top 25"]),
                  fmt(half_lives["Coaches Poll"])],
    })
    written["analysis_params"] = upsert(engine, "analysis_params", params, ["key"])
    return written

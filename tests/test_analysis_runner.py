import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sqlalchemy import text
from typer.testing import CliRunner

from cfb_bias import cli
from cfb_bias.analysis import dml, runner
from cfb_bias.analysis.runner import ANALYSES, load_frame, run_analyses, run_one
from cfb_bias.cli import app
from cfb_bias.db import read_sql, upsert

pytestmark = [
    pytest.mark.db,
    pytest.mark.filterwarnings("ignore:brand column 'enrollment' is entirely missing"),
    # 12 seeded teams = 12 clusters: the cluster-robust covariance has rank <= 11, below
    # the 12-13 brand columns, so the joint Wald test uses its rank (real data: ~130 teams)
    pytest.mark.filterwarnings("ignore:covariance of constraints does not have full rank"),
]

SEASONS = range(2501, 2504)
POLL_WEEKS = range(1, 9)
PLANTED = "RunPlanted"
CONFERENCES = ["SEC", "SEC", "Big Ten", "Big Ten", "Big 12", "Big 12", "ACC", "ACC",
               "Pac-12", "Pac-12", "Sun Belt", "MAC"]
TEAMS = [PLANTED] + [f"RunTeam{i:02d}" for i in range(1, 12)]
CFP_WEEKS = range(6, 9)
COMPUTED_START_WEEK = 4  # ratings weeks >= 3 match the final ranking exactly


@pytest.fixture(autouse=True)
def fast_learner(monkeypatch):
    monkeypatch.setattr(
        dml, "make_learner",
        lambda seed: HistGradientBoostingRegressor(max_iter=20, random_state=seed),
    )


def clean(engine):
    with engine.begin() as conn:
        for t in ("team_week_features", "team_brand", "ratings_weekly"):
            conn.execute(text(f"DELETE FROM {t} WHERE season BETWEEN 2501 AND 2503"))
        for t in ("bias_coefficients", "team_bias"):
            conn.execute(text(f"DELETE FROM {t} WHERE analysis = ANY(:a)"), {"a": list(ANALYSES)})
        conn.execute(text("DELETE FROM preseason_decay"))
        conn.execute(text("DELETE FROM analysis_params"))


@pytest.fixture
def seeded(pg_engine):
    clean(pg_engine)
    rng = np.random.default_rng(7)
    feats, brand, ratings = [], [], []
    for season in SEASONS:
        strength = rng.normal(0, 1, len(TEAMS))
        for i, team in enumerate(TEAMS):
            brand.append({
                "season": season, "team": team, "conference": CONFERENCES[i],
                "alltime_win_pct": rng.uniform(0.3, 0.7), "ap_titles": int(rng.integers(0, 4)),
                "prior_colley": rng.uniform(0.3, 0.7), "preseason_ap_score": rng.uniform(0, 1),
                "recruiting_4yr": rng.uniform(150, 300), "talent": rng.uniform(500, 1000),
                "enrollment": np.nan,
            })
            for w in list(range(1, 9)) + [99]:
                rating = strength[i] if w >= 3 else -strength[i]  # weeks 1-2 reversed
                ratings.append({"season": season, "week": w, "team": team,
                                "method": "colley", "rating": rating})
            for pw in POLL_WEEKS:
                played = max(pw - 1 - int(rng.integers(0, 2)), 0)  # some byes
                wins = int(rng.binomial(played, 1 / (1 + np.exp(-strength[i]))))
                colley = 0.5 + 0.1 * strength[i] + rng.normal(0, 0.02)
                base = 0.4 + 0.25 * np.tanh(strength[i]) + rng.normal(0, 0.03)
                y = base + (0.15 if team == PLANTED else 0.0)
                feats.append({
                    "season": season, "poll_week": pw, "team": team,
                    "ratings_week": pw - 1 if pw > 1 else None,
                    "ap_score": y, "ap_rank_score": y + rng.normal(0, 0.01),
                    "coaches_rank_score": y + rng.normal(0, 0.01),
                    "cfp_rank_score": y + rng.normal(0, 0.01) if pw in CFP_WEEKS else None,
                    "wins": wins, "losses": played - wins, "games_played": played,
                    "colley": colley, "colley_sos": rng.uniform(0.4, 0.6),
                    "massey": 10 * strength[i] + rng.normal(0, 1), "massey_sos": rng.normal(),
                    "quality_wins": int(rng.integers(0, 2)), "bad_losses": int(rng.integers(0, 2)),
                    "won_last": bool(rng.integers(0, 2)) if played else None,
                    "market_rating": strength[i] + rng.normal(0, 0.5),
                    "national_tv_share": rng.uniform(0, 1),
                })
    upsert(pg_engine, "team_week_features", pd.DataFrame(feats), ["season", "poll_week", "team"])
    upsert(pg_engine, "team_brand", pd.DataFrame(brand), ["season", "team"])
    upsert(pg_engine, "ratings_weekly", pd.DataFrame(ratings), ["season", "week", "team", "method"])
    yield pg_engine
    clean(pg_engine)


def params(engine) -> dict[str, str]:
    return read_sql(engine, "SELECT key, value FROM analysis_params").set_index("key")["value"].to_dict()


def test_run_analyses_writes_all_six_and_recovers_planted_team(seeded):
    # stale rows from an earlier run must not survive a rerun
    upsert(seeded, "bias_coefficients",
           pd.DataFrame({"analysis": ["main"], "variable": ["stale_var"], "coef": [1.0]}),
           ["analysis", "variable"])
    upsert(seeded, "team_bias",
           pd.DataFrame({"analysis": ["main"], "team": ["StaleTeam"], "mean_resid": [9.0]}),
           ["analysis", "team"])
    upsert(seeded, "preseason_decay",
           pd.DataFrame({"poll": ["AP Top 25"], "poll_week": [77], "coef": [1.0]}),
           ["poll", "poll_week"])

    written = run_analyses(seeded, SEASONS, start_week=3)

    coefs = read_sql(seeded, "SELECT * FROM bias_coefficients")
    assert set(coefs["analysis"]) == set(ANALYSES)
    by = coefs.set_index(["analysis", "variable"])
    assert "talent" in set(coefs.loc[coefs["analysis"] == "talent_2015plus", "variable"])
    assert "talent" not in set(coefs.loc[coefs["analysis"] == "main", "variable"])
    assert "enrollment" not in set(coefs["variable"])  # entirely NaN -> dropped
    assert "stale_var" not in set(coefs["variable"])
    n_main = len(SEASONS) * len(TEAMS) * len(range(3, 9))
    assert by.loc[("main", "sec"), "n_obs"] == n_main
    assert by.loc[("cfp_rank", "sec"), "n_obs"] == len(SEASONS) * len(TEAMS) * len(CFP_WEEKS)
    assert by.loc[("main", "sec"), "n_teams"] == len(TEAMS)

    tb = read_sql(seeded, "SELECT * FROM team_bias")
    assert set(tb["analysis"]) == set(ANALYSES)
    main = tb[tb["analysis"] == "main"].sort_values("mean_resid", ascending=False)
    assert main["team"].iloc[0] == PLANTED
    assert "StaleTeam" not in set(tb["team"])

    decay = read_sql(seeded, "SELECT * FROM preseason_decay")
    assert set(decay["poll"]) == {"AP Top 25", "Coaches Poll"}
    assert decay["poll_week"].min() >= 2
    assert 77 not in set(decay["poll_week"])

    p = params(seeded)
    assert p["start_week"] == "3"
    assert p["season_from"] == "2501" and p["season_to"] == "2503"
    assert {"stability_threshold", "half_life_ap", "half_life_coaches"} <= set(p)
    assert written == {
        "bias_coefficients": len(coefs), "team_bias": len(tb),
        "preseason_decay": len(decay), "analysis_params": len(p),
    }


OUTPUT_TABLES = {
    "bias_coefficients": ["analysis", "variable"], "team_bias": ["analysis", "team"],
    "preseason_decay": ["poll", "poll_week"], "analysis_params": ["key"],
}


def snapshot(engine) -> dict[str, pd.DataFrame]:
    return {t: read_sql(engine, f"SELECT * FROM {t} ORDER BY {', '.join(k)}")
            for t, k in OUTPUT_TABLES.items()}


def test_failed_rerun_leaves_previous_results_intact(seeded, monkeypatch):
    prior = {
        "bias_coefficients": pd.DataFrame({"analysis": list(ANALYSES), "variable": "sec",
                                           "coef": 0.5, "n_obs": 10}),
        "team_bias": pd.DataFrame({"analysis": list(ANALYSES), "team": PLANTED,
                                   "mean_resid": 0.2, "significant": True}),
        "preseason_decay": pd.DataFrame({"poll": ["AP Top 25", "Coaches Poll"],
                                         "poll_week": [2, 2], "coef": [0.4, 0.3]}),
        "analysis_params": pd.DataFrame({"key": ["start_week", "season_from"],
                                         "value": ["5", "2014"]}),
    }
    for table, df in prior.items():
        upsert(seeded, table, df, OUTPUT_TABLES[table])
    before = snapshot(seeded)

    real = runner.partial_out

    def failing(df, y, x_cols, b_cols, **kw):
        if y == "cfp_rank_score":  # last analysis: the others have already run
            raise ValueError("boom")
        return real(df, y, x_cols, b_cols, **kw)

    monkeypatch.setattr(runner, "partial_out", failing)
    with pytest.raises(ValueError, match="boom"):
        run_analyses(seeded, SEASONS, start_week=3)

    after = snapshot(seeded)
    for table in OUTPUT_TABLES:
        pd.testing.assert_frame_equal(after[table], before[table], obj=table)


def test_start_week_computed_from_colley_stability(seeded):
    run_analyses(seeded, SEASONS)
    p = params(seeded)
    assert p["start_week"] == str(COMPUTED_START_WEEK)
    assert p["stability_threshold"] == "0.75"
    n = read_sql(seeded, "SELECT n_obs FROM bias_coefficients "
                         "WHERE analysis = 'main' AND variable = 'sec'")["n_obs"].iloc[0]
    assert n == len(SEASONS) * len(TEAMS) * len(range(COMPUTED_START_WEEK, 9))


def test_cli_analyze_prints_start_week_and_rows(seeded, monkeypatch):
    calls = {}

    def fake_run(engine, seasons, start_week=None):
        calls.update(seasons=seasons, start_week=start_week)
        upsert(engine, "analysis_params",
               pd.DataFrame({"key": ["start_week", "stability_threshold"],
                             "value": ["3", "override"]}), ["key"])
        return {"bias_coefficients": 72, "team_bias": 66, "preseason_decay": 14,
                "analysis_params": 4}

    monkeypatch.setattr(cli, "run_analyses", fake_run)
    monkeypatch.setenv("DATABASE_URL", seeded.url.render_as_string(hide_password=False))
    monkeypatch.setenv("CFBD_API_KEY", "x")
    r = CliRunner().invoke(app, ["analyze", "--from", "2501", "--to", "2503", "--start-week", "3"])
    assert r.exit_code == 0, r.output
    assert calls == {"seasons": range(2501, 2504), "start_week": 3}
    assert "Start week: 3" in r.output
    assert "bias_coefficients: 72 rows" in r.output
    assert "preseason_decay: 14 rows" in r.output


def test_joint_wald_row_written_per_analysis(seeded):
    run_analyses(seeded, SEASONS, start_week=3)
    coefs = read_sql(seeded, "SELECT * FROM bias_coefficients")
    joint = coefs[coefs["variable"] == "__joint__"].set_index("analysis")
    assert set(joint.index) == set(ANALYSES)
    assert (joint["coef"] > 0).all()  # Wald statistic
    assert joint[["se", "ci_lo", "ci_hi"]].isna().all().all()
    assert joint["p_value"].between(0, 1).all()
    main = coefs[coefs["analysis"] == "main"]
    assert joint.loc["main", "n_obs"] == main["n_obs"].iloc[0]


def test_never_ranked_teams_excluded_from_team_tests(seeded):
    df = load_frame(seeded, SEASONS)
    floor = "RunTeam11"
    df.loc[df["team"] == floor, "ap_score"] = 0.0  # never ranked in the AP points poll
    window = df[df["poll_week"] >= 3]
    _, teams = run_one("main", ANALYSES["main"], window)
    assert floor not in set(teams["team"])
    assert set(teams["team"]) == set(TEAMS) - {floor}
    # still in the regression sample: only the team tests drop it
    coefs, _ = run_one("main", ANALYSES["main"], window)
    assert coefs["n_obs"].iloc[0] == len(window)


def test_rows_without_games_played_not_used(seeded):
    df = load_frame(seeded, SEASONS)
    window = df[df["poll_week"] >= 2]
    unplayed = int((window["games_played"] == 0).sum())
    assert unplayed > 0  # byes in poll week 2 of the seed
    coefs, _ = run_one("main", ANALYSES["main"], window)
    row = coefs.set_index("variable").loc["sec"]
    assert row["n_obs"] == len(window) - unplayed
    assert row["n_dropped"] == unplayed

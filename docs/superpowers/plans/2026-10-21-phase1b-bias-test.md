# Phase 1b — Brand Variables & Bias Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer "is there persistent poll bias, and for which teams?" — build per-team-week résumé features and brand/perceived-quality variables, run a cross-fitted partialling-out (double-ML) bias test with team-clustered inference, per-team persistent-bias tests with FDR control, an early-season preseason-persistence analysis, a margin sensitivity check, and an AP/Coaches/CFP comparison — all written to Postgres.

**Architecture:** New `cfb_bias.analysis` package. Builders (`brand`, `market`, `features`) turn existing tables into `team_brand`, `market_weekly` and `team_week_features`; statistical modules (`stability`, `dml`, `team_effects`, `decay`) are pure functions on DataFrames tested against synthetic data with known answers; `runner` wires them together behind `cfb analyze`. Two small ingest additions: TV media per game (CFBD) and undergraduate enrollment (College Scorecard).

**Tech Stack:** existing stack + scikit-learn (HistGradientBoostingRegressor, GroupKFold) + statsmodels (OLS with cluster-robust SEs, `multipletests`).

**Spec:** `docs/superpowers/specs/2026-10-20-cfb-ranking-bias-design.md` (bias definition, method steps 4–7, poll comparison). Data facts: `docs/data-coverage.md`.

## Global Constraints

- Work directly on `main`; no branches or PRs (author's standing instruction).
- Phase-1 analysis window: seasons **2014–2025** (author's decision). Talent composite exists only from **2015**.
- **Poll week ↔ ratings alignment:** CFBD poll week `w` reflects games through game week `w−1`. Join each poll week to the latest `ratings_weekly` week `≤ w−1` (as-of join); poll week 1 (preseason) has no games behind it.
- Main bias test uses **regular-season** AP poll weeks `≥ start_week` only; postseason polls are excluded. `start_week` is computed (Task 6), never hard-coded.
- **Résumé features X are margin-free and poll-free**: wins, losses, games played, Colley rating, Colley SOS, quality wins, bad losses, last-game result. Never in X: margin/Massey (except the margin sensitivity run), betting lines, recruiting, talent, preseason rank, prior seasons, conference, TV, enrollment.
- **Brand/perceived-quality variables B** (all tested as possible bias, never controlled as legitimate): all-time win % (1936→S−1), AP titles (1936→S−1), prior-season final Colley, preseason AP score, 4-year recruiting average, market rating, national-TV share to date, conference dummies (SEC, Big Ten, Big 12, ACC, Pac-12; baseline = everyone else), enrollment (when available), talent (2015+ run only).
- Poll targets: **AP main** = points ÷ max points in that poll-week (0 if unranked). **Cross-poll comparison** = rank score `(26 − rank)/25` (0 if unranked) for AP, Coaches and CFP (CFP has no points).
- Sample = **all FBS teams** each poll week (no selection on being ranked).
- Cross-fitting: **leave-one-season-out** folds; learner `HistGradientBoostingRegressor(random_state=0)`; final-stage OLS with **cluster-robust SEs clustered by team**; B standardised (z-score over the sample) before residualising, so coefficients are "per 1 SD".
- Per-team tests: unit = team-season mean residual; teams need **≥ 3 seasons**; Benjamini–Hochberg **q = 0.05**.
- National TV outlets: exactly `{"ABC", "CBS", "NBC", "FOX", "ESPN"}` (Claude's call).
- Free CFBD tier: 1,000 calls/month, ~185 used. Live pulls in this plan budget ~205 more.

## Review Focus

1. **Teams that change conference** (e.g. Texas/Oklahoma to the SEC in 2024) — conference dummies come from that season's `team_seasons` row. → test in Task 3.
2. **Poll weeks with no ratings week exactly `w−1`** (bye weeks, week-0 games, poll week after championship week) — as-of join picks the latest ratings week `≤ w−1`; preseason gets neutral résumé (0 wins, Colley 0.5). → test in Task 5.
3. **Teams new to FBS** (no history, no prior-season Colley) — rows with any missing B are dropped from that analysis and the dropped count is reported, never silently imputed. → test in Task 7.
4. **Games without betting lines / teams with no lined game in the window** — market rating NaN for that team-week, handled as in (3). → test in Task 4.
5. **CFP poll has 0 points for every team** — CFP must use the rank score, never points share (would be all-zero). → test in Task 5.

---

### Task 1: Dependencies, schema additions, TV media ingest

**Files:**
- Modify: `pyproject.toml` (add `scikit-learn`, `statsmodels`), `src/cfb_bias/schema.sql`, `src/cfb_bias/ingest/runner.py` (add `"media"` source), `src/cfb_bias/cli.py` (nothing beyond `--only media` working)
- Create: `src/cfb_bias/ingest/media.py`, `tests/fixtures/media_2023_sample.json`, `tests/test_ingest_media.py`

**Interfaces:**
- New tables (all `CREATE TABLE IF NOT EXISTS`):
  - `game_media` PK `(game_id, outlet)`: `game_id, season, week, season_type, home_team, away_team, media_type, outlet`
  - `team_brand` PK `(season, team)`: `season, team, conference, alltime_win_pct, ap_titles, prior_colley, preseason_ap_score, recruiting_4yr, talent, enrollment`
  - `market_weekly` PK `(season, week, team)`: `season, week, team, market_rating, n_games`
  - `team_week_features` PK `(season, poll_week, team)`: `season, poll_week, team, ratings_week, ap_score, ap_rank_score, coaches_rank_score, cfp_rank_score, wins, losses, games_played, colley, colley_sos, massey, massey_sos, quality_wins, bad_losses, won_last, market_rating, national_tv_share`
  - `school_enrollment` PK `(team)`: `team, unitid, institution, undergrad_enrollment, source_year` (filled by Task 11; created now so Task 3 can read it while empty)
  - `analysis_params` PK `(key)`: `key, value` (text)
  - `bias_coefficients` PK `(analysis, variable)`: `analysis, variable, coef, se, ci_lo, ci_hi, p_value, n_obs, n_teams, n_dropped`
  - `team_bias` PK `(analysis, team)`: `analysis, team, n_seasons, n_weeks, mean_resid, se, ci_lo, ci_hi, p_value, q_value, significant`
  - `preseason_decay` PK `(poll, poll_week)`: `poll, poll_week, coef, se, ci_lo, ci_hi, n_obs`
- `media_to_frame(records: list[dict]) -> pd.DataFrame` — CFBD `/games/media` records (`id, season, week, seasonType, homeTeam, awayTeam, mediaType, outlet`) → `game_media` columns; rows with null `outlet` dropped; duplicates on `(game_id, outlet)` dropped.
- `pull_media(client, engine, season: int, refresh: bool = False) -> int` — uses `get_season_types(client, "/games/media", season, refresh)`.

- [ ] **Step 1:** `uv add scikit-learn statsmodels`.
- [ ] **Step 2: Write failing tests** — `test_media_to_frame_columns` (column list = `game_media` columns), `test_media_drops_null_outlet`, `test_media_dedupes_game_outlet`, `test_parse_only_accepts_media`, and a `@pytest.mark.db` `test_new_tables_exist` (each new table selectable after `init_schema`).
- [ ] **Step 3:** Run → FAIL. **Step 4:** Implement. **Step 5:** Run → PASS.
- [ ] **Step 6:** Commit: `feat: phase 1b schema, analysis deps, TV media ingest`

---

### Task 2: Historical success helpers

**Files:**
- Create: `src/cfb_bias/analysis/__init__.py`, `src/cfb_bias/analysis/history.py`, `tests/test_history.py`

**Interfaces:**
- `alltime_win_pct(games: pd.DataFrame, season: int, first_season: int = 1936) -> pd.Series` — index team; win % over completed games with non-null points in seasons `[first_season, season−1]`, any opponent classification; ties count ½. Teams with no games absent from the result.
- `ap_titles(polls: pd.DataFrame, season: int, first_season: int = 1936) -> pd.Series` — index team; number of seasons in `[first_season, season−1]` the team was rank 1 in that season's **final AP poll** = the `"AP Top 25"` poll-week with the latest `(season_type order regular < postseason, week)`.
- Input frames use the `games` / `poll_ranks` table columns.

- [ ] **Step 1: Write failing tests**

```python
def test_alltime_win_pct_counts_ties_half_and_excludes_current_season():
    g = games_frame([(1999, "A", "B", 21, 21), (1999, "A", "C", 10, 3), (2000, "A", "B", 0, 7)])
    s = alltime_win_pct(g, season=2000, first_season=1936)
    assert s["A"] == pytest.approx(0.75)  # 1 win + ½ tie over 2 games; 2000 excluded
    assert s["B"] == pytest.approx(0.25)

def test_ap_titles_uses_final_poll_only():
    p = polls_frame([(1999, "regular", 15, "B", 1), (1999, "postseason", 1, "A", 1),
                     (2000, "postseason", 1, "A", 1)])
    t = ap_titles(p, season=2001)
    assert t["A"] == 2 and t.get("B", 0) == 0
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: all-time win % and AP titles`

---

### Task 3: Team brand table

**Files:**
- Create: `src/cfb_bias/analysis/brand.py`, `tests/test_brand.py`
- Modify: `src/cfb_bias/cli.py` (`cfb brand --from 2014 --to 2025`)

**Interfaces:**
- Consumes: Task 2 helpers; tables `games`, `poll_ranks`, `team_seasons`, `team_recruiting`, `team_talent`, `ratings_weekly`, `school_enrollment` (Task 11; treat as absent/empty if the table has no rows).
- Produces: `build_team_brand(engine, season: int) -> pd.DataFrame` with `team_brand` columns for every FBS team in `team_seasons` that season:
  - `conference` = that season's `team_seasons.conference`
  - `prior_colley` = `ratings_weekly` `season−1`, `week = 99`, `method = 'colley'` (NaN if absent)
  - `preseason_ap_score` = AP points share in poll week 1 regular season of `season` (0 if unranked)
  - `recruiting_4yr` = mean `team_recruiting.points` for classes `season−3 … season` (mean of available; NaN if none)
  - `talent` = `team_talent.talent` for `season` (NaN if absent)
  - `enrollment` = from `school_enrollment` (NaN if absent)
- `conference_dummies(conf: pd.Series) -> pd.DataFrame` columns `sec, big_ten, big_12, acc, pac_12` (exact CFBD names `"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"`).
- CLI upserts into `team_brand`.

- [ ] **Step 1: Write failing tests** — `@db` seed tiny tables, assert each column for two teams; `test_conference_from_that_season` (Texas `Big 12` in 2023, `SEC` in 2024 → `big_12`=1 then `sec`=1); `test_preseason_unranked_is_zero`; `test_recruiting_4yr_window`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: team brand variables`

---

### Task 4: Market rating from betting lines

**Files:**
- Create: `src/cfb_bias/analysis/market.py`, `tests/test_market.py`
- Modify: `src/cfb_bias/cli.py` (`cfb market --from 2014 --to 2025`)

**Interfaces:**
- Consumes: `betting_lines`, `games` (for `neutral_site`), `cfb_bias.ratings.massey.massey(games, teams, cap, ridge)`.
- Produces:
  - `consensus_spread(lines: pd.DataFrame) -> pd.Series` — index `game_id`; median non-null `spread` across providers (CFBD home-perspective: negative = home favoured).
  - `market_ratings(games: pd.DataFrame, spreads: pd.Series, teams: list[str], poll_week: int) -> pd.DataFrame` columns `team, market_rating, n_games` — Massey fit (`cap=100`, `ridge=1.0`) with `points_a − points_b` replaced by expected margin `−spread`, on games with a spread in game weeks `[max(1, poll_week−3), poll_week]` (Claude's call: a 4-week window including the upcoming week, the market's view around poll release). Teams with no lined game in the window → `market_rating` NaN, `n_games` 0.
  - CLI writes `market_weekly` for every regular-season poll week of each season.

- [ ] **Step 1: Write failing tests** — `test_consensus_spread_median_ignores_null`; `test_market_rating_orders_favourites` (A −14 vs B, B −14 vs C → A > B > C); `test_window_excludes_old_games` (a week-1 game absent at poll_week 6); `test_team_without_lines_is_nan`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: market rating from consensus spreads`

---

### Task 5: Team-week features

**Files:**
- Create: `src/cfb_bias/analysis/features.py`, `tests/test_features.py`
- Modify: `src/cfb_bias/cli.py` (`cfb features --from 2014 --to 2025`)

**Interfaces:**
- Consumes: `games`, `poll_ranks`, `team_seasons`, `ratings_weekly`, `market_weekly`, `game_media`.
- Produces: `build_features(engine, season: int) -> pd.DataFrame` with `team_week_features` columns, one row per FBS team × regular-season AP poll week:
  - `ratings_week` = latest `ratings_weekly.week ≤ poll_week − 1` (NULL for preseason → `colley = 0.5`, `massey = 0`, counts 0, sos NULL)
  - `ap_score` = points ÷ max points that poll-week; `*_rank_score` = `(26 − rank)/25`; 0 if unranked; CFP rank score NULL for weeks with no CFP poll
  - `wins, losses, games_played, won_last` from completed regular-season games with `week ≤ poll_week − 1` (FCS opponents count)
  - `quality_wins` = wins over opponents with Colley rank ≤ 25 at `ratings_week`; `bad_losses` = losses to opponents with Colley rank > 60 at `ratings_week` or non-FBS
  - `national_tv_share` = share of the team's games through `poll_week − 1` with a `game_media.outlet` in the national set (NaN if no games)
  - `market_rating` from `market_weekly` at `poll_week`
- Pure helpers `asof_ratings_week(poll_weeks, ratings_weeks) -> dict[int, int | None]` and `poll_scores(polls, poll_name, use_points) -> DataFrame[poll_week, team, score]` are unit-tested without the DB.

- [ ] **Step 1: Write failing tests** — `test_asof_join_handles_bye_and_preseason` (ratings weeks {1,2,4}: poll 1→None, 3→2, 5→4, 6→4); `test_cfp_uses_rank_score_not_points` (CFP points all 0 → scores > 0 for ranked teams); `test_ap_points_share`; `@db test_build_features_counts_wins_through_prior_week`; `test_quality_wins_and_bad_losses`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: team-week résumé and exposure features`

---

### Task 6: Start week from rating stability

**Files:**
- Create: `src/cfb_bias/analysis/stability.py`, `tests/test_stability.py`

**Interfaces:**
- `rating_stability(ratings: pd.DataFrame) -> pd.DataFrame` columns `ratings_week, mean_spearman, n_seasons` — per season, Spearman correlation between Colley at each regular ratings week and Colley at `week = 99`, averaged over seasons.
- `choose_start_week(stability: pd.DataFrame, threshold: float = 0.75) -> int` — smallest poll week `w` such that `mean_spearman` at ratings week `w − 1` ≥ threshold (Claude's call for the threshold; stored with the result). Raises `ValueError` if never reached.

- [ ] **Step 1: Write failing tests** — synthetic ratings where week 4 first correlates ≥ 0.75 → start week 5; `test_never_stable_raises`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement with `scipy.stats.spearmanr` (scipy arrives with scikit-learn). **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: start week from rating stability`

---

### Task 7: Cross-fitted partialling-out estimator

**Files:**
- Create: `src/cfb_bias/analysis/dml.py`, `tests/test_dml.py`

**Interfaces:**
- `@dataclass DmlResult`: `coefficients: pd.DataFrame` (columns `variable, coef, se, ci_lo, ci_hi, p_value`), `y_resid: pd.Series` (aligned to the used rows' index), `n_obs: int`, `n_teams: int`, `n_dropped: int`.
- `partial_out(df: pd.DataFrame, y: str, x_cols: list[str], b_cols: list[str], group_col: str = "season", cluster_col: str = "team", seed: int = 0) -> DmlResult`:
  1. drop rows with NaN in `y` or any `b_cols` (count → `n_dropped`; NaN in `x_cols` is allowed — the learner handles it);
  2. z-score each `b_col` over the remaining rows (drop a `b_col` that is constant, e.g. an all-NaN enrollment, with a `warnings.warn`);
  3. leave-one-group-out cross-fitting: for each fold fit `HistGradientBoostingRegressor(random_state=seed)` of `y` on X and of each `b` on X, predict on the held-out fold;
  4. OLS of `ỹ` on all `b̃` (no intercept), `cov_type="cluster"` by `cluster_col`; 95% CIs.

- [ ] **Step 1: Write failing tests**

```python
def test_recovers_known_effect():
    df = synthetic(n_seasons=8, n_teams=60, theta=0.3, seed=1)  # y = sin(x1) + x2**2 + 0.3*z(b) + noise, b depends on x1
    r = partial_out(df, "y", ["x1", "x2"], ["b"])
    row = r.coefficients.set_index("variable").loc["b"]
    assert row.ci_lo < 0.3 < row.ci_hi

def test_null_effect_not_significant():  # theta = 0 → p > 0.05 for fixed seed
    ...

def test_rows_with_missing_brand_dropped_and_counted(): ...
def test_constant_brand_column_dropped_with_warning(): ...
def test_naive_regression_is_biased_but_dml_is_not():  # confounded b: OLS of y on b alone misses 0.3; partial_out covers it
    ...
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: cross-fitted partialling-out estimator with clustered SEs`

---

### Task 8: Per-team persistent bias with FDR control

**Files:**
- Create: `src/cfb_bias/analysis/team_effects.py`, `tests/test_team_effects.py`

**Interfaces:**
- `team_bias(resid: pd.Series, teams: pd.Series, seasons: pd.Series, min_seasons: int = 3, q: float = 0.05) -> pd.DataFrame` columns `team, n_seasons, n_weeks, mean_resid, se, ci_lo, ci_hi, p_value, q_value, significant` — team-season means of `resid`, then per team a one-sample t-test across its seasons; teams with `< min_seasons` excluded; `q_value` via `statsmodels.stats.multitest.multipletests(method="fdr_bh")`; `significant = q_value < q`. Sorted by `mean_resid` descending.

- [ ] **Step 1: Write failing tests** — `test_persistent_team_flagged` (team A +0.1 every season, others noise → A significant); `test_one_big_season_not_persistent` (team B +0.5 in one of 6 seasons, 0 otherwise → not significant); `test_min_seasons_excludes`; `test_bh_controls_false_positives` (130 null teams, seed fixed → ≤ 6 flagged).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: per-team persistent bias tests with BH FDR`

---

### Task 9: Preseason persistence curve

**Files:**
- Create: `src/cfb_bias/analysis/decay.py`, `tests/test_decay.py`

**Interfaces:**
- `preseason_decay(df: pd.DataFrame, y: str, poll: str) -> pd.DataFrame` columns `poll, poll_week, coef, se, ci_lo, ci_hi, n_obs` — for each regular poll week ≥ 2, OLS of `y` on `[colley, colley_sos, wins, losses, preseason_ap_score]` + intercept, cluster-robust by team; reports the `preseason_ap_score` coefficient.
- `half_life(decay: pd.DataFrame) -> int | None` — first poll week whose coefficient is ≤ half the week-2 coefficient (None if never).

- [ ] **Step 1: Write failing tests** — synthetic data where preseason weight halves every 3 weeks → coefficients decrease and `half_life` = 5; `test_half_life_none_when_flat`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: preseason persistence curve`

---

### Task 10: Analysis runner and `cfb analyze`

**Files:**
- Create: `src/cfb_bias/analysis/runner.py`, `tests/test_analysis_runner.py`
- Modify: `src/cfb_bias/cli.py` (`cfb analyze --from 2014 --to 2025`)

**Interfaces:**
- Consumes everything above. Loads `team_week_features` ⨝ `team_brand` (+ `conference_dummies`) for the window, computes `start_week` (Task 6, stored in `analysis_params` as `start_week` and `stability_threshold`), then runs and upserts:

| `analysis` | rows | y | X | B |
|---|---|---|---|---|
| `main` | AP regular weeks ≥ start_week, 2014–2025 | `ap_score` | résumé X | all B except `talent` |
| `margin_sensitivity` | as main | `ap_score` | résumé X + `massey, massey_sos` | as main |
| `talent_2015plus` | as main, seasons ≥ 2015 | `ap_score` | résumé X | main B + `talent` |
| `ap_rank`, `coaches_rank`, `cfp_rank` | weeks with that poll ≥ start_week | `*_rank_score` | résumé X | as main |

  For every analysis: `bias_coefficients` rows (one per B variable, with `n_obs, n_teams, n_dropped`) and `team_bias` rows from `team_bias(dml.y_resid, …)`. `preseason_decay` for `AP Top 25` (`ap_score`) and `Coaches Poll` (`coaches_rank_score`) over all regular weeks ≥ 2.
- `run_analyses(engine, seasons: range) -> dict[str, int]` returns rows written per table.

- [ ] **Step 1: Write failing test** — `@db` seed a small synthetic-but-complete dataset (3 seasons × 12 teams × 8 weeks with a planted +0.15 bias for one team) through the tables; `run_analyses` writes all six analyses, `analysis_params.start_week` exists, and the planted team has the largest `mean_resid` in `main`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: analysis runner and cfb analyze`

---

### Task 11: Undergraduate enrollment (College Scorecard)

**Files:**
- Create: `src/cfb_bias/ingest/scorecard.py`, `data/reference/team_institutions.csv`, `tests/test_scorecard.py`
- Modify: `src/cfb_bias/cli.py` (`cfb enrollment`), `.env.example` (`SCORECARD_API_KEY=`), `src/cfb_bias/config.py` (`scorecard_api_key: str | None`)

**Interfaces:**
- `team_institutions.csv` columns `team, unitid, institution` — one row per team in `team_seasons` 2014–2025; hand-verified (e.g. `Miami` → University of Miami, FL; `Miami (OH)` → Miami University-Oxford).
- `fetch_enrollment(api_key: str, unitids: list[int], transport=None) -> pd.DataFrame` columns `unitid, institution, undergrad_enrollment, source_year` — `GET https://api.data.gov/ed/collegescorecard/v1/schools?id=<comma ids>&fields=id,school.name,latest.student.size&api_key=…`, batches of ≤ 100 ids; `source_year` = the API's `latest` data year if returned, else NULL. Missing key → clear error naming `SCORECARD_API_KEY`.
- Enrollment is a single latest snapshot treated as time-invariant (Claude's call; documented in the results).

- [ ] **Step 1: Write failing tests** with `httpx.MockTransport`: field mapping, batching at 100, missing-key error; `test_reference_csv_covers_teams` (CSV has unique teams and integer unitids).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement; build the CSV (Scorecard name search for drafts, then hand-check every row against the institution's football program). **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: undergraduate enrollment from College Scorecard`

---

### Task 12: Live run and results summary

**Files:**
- Create: `docs/phase1b-results.md`

- [ ] **Step 1 (calls ≈ 205: 64 seasons × 3 + 12):** pull history and media:
  `for y in $(seq 1936 1999); do uv run cfb pull --season $y --only games,polls; done` and `for y in $(seq 2014 2025); do uv run cfb pull --season $y --only media; done`. Check `/info` remaining calls before and after.
- [ ] **Step 2:** `uv run cfb ratings --season 2013` (prior-season Colley for 2014). Verify the media outlet strings actually contain the five national outlets; if CFBD spells them differently, fix the set in Task 5's module with a test and note it.
- [ ] **Step 3:** If `SCORECARD_API_KEY` is set: `uv run cfb enrollment`; otherwise record "enrollment not run" and continue (the estimator drops the empty column with a warning).
- [ ] **Step 4:** `uv run cfb brand --from 2014 --to 2025 && uv run cfb market --from 2014 --to 2025 && uv run cfb features --from 2014 --to 2025 && uv run cfb analyze --from 2014 --to 2025`.
- [ ] **Step 5:** Write `docs/phase1b-results.md`: start week and stability threshold; `main` coefficient table; significant teams (q < 0.05) with mean residual and CI for `main`, plus whether each survives `margin_sensitivity` and appears in `coaches_rank`/`cfp_rank`; preseason half-life for AP and Coaches; dropped-row counts; any data caveats. Numbers only — narrative and charts are phase 1d.
- [ ] **Step 6:** Commit on `main` and push: `docs: phase 1b bias test results`.

## Deferred to later plans

- Individual AP ballots and voter report cards (1c), including the shared vs. idiosyncratic decomposition (spec step 9).
- Parquet snapshot, notebook and Quarto report (1d).

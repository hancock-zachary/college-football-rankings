# Phase 1a — Data Foundation & Poll-Free Ratings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A uv-managed Python package that pulls CollegeFootballData.com (CFBD) data into PostgreSQL (Docker), computes poll-free weekly résumé (Colley-style) and margin (Massey-style) ratings with uncertainty, and reports data coverage so the phase-1 start year can be chosen.

**Architecture:** `cfb_bias` package with four layers: a cached CFBD HTTP client → pure "records → DataFrame" transforms → a thin Postgres loader (idempotent upserts) → rating modules that read games from Postgres and write `ratings_weekly`. A Typer CLI (`cfb`) drives everything. Pure functions are unit-tested with fixtures; DB code has integration tests that skip when Postgres isn't running.

**Tech Stack:** Python 3.14, uv, httpx, pandas, numpy, SQLAlchemy 2 + psycopg 3, Typer, python-dotenv, pytest, ruff; PostgreSQL 17 via docker compose.

**Spec:** `docs/superpowers/specs/2026-10-20-cfb-ranking-bias-design.md`

**Scope note:** The spec's phase 1 is split into four plans, each producing working software: **1a (this plan)** data foundation + ratings; **1b** brand variables (enrollment, historical success, conference), résumé model, bias test, early-season persistence, poll comparison; **1c** AP ballot scraping, voter→outlet table, voter report cards, shared vs. idiosyncratic decomposition; **1d** Parquet snapshot, shareable notebook, Quarto report. 1b–1d are planned after 1a's coverage audit, because their details (start year, poll-week alignment, ballot site structure) depend on what the data actually contains.

## Global Constraints

- Python **3.14** (`requires-python = ">=3.14"`), managed with **uv**.
- **PostgreSQL in Docker** (docker compose) is the pipeline's primary store.
- Analysis code uses **pandas** (scikit-learn/statsmodels/seaborn arrive in later plans).
- Ratings are **poll-free and prior-free**: no poll, preseason, recruiting, or betting input may reach `cfb_bias.ratings`. Teams start each season equal.
- The résumé rating is **win/loss-based** (a win is a win regardless of margin); margin enters only the secondary Massey rating.
- Postseason weighting: seasons **≤ 2019** — all bowls and CFP games weight `0.5`; seasons **≥ 2020** — CFP games weight `0.25`, all other bowls weight `0` (excluded). Regular season and conference championships weight `1.0`. *(Exact values are the builder's call per spec; these are the chosen values.)*
- Raw API responses are cached on disk under `data/raw/` (gitignored); secrets come from `.env` (gitignored). `CFBD_API_KEY` is the env var.
- Be a polite client: cache every response, never re-fetch a cached response unless `--refresh`, sleep ≥ 0.5 s between live requests.

## Review Focus

1. **Cancelled / not-yet-played games** (`completed: false`, null points) must be stored but never reach a rating — expect them skipped, not counted as 0–0. → test in Task 6.
2. **Teams that change classification or conference between seasons** (e.g. an FCS team moving up) — classification/conference come from each game record for that season, never a global team table. → test in Task 4.
3. **Re-running `cfb pull` for a season** must not duplicate rows — upserts on natural keys. → test in Task 3.
4. **Bad/missing API key or exhausted quota** must fail with a clear message naming `CFBD_API_KEY`, and must not write a cache file for the failed response. → test in Task 2.
5. **Early-season sparse graphs** (week 1: many teams with zero FBS games, disconnected components) — ratings still solve; a team with no games gets the neutral Colley rating `0.5`. → test in Task 7.

---

### Task 1: Project scaffold, Docker Postgres, CLI skeleton

**Files:**
- Create: `pyproject.toml`, `.python-version`, `docker-compose.yml`, `docker/initdb/01-test-db.sql`, `.env.example`, `src/cfb_bias/__init__.py`, `src/cfb_bias/config.py`, `src/cfb_bias/cli.py`, `tests/conftest.py`, `tests/test_cli.py`
- Modify: `.gitignore` (add `data/raw/`, `.env`), `README.md` (setup: install uv, `docker compose up -d`, copy `.env.example`, get a free key at collegefootballdata.com/key)

**Interfaces:**
- Produces: `Settings` dataclass in `config.py` with `cfbd_api_key: str | None`, `database_url: str`, `cache_dir: Path`; `load_settings() -> Settings` (reads `.env` via python-dotenv; defaults `database_url="postgresql+psycopg://cfb:cfb@localhost:5432/cfb"`, `cache_dir=Path("data/raw")`). `cli.app: typer.Typer`; console script `cfb = "cfb_bias.cli:app"`.
- Compose: service `db`, image `postgres:17`, user/password/db `cfb`/`cfb`/`cfb`, port 5432, named volume, mounts `docker/initdb` to `/docker-entrypoint-initdb.d` (creates database `cfb_test`).
- `tests/conftest.py` provides fixture `pg_engine` connecting to `TEST_DATABASE_URL` (default `postgresql+psycopg://cfb:cfb@localhost:5432/cfb_test`), calling `pytest.skip("Postgres not running")` if the connection fails, and dropping/recreating all tables per test session.

- [ ] **Step 1:** `uv init --package --name cfb-bias --python 3.14`, add deps (`httpx pandas numpy sqlalchemy "psycopg[binary]" typer python-dotenv`) and dev deps (`pytest ruff`). Configure `[tool.pytest.ini_options] testpaths=["tests"]`, marker `db`.
- [ ] **Step 2: Write the failing test**

```python
from typer.testing import CliRunner
from cfb_bias.cli import app

def test_cli_help_lists_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("init-db", "pull", "ratings", "coverage"):
        assert cmd in result.output
```

- [ ] **Step 3:** Run `uv run pytest tests/test_cli.py -v` → FAIL (import error).
- [ ] **Step 4:** Implement `cli.py` with four stub commands (`init-db`, `pull`, `ratings`, `coverage`) that later tasks fill in; implement `config.py`.
- [ ] **Step 5:** `uv run pytest -v` → PASS; `docker compose up -d` then `docker compose exec db psql -U cfb -l` lists `cfb` and `cfb_test`.
- [ ] **Step 6:** Commit: `git commit -m "feat: scaffold uv project, docker postgres, cli skeleton"`

---

### Task 2: Cached CFBD client

**Files:**
- Create: `src/cfb_bias/cfbd.py`, `tests/test_cfbd.py`

**Interfaces:**
- Produces: `class CfbdError(RuntimeError)`; `class CfbdClient` with `__init__(self, api_key: str | None, cache_dir: Path, transport: httpx.BaseTransport | None = None, min_interval: float = 0.5)` and `get(self, path: str, params: dict | None = None, refresh: bool = False) -> list[dict]`.
- Base URL `https://api.collegefootballdata.com`; header `Authorization: Bearer <key>`.
- Cache file: `cache_dir / "cfbd" / path.strip("/").replace("/", "_") / f"{sha256(sorted params json)[:16]}.json"`, written atomically (temp file + rename) **only on HTTP 200**.
- Errors: no key → `CfbdError` mentioning `CFBD_API_KEY` before any request; 401/403 → `CfbdError` mentioning `CFBD_API_KEY`; 429 or quota response → `CfbdError` mentioning quota; 5xx → retry 3× with backoff (1 s, 2 s, 4 s) then `CfbdError`.

- [ ] **Step 1: Write failing tests** (use `httpx.MockTransport`, `min_interval=0`):

```python
def test_get_caches_response(tmp_path):
    calls = []
    def handler(req):
        calls.append(req); return httpx.Response(200, json=[{"id": 1}])
    c = CfbdClient("k", tmp_path, httpx.MockTransport(handler), min_interval=0)
    assert c.get("/games", {"year": 2023}) == [{"id": 1}]
    assert c.get("/games", {"year": 2023}) == [{"id": 1}]
    assert len(calls) == 1
    assert calls[0].headers["Authorization"] == "Bearer k"

def test_refresh_bypasses_cache(tmp_path): ...  # 2 calls when refresh=True on second get

def test_401_raises_and_writes_no_cache(tmp_path):
    c = CfbdClient("bad", tmp_path, httpx.MockTransport(lambda r: httpx.Response(401)), min_interval=0)
    with pytest.raises(CfbdError, match="CFBD_API_KEY"):
        c.get("/games", {"year": 2023})
    assert not list(tmp_path.rglob("*.json"))

def test_missing_key_raises_before_request(tmp_path): ...  # api_key=None, handler asserts never called

def test_5xx_retries_then_succeeds(tmp_path, monkeypatch): ...  # 500, 500, 200 → returns body; patch time.sleep
```

- [ ] **Step 2:** `uv run pytest tests/test_cfbd.py -v` → FAIL.
- [ ] **Step 3:** Implement `CfbdClient` per Interfaces.
- [ ] **Step 4:** `uv run pytest tests/test_cfbd.py -v` → PASS.
- [ ] **Step 5:** Commit: `feat: cached CFBD client with clear auth/quota errors`

---

### Task 3: Database schema and idempotent loader

**Files:**
- Create: `src/cfb_bias/db.py`, `src/cfb_bias/schema.sql`, `tests/test_db.py`
- Modify: `src/cfb_bias/cli.py` (`init-db` runs `init_schema`)

**Interfaces:**
- Produces: `get_engine(url: str) -> sqlalchemy.Engine`; `init_schema(engine) -> None` (executes `schema.sql`, all `CREATE TABLE IF NOT EXISTS`); `upsert(engine, table: str, df: pd.DataFrame, key: list[str]) -> int` (INSERT … ON CONFLICT (key) DO UPDATE; returns rows written); `read_sql(engine, sql: str, params: dict | None = None) -> pd.DataFrame`.
- Tables and primary keys (columns are exactly the DataFrame columns defined in Tasks 4–5 and 9):
  - `games` PK `(game_id)`
  - `poll_ranks` PK `(season, season_type, week, poll, team)`
  - `betting_lines` PK `(game_id, provider)`
  - `team_recruiting` PK `(season, team)`
  - `team_talent` PK `(season, team)`
  - `team_seasons` PK `(season, team)` — FBS membership per season
  - `coach_seasons` PK `(season, team, coach)`
  - `venues` PK `(venue_id)`
  - `ratings_weekly` PK `(season, week, team, method)`

- [ ] **Step 1: Write failing tests** (`@pytest.mark.db`, use `pg_engine`):

```python
def test_upsert_is_idempotent(pg_engine):
    df = pd.DataFrame({"season": [2023], "team": ["Ohio State"], "rank": [4], "points": [300.1]})
    upsert(pg_engine, "team_recruiting", df, ["season", "team"])
    df.loc[0, "rank"] = 3
    upsert(pg_engine, "team_recruiting", df, ["season", "team"])
    out = read_sql(pg_engine, "select * from team_recruiting")
    assert len(out) == 1 and out.loc[0, "rank"] == 3

def test_init_schema_twice_is_safe(pg_engine):
    init_schema(pg_engine); init_schema(pg_engine)
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run (with `docker compose up -d`) → PASS.
- [ ] **Step 5:** Commit: `feat: postgres schema and idempotent upsert`

---

### Task 4: Games and FBS team-season ingestion

**Files:**
- Create: `src/cfb_bias/ingest/__init__.py`, `src/cfb_bias/ingest/games.py`, `tests/fixtures/games_2023_sample.json`, `tests/fixtures/teams_fbs_2023_sample.json`, `tests/test_ingest_games.py`
- Modify: `src/cfb_bias/cli.py` (`pull --season N [--refresh]` calls ingest functions)

**Interfaces:**
- Consumes: `CfbdClient.get`, `upsert`.
- Produces:
  - `games_to_frame(records: list[dict]) -> pd.DataFrame` with columns `game_id, season, week, season_type, start_date, completed, neutral_site, conference_game, home_team, home_conference, home_classification, home_points, away_team, away_conference, away_classification, away_points, venue_id, notes, is_cfp`. Source fields (CFBD camelCase): `id, season, week, seasonType, startDate, completed, neutralSite, conferenceGame, homeTeam, homeConference, homeClassification, homePoints, awayTeam, awayConference, awayClassification, awayPoints, venueId, notes`. Missing keys → null (never KeyError).
  - `is_cfp` = `bool(record.get("playoff"))` **or** `notes` matches `r"college football playoff|\bcfp\b"` (case-insensitive). *Verify against real 2019 and 2024 postseason data in Task 10 and adjust the rule there if needed.*
  - `teams_to_frame(records: list[dict], season: int) -> pd.DataFrame` columns `season, team, conference, classification`.
  - `pull_games(client, engine, season: int, refresh: bool = False) -> int` — fetches `/games` with `year=season, seasonType="both"` (fall back to two calls `regular` + `postseason` if `both` is rejected) and `/teams/fbs?year=season`; upserts both.
- Fixture: 6–8 trimmed real-shape records including one FCS opponent, one neutral-site game, one `completed: false` game with null points, one postseason game whose notes contain "CFP Semifinal".

- [ ] **Step 1: Write failing tests**

```python
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

def test_cfp_flag_from_notes():
    df = games_to_frame(load_fixture("games_2023_sample.json")).set_index("game_id")
    assert df.loc[FIXTURE_CFP_ID, "is_cfp"]
    assert not df.loc[FIXTURE_NEUTRAL_ID, "is_cfp"]

def test_incomplete_game_kept_with_null_points():
    df = games_to_frame(load_fixture("games_2023_sample.json")).set_index("game_id")
    assert not df.loc[FIXTURE_INCOMPLETE_ID, "completed"]
    assert pd.isna(df.loc[FIXTURE_INCOMPLETE_ID, "home_points"])
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: ingest games and FBS team-seasons`

---

### Task 5: Polls, betting lines, recruiting, talent, coaches, venues

**Files:**
- Create: `src/cfb_bias/ingest/polls.py`, `src/cfb_bias/ingest/lines.py`, `src/cfb_bias/ingest/teams_meta.py`, matching fixtures under `tests/fixtures/`, `tests/test_ingest_other.py`
- Modify: `src/cfb_bias/cli.py` (`pull` also runs these; `--only games,polls,...` option)

**Interfaces:**
- Produces (each `*_to_frame` is pure; each `pull_*(client, engine, season, refresh=False) -> int` upserts):
  - `polls_to_frame(records) -> DataFrame` — flattens `PollWeek{season, seasonType, week, polls[{poll, isFinal, ranks[{rank, school, conference, firstPlaceVotes, points}]}]}` to columns `season, season_type, week, poll, is_final, rank, team, conference, first_place_votes, points`. Fetch `/rankings?year=season&seasonType=both` (or regular + postseason).
  - `lines_to_frame(records) -> DataFrame` — one row per `(game_id, provider)`: `game_id, season, week, season_type, home_team, away_team, provider, spread, spread_open, over_under, home_moneyline, away_moneyline`. `spread` is from the home team's perspective as CFBD reports it; store as-is.
  - `recruiting_to_frame(records) -> season, team, rank, points`; `talent_to_frame(records) -> season, team, talent`; `coaches_to_frame(records) -> season, team, coach, games, wins, losses, preseason_rank, postseason_rank` (one row per `seasons[]` entry, `coach = f"{firstName} {lastName}"`); `venues_to_frame(records) -> venue_id, name, city, state, capacity, latitude, longitude, dome, grass, construction_year`.

- [ ] **Step 1: Write failing tests** — one per transform, asserting column list and one known value from the fixture, e.g.:

```python
def test_polls_flatten():
    df = polls_to_frame(load_fixture("rankings_2023_sample.json"))
    ap = df[(df.poll == "AP Top 25") & (df.week == 1)]
    assert len(ap) == 25 and ap.rank.min() == 1

def test_lines_one_row_per_provider():
    df = lines_to_frame(load_fixture("lines_2023_sample.json"))
    assert not df.duplicated(["game_id", "provider"]).any()

def test_coaches_one_row_per_season(): ...
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: ingest polls, lines, recruiting, talent, coaches, venues`

---

### Task 6: Rating inputs — game weights and FBS filtering

**Files:**
- Create: `src/cfb_bias/ratings/__init__.py`, `src/cfb_bias/ratings/inputs.py`, `tests/test_rating_inputs.py`

**Interfaces:**
- Consumes: `games` DataFrame (Task 4 columns).
- Produces:
  - `game_weight(season: int, season_type: str, is_cfp: bool) -> float` — implements the Global Constraints weights.
  - `rating_games(games: pd.DataFrame, season: int, through_week: int | None, include_postseason: bool) -> pd.DataFrame` with columns `game_id, week, team_a, team_b, points_a, points_b, a_won, neutral_site, a_home, weight`. Rules: only `completed` games with non-null points; any team whose classification ≠ `"fbs"` **for that game** is renamed `"FCS"` (one pooled pseudo-team); games with weight 0 dropped; regular-season games with `week > through_week` dropped (`None` = all); postseason games only when `include_postseason`. `team_a` is the home team (or first-listed at neutral sites).

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("season,stype,cfp,w", [
    (2019, "regular", False, 1.0), (2019, "postseason", False, 0.5), (2019, "postseason", True, 0.5),
    (2020, "postseason", False, 0.0), (2024, "postseason", True, 0.25), (2024, "regular", False, 1.0)])
def test_game_weight(season, stype, cfp, w):
    assert game_weight(season, stype, cfp) == w

def test_incomplete_games_excluded(): ...      # completed=False row absent from output
def test_fcs_opponents_pooled(): ...            # FCS team names all → "FCS"
def test_2021_non_cfp_bowl_dropped(): ...
def test_through_week_cutoff(): ...             # week 5 game absent when through_week=4
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: rating inputs with postseason weighting rules`

---

### Task 7: Weighted Colley résumé rating with bootstrap uncertainty

**Files:**
- Create: `src/cfb_bias/ratings/colley.py`, `tests/test_colley.py`

**Interfaces:**
- Consumes: `rating_games` output (Task 6), plus `teams: list[str]` (all FBS teams that season + `"FCS"`).
- Produces:
  - `colley(games: pd.DataFrame, teams: list[str]) -> pd.Series` (index team, float rating). Weighted Colley: `C[i,i] = 2 + Σ w over i's games`, `C[i,j] = −Σ w over i–j games`, `b[i] = 1 + (Σ w·win − Σ w·loss)/2`; solve `C r = b` with `numpy.linalg.solve`. Margin is ignored.
  - `colley_bootstrap(games, teams, n: int = 500, seed: int = 0) -> pd.DataFrame` columns `team, rating, lo, hi` — `rating` from the full sample; `lo`/`hi` are the 5th/95th percentiles over `n` resamples of games with replacement.

- [ ] **Step 1: Write failing tests**

```python
def g(*rows):  # rows of (a, b, a_won, weight)
    return pd.DataFrame(rows, columns=["team_a", "team_b", "a_won", "weight"])

def test_unweighted_mean_is_half():
    r = colley(g(("A","B",True,1), ("B","C",True,1), ("C","D",False,1)), list("ABCD"))
    assert r.mean() == pytest.approx(0.5)

def test_team_with_no_games_is_neutral():
    r = colley(g(("A","B",True,1)), ["A","B","C"])
    assert r["C"] == pytest.approx(0.5)

def test_transitive_order():
    r = colley(g(("A","B",True,1), ("B","C",True,1)), list("ABC"))
    assert r["A"] > r["B"] > r["C"]

def test_margin_is_ignored():  # same result whatever points columns contain
    ...

def test_weight_zero_equals_absent_game(): ...
def test_bootstrap_interval_contains_rating(): ...  # lo <= rating <= hi for all teams; deterministic for fixed seed
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: weighted Colley résumé rating with bootstrap CIs`

---

### Task 8: Massey-style margin rating (secondary)

**Files:**
- Create: `src/cfb_bias/ratings/massey.py`, `tests/test_massey.py`

**Interfaces:**
- Produces: `massey(games: pd.DataFrame, teams: list[str], cap: int = 28, ridge: float = 1.0) -> pd.Series` — weighted least squares on `clip(points_a − points_b, −cap, cap) = r_a − r_b + h·a_home`, ridge penalty `ridge` on team ratings (not on `h`), ratings centred to mean 0. Also `massey_bootstrap(games, teams, n=500, seed=0, **kw) -> DataFrame[team, rating, lo, hi]` mirroring Task 7.

- [ ] **Step 1: Write failing tests** — `test_margin_order` (A beats B by 14, B beats C by 14 → A > B > C, A−C > A−B); `test_margin_cap` (a 70-point win produces the same ratings as a 28-point win); `test_centred` (mean ≈ 0); `test_team_with_no_games_is_zero`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement with `numpy.linalg.lstsq` on the augmented (ridge) system. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: Massey margin rating as sensitivity check`

---

### Task 9: Weekly ratings, SOS, conference strength; `cfb ratings`

**Files:**
- Create: `src/cfb_bias/ratings/weekly.py`, `tests/test_weekly.py`
- Modify: `src/cfb_bias/cli.py` (`ratings --season N [--bootstrap 500]`)

**Interfaces:**
- Consumes: `rating_games`, `colley_bootstrap`, `massey_bootstrap`, `read_sql`, `upsert`.
- Produces:
  - `strength_of_schedule(games: pd.DataFrame, ratings: pd.Series) -> pd.Series` — per team, weight-averaged rating of opponents faced (FCS pseudo-team included at its own rating).
  - `conference_strength(ratings: pd.Series, team_conf: pd.Series) -> pd.DataFrame` columns `conference, median_rating, n_teams` (median, per spec).
  - `weekly_ratings(games: pd.DataFrame, teams: list[str], season: int, n_boot: int = 500) -> pd.DataFrame` — for every regular-season week `w` present in `games`, ratings using games through week `w` (`include_postseason=False`), plus one final row-set with `week = 99` using all games including weighted postseason. Columns: `season, week, team, method ("colley"|"massey"), rating, lo, hi, sos`. Excludes the `"FCS"` pseudo-team from output.
  - CLI `cfb ratings --season N` reads `games` and `team_seasons` from Postgres, runs `weekly_ratings`, upserts into `ratings_weekly`.

- [ ] **Step 1: Write failing tests** — `test_sos_uses_opponent_ratings` (toy: A played B (0.7) and C (0.3) → 0.5); `test_conference_median_resists_outlier` (conference ratings [0.9, 0.5, 0.5, 0.5] → median 0.5); `test_weekly_cutoff_is_monotonic_in_games` (week-1 ratings use only week-1 games: a team idle in week 1 is 0.5 at week 1); `test_final_week_99_includes_cfp_only_with_weight` (2024 toy season: non-CFP bowl doesn't change week-99 ratings vs. regular-season-final; a CFP game does).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: weekly poll-free ratings, SOS and conference strength`

---

### Task 10: Coverage audit and first live pull

**Files:**
- Create: `src/cfb_bias/coverage.py`, `tests/test_coverage.py`, `docs/data-coverage.md` (generated)
- Modify: `src/cfb_bias/cli.py` (`coverage --from 2000 --to 2025 --out docs/data-coverage.md`)

**Interfaces:**
- Produces: `coverage_table(engine, seasons: range) -> pd.DataFrame` columns `season, fbs_teams, games_completed, games_with_lines, pct_games_with_lines, ap_poll_weeks, ap_has_preseason, coaches_poll_weeks, cfp_poll_weeks, cfp_games_flagged, recruiting_teams, talent_teams`; `write_coverage_md(df, path: Path) -> None` (markdown table plus a short "first week label" note per poll showing which CFBD `week` value holds the preseason AP poll).

- [ ] **Step 1: Write failing test** — seed `pg_engine` with a tiny two-season dataset; assert `pct_games_with_lines` and `ap_poll_weeks` per season.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. Commit: `feat: data coverage audit`
- [ ] **Step 5 (live, needs the author's `CFBD_API_KEY` in `.env`):** `docker compose up -d && uv run cfb init-db && for y in $(seq 2000 2025); do uv run cfb pull --season $y; done` then `uv run cfb coverage --from 2000 --to 2025`. Check API quota first via the CFBD `/info` endpoint; if 26 seasons × ~8 endpoints exceeds the free tier, pull 2010–2025 first.
- [ ] **Step 6:** Verify the CFP rule: in `docs/data-coverage.md`, `cfp_games_flagged` must be 2 or 3 for 2014–2023 and 11 for 2024–2025 (12-team format: 4 first-round + 4 quarterfinals + 2 semifinals + 1 final). If not, fix `is_cfp` in Task 4's module, add the failing real-record case to the fixture, re-run.
- [ ] **Step 7:** `uv run cfb ratings --season 2023` and spot-check: the 2023 final (`week = 99`) Colley top 5 should include Michigan and Washington (both entered the CFP undefeated). Record the result in `docs/data-coverage.md` under "Sanity check".
- [ ] **Step 8:** Commit `docs/data-coverage.md`: `docs: data coverage audit and ratings sanity check`. Bring the coverage table to the author to fix the phase-1 start year before writing plan 1b.

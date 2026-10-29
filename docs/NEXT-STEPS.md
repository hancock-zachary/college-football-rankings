# Where this project stands (2026-10-29)

## Done

- **Phase 1a** (data pipeline + poll-free ratings): complete. See `docs/superpowers/plans/2026-10-20-phase1a-data-foundation-and-ratings.md` and `docs/data-coverage.md`.
- **Phase 1b** (bias test): all code built, reviewed and committed (tasks 1–11 of `docs/superpowers/plans/2026-10-21-phase1b-bias-test.md`). 187 tests pass.
- Analysis window chosen: **2014–2025**.

## Not done

- **Phase 1b task 12:** `cfb analyze` has never completed on real data, and `docs/phase1b-results.md` is not written. The first live run failed while saving because the Postgres container stopped; the second was cancelled. Nothing was saved either time.
- **Enrollment:** needs a free College Scorecard key (api.data.gov) as `SCORECARD_API_KEY` in `.env`. Without it the analysis leaves enrollment out. Expect some teams in `data/reference/team_institutions.csv` to come back unresolved on the first `cfb enrollment` run; fix those names in the CSV.
- **Phases 1c** (AP ballots, voter report cards) and **1d** (notebook, data snapshot, Quarto report): not planned yet.

## Rebuilding on a new machine

The database and the API cache (`data/raw/`, ~112 MB) are not in git. Either copy `data/raw/` over from the old machine (then no API calls are needed), or re-pull (~400 CollegeFootballData.com calls; the free tier allows 1,000 a month).

```bash
uv sync
docker compose up -d
cp .env.example .env   # add CFBD_API_KEY (and SCORECARD_API_KEY if you have one)
uv run cfb init-db
for y in $(seq 1936 2013); do uv run cfb pull --season $y --only games,polls; done
for y in $(seq 2014 2025); do uv run cfb pull --season $y; done
for y in $(seq 2013 2025); do uv run cfb ratings --season $y; done
uv run cfb enrollment   # optional, needs SCORECARD_API_KEY
uv run cfb brand --from 2014 --to 2025
uv run cfb market --from 2014 --to 2025
uv run cfb features --from 2014 --to 2025
uv run cfb analyze --from 2014 --to 2025   # ~15 min; writes all results in one transaction
```

Then write `docs/phase1b-results.md` (plan task 12, step 5): start week, the `main` coefficients including the `__joint__` Wald row, significant teams and whether they hold up under `margin_sensitivity` and the Coaches/CFP runs, the preseason half-lives, and dropped-row counts.

## Decisions made during phase 1b (beyond the plan)

- `run_analyses` takes an optional `start_week` override (used by tests).
- Enrollment reference CSV stores institution name + state, not UNITIDs; `cfb enrollment` looks each school up and reports any that don't match exactly once.
- Per-team bias tests cover only teams ranked at least once in that analysis (never-ranked teams otherwise show tiny but "significant" effects).
- Each analysis has a joint Wald row `__joint__` (all brand variables = 0), since individual brand coefficients overlap heavily.
- The gradient-boosting learner has early stopping off, so every analysis uses the same learner.
- The market rating window includes the upcoming week's spreads (pre-game, but set after the poll is released).

## Caveats to state in the results

- Market and TV variables partly reflect earlier rankings, so describe them as "predicts", not "causes". Worth a sensitivity run with the market window ending at week w−1.
- Standard errors cluster by team only; a team × season sensitivity check is worth reporting.
- 2020 teams that were ranked before playing are dropped (no games yet); consider a 2020-excluded sensitivity run.
- The Coaches Poll decay curve uses the AP preseason score.
- The conference baseline mixes Group of Five teams with independents, including Notre Dame; the Pac-12 dummy covers only 2 teams in 2024–25.
- Team effects are tested against 0, not against the average residual.

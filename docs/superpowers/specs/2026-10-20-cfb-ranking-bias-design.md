# College Football Ranking Bias — Design

Date: 2026-10-20
Status: Draft for review

## Intent

Answer, with evidence a technical reader would accept: **is there persistent, unfair bias in college football polls?** The author suspects some programs (e.g. USC, Notre Dame) are ranked above what they earn on the field, but recognises that suspicion is itself a bias. The design therefore must let the data identify biased programs without any team being singled out in advance.

"Unfair bias" is defined operationally: **anything that predicts a team's poll position after accounting for what the team has done on the field.**

**A fair poll ranks résumé, not perceived quality (author's decision).** Rankings should reflect what a team has *done* — wins and losses against the schedule it actually played — not how good it is believed to be. "How good they are" (talent, recruiting, betting-market strength) is itself a bias: a talented team that loses games it shouldn't has not earned its rank, and a less talented team that finds ways to win has. Consequently, recruiting rank, talent composite, preseason rank and market expectations are all **tested as possible sources of bias**, never controlled for as legitimate signal.

The analysis only needs to show that bias exists and how large it is — not explain where it comes from (e.g. it need not prove USC's boost stems from the early-2000s era).

Follow-on questions, handled in later phases on the same pipeline:

1. Does last year's success determine next year's? (Author's hypothesis: yes.)
2. Does school size matter? (Hypothesis: there's a threshold — above/below it, size matters.)
3. Does historical program prestige inflate rankings? (Largely answered by phase 1's brand variables.)
4. Do recruiting classes still predict success, or is the transfer portal a better indicator?

## Deliverables

1. **Public report with charts** — hosted on the author's personal portfolio. Audience is technical: methods, model specs, uncertainty intervals, and multiple-comparison-adjusted results are shown, but every chart must still read clearly at a glance.
2. **Shareable, runnable notebook** — others can reproduce/explore the analysis. Runs from a **bundled data snapshot**; no API keys, Docker, or database needed by the reader.
3. **Background data pipeline** — pulls live from APIs and scraped sources for the author's own work. Not user-facing.

## Goals / Non-goals / Anti-goals

Goals
- A defensible, quantified answer to "is there persistent poll bias, and for which teams/conferences/voters?"
- Per-voter "report cards" and a decomposition of whole-poll bias into shared vs. idiosyncratic voter bias.
- A reusable pipeline that later phases build on.

Non-goals
- Explaining *why* a bias exists.
- Modelling injuries (treated as noise; assumed roughly evenly distributed across programs). May revisit later.
- Real-time / in-season dashboards.

Anti-goals (would make it a failure even if it "works")
- **Circularity:** any "neutral" benchmark that uses poll data (e.g. strength of schedule built from ranked wins) — the thing being tested must not leak into the yardstick.
- **Conference-brand leakage:** strength-of-schedule or conference strength inferred from reputation rather than results.
- **Cherry-picking:** declaring a team biased from one season or without correcting for testing ~130 teams.
- Presenting voter scores as accusations rather than estimates with uncertainty.

## Data

### Sources (to be verified during implementation)

| Data | Likely source | Notes |
|---|---|---|
| Games & scores, team records, venues/stadium capacity, coaches, weekly AP/Coaches/CFP polls incl. preseason, betting lines, recruiting rankings, team talent composite, transfer portal (2021+), game TV network/media | CollegeFootballData.com API (free key) | Main source. |
| School enrollment | College Scorecard API / IPEDS | Federal data. |
| Individual AP voter ballots | Scrape (e.g. collegepollarchive.com) | No standard API. Cache every page; rate-limit; respect site terms; credit sources in the report. |
| Voter → outlet → state/region | Hand-maintained table | Built once, updated each season. Needed for home/region bias. |
| "Marquee game of the week" | Proxy (TBD in a later phase) | No official list. Candidates: College GameDay sites, national TV windows, Fox Big Noon. Not needed for phase 1. |

### Time window

"Go back only as far as the data is good."
- Phase 1 primary window: approx. **2014 → present**, where betting lines, individual ballots, and detailed stats all coexist. Exact start year is decided after checking actual coverage.
- Portal vs. recruiting: 2021+.
- Questions where older data is solid (e.g. year-over-year success) may extend further back.

### Bowl / postseason rules (author's decision)

- **Seasons through 2019:** all bowl games may feed the score-based ratings, **down-weighted** relative to regular-season games.
- **2020 onward:** only **CFP games** are used, and only lightly weighted; other bowls are excluded (opt-outs and motivation changes make them unreliable).
- The **poll-bias test itself** uses regular season + conference championships only; postseason is excluded from it.

## Method (phase 1: bias)

1. **Résumé ratings (our own, poll-free, primary).** Built from game **results** — wins and losses — and the schedule's game graph, with no poll inputs and no preseason priors; teams start each season equal. *Claude's call:* a win/loss-based method in the spirit of the Colley matrix, so a win counts as a win regardless of margin. Strength of schedule is derived from these ratings (opponents' own results, recursively), so conference strength comes only from the games actually played.
2. **Margin-based ratings (secondary).** An Elo/Massey-style rating using score margins, also poll-free and prior-free. Used as a sensitivity check ("does the conclusion change if margin counts?"), not as the yardstick.
3. **Conference strength** is a by-product of team ratings, summarised with a robust statistic (median) so one outlier team (e.g. a dominant Miami) doesn't inflate its conference. Cross-conference calibration comes from non-conference games; its thin sample size is reported honestly.
4. **Betting markets are not the yardstick.** Under the résumé definition, market lines measure perceived quality, which is one of the things being tested. They are used to show *where* polls track perceived quality instead of résumé (e.g. a team ranked above its résumé but in line with the market is evidence of quality-based bias, not evidence of fairness).
5. **Résumé model.** Model each team's weekly poll position from on-field résumé factors only (record, poll-free SOS, quality of wins and losses, recency). Margin of victory is excluded from the primary model and included only in the sensitivity check.
6. **Bias test.** Regress the residual on brand and perceived-quality variables: historical success (titles, all-time win %), preseason rank, recruiting stars / talent composite, betting-market strength, enrollment, conference membership (e.g. "Big Ten"), TV exposure. Also compute per-team persistent residuals across weeks and seasons, with multiple-comparison correction. A team is "biased-for" only if the effect is consistent and survives correction.
7. **Early season (Claude's call).** The preseason poll and the first weeks, before ratings carry enough games, are mostly guesswork by construction. They are excluded from the main bias test (start week chosen once rating stability is measured) and analysed separately as **preseason persistence**: how long the preseason ordering lingers in the poll after résumés diverge from it.
8. **Voter report cards** (individual AP ballots). Per voter, each with confidence intervals and a minimum-ballot threshold below which a voter is flagged as "insufficient data":
   - deviation from the résumé benchmark (and, separately, how closely they track betting-market quality)
   - brand lean
   - home/region and conference favouritism
   - preseason stickiness (how slowly they move off preseason picks)
   - deviation from consensus (reported separately from accuracy — contrarian ≠ wrong)
9. **Aggregation.** Decompose whole-poll bias into **shared** bias (most voters lean the same way; a bigger panel won't fix it) vs. **idiosyncratic** bias (voters lean differently and largely cancel). Quantify how much cancels.

## Tooling (decisions)

Author's decisions:
- **Python 3.14**, managed with **uv**.
- **PostgreSQL in Docker** (docker compose) as the pipeline's primary store.
- **pandas** and **scikit-learn** for analysis.
- **seaborn** as primary plotting library (author is "not married" to it).

Claude's calls (author had no preference; open to change):
- **statsmodels** added for inference (p-values, CIs) — scikit-learn doesn't provide them.
- A few **Plotly** charts where hover materially helps (e.g. team-season scatter).
- **Quarto** to render the report as a static site (GitHub Pages / portfolio).
- **Jupyter** notebook with an "Open in Colab" badge for the shareable notebook.
- Raw API/scrape responses cached on disk (JSON/HTML); cleaned tables loaded into Postgres; the shareable snapshot is exported from Postgres to **Parquet**.
- Pipeline is a Python package in this repo with a CLI (e.g. `uv run cfb pull --season 2025`).
- The shareable notebook must run on Colab's (older) Python, so it avoids 3.14-only syntax and is tested on an older interpreter.

## Scope calls (Claude's, after builder review)

- **Polls:** AP is primary (only poll with public individual ballots). Coaches Poll and CFP rankings get a lighter side-by-side comparison of team-level bias.
- **Season coverage:** analysis runs through the **completed 2025 season**. 2026 (in progress as of this writing) is added once complete; no weekly re-runs.
- **Snapshot redistribution:** check CollegeFootballData.com and ballot-source terms before publishing. If raw redistribution isn't allowed, the snapshot ships only derived/aggregated tables sufficient to run the notebook — the notebook must still run without an API key.

## Phasing

1. **Phase 1 — Bias.** Pipeline for games, polls, betting lines, brand variables, ballots + voter table; poll-free ratings; résumé model; bias test; voter report cards; aggregation; report + notebook + snapshot.
2. Later phases (order TBD with the author): year-over-year success; school size threshold; recruiting vs. portal; marquee-game exposure.

## Left to the builder

- Exact résumé-rating and margin-rating algorithms and tuning, as long as they are poll-free, prior-free, and handle uncertainty (the résumé rating must be win/loss-based).
- Early-season cutoff week for the main bias test.
- Exact bowl down-weighting values.
- Résumé-model form (e.g. ordinal/rank regression vs. points-based), feature engineering, and how unranked teams are handled.
- Multiple-comparison method.
- Database schema, package layout, caching layout, CLI shape.
- Exact phase-1 start year, once data coverage is checked.

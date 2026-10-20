# college-football-rankings

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and Docker.
2. Install dependencies: `uv sync`
3. Start Postgres: `docker compose up -d` (creates databases `cfb` and `cfb_test`)
4. Copy the env template: `cp .env.example .env`
5. Get a free API key at <https://collegefootballdata.com/key> and put it in `.env` as `CFBD_API_KEY`.
6. Run `uv run cfb --help`; run tests with `uv run pytest`.

CREATE TABLE IF NOT EXISTS games (
    game_id bigint PRIMARY KEY,
    season integer,
    week integer,
    season_type text,
    start_date timestamptz,
    completed boolean,
    neutral_site boolean,
    conference_game boolean,
    home_team text,
    home_conference text,
    home_classification text,
    home_points double precision,
    away_team text,
    away_conference text,
    away_classification text,
    away_points double precision,
    venue_id bigint,
    notes text,
    is_cfp boolean
);

CREATE TABLE IF NOT EXISTS poll_ranks (
    season integer,
    season_type text,
    week integer,
    poll text,
    is_final boolean,
    rank integer,
    team text,
    conference text,
    first_place_votes integer,
    points double precision,
    PRIMARY KEY (season, season_type, week, poll, team)
);

CREATE TABLE IF NOT EXISTS betting_lines (
    game_id bigint,
    season integer,
    week integer,
    season_type text,
    home_team text,
    away_team text,
    provider text,
    spread double precision,
    spread_open double precision,
    over_under double precision,
    home_moneyline double precision,
    away_moneyline double precision,
    PRIMARY KEY (game_id, provider)
);

CREATE TABLE IF NOT EXISTS team_recruiting (
    season integer,
    team text,
    rank integer,
    points double precision,
    PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS team_talent (
    season integer,
    team text,
    talent double precision,
    PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS team_seasons (
    season integer,
    team text,
    conference text,
    classification text,
    PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS coach_seasons (
    season integer,
    team text,
    coach text,
    games integer,
    wins integer,
    losses integer,
    preseason_rank integer,
    postseason_rank integer,
    PRIMARY KEY (season, team, coach)
);

CREATE TABLE IF NOT EXISTS venues (
    venue_id bigint PRIMARY KEY,
    name text,
    city text,
    state text,
    capacity integer,
    latitude double precision,
    longitude double precision,
    dome boolean,
    grass boolean,
    construction_year integer
);

CREATE TABLE IF NOT EXISTS ratings_weekly (
    season integer,
    week integer,
    team text,
    method text,
    rating double precision,
    lo double precision,
    hi double precision,
    sos double precision,
    PRIMARY KEY (season, week, team, method)
);

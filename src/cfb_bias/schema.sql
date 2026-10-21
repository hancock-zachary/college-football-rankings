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

CREATE TABLE IF NOT EXISTS game_media (
    game_id bigint,
    season integer,
    week integer,
    season_type text,
    home_team text,
    away_team text,
    media_type text,
    outlet text,
    PRIMARY KEY (game_id, outlet)
);

CREATE TABLE IF NOT EXISTS team_brand (
    season integer,
    team text,
    conference text,
    alltime_win_pct double precision,
    ap_titles integer,
    prior_colley double precision,
    preseason_ap_score double precision,
    recruiting_4yr double precision,
    talent double precision,
    enrollment double precision,
    PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS market_weekly (
    season integer,
    week integer,
    team text,
    market_rating double precision,
    n_games integer,
    PRIMARY KEY (season, week, team)
);

CREATE TABLE IF NOT EXISTS team_week_features (
    season integer,
    poll_week integer,
    team text,
    ratings_week integer,
    ap_score double precision,
    ap_rank_score double precision,
    coaches_rank_score double precision,
    cfp_rank_score double precision,
    wins integer,
    losses integer,
    games_played integer,
    colley double precision,
    colley_sos double precision,
    massey double precision,
    massey_sos double precision,
    quality_wins integer,
    bad_losses integer,
    won_last boolean,
    market_rating double precision,
    national_tv_share double precision,
    PRIMARY KEY (season, poll_week, team)
);

CREATE TABLE IF NOT EXISTS school_enrollment (
    team text PRIMARY KEY,
    unitid bigint,
    institution text,
    undergrad_enrollment double precision,
    source_year integer
);

CREATE TABLE IF NOT EXISTS analysis_params (
    key text PRIMARY KEY,
    value text
);

CREATE TABLE IF NOT EXISTS bias_coefficients (
    analysis text,
    variable text,
    coef double precision,
    se double precision,
    ci_lo double precision,
    ci_hi double precision,
    p_value double precision,
    n_obs integer,
    n_teams integer,
    n_dropped integer,
    PRIMARY KEY (analysis, variable)
);

CREATE TABLE IF NOT EXISTS team_bias (
    analysis text,
    team text,
    n_seasons integer,
    n_weeks integer,
    mean_resid double precision,
    se double precision,
    ci_lo double precision,
    ci_hi double precision,
    p_value double precision,
    q_value double precision,
    significant boolean,
    PRIMARY KEY (analysis, team)
);

CREATE TABLE IF NOT EXISTS preseason_decay (
    poll text,
    poll_week integer,
    coef double precision,
    se double precision,
    ci_lo double precision,
    ci_hi double precision,
    n_obs integer,
    PRIMARY KEY (poll, poll_week)
);

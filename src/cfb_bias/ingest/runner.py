"""Run a selection of ingest sources for one season."""

from sqlalchemy import Engine

from cfb_bias.cfbd import CfbdClient
from cfb_bias.ingest.games import pull_games
from cfb_bias.ingest.lines import pull_lines
from cfb_bias.ingest.media import pull_media
from cfb_bias.ingest.polls import pull_polls
from cfb_bias.ingest.teams_meta import (
    pull_coaches,
    pull_recruiting,
    pull_talent,
    pull_venues,
)

# Season-scoped sources take (client, engine, season, refresh); venues has no season.
_SEASONAL = {
    "games": pull_games, "polls": pull_polls, "lines": pull_lines,
    "recruiting": pull_recruiting, "talent": pull_talent, "coaches": pull_coaches,
    "media": pull_media,
}
ALL_SOURCES = ("games", "polls", "lines", "recruiting", "talent", "coaches", "media", "venues")


def parse_only(value: str | None) -> list[str]:
    """Parse a comma-separated --only value into sources in canonical order."""
    if value is None:
        return list(ALL_SOURCES)
    names = {n.strip() for n in value.split(",") if n.strip()}
    unknown = sorted(names - set(ALL_SOURCES))
    if unknown or not names:
        raise ValueError(
            f"unknown source(s) {', '.join(unknown) or '(none given)'}; "
            f"choose from {', '.join(ALL_SOURCES)}"
        )
    return [s for s in ALL_SOURCES if s in names]


def pull_all(
    client: CfbdClient, engine: Engine, season: int, sources: list[str], refresh: bool = False
) -> dict[str, int]:
    """Pull each source; returns rows written per source."""
    written = {}
    for name in sources:
        if name == "venues":
            written[name] = pull_venues(client, engine, refresh=refresh)
        else:
            written[name] = _SEASONAL[name](client, engine, season, refresh=refresh)
    return written

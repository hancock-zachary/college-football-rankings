"""Shared fetch helper for CFBD endpoints that take a seasonType parameter."""

from cfb_bias.cfbd import CfbdClient, CfbdError

# Substrings of the CfbdError messages raised for a missing/rejected key or an exhausted
# quota (see cfb_bias.cfbd). Retrying with other seasonType values cannot fix these.
_NON_RETRYABLE_MARKERS = ("CFBD_API_KEY", "quota exceeded")


def get_season_types(client: CfbdClient, path: str, season: int, refresh: bool) -> list[dict]:
    """GET seasonType=both; if CFBD rejects it, fetch regular + postseason instead.

    Auth and quota errors propagate immediately rather than triggering the fallback.
    """
    base = {"year": season}
    try:
        return client.get(path, base | {"seasonType": "both"}, refresh=refresh)
    except CfbdError as exc:
        if any(marker in str(exc) for marker in _NON_RETRYABLE_MARKERS):
            raise
    regular = client.get(path, base | {"seasonType": "regular"}, refresh=refresh)
    post = client.get(path, base | {"seasonType": "postseason"}, refresh=refresh)
    return regular + post

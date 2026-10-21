"""Shared fetch helper for CFBD endpoints that take a seasonType parameter."""

from cfb_bias.cfbd import CfbdAuthError, CfbdClient, CfbdError, CfbdQuotaError


def get_season_types(client: CfbdClient, path: str, season: int, refresh: bool) -> list[dict]:
    """GET seasonType=both; if CFBD rejects it, fetch regular + postseason instead.

    Auth and quota errors propagate immediately rather than triggering the fallback,
    since retrying with other seasonType values cannot fix them.
    """
    base = {"year": season}
    try:
        return client.get(path, base | {"seasonType": "both"}, refresh=refresh)
    except (CfbdAuthError, CfbdQuotaError):
        raise
    except CfbdError:
        pass
    regular = client.get(path, base | {"seasonType": "regular"}, refresh=refresh)
    post = client.get(path, base | {"seasonType": "postseason"}, refresh=refresh)
    return regular + post

"""Undergraduate enrollment from the College Scorecard API.

Each team's institution (official name + state, from data/reference/team_institutions.csv)
is resolved with one name search. Only an unambiguous single match is accepted; zero or
multiple matches are returned as unresolved rather than guessed.
"""

import time

import httpx
import pandas as pd

SCHOOLS_URL = "https://api.data.gov/ed/collegescorecard/v1/schools"
FIELDS = "id,school.name,latest.student.size"
RETRY_BACKOFF_SECONDS = (1, 2, 4)  # same schedule as cfbd.py

RESOLVED_COLUMNS = ["team", "unitid", "institution", "undergrad_enrollment", "source_year"]
UNRESOLVED_COLUMNS = ["team", "institution", "state", "n_matches"]


class ScorecardError(RuntimeError):
    """Raised for any College Scorecard client failure."""


def _search(http: httpx.Client, api_key: str, institution: str, state: str) -> list[dict]:
    params = {
        "school.name": institution,
        "school.state": state,
        "fields": FIELDS,
        "api_key": api_key,
    }
    for attempt in range(len(RETRY_BACKOFF_SECONDS) + 1):
        retry = attempt < len(RETRY_BACKOFF_SECONDS)
        try:
            resp = http.get(SCHOOLS_URL, params=params)
        except httpx.TransportError as exc:
            if retry:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                continue
            raise ScorecardError(
                f"Scorecard request for {institution!r} failed ({type(exc).__name__}) after "
                f"{len(RETRY_BACKOFF_SECONDS)} retries."
            ) from exc
        status = resp.status_code
        if status == 200:
            return resp.json().get("results", [])
        if status in (401, 403):
            raise ScorecardError(
                f"Scorecard rejected the API key (HTTP {status}); check SCORECARD_API_KEY."
            )
        if status == 429:
            raise ScorecardError(
                "Scorecard quota exceeded for SCORECARD_API_KEY (HTTP 429); try again later."
            )
        if status >= 500:
            if retry:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                continue
            raise ScorecardError(
                f"Scorecard server error (HTTP {status}) for {institution!r} after "
                f"{len(RETRY_BACKOFF_SECONDS)} retries."
            )
        raise ScorecardError(f"Scorecard request failed (HTTP {status}) for {institution!r}.")
    raise AssertionError("unreachable")  # pragma: no cover


def resolve_and_fetch(
    api_key: str | None,
    institutions: pd.DataFrame,
    transport: httpx.BaseTransport | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (resolved, unresolved) frames for `institutions` (team, institution, state)."""
    if not api_key:
        raise ScorecardError(
            "SCORECARD_API_KEY is not set; add it to .env or the environment "
            "(free key: https://api.data.gov/signup/)."
        )
    resolved: list[dict] = []
    unresolved: list[dict] = []
    with httpx.Client(transport=transport, timeout=60.0) as http:
        for rec in institutions.itertuples(index=False):
            results = _search(http, api_key, rec.institution, rec.state)
            if len(results) == 1:
                hit = results[0]
                resolved.append({
                    "team": rec.team,
                    "unitid": hit["id"],
                    "institution": hit["school.name"],
                    "undergrad_enrollment": hit.get("latest.student.size"),
                    "source_year": None,
                })
            else:
                unresolved.append({
                    "team": rec.team,
                    "institution": rec.institution,
                    "state": rec.state,
                    "n_matches": len(results),
                })
    resolved_df = pd.DataFrame(resolved, columns=RESOLVED_COLUMNS).astype({
        "unitid": "Int64",
        "undergrad_enrollment": "Float64",
        "source_year": "Int64",
    })
    unresolved_df = pd.DataFrame(unresolved, columns=UNRESOLVED_COLUMNS)
    return resolved_df, unresolved_df

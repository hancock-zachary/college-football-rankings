from pathlib import Path

import httpx
import pandas as pd
import pytest

from cfb_bias.ingest import scorecard
from cfb_bias.ingest.scorecard import ScorecardError, resolve_and_fetch

CSV = Path(__file__).resolve().parents[1] / "data" / "reference" / "team_institutions.csv"


def _institutions(*rows):
    return pd.DataFrame(rows, columns=["team", "institution", "state"])


def _transport(handler, calls=None):
    def wrapped(request):
        if calls is not None:
            calls.append(request)
        return handler(request)

    return httpx.MockTransport(wrapped)


def _ok(results):
    return httpx.Response(200, json={"metadata": {"total": len(results)}, "results": results})


def test_single_match_maps_fields_and_sends_params():
    calls = []

    def handler(request):
        return _ok([{"id": 135726, "school.name": "University of Miami",
                     "latest.student.size": 11000}])

    resolved, unresolved = resolve_and_fetch(
        "KEY", _institutions(("Miami", "University of Miami", "FL")),
        transport=_transport(handler, calls),
    )
    assert unresolved.empty
    row = resolved.iloc[0]
    assert row["team"] == "Miami"
    assert row["unitid"] == 135726
    assert row["institution"] == "University of Miami"
    assert row["undergrad_enrollment"] == 11000
    assert pd.isna(row["source_year"])
    params = calls[0].url.params
    assert params["school.name"] == "University of Miami"
    assert params["school.state"] == "FL"
    assert params["fields"] == "id,school.name,latest.student.size"
    assert params["api_key"] == "KEY"
    assert calls[0].url.path == "/ed/collegescorecard/v1/schools"


def test_zero_and_multiple_matches_are_unresolved():
    def handler(request):
        name = request.url.params["school.name"]
        if name == "Nowhere U":
            return _ok([])
        if name == "Twin U":
            return _ok([{"id": 1, "school.name": "Twin U", "latest.student.size": 5},
                        {"id": 2, "school.name": "Twin U Two", "latest.student.size": 6}])
        return _ok([{"id": 3, "school.name": "Good U", "latest.student.size": 7}])

    resolved, unresolved = resolve_and_fetch(
        "KEY",
        _institutions(("A", "Nowhere U", "OH"), ("B", "Twin U", "OH"), ("C", "Good U", "OH")),
        transport=_transport(handler),
    )
    assert resolved["team"].tolist() == ["C"]
    assert unresolved["team"].tolist() == ["A", "B"]
    assert unresolved["n_matches"].tolist() == [0, 2]
    assert list(unresolved.columns) == ["team", "institution", "state", "n_matches"]


def test_missing_key_names_env_var():
    with pytest.raises(ScorecardError, match="SCORECARD_API_KEY"):
        resolve_and_fetch(None, _institutions(("A", "X", "OH")))


def test_quota_error_on_429():
    transport = _transport(lambda r: httpx.Response(429))
    with pytest.raises(ScorecardError, match="quota"):
        resolve_and_fetch("KEY", _institutions(("A", "X", "OH")), transport=transport)


def test_5xx_is_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr(scorecard.time, "sleep", sleeps.append)
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503)
        return _ok([{"id": 3, "school.name": "Good U", "latest.student.size": 7}])

    resolved, _ = resolve_and_fetch(
        "KEY", _institutions(("C", "Good U", "OH")), transport=_transport(handler)
    )
    assert len(resolved) == 1
    assert sleeps == [1, 2]


def test_persistent_5xx_raises(monkeypatch):
    monkeypatch.setattr(scorecard.time, "sleep", lambda s: None)
    with pytest.raises(ScorecardError, match="after"):
        resolve_and_fetch(
            "KEY", _institutions(("C", "Good U", "OH")),
            transport=_transport(lambda r: httpx.Response(500)),
        )


def test_transport_error_is_retried(monkeypatch):
    monkeypatch.setattr(scorecard.time, "sleep", lambda s: None)
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("boom")
        return _ok([])

    _, unresolved = resolve_and_fetch(
        "KEY", _institutions(("C", "Good U", "OH")), transport=_transport(handler)
    )
    assert len(attempts) == 2
    assert unresolved["n_matches"].tolist() == [0]


def test_reference_csv_covers_teams():
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    assert list(df.columns) == ["team", "institution", "state"]
    assert len(df) >= 130
    assert df["team"].is_unique
    assert (df["team"].str.strip() != "").all()
    assert (df["institution"].str.strip() != "").all()
    assert df["state"].str.fullmatch(r"[A-Z]{2}").all()
    by_team = df.set_index("team")
    assert by_team.loc["Miami", "institution"] == "University of Miami"
    assert by_team.loc["Miami (OH)", "institution"] == "Miami University-Oxford"


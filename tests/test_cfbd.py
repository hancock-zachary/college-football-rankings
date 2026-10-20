import httpx
import pytest

from cfb_bias.cfbd import CfbdClient, CfbdError


def make_client(tmp_path, handler, key="k", min_interval=0):
    return CfbdClient(key, tmp_path, httpx.MockTransport(handler), min_interval=min_interval)


def test_get_caches_response(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json=[{"id": 1}])

    c = make_client(tmp_path, handler)
    assert c.get("/games", {"year": 2023}) == [{"id": 1}]
    assert c.get("/games", {"year": 2023}) == [{"id": 1}]
    assert len(calls) == 1
    assert calls[0].headers["Authorization"] == "Bearer k"
    assert len(list((tmp_path / "cfbd" / "games").glob("*.json"))) == 1


def test_cache_key_ignores_param_order(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json=[])

    c = make_client(tmp_path, handler)
    c.get("/games", {"year": 2023, "week": 1})
    c.get("/games", {"week": 1, "year": 2023})
    assert len(calls) == 1


def test_refresh_bypasses_cache(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json=[{"n": len(calls)}])

    c = make_client(tmp_path, handler)
    assert c.get("/games", {"year": 2023}) == [{"n": 1}]
    assert c.get("/games", {"year": 2023}, refresh=True) == [{"n": 2}]
    assert len(calls) == 2
    assert c.get("/games", {"year": 2023}) == [{"n": 2}]


def test_401_raises_and_writes_no_cache(tmp_path):
    c = make_client(tmp_path, lambda r: httpx.Response(401), key="bad")
    with pytest.raises(CfbdError, match="CFBD_API_KEY"):
        c.get("/games", {"year": 2023})
    assert not list(tmp_path.rglob("*.json"))


def test_403_raises_key_error(tmp_path):
    c = make_client(tmp_path, lambda r: httpx.Response(403))
    with pytest.raises(CfbdError, match="CFBD_API_KEY"):
        c.get("/games")


def test_429_raises_quota_error(tmp_path):
    c = make_client(tmp_path, lambda r: httpx.Response(429))
    with pytest.raises(CfbdError, match="quota"):
        c.get("/games")
    assert not list(tmp_path.rglob("*.json"))


def test_other_4xx_raises_with_status_and_path_no_retry(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(404)

    c = make_client(tmp_path, handler)
    with pytest.raises(CfbdError, match=r"404.*/games"):
        c.get("/games")
    assert len(calls) == 1
    assert not list(tmp_path.rglob("*.json"))


def test_missing_key_raises_before_request(tmp_path):
    def handler(req):
        raise AssertionError("no request should be made")

    c = make_client(tmp_path, handler, key=None)
    with pytest.raises(CfbdError, match="CFBD_API_KEY"):
        c.get("/games")


def test_5xx_retries_then_succeeds(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    responses = iter([500, 500, 200])

    def handler(req):
        status = next(responses)
        return httpx.Response(status, json=[{"ok": True}] if status == 200 else None)

    c = make_client(tmp_path, handler)
    assert c.get("/games") == [{"ok": True}]
    assert sleeps == [1, 2]


def test_5xx_exhausts_retries(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(503)

    c = make_client(tmp_path, handler)
    with pytest.raises(CfbdError, match="503"):
        c.get("/games")
    assert len(calls) == 4
    assert sleeps == [1, 2, 4]
    assert not list(tmp_path.rglob("*.json"))


def test_min_interval_only_between_live_requests(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    c = make_client(tmp_path, lambda r: httpx.Response(200, json=[]), min_interval=0.5)
    c.get("/games", {"year": 1})
    assert sleeps == []
    c.get("/games", {"year": 1})  # cache hit
    assert sleeps == []
    c.get("/games", {"year": 2})  # live
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= 0.5

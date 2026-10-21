"""Cached, polite HTTP client for the CollegeFootballData.com API."""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import httpx

BASE_URL = "https://api.collegefootballdata.com"
RETRY_BACKOFF_SECONDS = (1, 2, 4)


class CfbdError(RuntimeError):
    """Raised for any CFBD client failure."""


class CfbdAuthError(CfbdError):
    """The API key is missing or was rejected (HTTP 401/403)."""


class CfbdQuotaError(CfbdError):
    """The API quota is exhausted (HTTP 429)."""


class CfbdClient:
    def __init__(
        self,
        api_key: str | None,
        cache_dir: Path,
        transport: httpx.BaseTransport | None = None,
        min_interval: float = 0.5,
    ):
        self._api_key = api_key
        self._cache_dir = Path(cache_dir)
        self._http = httpx.Client(base_url=BASE_URL, transport=transport, timeout=60.0)
        self._min_interval = min_interval
        self._last_request: float | None = None

    def get(self, path: str, params: dict | None = None, refresh: bool = False) -> list[dict]:
        if not self._api_key:
            raise CfbdAuthError("CFBD_API_KEY is not set; add it to .env or the environment.")
        cache_file = self._cache_path(path, params)
        if not refresh and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        data = self._fetch(path, params)
        self._write_atomic(cache_file, data)
        return data

    def _cache_path(self, path: str, params: dict | None) -> Path:
        digest = hashlib.sha256(
            json.dumps(params or {}, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        return self._cache_dir / "cfbd" / path.strip("/").replace("/", "_") / f"{digest}.json"

    def _fetch(self, path: str, params: dict | None) -> list[dict]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for attempt in range(len(RETRY_BACKOFF_SECONDS) + 1):
            self._throttle()
            try:
                resp = self._http.get(path, params=params, headers=headers)
            except httpx.TransportError as exc:
                self._last_request = time.monotonic()
                if attempt < len(RETRY_BACKOFF_SECONDS):
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                    continue
                raise CfbdError(
                    f"CFBD request to {path} failed ({type(exc).__name__}) after "
                    f"{len(RETRY_BACKOFF_SECONDS)} retries."
                ) from exc
            self._last_request = time.monotonic()
            status = resp.status_code
            if status == 200:
                return resp.json()
            if status in (401, 403):
                raise CfbdAuthError(
                    f"CFBD rejected the API key (HTTP {status}) for {path}; check CFBD_API_KEY."
                )
            if status == 429:
                raise CfbdQuotaError(
                    f"CFBD quota exceeded for CFBD_API_KEY (HTTP 429) for {path}; try again later."
                )
            if status >= 500:
                if attempt < len(RETRY_BACKOFF_SECONDS):
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                    continue
                raise CfbdError(
                    f"CFBD server error (HTTP {status}) for {path} after "
                    f"{len(RETRY_BACKOFF_SECONDS)} retries."
                )
            raise CfbdError(f"CFBD request failed (HTTP {status}) for {path}.")
        raise AssertionError("unreachable")  # pragma: no cover

    def _throttle(self) -> None:
        if self._last_request is None:
            return
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    @staticmethod
    def _write_atomic(target: Path, data: list[dict]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

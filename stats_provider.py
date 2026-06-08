"""
stats_provider.py
=================
Pluggable live-stats layer for the hero database.

The engine doesn't care *where* win-rates / ban-rates / counters come from - it
just reads a ``HeroDB``.  This module lets you keep ``heroes.json`` as the
always-present seed/fallback while transparently overlaying fresh numbers from
any source you like:

    StatsProvider              abstract: ``fetch() -> {hero_name: {field: val}}``
      ├─ CallableStatsProvider   wrap any python function (great for tests)
      └─ HttpJsonStatsProvider   GET a JSON endpoint, map its fields to ours
    CachedStatsProvider        on-disk TTL cache around ANY provider, with
                               graceful "serve stale on network failure"
    StatsRepository            builds a HeroDB from seed + provider, and can
                               refresh it in place on a schedule

No third-party dependency: HTTP uses the standard-library ``urllib``.  Point
``HttpJsonStatsProvider`` at whatever source you choose and describe its shape
with ``record_path`` + ``field_map`` - nothing about a specific website is
hard-coded here.

Wiring (see main.py):
    provider = CachedStatsProvider(HttpJsonStatsProvider(url, field_map=...))
    repo = StatsRepository("heroes.json", provider)
    db = repo.build()                 # seed overlaid with live data (or cache)
    ...
    repo.refresh(db)                  # later, on a QTimer, mutates db in place
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence

from engine import HeroDB


# ===========================================================================
#  PROVIDERS
# ===========================================================================
class StatsProvider(ABC):
    """Returns ``{hero_name: {field: value, ...}}`` with partial records OK.

    Recognised fields mirror ``heroes.json``: win_rate, ban_rate, base_role,
    damage_type, owned, counters, countered_by, synergies, archetypes.
    """

    @abstractmethod
    def fetch(self, force: bool = False) -> Dict[str, dict]:
        ...


class CallableStatsProvider(StatsProvider):
    """Adapt any zero-arg callable into a provider (handy for tests / custom
    scrapers you write yourself)."""

    def __init__(self, fn: Callable[[], Dict[str, dict]]):
        self._fn = fn

    def fetch(self, force: bool = False) -> Dict[str, dict]:
        return self._fn() or {}


class HttpJsonStatsProvider(StatsProvider):
    """Fetch a JSON document over HTTP and normalise it to our schema.

    Parameters
    ----------
    url         : endpoint returning JSON.
    record_path : dotted path to the list/dict of hero records inside the
                  payload, e.g. ``"data.heroes"``.  ``None`` = payload itself.
    name_key    : key holding the hero name when records are a list.
    field_map   : maps OUR field name -> the source's key, e.g.
                  ``{"win_rate": "winRate", "ban_rate": "banRate"}``.
                  Any of our fields left unmapped default to the same key name.
    """

    OUR_FIELDS = ("win_rate", "ban_rate", "base_role", "damage_type", "owned",
                  "counters", "countered_by", "synergies", "archetypes")

    def __init__(self, url: str, record_path: Optional[str] = None,
                 name_key: str = "name",
                 field_map: Optional[Dict[str, str]] = None,
                 timeout: float = 8.0,
                 headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.record_path = record_path
        self.name_key = name_key
        self.field_map = field_map or {}
        self.timeout = timeout
        self.headers = headers or {"User-Agent": "mlbb-draft-overlay/1.0"}

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _dig(payload: Any, dotted: Optional[str]) -> Any:
        if not dotted:
            return payload
        node = payload
        for part in dotted.split("."):
            node = node[part]
        return node

    def _normalise_record(self, rec: dict) -> dict:
        out = {}
        for our in self.OUR_FIELDS:
            src = self.field_map.get(our, our)
            if src in rec and rec[src] is not None:
                out[our] = rec[src]
        return out

    # -- StatsProvider -----------------------------------------------------
    def fetch(self, force: bool = False) -> Dict[str, dict]:
        req = urllib.request.Request(self.url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        records = self._dig(payload, self.record_path)
        result: Dict[str, dict] = {}
        if isinstance(records, dict):
            # already name -> record
            for name, rec in records.items():
                if isinstance(rec, dict):
                    result[name] = self._normalise_record(rec)
        elif isinstance(records, Sequence):
            for rec in records:
                if isinstance(rec, dict) and rec.get(self.name_key):
                    result[str(rec[self.name_key])] = self._normalise_record(rec)
        else:
            raise ValueError("unexpected JSON shape at record_path")
        return result


class CachedStatsProvider(StatsProvider):
    """On-disk TTL cache wrapping any provider.

    * Fresh cache (age < ttl) is returned without hitting the inner provider.
    * On a successful inner fetch the cache is rewritten.
    * If the inner provider raises (e.g. the network is down) we serve the
      STALE cache rather than crashing the draft overlay.
    """

    def __init__(self, inner: StatsProvider, cache_path: str = ".stats_cache.json",
                 ttl: float = 3600.0):
        self.inner = inner
        self.cache_path = cache_path
        self.ttl = ttl

    def _read_cache(self, respect_ttl: bool) -> Optional[Dict[str, dict]]:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            if respect_ttl and (time.time() - blob.get("ts", 0)) > self.ttl:
                return None
            data = blob.get("data")
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _write_cache(self, data: Dict[str, dict]) -> None:
        try:
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"ts": time.time(), "data": data}, fh)
            os.replace(tmp, self.cache_path)              # atomic
        except OSError as exc:
            sys.stderr.write(f"[stats] cache write failed: {exc}\n")

    def fetch(self, force: bool = False) -> Dict[str, dict]:
        if not force:
            fresh = self._read_cache(respect_ttl=True)
            if fresh is not None:
                return fresh
        try:
            data = self.inner.fetch(force=force)
            self._write_cache(data)
            return data
        except Exception as exc:                          # network / parse error
            stale = self._read_cache(respect_ttl=False)
            if stale is not None:
                sys.stderr.write(f"[stats] live fetch failed ({exc}); "
                                 "serving cached stats.\n")
                return stale
            raise


# ===========================================================================
#  REPOSITORY
# ===========================================================================
class StatsRepository:
    """Owns the seed file + provider, and produces/refreshes a HeroDB."""

    def __init__(self, base_path: str = "heroes.json",
                 provider: Optional[StatsProvider] = None):
        self.base_path = base_path
        self.provider = provider

    def build(self) -> HeroDB:
        """Load the seed DB, then overlay live data if a provider is set.
        Always returns a usable DB - provider errors degrade to the seed."""
        db = HeroDB.load(self.base_path)
        if self.provider is not None:
            try:
                n = db.apply_updates(self.provider.fetch())
                sys.stderr.write(f"[stats] overlaid live data on {n} heroes.\n")
            except Exception as exc:
                sys.stderr.write(f"[stats] provider unavailable ({exc}); "
                                 "using seed heroes.json.\n")
        return db

    def refresh(self, db: HeroDB) -> int:
        """Force a fresh fetch and apply it to ``db`` in place.  Returns the
        number of heroes updated (0 on failure - never raises)."""
        if self.provider is None:
            return 0
        try:
            return db.apply_updates(self.provider.fetch(force=True))
        except Exception as exc:
            sys.stderr.write(f"[stats] refresh failed: {exc}\n")
            return 0


# ===========================================================================
#  SELF-TEST  (no network: uses a callable provider + temp cache)
# ===========================================================================
if __name__ == "__main__":
    import tempfile

    db = HeroDB.load("heroes.json")
    before = db.get("Tigreal").win_rate
    print(f"Tigreal seed win_rate = {before}")

    feed = {"tigreal": {"win_rate": 57.3, "ban_rate": 41.0,
                        "counters": ["Layla", "Hanabi"]}}
    cache = os.path.join(tempfile.gettempdir(), "mlbb_stats_selftest.json")
    if os.path.exists(cache):
        os.remove(cache)

    flaky = {"down": False}

    def source():
        if flaky["down"]:
            raise RuntimeError("network down")
        return feed

    provider = CachedStatsProvider(CallableStatsProvider(source),
                                   cache_path=cache, ttl=9999)
    repo = StatsRepository("heroes.json", provider)

    db = repo.build()
    print(f"after live overlay   = {db.get('Tigreal').win_rate} "
          f"(ban {db.get('Tigreal').ban_rate}, counters {db.get('Tigreal').counters})")
    assert db.get("Tigreal").win_rate == 57.3
    assert db.get("Tigreal").base_role == "Tank"      # untouched field preserved

    # now simulate the network dying on refresh -> stale cache should be served
    flaky["down"] = True
    n = repo.refresh(db)
    print(f"refresh while offline updated {n} heroes (served from cache)")
    assert db.get("Tigreal").win_rate == 57.3

    os.remove(cache)
    print("STATS PROVIDER SELF-TEST: PASS")

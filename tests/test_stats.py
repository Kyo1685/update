"""
Tests for the live-stats overlay (HeroDB.apply_updates + stats_provider).

Run with:
    pytest -q
    python tests/test_stats.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import HeroDB
from stats_provider import (CachedStatsProvider, CallableStatsProvider,
                            StatsRepository)

HEROES = os.path.join(os.path.dirname(__file__), "..", "heroes.json")


def _db():
    return HeroDB.load(HEROES)


def _tmp_cache():
    return os.path.join(tempfile.gettempdir(),
                        f"mlbb_stats_test_{os.getpid()}.json")


def test_apply_updates_overlays_and_preserves():
    db = _db()
    assert db.get("Tigreal").base_role == "Tank"
    n = db.apply_updates({"tigreal": {"win_rate": 57.3, "ban_rate": 41.0,
                                      "counters": ["Layla", "Hanabi"]}})
    assert n == 1
    t = db.get("Tigreal")
    assert t.win_rate == 57.3 and t.ban_rate == 41.0
    assert t.counters == ("Layla", "Hanabi")          # list -> tuple
    assert t.base_role == "Tank"                       # untouched field kept


def test_apply_updates_ignores_unknown_and_bad():
    db = _db()
    assert db.apply_updates({"NotAHero": {"win_rate": 50}}) == 0
    assert db.apply_updates({"tigreal": {"win_rate": "oops"}}) == 0  # uncoercible


def test_callable_provider_build():
    provider = CallableStatsProvider(lambda: {"gord": {"ban_rate": 88.0}})
    repo = StatsRepository(HEROES, provider)
    db = repo.build()
    assert db.get("Gord").ban_rate == 88.0


def test_cached_provider_serves_stale_on_failure():
    cache = _tmp_cache()
    if os.path.exists(cache):
        os.remove(cache)
    flaky = {"down": False}

    def source():
        if flaky["down"]:
            raise RuntimeError("network down")
        return {"alice": {"win_rate": 60.0}}

    provider = CachedStatsProvider(CallableStatsProvider(source),
                                   cache_path=cache, ttl=9999)
    first = provider.fetch(force=True)                  # writes cache
    assert first["alice"]["win_rate"] == 60.0
    flaky["down"] = True
    stale = provider.fetch(force=True)                  # inner fails -> cache
    assert stale["alice"]["win_rate"] == 60.0
    os.remove(cache)


def test_repository_refresh_in_place():
    db = _db()
    base = db.get("Akai").ban_rate
    provider = CallableStatsProvider(lambda: {"akai": {"ban_rate": base + 25.0}})
    repo = StatsRepository(HEROES, provider)
    updated = repo.refresh(db)
    assert updated >= 1
    assert db.get("Akai").ban_rate == base + 25.0


def test_build_degrades_to_seed_on_provider_error():
    def boom():
        raise RuntimeError("boom")
    repo = StatsRepository(HEROES, CallableStatsProvider(boom))
    db = repo.build()                                   # must not raise
    assert len(db.all()) >= 50                          # seed still loaded


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)

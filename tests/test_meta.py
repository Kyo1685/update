"""
Tests for meta.py (the live meta from the official hero rank).  No network:
a fake client returns records in the official API's shape.

Run:  python tests/test_meta.py   (or: pytest -q)
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meta
from engine import HeroDB
from stats_provider import CachedStatsProvider, StatsRepository


def _rank_row(hid, name, win, ban, pick, best, worst):
    return {"main_heroid": hid, "main_hero": {"data": {"name": name}},
            "main_hero_win_rate": win, "main_hero_ban_rate": ban,
            "main_hero_appearance_rate": pick,
            "sub_hero": [{"heroid": h, "increase_win_rate": 0.03} for h in best],
            "sub_hero_last": [{"heroid": h, "increase_win_rate": -0.03} for h in worst]}


class FakeClient(meta.MoontonClient):
    """Official record shapes, canned.  Heroes: 1 Tigreal, 2 Minsitthar (the
    official spelling), 3 Layla, 4 Hirara (not in the local roster)."""
    calls = 0

    def hero_rank(self, days=7, rank="all", teammates=False):
        FakeClient.calls += 1
        if teammates:
            return [_rank_row(1, "Tigreal", .5, 0, 0, [3], []),
                    _rank_row(2, "Minsitthar", .5, 0, 0, [1], []),
                    _rank_row(3, "Layla", .5, 0, 0, [1, 2], []),
                    _rank_row(4, "Hirara", .5, 0, 0, [2], [])]
        return [_rank_row(1, "Tigreal", 0.5312, 0.0412, 0.0101, [3, 4], [2]),
                _rank_row(2, "Minsitthar", 0.506, 0.049, 0.02, [1], [3]),
                _rank_row(3, "Layla", 0.49, 0.001, 0.03, [], [1, 4]),
                _rank_row(4, "Hirara", 0.5507, 0.5897, 0.0079, [3], [2])]

    def _post(self, source, body):              # the roster endpoint
        role = lambda t: {"data": {"sort_title": t}}
        lane = lambda t: {"data": {"road_sort_title": t}}
        return [{"hero_id": 4, "hero": {"data": {
                    "name": "Hirara", "head": "https://example/hirara.png",
                    "sortid": [role("assassin"), ""],           # '' = no 2nd role
                    "roadsort": [lane("Jungle"), ""]}}},
                {"hero_id": 3, "hero": {"data": {
                    "name": "Layla", "head": "https://example/layla.png",
                    "sortid": [role("Marksman")], "roadsort": [lane("Gold Lane")]}}},
                {"hero_id": 2, "hero": {"data": {
                    "name": "Minsitthar", "head": "",
                    "sortid": [role("fighter")],
                    "roadsort": [lane("Exp Lane"), lane("Roam")]}}}]


LOCAL = ["Tigreal", "Minsithar", "Layla"]


def test_names_map_to_the_local_spelling():
    names = meta.NameMap(LOCAL)
    assert names("Minsitthar") == "Minsithar"       # official spelling
    assert names("tigreal") == "Tigreal"
    assert names("Hirara") == "Hirara"               # unknown: kept
    assert names.knows("Minsithar") and not names.knows("Hirara")


def test_fetch_meta_percentages_and_matchups():
    m = meta.fetch_meta(FakeClient(), 7, "all", meta.NameMap(LOCAL))
    t = m["Tigreal"]
    assert (t["win_rate"], t["ban_rate"], t["pick_rate"]) == (53.12, 4.12, 1.01)
    assert t["counters"] == ["Layla", "Hirara"]      # enemies it does best against
    assert t["countered_by"] == ["Minsithar"]        # mapped to the local name
    assert t["synergies"] == ["Layla"]               # best teammates
    assert m["Minsithar"]["counters"] == ["Tigreal"]


def test_roster_reads_roles_lanes_and_portraits():
    rows = {r["name"]: r for r in FakeClient().heroes()}
    assert rows["Hirara"]["roles"] == ["Assassin"] and rows["Hirara"]["lanes"] == ["JUNGLE"]
    assert rows["Minsitthar"]["lanes"] == ["EXP", "ROAM"]
    assert rows["Layla"]["portrait"].endswith("layla.png")


def test_live_provider_reports_new_heroes():
    p = meta.MoontonStatsProvider(LOCAL, client=FakeClient())
    data = p.fetch()
    assert "Minsithar" in data and p.new_heroes == ["Hirara"]
    db = HeroDB([])
    assert db.apply_updates(data) == 0               # unknown heroes are ignored


def test_update_heroes_keeps_manual_fields_and_adds_new_heroes():
    payload = {"version": 2, "heroes": [
        {"name": "Tigreal", "base_role": "Tank", "win_rate": 50.0, "ban_rate": 0.0,
         "counters": ["X"], "countered_by": [], "synergies": [],
         "damage_type": "Physical", "archetypes": ["Utility"], "owned": True,
         "lanes": ["ROAM"], "roles": ["Tank"]},
        {"name": "Layla", "base_role": "Marksman", "win_rate": 49.0, "ban_rate": 0.0,
         "counters": [], "countered_by": [], "synergies": [], "damage_type": "Physical",
         "archetypes": [], "owned": False, "lanes": ["GOLD", "MID"], "roles": ["Marksman"]}]}
    names = meta.NameMap([h["name"] for h in payload["heroes"]])
    client = FakeClient()
    out, changes, added = meta.update_heroes(
        payload, meta.fetch_meta(client, names=names), client.heroes(), names)
    by = {h["name"]: h for h in out["heroes"]}
    assert set(added) == {"Minsitthar", "Hirara"}      # not in this roster
    t = by["Tigreal"]
    assert t["win_rate"] == 53.12 and t["counters"] == ["Layla", "Hirara"]
    assert t["owned"] is True and t["archetypes"] == ["Utility"]    # manual fields kept
    assert t["lanes"] == ["ROAM"]                    # not in the roster: untouched
    assert by["Layla"]["lanes"] == ["GOLD"]          # official lanes
    assert any("Layla: lanes" in c for c in changes)
    h = by["Hirara"]
    assert (h["owned"], h["base_role"], h["lanes"], h["ban_rate"]) == (
        False, "Assassin", ["JUNGLE"], 58.97)
    assert out["version"] == 2 and out["meta"]["source"]
    # The result is a heroes.json the app loads.
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "heroes.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        db = HeroDB.load(path)
        assert db.get("Hirara").win_rate == 55.07 and db.get("Tigreal").owned
    finally:
        shutil.rmtree(tmp)


def test_app_starts_from_the_last_good_copy_without_network():
    """Startup must never wait on the website: build(network=False) uses the
    cached copy (or heroes.json) and makes no request."""
    tmp = tempfile.mkdtemp()
    try:
        heroes = os.path.join(tmp, "heroes.json")
        with open(heroes, "w", encoding="utf-8") as fh:
            json.dump({"heroes": [{"name": n, "base_role": "Tank"} for n in LOCAL]}, fh)
        cache = os.path.join(tmp, "meta_cache.json")
        live = meta.MoontonStatsProvider(LOCAL, client=FakeClient())
        repo = StatsRepository(heroes, CachedStatsProvider(live, cache_path=cache, ttl=3600))
        FakeClient.calls = 0
        db = repo.build(network=False)                   # no cache yet: the seed
        assert FakeClient.calls == 0 and db.get("Tigreal").win_rate == 50.0
        db.apply_updates(repo.provider.fetch(force=True))   # the background refresh
        assert db.get("Tigreal").win_rate == 53.12
        FakeClient.calls = 0
        db2 = repo.build(network=False)                  # next start: cached copy
        assert FakeClient.calls == 0 and db2.get("Tigreal").win_rate == 53.12
    finally:
        shutil.rmtree(tmp)


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

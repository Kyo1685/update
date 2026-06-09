"""
Lightweight tests for the scoring engine and lane optimiser.

Run with either:
    pytest -q
    python tests/test_engine.py        # falls back to a tiny runner
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from engine import DraftState, HeroDB, ScoringEngine, Settings
from detector import assign_lanes

DB = HeroDB.load(os.path.join(os.path.dirname(__file__), "..", "heroes.json"))
ENG = ScoringEngine(DB)


def _state():
    return DraftState(
        ally_picks=["Guinevere", "Zetian", "Minsithar", None, None],
        enemy_picks=["Alice", "Khufra", "Akai", "Gord", None],
        ally_bans=["Harley", "Karrie", "Sora", "Gloo", "Freya"],
        enemy_bans=["Freya", "Gusion", "Yi Sun-shin", "Sora", "Zhuxin"],
        ally_lanes={"EXP": "Guinevere", "JUNGLE": "Minsithar"},   # MID/GOLD/ROAM open
        enemy_lanes={"JUNGLE": "Alice", "ROAM": "Akai", "EXP": "Khufra", "MID": "Gord"},
    )


def test_database_loads():
    assert len(DB.all()) >= 50
    assert DB.get("guinevere").name == "Guinevere"        # case-insensitive
    assert DB.get("GORD").damage_type == "Magical"


def test_no_picked_or_banned_in_suggestions():
    res = ENG.evaluate(_state(), Settings())
    taken = _state().taken()
    for lane in config.LANES:
        for s in res.suggestions[lane]:
            assert s.name.lower() not in taken, f"{s.name} already taken"


def test_filled_lanes_get_no_suggestions():
    res = ENG.evaluate(_state(), Settings())
    # lanes already taken by an ally pick are grayed out (no recommendations)
    assert res.suggestions["EXP"] == [] and res.suggestions["JUNGLE"] == []
    assert res.filled_lanes.get("EXP") == "Guinevere"
    assert res.filled_lanes.get("JUNGLE") == "Minsithar"
    # open lanes still get recommendations
    assert len(res.suggestions["MID"]) > 0
    assert len(res.suggestions["GOLD"]) > 0


def test_dark_mode_inverts_ranking():
    st = _state()
    normal = ENG.evaluate(st, Settings()).suggestions["MID"]
    dark = ENG.evaluate(st, Settings(dark_mode=True)).suggestions["MID"]
    # The worst normal pick should bubble to (near) the top under dark mode.
    assert normal[0].name != dark[0].name
    assert dark[0].score < 0                                # inverted sign


def test_pool_filter_only_owned():
    res = ENG.evaluate(_state(), Settings(hero_pool_filter=True))
    for lane in config.LANES:
        for s in res.suggestions[lane]:
            assert s.hero.owned is True


def test_lane_counter_amplifies_direct_matchup():
    # Enemy JUNGLE = Gord; Joy (a jungler) hard-counters Gord, so Lane Counter
    # mode (3x on the direct match-up) must boost Joy hard.
    st = DraftState(enemy_picks=["Gord", None, None, None, None],
                    enemy_lanes={"JUNGLE": "Gord"})

    def joy(settings):
        res = ENG.evaluate(st, settings)
        return next((s.score for s in res.suggestions["JUNGLE"]
                     if s.name == "Joy"), None)

    base, lc = joy(Settings()), joy(Settings(lane_counter_mode=True))
    assert base is not None and lc is not None
    assert lc > base + 5


def test_assign_lanes_unique_and_optimal():
    lanes = assign_lanes(["Guinevere", "Zetian", "Minsithar", "Aamon", "Estes"], DB)
    assert len(set(lanes.values())) == len(lanes)          # no double-booked lane
    assert lanes.get("MID") == "Zetian"                    # only mage -> MID
    assert lanes.get("JUNGLE") == "Aamon"                  # only assassin -> JUNGLE
    assert lanes.get("ROAM") == "Estes"                    # only support -> ROAM


def test_build_path_reacts_to_enemy_profile():
    # 0 enemies -> waiting
    empty = ENG.evaluate(DraftState(), Settings())
    assert "AWAITING" in empty.build_path
    # magic-heavy enemy -> advise magic defence
    st = DraftState(enemy_picks=["Alice", "Gord", "Kadita", "Pharsa", None])
    res = ENG.evaluate(st, Settings())
    assert "MAGIC DEF" in res.build_path


def test_ban_relief_boosts_threatened_hero():
    # Lancelot is countered_by Khufra; banning Khufra should relieve Lancelot.
    def lanc(state):
        res = ENG.evaluate(state, Settings())
        return next((s.score for s in res.suggestions["JUNGLE"]
                     if s.name == "Lancelot"), None)
    base = lanc(DraftState())
    relieved = lanc(DraftState(enemy_bans=["Khufra", None, None, None, None]))
    assert base is not None and relieved is not None
    assert relieved > base


def test_probability_is_complementary():
    res = ENG.evaluate(_state(), Settings())
    assert abs(res.ally_win_pct + res.enemy_win_pct - 100.0) < 0.2


# --- tiny runner so the file works without pytest --------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)

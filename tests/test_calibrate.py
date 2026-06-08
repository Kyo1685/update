"""
Tests for calibration geometry + layout JSON persistence (no GUI, no network).

Run:  python tests/test_calibrate.py   (or: pytest -q)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from calibrate import layout_from_anchors, _interp, _pitch


def test_interp_evenly_spaces_five():
    pts = _interp((100, 100), (100, 500))
    assert pts == [(100, 100), (100, 200), (100, 300), (100, 400), (100, 500)]


def test_pitch():
    assert abs(_pitch((0, 0), (0, 400)) - 100.0) < 1e-6


def test_layout_from_anchors_shapes_and_size():
    L = layout_from_anchors(
        ally_pick=((100, 100), (100, 500)),     # vertical, pitch 100
        enemy_pick=((900, 100), (900, 500)),
        ally_ban=((50, 30), (450, 30)),         # horizontal, pitch 100
        enemy_ban=((900, 30), (1300, 30)),
        pick_factor=0.82, ban_factor=0.82,
    )
    assert len(L.ally_picks) == len(L.enemy_picks) == 5
    assert len(L.ally_bans) == len(L.enemy_bans) == 5
    b0 = L.ally_picks[0]
    assert b0.w == b0.h == 82                     # 0.82 * pitch(100)
    # centered on the first clicked point (100,100)
    assert b0.x + b0.w // 2 == 100 and b0.y + b0.h // 2 == 100
    # last ally pick centered at (100,500)
    b4 = L.ally_picks[4]
    assert b4.x + b4.w // 2 == 100 and b4.y + b4.h // 2 == 500


def test_size_factor_scales_box():
    small = layout_from_anchors(((0, 0), (0, 400)), ((0, 0), (0, 400)),
                                ((0, 0), (400, 0)), ((0, 0), (400, 0)),
                                pick_factor=0.5)
    assert small.ally_picks[0].w == 50            # 0.5 * 100


def test_layout_json_roundtrip():
    L = config.build_layout(1366, 768)
    path = os.path.join(tempfile.gettempdir(), f"layout_test_{os.getpid()}.json")
    try:
        config.save_layout(L, path)
        L2 = config.load_layout(path)
        for a, b in zip(L.all_picks + L.all_bans, L2.all_picks + L2.all_bans):
            assert a.as_tuple() == b.as_tuple()
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_apply_region_updates_globals():
    original = dict(config.ACTIVE_REGION)
    try:
        L = config.build_layout(800, 600)
        config.apply_region({"left": 10, "top": 20, "width": 800, "height": 600}, L)
        assert config.ACTIVE_REGION["width"] == 800
        assert config.LAYOUT.ally_picks[0].as_tuple() == L.ally_picks[0].as_tuple()
    finally:
        config.apply_region(original, None)        # restore for other tests


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

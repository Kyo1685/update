"""
Tests for the detector: separate ban library + lane assignment.
Vision parts need numpy+opencv (skipped cleanly if absent).

Run:  python tests/test_detector.py   (or: pytest -q)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from engine import HeroDB
from detector import assign_lanes, DraftDetector, TemplateLibrary, np, cv2

DB = HeroDB.load(os.path.join(os.path.dirname(__file__), "..", "heroes.json"))


def _have_cv():
    if np is None or cv2 is None:
        print("(skipped: opencv not installed)")
        return False
    return True


def _patch(seed, h, w):
    r = np.random.default_rng(abs(hash(seed)) % (2 ** 32))
    return r.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_lane_assignment_one_per_role():
    lanes = assign_lanes(["Guinevere", "Aamon", "Kadita", "Hanabi", "Estes"], DB)
    assert lanes["MID"] == "Kadita"        # only mage
    assert lanes["GOLD"] == "Hanabi"       # only marksman
    assert lanes["JUNGLE"] == "Aamon"      # only assassin
    assert lanes["ROAM"] == "Estes"        # only support


def test_ban_row_uses_ban_library():
    if not _have_cv():
        return
    config.apply_region({"left": 0, "top": 0,
                         "width": config.RES_W, "height": config.RES_H},
                        config.build_layout())
    L = config.LAYOUT
    frame = np.full((config.RES_H, config.RES_W, 3), 18, np.uint8)

    pick_lib, ban_lib = TemplateLibrary(), TemplateLibrary()
    # A pick hero only the PICK library knows...
    pb = L.ally_picks[0]
    akai = _patch("Akai", pb.h, pb.w)
    frame[pb.y:pb.y2, pb.x:pb.x2] = akai
    pick_lib.add("Akai", akai)
    # ...and a DIFFERENT hero only the BAN library knows, planted in a ban slot.
    bb = L.ally_bans[0]
    tig = _patch("Tigreal", bb.h, bb.w)
    frame[bb.y:bb.y2, bb.x:bb.x2] = tig
    ban_lib.add("Tigreal", tig)

    det = DraftDetector(DB, library=pick_lib, ban_library=ban_lib)
    state = det.detect(frame)
    assert state.ally_picks[0] == "Akai"
    # Only resolvable through the ban library -> proves the ban row uses it.
    assert state.ally_bans[0] == "Tigreal"


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

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
from detector import (assign_lanes, predicted_role_label, DraftDetector,
                      TemplateLibrary, np, cv2)

DB = HeroDB.load(os.path.join(os.path.dirname(__file__), "..", "heroes.json"))


def _have_cv():
    if np is None or cv2 is None:
        print("(skipped: opencv not installed)")
        return False
    return True


def _patch(seed, h, w):
    # Smooth, colourful, low-frequency patch (survives resize-correlation, like
    # real hero art - unlike pure noise once the histogram fallback is off).
    r = np.random.default_rng(abs(hash(seed)) % (2 ** 32))
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:, :, 0] = int(r.integers(0, 180)); hsv[:, :, 1] = 200; hsv[:, :, 2] = 190
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for _ in range(3):
        col = tuple(int(x) for x in r.integers(0, 255, (3,)))
        x0, y0 = int(r.integers(0, w // 2)), int(r.integers(0, h // 2))
        cv2.rectangle(img, (x0, y0), (x0 + int(w * 0.4), y0 + int(h * 0.4)), col, -1)
    return img


def test_lane_assignment_one_per_role():
    lanes = assign_lanes(["Guinevere", "Aamon", "Kadita", "Hanabi", "Estes"], DB)
    assert lanes["MID"] == "Kadita"        # only mage
    assert lanes["GOLD"] == "Hanabi"       # only marksman
    assert lanes["JUNGLE"] == "Aamon"      # only assassin
    assert lanes["ROAM"] == "Estes"        # only support


def _lane_of(name, lanes):
    return next((ln for ln, hn in lanes.items() if hn == name), None)


def test_label_never_shows_a_lane_the_hero_cannot_play():
    # Two MID-only mages (Nana + Gord) collide, so the 1-to-1 optimiser must
    # bump one off MID.  The displayed tag must still be each hero's REAL lane -
    # Nana must read [MID], never [JUG] (the screenshot bug).
    lanes = assign_lanes(["Nana", "Melissa", "Lukas", "Gord"], DB)
    assert predicted_role_label("Nana", _lane_of("Nana", lanes), DB) == "MID"
    assert predicted_role_label("Gord", _lane_of("Gord", lanes), DB) == "MID"
    # Heroes sitting in a lane they actually play are unaffected.
    assert predicted_role_label("Melissa", _lane_of("Melissa", lanes), DB) == "GLD"
    assert predicted_role_label("Lukas", _lane_of("Lukas", lanes), DB) in ("EXP", "JUG")


def test_template_overlay_replaces_base():
    if not _have_cv():
        return
    base = TemplateLibrary()
    a = _patch("GordBase", 96, 96)
    b = _patch("GordOverride", 96, 96)      # visually different "side" crop
    base.add("Gord", a)
    ovr = TemplateLibrary()
    ovr.add("Gord", b)
    assert base.overlay(ovr) == 1
    assert len(base) == 1                    # replaced, not appended
    # The override art now self-matches better than the old base art.
    score_b = base.top_matches(b, 1)[0][1]
    score_a = base.top_matches(a, 1)[0][1]
    assert score_b > score_a


def test_grayed_slot_reads_as_not_picked():
    if not _have_cv():
        return
    config.apply_region({"left": 0, "top": 0,
                         "width": config.RES_W, "height": config.RES_H},
                        config.build_layout())
    L = config.LAYOUT
    frame = np.full((config.RES_H, config.RES_W, 3), 18, np.uint8)

    lib = TemplateLibrary()
    # Slot 0: a fully-coloured, LOCKED pick.
    b0 = L.ally_picks[0]
    pick = _patch("Melissa", b0.h, b0.w)
    frame[b0.y:b0.y2, b0.x:b0.x2] = pick
    lib.add("Melissa", pick)
    # Slot 1: same art but GRAYED (desaturated + dimmed) = hovered, not locked.
    b1 = L.ally_picks[1]
    gray = cv2.cvtColor(cv2.cvtColor(pick, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    gray = (gray.astype(np.float32) * 0.5).astype(np.uint8)
    frame[b1.y:b1.y2, b1.x:b1.x2] = gray

    det = DraftDetector(DB, ally_library=lib)
    state = det.detect(frame)
    assert state.ally_picks[0] == "Melissa"      # locked pick detected
    assert state.ally_pending[0] is False
    assert state.ally_pending[1] is True         # grayed slot flagged
    assert state.ally_picks[1] is None           # ...and NOT matched to a hero


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

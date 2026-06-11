"""
Tests for the detector: separate ban library + lane assignment.
Vision parts need numpy+opencv (skipped cleanly if absent).

Run:  python tests/test_detector.py   (or: pytest -q)
"""
import contextlib
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from engine import HeroDB, ScoringEngine, Settings
from detector import (assign_lanes, predicted_role_label, DraftDetector,
                      TemplateLibrary, np, cv2)

ROOT = os.path.join(os.path.dirname(__file__), "..")
DB = HeroDB.load(os.path.join(ROOT, "heroes.json"))


def _have_cv():
    if np is None or cv2 is None:
        print("(skipped: opencv not installed)")
        return False
    return True


def _patch(seed, h, w):
    # Smooth, colourful, low-frequency patch (survives resize-correlation, like
    # real hero art - unlike pure noise once the histogram fallback is off).
    # Seed from a STABLE hash (crc32), not Python's per-process-randomised
    # hash(), so the synthetic art - and the matches - are reproducible.
    r = np.random.default_rng(zlib.crc32(str(seed).encode()))
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


@contextlib.contextmanager
def _grayed_gate(sat=0.60, val=0.75):
    """Temporarily enable the 'NOT PICKED' gate (it ships OFF by default)."""
    saved = (config.LOCKED_REL_SATURATION, config.LOCKED_REL_VALUE)
    config.LOCKED_REL_SATURATION, config.LOCKED_REL_VALUE = sat, val
    try:
        yield
    finally:
        config.LOCKED_REL_SATURATION, config.LOCKED_REL_VALUE = saved


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


def test_hist_confirmed_fallback_without_seeds():
    """Two-signal fallback: with NO learned seeds (fresh install / right after
    a re-calibration), a hero whose downloaded template is weak but whose
    colour winner agrees with a template top-K candidate (Johnson) must still
    resolve - while colour-only impostors (name-bar crop reading as Gord, the
    squad-logo slot) stay BLANK.  Skinned heroes may blank; nothing may be
    mislabelled."""
    if not _have_cv() or not _templates_present():
        print("(skipped: cv2/templates absent)"); return
    fdir = os.path.join(ROOT, "tests", "fixtures", "real_draft")
    if not os.path.isdir(fdir):
        print("(skipped: real_draft fixtures absent)"); return
    # Build libs exactly like main.start_live but WITHOUT the learned overlays.
    f = lambda d: os.path.join(ROOT, d)
    sq = TemplateLibrary.from_dir(f(config.TEMPLATE_DIR))
    ci = TemplateLibrary.from_dir(f(config.TEMPLATE_CIRCLE_DIR), circular=True,
                                  flip=not config.PACK_FACES_ALLY)
    ci.overlay(TemplateLibrary.from_dir(f(config.TEMPLATE_ALLY_DIR), circular=True))
    sq.overlay(TemplateLibrary.from_dir(f(config.TEMPLATE_ENEMY_DIR)))
    libs = {"ally_pick": (ci, config.TEMPLATE_MATCH_THRESHOLD),
            "enemy_pick": (sq, config.ENEMY_MATCH_THRESHOLD),
            "ally_ban": (ci, config.BAN_MATCH_THRESHOLD),
            "enemy_ban": (ci, config.BAN_MATCH_THRESHOLD)}
    wrong = []
    resolved = {}
    for grp, truths in REAL_DRAFT_TRUTH.items():
        lib, thr = libs[grp]
        for i, want in enumerate(truths):
            crop = cv2.imread(os.path.join(fdir, f"{grp}_{i}.png"))
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if float(g.std()) < config.EMPTY_SLOT_STDDEV:
                got = None
            else:
                n, _s, m = lib.match(crop, thr)
                got = None if m == "low" else n
            resolved[f"{grp}_{i}"] = got
            if got is not None and got != want:
                wrong.append(f"{grp}_{i}: got {got}, want {want}")
    assert not wrong, f"mislabelled without seeds: {wrong}"
    # The two-signal fallback must recover Johnson (weak downloaded template).
    assert resolved["enemy_pick_1"] == "Johnson"
    # Garbage crops must stay blank (colour-only Gord is rejected).
    assert resolved["ally_pick_3"] is None and resolved["ally_pick_4"] is None
    # All ten bans must still resolve seed-free.
    for grp in ("ally_ban", "enemy_ban"):
        for i, want in enumerate(REAL_DRAFT_TRUTH[grp]):
            assert resolved[f"{grp}_{i}"] == want


def test_full_screen_1366x768_end_to_end():
    """THE user-setup test: a full 1366x768 desktop frame (their resolution,
    100% scaling), a layout shaped like their calibration (ally column left,
    enemy column right, ban rows top, native crop sizes), the REAL screen
    crops pasted in, and the REAL DraftDetector + libraries exactly as
    main.start_live builds them.  Every pick and ban must resolve - Johnson
    included - and nothing may be mislabelled."""
    if not _have_cv() or not _templates_present():
        print("(skipped: cv2/templates absent)"); return
    fdir = os.path.join(ROOT, "tests", "fixtures", "real_draft")
    if not os.path.isdir(fdir):
        print("(skipped: real_draft fixtures absent)"); return

    W, H = 1366, 768                      # the user's exact display
    frame = np.full((H, W, 3), 17, np.uint8)
    img = {f"{g}_{i}": cv2.imread(os.path.join(fdir, f"{g}_{i}.png"))
           for g in ("ally_pick", "enemy_pick", "ally_ban", "enemy_ban")
           for i in range(5)}

    def col(prefix, x, y0, pitch):
        boxes = []
        for i in range(5):
            c = img[f"{prefix}_{i}"]
            h, w = c.shape[:2]
            y = y0 + i * pitch
            frame[y:y + h, x:x + w] = c
            boxes.append(config.Box(x, y, w, h))
        return boxes

    def row(prefix, y, x0, pitch):
        boxes = []
        for i in range(5):
            c = img[f"{prefix}_{i}"]
            h, w = c.shape[:2]
            x = x0 + i * pitch
            frame[y:y + h, x:x + w] = c
            boxes.append(config.Box(x, y, w, h))
        return boxes

    layout = config.Layout(
        ally_picks=col("ally_pick", 96, 168, 104),     # left column, 82px boxes
        enemy_picks=col("enemy_pick", 1196, 168, 100),  # right column, 84px boxes
        ally_bans=row("ally_ban", 100, 80, 60),         # top-left ban row
        enemy_bans=row("enemy_ban", 100, 1000, 60),     # top-right ban row
    )
    ally, enemy, ban = _build_side_libs()
    det = DraftDetector(DB, ally_library=ally, enemy_library=enemy,
                        ban_library=ban, layout=layout)
    s = det.detect(frame)

    assert s.enemy_picks[1] == "Johnson", \
        f"JOHNSON failed on the full-screen test: {s.enemy_picks}"
    assert s.ally_picks == ["Nana", "Melissa", "Lukas", None, None], s.ally_picks
    assert s.enemy_picks == ["Vexana", "Johnson", "Layla", None, None], s.enemy_picks
    assert s.ally_bans == ["Sora", "Harley", "Gloo", "Hilda", "Minsithar"], s.ally_bans
    assert s.enemy_bans == ["Chou", "Selena", "Saber", "Hayabusa", "Miya"], s.enemy_bans
    # And the labels must SURVIVE a global animation pulse (sticky slots).
    pulsed = np.clip(frame.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    s2 = det.detect(pulsed)
    assert s2.enemy_picks[1] == "Johnson" and s2.ally_picks[:3] == s.ally_picks[:3]


def test_auto_learn_persists_confident_match(tmp_path=None):
    if not _have_cv():
        return
    import tempfile
    d = tempfile.mkdtemp(prefix="learned_")
    lib = TemplateLibrary()
    art = _patch("Helcurt", 96, 96)
    lib.add("Helcurt", art)
    lib.learn_dir = d
    # A confident template hit must be remembered (saved + reusable).
    assert lib.maybe_learn("Helcurt", art, 0.91) is True
    assert os.path.exists(os.path.join(d, "helcurt.png"))
    # A near-identical re-sight skips the write; a drifted-but-confident one
    # REFRESHES the memory so it tracks the screen.
    assert lib.maybe_learn("Helcurt", art, 0.99) is False      # already fresh
    assert lib.maybe_learn("Helcurt", art, 0.85) is True       # refreshed
    assert lib.maybe_learn("Gord", art, 0.50) is False         # below LEARN_MIN_SCORE
    # A library with no learn_dir never persists.
    lib2 = TemplateLibrary()
    assert lib2.maybe_learn("Helcurt", art, 0.99) is False


def test_template_overlay_adds_without_destroying():
    """An override/learned crop becomes the primary art but the PRIOR art is
    kept as an anchor: both the new and the old rendering of the hero must
    still self-match strongly (a bad replacement can only add evidence)."""
    if not _have_cv():
        return
    base = TemplateLibrary()
    a = _patch("GordBase", 96, 96)
    b = _patch("GordOverride", 96, 96)      # visually different "side" crop
    base.add("Gord", a)
    ovr = TemplateLibrary()
    ovr.add("Gord", b)
    assert base.overlay(ovr) == 1
    assert len(base) == 1                    # one hero entry, multi-source art
    score_b = base.top_matches(b, 1)[0][1]   # new art self-matches...
    score_a = base.top_matches(a, 1)[0][1]   # ...and the old art STILL does
    assert score_b > 0.9 and score_a > 0.9


def _vivid_patch(h, w, hue=110):
    """Deterministic, fully-saturated patch with HIGH-CONTRAST quadrants so its
    grayscale std stays well above the empty-slot gate even after dimming."""
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:, :, 0] = hue; hsv[:, :, 1] = 220; hsv[:, :, 2] = 235
    cv2.rectangle(hsv, (0, 0), (w // 2, h // 2),
                  ((hue + 60) % 180, 220, 40), -1)     # dark quadrant
    cv2.rectangle(hsv, (w // 2, h // 2), (w, h),
                  ((hue + 30) % 180, 200, 255), -1)    # bright quadrant
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


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
    pick = _vivid_patch(b0.h, b0.w)
    frame[b0.y:b0.y2, b0.x:b0.x2] = pick
    lib.add("Melissa", pick)
    # Slot 1: the SAME art but GRAYED (desaturated + dimmed) = hovered, not
    # locked.  Desaturating in HSV keeps the texture so it isn't read as empty.
    b1 = L.ally_picks[1]
    hsv = cv2.cvtColor(pick, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.15                          # strip saturation
    hsv[:, :, 2] *= 0.45                          # dim it
    gray = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    frame[b1.y:b1.y2, b1.x:b1.x2] = gray

    det = DraftDetector(DB, ally_library=lib)
    with _grayed_gate():                          # gate ships OFF; enable to test
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


def _build_side_libs():
    """Build ally / enemy / ban libraries exactly like main.start_live(),
    against the REAL bundled templates (so confusions are tested for real)."""
    f = lambda d: os.path.join(ROOT, d)
    ally_flip = not config.PACK_FACES_ALLY
    sq = TemplateLibrary.from_dir(f(config.TEMPLATE_DIR))
    ci_ally = TemplateLibrary.from_dir(f(config.TEMPLATE_CIRCLE_DIR),
                                       circular=True, flip=ally_flip)
    ally, enemy, ban = ci_ally, sq, ci_ally
    a_ovr = TemplateLibrary.from_dir(f(config.TEMPLATE_ALLY_DIR), circular=True)
    e_ovr = TemplateLibrary.from_dir(f(config.TEMPLATE_ENEMY_DIR))
    if len(a_ovr):
        ally.overlay(a_ovr)
    if len(e_ovr):
        enemy.overlay(e_ovr)
    # Highest-priority learned memory (same order as main.start_live), but
    # from FROZEN fixture seeds: the live templates_learned/ folders are
    # auto-refreshed during play, so tests must not depend on them.
    learned = TemplateLibrary.from_dir(
        f("tests/fixtures/learned_circle"), circular=True)
    learned_e = TemplateLibrary.from_dir(f("tests/fixtures/learned_enemy"))
    if len(learned):
        ally.overlay(learned, learned=True)
    if len(learned_e):
        enemy.overlay(learned_e, learned=True)
    return ally, enemy, ban


def _templates_present():
    return os.path.isdir(os.path.join(ROOT, config.TEMPLATE_CIRCLE_DIR))


def test_real_templates_have_no_named_confusions():
    """Helcurt must not read as Gord; Sora must not read as Ixia/Kalea/Ling."""
    if not _have_cv():
        return
    if not _templates_present():
        print("(skipped: bundled templates not present)"); return
    ally, enemy, _ = _build_side_libs()
    cases = [
        ("templates_ally/helcurt.png",  ally,  config.TEMPLATE_MATCH_THRESHOLD,
         "Helcurt", {"Gord"}),
        ("templates_enemy/helcurt.png", enemy, config.ENEMY_MATCH_THRESHOLD,
         "Helcurt", {"Gord"}),
        ("templates_circle/sora.png",   ally,  config.TEMPLATE_MATCH_THRESHOLD,
         "Sora", {"Ixia", "Kalea", "Ling"}),
        ("templates_enemy/sora.png",    enemy, config.ENEMY_MATCH_THRESHOLD,
         "Sora", {"Ixia", "Kalea", "Ling"}),
    ]
    for path, lib, thr, want, confus in cases:
        crop = cv2.imread(os.path.join(ROOT, path), cv2.IMREAD_COLOR)
        assert crop is not None, f"missing fixture {path}"
        name, _score, _m = lib.match(crop, thr)
        assert name == want, f"{path}: matched {name}, expected {want}"
        # Named look-alikes must trail the true hero decisively.
        tops = dict(lib.top_matches(crop, 16))
        for c in confus:
            assert tops.get(c, 0.0) < tops[want], \
                f"{path}: look-alike {c} ({tops.get(c):.2f}) >= {want} ({tops[want]:.2f})"


REAL_DRAFT_TRUTH = {
    # Ground truth for tests/fixtures/real_draft/ - genuine 1366x768 screen
    # crops captured by tools/diagnose.py from a live draft (real skins, the
    # red ban-slash overlay, tiny 51px ban icons and all).
    #   - ally_pick[3] is a SQUAD-LOGO placeholder slot -> must stay blank.
    #   - ally_pick[4] is EXCLUDED: that capture's box sat ~25px too low so the
    #     crop is the player-name bar ("BD 'Walking Fanny'"), not Helcurt's
    #     face.  No matcher can fix pixels that aren't in the box - the fix is
    #     re-calibrating that one slot.  Kept here only to document it.
    #   - "Sora" is a 2025 hero with NO portrait in any public DB, so its
    #     template is the user's own ban crop (the only Sora image that exists).
    #   - ally_pick[4] = None: Helcurt's box sits ~25px low so the crop is the
    #     player-name bar, not the avatar.  With the colour fallback OFF it stays
    #     BLANK (never mislabelled) - the fix is re-calibrating that one slot.
    "ally_pick": ["Nana", "Melissa", "Lukas", None, None],
    "enemy_pick": ["Vexana", "Johnson", "Layla", None, None],
    "ally_ban": ["Sora", "Harley", "Gloo", "Hilda", "Minsithar"],
    "enemy_ban": ["Chou", "Selena", "Saber", "Hayabusa", "Miya"],
}


def test_real_screen_crops_ground_truth():
    """End-to-end on REAL screen crops with the colour fallback OFF: every
    pick/ban resolves exactly via TEMPLATE (skins/Sora via learned memory) and
    NOTHING is mislabelled - the one hard slot (mis-calibrated Helcurt) stays
    blank, never wrong."""
    if not _have_cv() or not _templates_present():
        print("(skipped: cv2/templates absent)"); return
    fdir = os.path.join(ROOT, "tests", "fixtures", "real_draft")
    if not os.path.isdir(fdir):
        print("(skipped: real_draft fixtures absent)"); return
    ally, enemy, ban = _build_side_libs()
    libs = {"ally_pick": (ally, config.TEMPLATE_MATCH_THRESHOLD),
            "enemy_pick": (enemy, config.ENEMY_MATCH_THRESHOLD),
            "ally_ban": (ban, config.BAN_MATCH_THRESHOLD),
            "enemy_ban": (ban, config.BAN_MATCH_THRESHOLD)}
    correct = wrong = 0
    for grp, truths in REAL_DRAFT_TRUTH.items():
        lib, thr = libs[grp]
        for i, want in enumerate(truths):
            crop = cv2.imread(os.path.join(fdir, f"{grp}_{i}.png"))
            assert crop is not None, f"missing fixture {grp}_{i}.png"
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if float(g.std()) < config.EMPTY_SLOT_STDDEV:
                got = None
            else:
                n, _s, m = lib.match(crop, thr)
                got = None if m == "low" else n
            if got == want:
                correct += 1
            elif got is not None:
                wrong += 1
                print(f"  WRONG {grp}[{i}]: got {got}, want {want}")
    assert wrong == 0, f"{wrong} slots MISLABELLED (must be 0)"
    assert correct == 20, f"expected 20/20 to match expected, got {correct}"


def test_ally_circular_orientation_not_reflipped():
    """Guards the flip fix: ally avatars match the un-flipped pack (12/12).
    If PACK_FACES_ALLY is ever reverted the ally side mis-IDs everyone again."""
    if not _have_cv():
        return
    if not _templates_present():
        print("(skipped: bundled templates not present)"); return
    ally, _, _ = _build_side_libs()
    # Use heroes WITHOUT a learned override, so this tests the base pack's
    # orientation (a learned crop deliberately differs from the downloaded one).
    for h in ["Gord", "Harley", "Tigreal", "Eudora", "Aurora", "Balmond", "Helcurt"]:
        crop = cv2.imread(os.path.join(ROOT, "templates_circle", f"{h.lower()}.png"),
                          cv2.IMREAD_COLOR)
        if crop is None:
            continue
        name, score, _m = ally.match(crop, config.TEMPLATE_MATCH_THRESHOLD)
        assert name == h, f"ally {h} -> {name} ({score:.2f}); flip direction regressed?"


def _place(frame, box, img):
    frame[box.y:box.y2, box.x:box.x2] = cv2.resize(img, (box.w, box.h))


def _grayed(img):
    """Desaturate + dim a portrait to mimic an un-locked (hovered) avatar."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.20
    hsv[:, :, 2] *= 0.45
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def test_overlay_behaviors_helcurt_pending_and_sora_ban():
    """End-to-end on the REAL templates (the two behaviours asked about):
      * a GRAYED Helcurt reads 'NOT PICKED' (not a wrong hero, not Gord),
      * a LOCKED Helcurt reads Helcurt,
      * a banned Sora is detected and reaches the overlay's ban row."""
    if not _have_cv() or not _templates_present():
        print("(skipped: cv2/templates absent)"); return
    config.apply_region({"left": 0, "top": 0,
                         "width": config.RES_W, "height": config.RES_H},
                        config.build_layout())
    L = config.LAYOUT
    ally, enemy, ban = _build_side_libs()
    det = DraftDetector(DB, ally_library=ally, enemy_library=enemy, ban_library=ban)
    circ = lambda n: cv2.imread(
        os.path.join(ROOT, "templates_circle", f"{n}.png"), cv2.IMREAD_COLOR)

    def frame():
        return np.full((config.RES_H, config.RES_W, 3), 18, np.uint8)

    # Filler heroes WITHOUT a learned override (so the downloaded crop matches).
    fill = ["tigreal", "eudora", "aurora"]
    # --- grayed Helcurt among locked allies, Sora banned --------------------
    f = frame()
    for i, n in enumerate(fill):
        _place(f, L.ally_picks[i], circ(n))
    _place(f, L.ally_picks[3], _grayed(circ("helcurt")))   # hovered, not locked
    _place(f, L.ally_bans[0], circ("sora"))
    with _grayed_gate():                                    # gate ships OFF here
        s = det.detect(f)
    assert s.ally_pending[3] is True and s.ally_picks[3] is None    # NOT PICKED
    assert s.ally_picks[:3] == ["Tigreal", "Eudora", "Aurora"]      # others fine
    assert not any(s.ally_pending[:3])                              # no false flags
    assert "Sora" in s.ally_bans                                    # ban detected
    res = ScoringEngine(DB).evaluate(s, Settings())
    assert "Sora" in res.ally_bans                                  # reaches overlay

    # --- locked (full-colour) Helcurt -> detected as Helcurt ---------------
    f2 = frame()
    for i, n in enumerate(fill):
        _place(f2, L.ally_picks[i], circ(n))
    _place(f2, L.ally_picks[3], circ("helcurt"))
    s2 = det.detect(f2)
    assert s2.ally_pending[3] is False and s2.ally_picks[3] == "Helcurt"


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

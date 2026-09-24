"""
Tests for hero recognition: recognizer.py (DINOv2 + the double-check),
icon_match.py (the registered art) and DraftDetector on REAL draft screenshots.

The decision rules, the reference index, memory and the icon matcher are tested
with numpy + opencv only (a fake embedder stands in for DINOv2).  The real-model
test needs the optional AI stack and skips cleanly without it.

Real data - genuine slot crops from the user's 1366x768 screen, pasted back at
their calibrated slot boxes (recovered by locating their diag/ crops):
  real_draft           draft A, lossless captures from tools/diagnose.py
  real_draft_a_screen  draft A from a chat screenshot: compressed, with the old
                       overlay's box lines + labels on the portraits, and
                       Helcurt HOVERED (pre-selected, not locked) in slot 5
  real_draft_b         draft B from a chat screenshot

Run:  python tests/test_recognizer.py   (or: pytest -q)
"""
import contextlib
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import recognizer
from engine import HeroDB
from detector import DraftDetector, TemplateLibrary, np, cv2
from icon_match import IconMatcher, decide_ban, is_hover

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIX = os.path.join(ROOT, "tests", "fixtures")
DB = HeroDB.load(os.path.join(ROOT, "heroes.json"))
ART_DIRS = [os.path.join(ROOT, "templates_circle"), os.path.join(ROOT, "templates")]
# Persistent across runs (outside the repo) so the roster is embedded once.
TEST_CACHE = os.path.join(tempfile.gettempdir(), "mlbb_dino_test_cache.npz")

B = config.Box
USER_LAYOUT = config.Layout(
    ally_picks=[B(95, 156, 82, 82), B(94, 256, 82, 82), B(93, 355, 82, 82),
                B(92, 455, 82, 82), B(91, 555, 82, 82)],
    enemy_picks=[B(1192, 157, 84, 84), B(1192, 260, 84, 84), B(1192, 362, 84, 84),
                 B(1193, 465, 84, 84), B(1193, 568, 84, 84)],
    ally_bans=[B(83, 88, 51, 51), B(145, 88, 51, 51), B(207, 89, 51, 51),
               B(268, 89, 51, 51), B(330, 89, 51, 51)],
    enemy_bans=[B(985, 87, 51, 51), B(1048, 87, 51, 51), B(1111, 88, 51, 51),
                B(1173, 89, 51, 51), B(1236, 89, 51, 51)])

TRUTH_A = dict(ally_picks=["Nana", "Melissa", "Lukas", None, None],
               enemy_picks=["Vexana", "Johnson", "Layla", None, None],
               ally_bans=["Sora", "Harley", "Gloo", "Hilda", "Minsithar"],
               enemy_bans=["Chou", "Selena", "Saber", "Hayabusa", "Miya"])
TRUTH_B = dict(ally_picks=["Helcurt", "Sun", None, "Hanabi", None],
               enemy_picks=["Julian", "Fredrinn", "Miya", None, None],
               ally_bans=["Chou", "Nana", "Harley", "Vexana", "Minsithar"],
               enemy_bans=["Kalea", "Vexana", "Floryn", "Gloo", "Yu Zhong"])
# fixture set -> (truth, ally "NOT PICKED" flags; None = don't care)
SETS = {"real_draft": (TRUTH_A, [False, False, False, False, None]),
        "real_draft_a_screen": (TRUTH_A, [False, False, False, False, True]),
        "real_draft_b": (TRUTH_B, [False] * 5)}
GROUPS = (("ally_picks", "ally_pick"), ("enemy_picks", "enemy_pick"),
          ("ally_bans", "ally_ban"), ("enemy_bans", "enemy_ban"))


def _have_cv():
    if np is None or cv2 is None:
        print("(skipped: opencv not installed)")
        return False
    return True


@contextlib.contextmanager
def _setting(**values):
    saved = {k: getattr(config, k) for k in values}
    for k, v in values.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def _crop(fixture_set, group, i):
    img = cv2.imread(os.path.join(FIX, fixture_set, f"{group}_{i}.png"))
    assert img is not None, f"missing fixture {fixture_set}/{group}_{i}.png"
    return img


def _frame(fixture_set):
    """The user's 1366x768 screen with a draft's real crops at their boxes."""
    frame = np.full((768, 1366, 3), 17, np.uint8)
    for field, group in GROUPS:
        for i, b in enumerate(getattr(USER_LAYOUT, field)):
            frame[b.y:b.y2, b.x:b.x2] = _crop(fixture_set, group, i)
    return frame


class FakeEmbedder:
    """Deterministic stand-in for DINOv2: the unit-norm 8x8 colour thumbnail."""
    signature = "fake/v1"

    def __init__(self):
        self.embedded = 0

    def embed(self, images):
        self.embedded += len(images)
        out = []
        for im in images:
            v = cv2.resize(im, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            v -= v.mean()
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.array(out, np.float32)


def _art(seed, size=160):
    r = np.random.default_rng(seed)
    return cv2.resize(r.integers(0, 255, (8, 8, 3), dtype=np.uint8), (size, size),
                      interpolation=cv2.INTER_NEAREST)


def _ban_icon(art, box=51, d=44, at=(3, 3)):
    icon = np.full((box, box, 3), 30, np.uint8)
    icon[at[1]:at[1] + d, at[0]:at[0] + d] = cv2.resize(art, (d, d),
                                                        interpolation=cv2.INTER_AREA)
    return icon


def _quiet(_msg):
    pass


# ---------------------------------------------------------------------------
#  The double-check rule for picks (pure python)
# ---------------------------------------------------------------------------
def test_decide_clear_win_never_asks_the_referee():
    def boom(_names):
        raise AssertionError("the referee must not run on a clear win")
    assert recognizer.decide([(0.86, "Nana"), (0.79, "Joy")], boom) == (
        "Nana", 0.86, "dino")


def test_decide_blank_below_min_similarity():
    assert recognizer.decide([(0.50, "Atlas"), (0.40, "Hanzo")]) == (
        None, 0.50, "dino-low")


def test_decide_near_tie_is_refereed_by_the_art():
    rank = [(0.789, "Hanabi"), (0.750, "Ixia"), (0.740, "Odette"), (0.60, "Gord")]
    seen = []

    def referee(scores):
        def judge(names):
            seen.append(list(names))
            return {n: scores[n] for n in names if n in scores}
        return judge
    # The real Hanabi case: the art agrees with DINOv2's best.
    assert recognizer.decide(rank, referee({"Hanabi": 0.88, "Ixia": 0.67,
                                            "Odette": 0.59})) == (
        "Hanabi", 0.789, "dino+ncc")
    assert seen[-1] == ["Hanabi", "Ixia", "Odette"]    # only the tied candidates
    # The art is sure of another tied candidate -> that one.
    assert recognizer.decide(rank, referee({"Hanabi": 0.50, "Ixia": 0.93,
                                            "Odette": 0.55})) == (
        "Ixia", 0.750, "ncc")
    # Neither sure -> blank, never a guess.
    assert recognizer.decide(rank, referee({"Hanabi": 0.50, "Ixia": 0.52,
                                            "Odette": 0.49})) == (
        None, 0.789, "dino-tie")
    assert recognizer.decide(rank) == (None, 0.789, "dino-tie")
    assert recognizer.decide([]) == (None, 0.0, "none")


# ---------------------------------------------------------------------------
#  Reference index + references
# ---------------------------------------------------------------------------
def test_screen_crops_only_count_as_near_exact_matches():
    q = np.array([1.0, 0.0, 0.0], np.float32)
    clean_a = np.array([0.70, 0.714, 0.0], np.float32)      # sim 0.70
    screen_c_weak = np.array([0.80, 0.0, 0.60], np.float32)  # sim 0.80 < 0.85
    idx = recognizer.HeroIndex()
    idx.add("A", clean_a)
    idx.add("C", screen_c_weak, screen=True)
    assert [h for _, h in idx.rank(q)] == ["A"]              # C filtered out
    idx.add("C", np.array([0.95, 0.312, 0.0], np.float32), screen=True)
    assert idx.rank(q)[0][1] == "C"                          # near-exact counts


def test_references_are_cached_deduplicated_and_sorted_by_kind():
    if not _have_cv():
        return
    tmp = tempfile.mkdtemp()
    try:
        clean, mirror, screen = (os.path.join(tmp, d) for d in ("clean", "mirror", "screen"))
        for d in (clean, mirror, screen):
            os.makedirs(d)
        a = _art(1)
        cv2.imwrite(os.path.join(clean, "alpha.png"), a)
        cv2.imwrite(os.path.join(mirror, "alpha.png"), cv2.flip(a, 1))  # same art, mirrored
        cv2.imwrite(os.path.join(screen, "cici__pick__x1.png"), _art(3, 82))
        icons = IconMatcher({"Alpha": a})
        cache = os.path.join(tmp, "cache.npz")
        emb = FakeEmbedder()
        refs = recognizer.build_references(emb, icons, [clean, mirror], [screen],
                                           cache, log=_quiet, ban_px=44)
        views = len(recognizer.pick_reference_views(a))
        assert views == 12                  # 2 orientations x 3 zooms x sharp/soft
        assert emb.embedded == views + 2 + 1    # + a ban icon per side + the memory
        assert sorted(refs.pick.heroes()) == ["Alpha", "Cici"] and refs.pick.n_screen == 1
        assert refs.ban["ally"].heroes() == ["Alpha"] and refs.ban["enemy"].n_refs == 1
        emb2 = FakeEmbedder()
        recognizer.build_references(emb2, icons, [clean, mirror], [screen],
                                    cache, log=_quiet, ban_px=44)
        assert emb2.embedded == 0           # everything cached
        cv2.imwrite(os.path.join(screen, "cici__pick__x1.png"), _art(4, 82))
        emb3 = FakeEmbedder()
        recognizer.build_references(emb3, icons, [clean, mirror], [screen],
                                    cache, log=_quiet, ban_px=44)
        assert emb3.embedded == 1           # only the changed memory
    finally:
        shutil.rmtree(tmp)


def test_mislabelled_memories_are_ignored():
    """A remembered crop that clearly shows another hero than the one it is
    filed under (an old mislabel) must not be allowed to repeat the mistake."""
    if not _have_cv():
        return
    tmp = tempfile.mkdtemp()
    try:
        clean, screen = os.path.join(tmp, "clean"), os.path.join(tmp, "screen")
        os.makedirs(clean)
        os.makedirs(screen)
        a, b = _art(1), _art(2)
        cv2.imwrite(os.path.join(clean, "alpha.png"), a)
        cv2.imwrite(os.path.join(clean, "beta.png"), b)
        portrait = cv2.resize(a, (82, 82), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(screen, "beta__pick__bad.png"), portrait)  # it's Alpha
        cv2.imwrite(os.path.join(screen, "alpha__pick__ok.png"),
                    np.clip(portrait.astype(int) + 9, 0, 255).astype(np.uint8))
        cv2.imwrite(os.path.join(screen, "ghost__ban__bad.png"), _ban_icon(a))  # Alpha's ban
        cv2.imwrite(os.path.join(screen, "sora__ban__ok.png"), _ban_icon(_art(9)))
        icons = IconMatcher({"Alpha": a, "Beta": b})
        refs = recognizer.build_references(FakeEmbedder(), icons, [clean], [screen],
                                           None, log=_quiet, ban_px=44)
        assert len(refs.ignored) == 2, refs.ignored
        assert any("beta__pick__bad" in m for m in refs.ignored)
        assert any("ghost__ban__bad" in m and "Alpha" in m for m in refs.ignored)
        assert refs.pick.n_screen == 1                      # alpha__pick__ok kept
        assert refs.screen_ban.heroes() == ["Sora"]         # no art -> memory kept
    finally:
        shutil.rmtree(tmp)


def test_memory_saves_clear_wins_once_and_caps_per_hero():
    if not _have_cv():
        return
    tmp = tempfile.mkdtemp()
    try:
        emb = FakeEmbedder()
        idx = recognizer.HeroIndex()
        idx.add("Nana", emb.embed([_art(10)])[0])
        refs = recognizer.References(idx, {}, recognizer.HeroIndex())
        rec = recognizer.DinoRecognizer(emb, refs, IconMatcher({}), learn_dir=tmp)
        crop = _art(11)
        vec = emb.embed([crop])[0]
        win = [(0.90, "Nana"), (0.70, "Joy")]
        assert rec.maybe_learn("Nana", crop, vec, win)
        assert len(os.listdir(tmp)) == 1 and idx.n_screen == 1
        assert "__pick__" in os.listdir(tmp)[0]
        assert not rec.maybe_learn("Nana", crop, vec, win)            # duplicate
        assert not rec.maybe_learn("Nana", _art(12), emb.embed([_art(12)])[0],
                                   [(0.90, "Nana"), (0.88, "Joy")])  # margin too small
        with _setting(DINO_LEARN_MAX_PER_HERO=1):
            assert not rec.maybe_learn("Nana", _art(13), emb.embed([_art(13)])[0], win)
        assert recognizer.hero_name_from_file(os.listdir(tmp)[0]) == "Nana"
    finally:
        shutil.rmtree(tmp)


def test_load_falls_back_when_ai_stack_missing():
    saved, logs = recognizer.torch, []
    recognizer.torch = None
    try:
        assert recognizer.DinoRecognizer.load(log=logs.append) is None
        assert "template matching" in logs[0]
    finally:
        recognizer.torch = saved


# ---------------------------------------------------------------------------
#  The registered art on REAL screenshots (numpy + opencv only)
# ---------------------------------------------------------------------------
def _icons():
    return IconMatcher.from_dirs(ART_DIRS)


def test_every_real_ban_icon_is_named_by_the_registered_art():
    """30 real ban icons from three screenshots: every hero with public art is
    named with a wide lead; Sora (no public art) stays blank, never wrong."""
    if not _have_cv():
        return
    icons = _icons()
    for fixture_set, (truth, _pending) in SETS.items():
        for field, group in GROUPS[2:]:
            side = "ally" if field == "ally_bans" else "enemy"
            for i, want in enumerate(truth[field]):
                match = icons.rank_ban(_crop(fixture_set, group, i), side)
                hero, score, _method = decide_ban(match)
                where = f"{fixture_set}/{group}_{i}"
                if want == "Sora":
                    assert hero is None, f"{where}: Sora read as {hero}"
                    continue
                assert hero == want, f"{where}: {hero} ({match.rank[:3]})"
                assert match.margin >= 0.15, f"{where}: margin {match.margin:.2f}"


def test_hovered_pick_is_not_a_locked_pick():
    """Helcurt hovered (pre-selected) in draft A vs locked in draft B: the
    hover is drawn at half contrast; no locked real pick trips the check."""
    if not _have_cv():
        return
    icons = _icons()
    hover = icons.fit_pick(_crop("real_draft_a_screen", "ally_pick", 4), "Helcurt")
    assert hover.ncc > 0.85 and is_hover(hover), hover
    for fixture_set, (truth, _pending) in SETS.items():
        for field, group in GROUPS[:2]:
            for i, hero in enumerate(truth[field]):
                if hero:
                    fit = icons.fit_pick(_crop(fixture_set, group, i), hero)
                    assert not is_hover(fit), f"{fixture_set}/{group}_{i} {hero}: {fit}"


def test_template_mode_names_real_bans_by_the_registered_art():
    """Without the AI stack the bans still go through the registered art."""
    if not _have_cv():
        return
    ci = TemplateLibrary.from_dir(ART_DIRS[0], circular=True,
                                  flip=not config.PACK_FACES_ALLY)
    sq = TemplateLibrary.from_dir(ART_DIRS[1])
    det = DraftDetector(DB, ally_library=ci, enemy_library=sq, ban_library=ci,
                        layout=USER_LAYOUT, icons=_icons())
    state = det.detect(_frame("real_draft_b"))
    assert state.ally_bans == TRUTH_B["ally_bans"], state.ally_bans
    assert state.enemy_bans == TRUTH_B["enemy_bans"], state.enemy_bans


# ---------------------------------------------------------------------------
#  Real model, real screenshots, the real DraftDetector (optional AI stack)
# ---------------------------------------------------------------------------
def _real_recognizer(tmp):
    """DINOv2 over DOWNLOADED art only; Sora's single reference is his ban
    crop (no public art exists), kept as a crop remembered from the screen."""
    if not _have_cv() or not recognizer.ai_stack_available():
        print("(skipped: torch/transformers not installed)")
        return None
    sora = os.path.join(tmp, "screen")
    os.makedirs(sora)
    shutil.copy(os.path.join(FIX, "learned_circle", "sora.png"),
                os.path.join(sora, "sora__ban__seed.png"))
    rec = recognizer.DinoRecognizer.load(
        clean_dirs=ART_DIRS, screen_dirs=[sora], cache_path=TEST_CACHE,
        learn_dir=os.path.join(tmp, "mem"), icons=_icons(), log=_quiet)
    if rec is None:
        print("(skipped: DINOv2 model unavailable)")
    return rec


def _assert_truth(state, truth, pending, label):
    for field, _group in GROUPS:
        assert getattr(state, field) == truth[field], \
            f"{label}: {field} = {getattr(state, field)}"
    for i, want in enumerate(pending):
        if want is not None:
            assert state.ally_pending[i] is want, \
                f"{label}: ally slot {i} NOT PICKED = {state.ally_pending[i]}"


def test_dino_names_every_real_slot_on_every_screenshot():
    """Three real screenshots through DraftDetector in DINOv2 mode: every
    pick and ban exact (skins, Johnson, Hanabi vs Ixia, Miya vs Layla, the
    tiny ban icons), empty and placeholder slots blank, the hovered Helcurt
    NOT PICKED.  The draft animation then costs no inference, and what the
    detector learned doesn't change a fresh detector's answers."""
    tmp = tempfile.mkdtemp()
    try:
        with _setting(LAYOUT=USER_LAYOUT):
            rec = _real_recognizer(tmp)
            if rec is None:
                return
            for fixture_set, (truth, pending) in SETS.items():
                frame = _frame(fixture_set)
                det = DraftDetector(DB, layout=USER_LAYOUT, recognizer=rec,
                                    icons=rec.icons)
                _assert_truth(det.detect(frame), truth, pending, f"{fixture_set} cold")
                before = rec.embed_calls
                pulsed = np.clip(frame.astype(np.int16) + 12, 0, 255).astype(np.uint8)
                _assert_truth(det.detect(pulsed), truth, pending, f"{fixture_set} animated")
                assert rec.embed_calls == before, "animation re-ran the model"
            for fixture_set, (truth, pending) in SETS.items():
                det = DraftDetector(DB, layout=USER_LAYOUT, recognizer=rec,
                                    icons=rec.icons)
                _assert_truth(det.detect(_frame(fixture_set)), truth, pending,
                              f"{fixture_set} after learning")
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

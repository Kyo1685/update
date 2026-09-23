"""
Tests for DINOv2 recognition (recognizer.py + DraftDetector in DINOv2 mode).

The rule, the index and the memory are tested with a fake embedder (numpy +
opencv only).  The real-crop tests need the optional AI stack and the model;
they skip cleanly without them.

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

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIX = os.path.join(ROOT, "tests", "fixtures", "real_draft")
DB = HeroDB.load(os.path.join(ROOT, "heroes.json"))
# Persistent across runs (outside the repo) so the roster is embedded once.
TEST_CACHE = os.path.join(tempfile.gettempdir(), "mlbb_dino_test_cache.npz")

TRUTH = dict(ally_picks=["Nana", "Melissa", "Lukas", None, None],
             enemy_picks=["Vexana", "Johnson", "Layla", None, None],
             ally_bans=["Sora", "Harley", "Gloo", "Hilda", "Minsithar"],
             enemy_bans=["Chou", "Selena", "Saber", "Hayabusa", "Miya"])


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


def _art(seed, size=64):
    r = np.random.default_rng(seed)
    return cv2.resize(r.integers(0, 255, (8, 8, 3), dtype=np.uint8), (size, size),
                      interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
#  The double-check rule (pure python)
# ---------------------------------------------------------------------------
def test_decide_clear_win_never_asks_the_tiebreaker():
    def boom():
        raise AssertionError("tiebreak must not run on a clear win")
    assert recognizer.decide([(0.86, "Nana"), (0.79, "Joy")], boom) == (
        "Nana", 0.86, "dino")


def test_decide_blank_below_min_similarity():
    assert recognizer.decide([(0.50, "Atlas"), (0.40, "Hanzo")]) == (
        None, 0.50, "dino-low")


def test_decide_near_tie_uses_template_only_among_top_candidates():
    rank = [(0.773, "Lesley"), (0.773, "Silvanna"), (0.772, "Miya"), (0.70, "Gord")]
    assert recognizer.decide(rank, lambda: "miya") == ("Miya", 0.772, "dino+tmpl")
    # A template pick outside DINOv2's top candidates can't win: blank.
    assert recognizer.decide(rank, lambda: "Gord") == (None, 0.773, "dino-tie")
    assert recognizer.decide(rank, lambda: None) == (None, 0.773, "dino-tie")
    assert recognizer.decide([]) == (None, 0.0, "none")


# ---------------------------------------------------------------------------
#  Reference index
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


def test_index_cache_dedupe_and_screen_tagging():
    if not _have_cv():
        return
    tmp = tempfile.mkdtemp()
    try:
        clean, mirror, screen = (os.path.join(tmp, d) for d in ("clean", "mirror", "screen"))
        for d in (clean, mirror, screen):
            os.makedirs(d)
        a = _art(1)
        cv2.imwrite(os.path.join(clean, "alpha.png"), a)
        cv2.imwrite(os.path.join(mirror, "alpha.png"), cv2.flip(a, 1))   # same pixels, mirrored
        cv2.imwrite(os.path.join(screen, "cici__x1.png"), _art(2))
        cache = os.path.join(tmp, "cache.npz")
        emb = FakeEmbedder()
        idx = recognizer.build_index(emb, [clean, mirror], [screen], cache, log=lambda m: None)
        assert emb.embedded == 4            # alpha + its mirror, cici + its mirror
        assert sorted(idx.heroes()) == ["Alpha", "Cici"] and idx.n_screen == 2
        emb2 = FakeEmbedder()
        recognizer.build_index(emb2, [clean, mirror], [screen], cache, log=lambda m: None)
        assert emb2.embedded == 0           # everything cached
        cv2.imwrite(os.path.join(screen, "cici__x1.png"), _art(3))
        emb3 = FakeEmbedder()
        recognizer.build_index(emb3, [clean, mirror], [screen], cache, log=lambda m: None)
        assert emb3.embedded == 2           # only the changed image (+ mirror)
    finally:
        shutil.rmtree(tmp)


def test_memory_saves_clear_wins_once_and_caps_per_hero():
    if not _have_cv():
        return
    tmp = tempfile.mkdtemp()
    try:
        emb = FakeEmbedder()
        base = _art(10)
        idx = recognizer.HeroIndex()
        idx.add("Nana", emb.embed([base])[0])
        rec = recognizer.DinoRecognizer(emb, idx, learn_dir=tmp)
        crop = _art(11)
        vec = emb.embed([crop])[0]
        win = [(0.90, "Nana"), (0.70, "Joy")]
        assert rec.maybe_learn("Nana", crop, vec, win)
        assert len(os.listdir(tmp)) == 1 and idx.n_screen == 1
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
#  Real crops, real model (optional)
# ---------------------------------------------------------------------------
def _real_recognizer(learn_dir):
    """DINOv2 over DOWNLOADED art only; Sora's single reference is his ban
    crop (no public art exists), passed as a screen crop."""
    if not _have_cv() or not recognizer.ai_stack_available():
        print("(skipped: torch/transformers not installed)")
        return None
    sora = os.path.join(learn_dir, "_sora_only")
    os.makedirs(sora)
    shutil.copy(os.path.join(ROOT, "tests", "fixtures", "learned_circle", "sora.png"), sora)
    rec = recognizer.DinoRecognizer.load(
        clean_dirs=[os.path.join(ROOT, "templates_circle"), os.path.join(ROOT, "templates")],
        screen_dirs=[sora], cache_path=TEST_CACHE, learn_dir=os.path.join(learn_dir, "mem"),
        log=lambda m: None)
    if rec is None:
        print("(skipped: DINOv2 model unavailable)")
    return rec


def _full_screen():
    """The user's setup: 1366x768 desktop, their real crops, a layout shaped
    like their calibration (ally column left, enemy right, ban rows on top)."""
    frame = np.full((768, 1366, 3), 17, np.uint8)

    def place(prefix, positions):
        boxes = []
        for i, (x, y) in enumerate(positions):
            c = cv2.imread(os.path.join(FIX, f"{prefix}_{i}.png"))
            h, w = c.shape[:2]
            frame[y:y + h, x:x + w] = c
            boxes.append(config.Box(x, y, w, h))
        return boxes

    layout = config.Layout(
        ally_picks=place("ally_pick", [(96, 168 + 104 * i) for i in range(5)]),
        enemy_picks=place("enemy_pick", [(1196, 168 + 100 * i) for i in range(5)]),
        ally_bans=place("ally_ban", [(80 + 60 * i, 100) for i in range(5)]),
        enemy_bans=place("enemy_ban", [(1000 + 60 * i, 100) for i in range(5)]))
    return frame, layout


def _detector(rec, layout):
    square = TemplateLibrary.from_dir(os.path.join(ROOT, "templates"))
    circle = TemplateLibrary.from_dir(os.path.join(ROOT, "templates_circle"), circular=True)
    return DraftDetector(DB, ally_library=circle, enemy_library=square,
                         ban_library=circle, layout=layout, recognizer=rec)


def _assert_truth(state, label):
    for k, want in TRUTH.items():
        assert getattr(state, k) == want, f"{label}: {k} = {getattr(state, k)}"


def test_dino_recognises_every_real_slot_end_to_end():
    """Full 1366x768 frame through DraftDetector in DINOv2 mode, downloaded art
    only: all 20 slots exact (skins, Johnson, Miya/Hilda near-ties, empties)."""
    tmp = tempfile.mkdtemp()
    try:
        rec = _real_recognizer(tmp)
        if rec is None:
            return
        frame, layout = _full_screen()
        det = _detector(rec, layout)
        _assert_truth(det.detect(frame), "cold")
        # Draft animation: a brightness pulse must keep every label WITHOUT
        # running the model again.
        before = rec.embed_calls
        pulsed = np.clip(frame.astype(np.int16) + 12, 0, 255).astype(np.uint8)
        _assert_truth(det.detect(pulsed), "animated")
        assert rec.embed_calls == before, "animation re-ran the model"
        # What it learned must not pollute a fresh detector.
        _assert_truth(_detector(rec, layout).detect(frame), "after learning")
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

"""
detector.py
===========
Computer-vision pipeline that turns the 2712x1220 Scrcpy mirror into a
``DraftState``.  No neural networks - just fast, deterministic OpenCV.

Stages
------
1. ScreenCapturer  - grabs the mirror window with ``mss`` (fast, zero-copy),
   throttled + cached so the UI thread never stutters waiting on a frame.
2. For each of the 20 slot boxes defined in config.LAYOUT we crop the pixels.
3. A cheap per-slot SIGNATURE (8x8 grey thumbnail) is compared to the previous
   frame.  Unchanged slots short-circuit straight to the cached result - this
   is the single most important anti-stutter optimisation, because template
   matching only runs on slots that actually changed.
4. Changed slots are classified by TemplateLibrary:
       a) primary  : multi-position ``cv2.matchTemplate`` (TM_CCOEFF_NORMED)
       b) fallback : HSV colour-histogram correlation
   Empty / locked-but-blank slots are rejected by a std-dev gate.
5. assign_lanes() runs an optimal 1-to-1 role->lane assignment so every
   detected hero gets a predicted lane even when base roles collide.

The heavy imports (numpy / cv2 / mss) are guarded: this module always *imports*
so that the pure-Python optimiser and the rest of the app can be exercised on
machines without the CV stack; the relevant class raises a clear error only
when you actually try to use it.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

import config
from engine import DraftState, HeroDB

# --- guarded heavy imports -------------------------------------------------
try:
    import numpy as np
except Exception:                       # pragma: no cover
    np = None

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None

try:
    import mss
except Exception:                       # pragma: no cover
    mss = None


def _require_cv():
    if np is None or cv2 is None:
        raise RuntimeError(
            "numpy + opencv-python are required for vision. "
            "Install with:  pip install numpy opencv-python"
        )


def _mss_instance():
    """Create an mss grabber, preferring the new ``MSS`` class (newer versions
    deprecate the lowercase ``mss.mss`` alias)."""
    if mss is None:
        raise RuntimeError("mss is required for screen capture. "
                           "Install with:  pip install mss")
    factory = getattr(mss, "MSS", None) or getattr(mss, "mss")
    return factory()


# ===========================================================================
#  ROLE / LANE OPTIMISER  (pure python - always available)
# ===========================================================================
def assign_lanes(names: Sequence[Optional[str]], db: HeroDB) -> Dict[str, str]:
    """Assign each detected hero to a UNIQUE lane, maximising total role-fit.

    With <=5 heroes we brute-force every lane permutation (<=120) which is both
    optimal and instant - no scipy/Hungarian dependency needed.  Heroes with an
    unknown role contribute 0 preference and simply absorb a leftover lane.
    """
    picks = [n for n in names if n]
    if not picks:
        return {}

    heroes = [(n, db.get(n)) for n in picks]

    def pref(hero, lane: str) -> int:
        # Use the hero's actual lanes (data-driven); fall back to role priority.
        order = (list(hero.lanes) if (hero and hero.lanes)
                 else config.ROLE_LANE_PRIORITY.get(hero.base_role if hero else "", []))
        # First choice scores highest; absent -> 0.
        return (len(order) - order.index(lane)) if lane in order else 0

    best_perm: Optional[Tuple[str, ...]] = None
    best_score = -1
    for perm in permutations(config.LANES, len(heroes)):
        score = sum(pref(h, perm[i]) for i, (_, h) in enumerate(heroes))
        if score > best_score:
            best_score, best_perm = score, perm

    assignment: Dict[str, str] = {}
    if best_perm:
        for i, (name, _) in enumerate(heroes):
            assignment[best_perm[i]] = name
    return assignment


def predicted_role_label(name: str, lane: Optional[str], db: HeroDB) -> str:
    """Short tag shown next to a portrait, e.g. 'GUINEVERE [EXP]'."""
    if lane:
        return config.LANE_ABBR.get(lane, lane[:3].upper())
    hero = db.get(name)
    if hero:
        order = config.ROLE_LANE_PRIORITY.get(hero.base_role, [])
        if order:
            return config.LANE_ABBR.get(order[0], order[0][:3].upper())
    return "???"


# ===========================================================================
#  SCREEN CAPTURE  (mss + cache + throttle)
# ===========================================================================
class ScreenCapturer:
    """Grabs the mirror window and caches the last frame.

    ``grab()`` is safe to spam from any thread - it returns the cached BGR
    ndarray until ``CAPTURE_MIN_INTERVAL`` has elapsed, then refreshes.
    """

    def __init__(self, region: Optional[dict] = None,
                 min_interval: float = config.CAPTURE_MIN_INTERVAL):
        _require_cv()
        self._sct = _mss_instance()
        if region is None:
            region = config.ACTIVE_REGION
        self._region = {"left": int(region["left"]), "top": int(region["top"]),
                        "width": int(region["width"]), "height": int(region["height"])}
        self._min_interval = min_interval
        self._last_frame = None
        self._last_t = 0.0

    @staticmethod
    def primary_monitor() -> dict:
        """Geometry of the primary monitor (handles any PC resolution)."""
        with _mss_instance() as sct:
            m = sct.monitors[1]
            return {"left": m["left"], "top": m["top"],
                    "width": m["width"], "height": m["height"]}

    def grab(self, force: bool = False):
        """Return a BGR ndarray of the capture region (cached/throttled)."""
        now = time.monotonic()
        if (not force and self._last_frame is not None
                and now - self._last_t < self._min_interval):
            return self._last_frame
        raw = self._sct.grab(self._region)            # BGRA
        frame = np.asarray(raw)[:, :, :3]             # drop alpha -> BGR
        self._last_frame = np.ascontiguousarray(frame)
        self._last_t = now
        return self._last_frame

    def close(self):
        try:
            self._sct.close()
        except Exception:
            pass


# ===========================================================================
#  TEMPLATE LIBRARY  (matchTemplate + histogram fallback)
# ===========================================================================
class TemplateLibrary:
    CANON = 96          # canonical avatar size used for correlation
    PAD = 10            # search slack so small mis-alignment still matches

    def __init__(self, circular: bool = False):
        _require_cv()
        # ``circular`` libs centre-crop to the inscribed square before matching
        # so the avatar's ring/background corners don't pollute the score.
        self.circular = circular
        self._tmpl: Dict[str, "np.ndarray"] = {}      # colour canonical (BGR)
        self._hist: Dict[str, "np.ndarray"] = {}

    # ----- loading ---------------------------------------------------------
    @classmethod
    def from_dir(cls, path: str = config.TEMPLATE_DIR,
                 circular: bool = False) -> "TemplateLibrary":
        lib = cls(circular=circular)
        if not os.path.isdir(path):
            return lib                                   # empty but valid
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        for pat in patterns:
            for fp in glob.glob(os.path.join(path, pat)):
                img = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                name = cls._name_from_file(fp)
                lib.add(name, img)
        return lib

    @staticmethod
    def _name_from_file(fp: str) -> str:
        stem = os.path.splitext(os.path.basename(fp))[0]
        return stem.replace("_", " ").replace("-", "-").strip().title()

    def add(self, name: str, bgr: "np.ndarray") -> None:
        self._tmpl[name] = self._prep_tmpl(bgr)
        self._hist[name] = self._prep_hist(bgr)

    def __len__(self) -> int:
        return len(self._tmpl)

    def merge_missing(self, other: "TemplateLibrary") -> int:
        """Add any heroes present in ``other`` but missing here (e.g. backfill
        the ban library with square portraits for heroes the circular pack
        lacks).  Existing entries are kept.  Returns how many were added."""
        added = 0
        for name, tmpl in other._tmpl.items():
            if name not in self._tmpl:
                self._tmpl[name] = tmpl
                self._hist[name] = other._hist[name]
                added += 1
        return added

    # ----- preprocessing ---------------------------------------------------
    def _inscribe(self, bgr: "np.ndarray") -> "np.ndarray":
        """For circular libs, take the largest square inside the circle (pure
        face, no ring / background corners).  No-op for square libs."""
        if not self.circular:
            return bgr
        h, w = bgr.shape[:2]
        side = int(min(h, w) / 1.41421356)
        cy, cx = h // 2, w // 2
        half = max(1, side // 2)
        return bgr[cy - half:cy + half, cx - half:cx + half]

    def _prep_tmpl(self, bgr: "np.ndarray") -> "np.ndarray":
        # Keep COLOUR: many heroes share a silhouette but differ in palette
        # (e.g. Helcurt vs Gord), so grayscale matching confuses them.
        img = bgr[:, :, :3] if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        img = self._inscribe(img)
        return cv2.resize(img, (self.CANON, self.CANON), interpolation=cv2.INTER_AREA)

    def _prep_hist(self, bgr: "np.ndarray") -> "np.ndarray":
        img = bgr[:, :, :3] if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(self._inscribe(img), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    # ----- matching --------------------------------------------------------
    def match(self, crop_bgr: "np.ndarray",
              threshold: Optional[float] = None,
              hist_threshold: Optional[float] = None
              ) -> Tuple[Optional[str], float, str]:
        """Aggressive 2-stage match: colour-histogram shortlist, then multi-scale
        template correlation over the shortlist.  Returns (name, score, method).
        """
        if threshold is None:
            threshold = config.TEMPLATE_MATCH_THRESHOLD
        if hist_threshold is None:
            hist_threshold = config.HISTOGRAM_MATCH_THRESHOLD
        if not self._tmpl:
            return None, 0.0, "none"

        # Stage 1: colour-histogram shortlist (cheap; keeps same-palette
        # look-alikes together so stage 2 can tell them apart).
        crop_hist = self._prep_hist(crop_bgr)
        hist_ranked = sorted(
            ((float(cv2.compareHist(crop_hist, h, cv2.HISTCMP_CORREL)), n)
             for n, h in self._hist.items()), reverse=True)
        shortlist = [n for _, n in hist_ranked[:config.MATCH_SHORTLIST]]

        # Stage 2: multi-scale template correlation over an expanded target.
        best_name, best = self._best_template(crop_bgr, shortlist)
        if best >= threshold:
            return best_name, best, "template"

        if config.USE_HISTOGRAM_FALLBACK and hist_ranked and \
                hist_ranked[0][0] >= hist_threshold:
            return hist_ranked[0][1], hist_ranked[0][0], "histogram"
        return best_name, max(best, 0.0), "low"

    def _best_template(self, crop_bgr, names):
        """Highest multi-scale TM_CCOEFF_NORMED over ``names``."""
        big = self.CANON + 40
        target = cv2.resize(self._inscribe(crop_bgr[:, :, :3]), (big, big),
                            interpolation=cv2.INTER_AREA)
        best_name, best = None, -1.0
        for name in names:
            full = self._tmpl.get(name)
            if full is None:
                continue
            for sc in config.MATCH_SCALES:
                s = max(8, int(round(self.CANON * sc)))
                if s > big:
                    continue
                tmpl = full if s == self.CANON else cv2.resize(
                    full, (s, s), interpolation=cv2.INTER_AREA)
                score = float(cv2.matchTemplate(
                    target, tmpl, cv2.TM_CCOEFF_NORMED).max())
                if score > best:
                    best_name, best = name, score
        return best_name, best

    def top_matches(self, crop_bgr: "np.ndarray", n: int = 3):
        """Top-n (name, score) multi-scale matches over the whole library
        (diagnostics)."""
        if not self._tmpl:
            return []
        big = self.CANON + 40
        target = cv2.resize(self._inscribe(crop_bgr[:, :, :3]), (big, big),
                            interpolation=cv2.INTER_AREA)
        out = []
        for name, full in self._tmpl.items():
            best = -1.0
            for sc in config.MATCH_SCALES:
                s = max(8, int(round(self.CANON * sc)))
                if s > big:
                    continue
                tmpl = full if s == self.CANON else cv2.resize(
                    full, (s, s), interpolation=cv2.INTER_AREA)
                best = max(best, float(cv2.matchTemplate(
                    target, tmpl, cv2.TM_CCOEFF_NORMED).max()))
            out.append((name, best))
        out.sort(key=lambda kv: kv[1], reverse=True)
        return out[:n]


# ===========================================================================
#  DRAFT DETECTOR  (orchestrates capture -> slots -> state, with caching)
# ===========================================================================
@dataclass
class _SlotCache:
    signature: Optional["np.ndarray"] = None
    name: Optional[str] = None
    confidence: float = 0.0


class DraftDetector:
    def __init__(self, db: HeroDB,
                 library: Optional[TemplateLibrary] = None,
                 ally_library: Optional[TemplateLibrary] = None,
                 enemy_library: Optional[TemplateLibrary] = None,
                 ban_library: Optional[TemplateLibrary] = None,
                 ally_libraries=None, enemy_libraries=None, ban_libraries=None,
                 layout: Optional[config.Layout] = None,
                 accept_low: bool = False):
        self.db = db
        # Resolve at call time so a calibrated config.LAYOUT is picked up.
        self.layout = layout if layout is not None else config.LAYOUT
        base = library if library is not None else TemplateLibrary()
        # Each slot group can be matched against SEVERAL template sets; the most
        # confident wins (e.g. ally picks try the square portrait AND the
        # circular icon, so the exact hero beats a look-alike).
        self.ally_libraries = ally_libraries or [ally_library or base]
        self.enemy_libraries = enemy_libraries or [enemy_library or base]
        self.ban_libraries = ban_libraries or [ban_library or base]
        self.accept_low = accept_low
        # One cache entry per slot, keyed by a stable slot id.
        self._cache: Dict[str, _SlotCache] = {}

    # ----- helpers ---------------------------------------------------------
    @staticmethod
    def _crop(frame: "np.ndarray", box: config.Box) -> "np.ndarray":
        h, w = frame.shape[:2]
        x1 = max(0, min(box.x, w - 1))
        y1 = max(0, min(box.y, h - 1))
        x2 = max(x1 + 1, min(box.x2, w))
        y2 = max(y1 + 1, min(box.y2, h))
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _signature(crop: "np.ndarray") -> "np.ndarray":
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _is_empty(crop: "np.ndarray") -> bool:
        # Locked portraits are busy splash-art (high variance); empty/dimmed
        # slots are comparatively flat.
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(g.std()) < config.EMPTY_SLOT_STDDEV

    @staticmethod
    def _mean_value(crop: "np.ndarray") -> float:
        return float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2].mean())

    @staticmethod
    def _is_grayed(crop: "np.ndarray") -> bool:
        # Un-locked / hovered slots render desaturated ("grayed"); a confirmed
        # pick is full colour.  Gate on mean HSV saturation so we only report
        # heroes that are actually locked in.
        if config.LOCKED_MIN_SATURATION <= 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 1].mean()) < config.LOCKED_MIN_SATURATION

    def _crop_expanded(self, frame: "np.ndarray", box: config.Box,
                       frac: float) -> "np.ndarray":
        """Crop the box plus a margin, so the template can be searched even if
        the calibrated box is a little off-centre."""
        dx, dy = int(box.w * frac), int(box.h * frac)
        h, w = frame.shape[:2]
        x1, y1 = max(0, box.x - dx), max(0, box.y - dy)
        x2, y2 = min(w, box.x2 + dx), min(h, box.y2 + dy)
        return frame[y1:y2, x1:x2]

    def _classify_slot(self, frame: "np.ndarray", box: config.Box, slot_id: str,
                       libraries, threshold: Optional[float] = None
                       ) -> Tuple[Optional[str], float]:
        """Detect the hero in one slot, trying every library and keeping the
        most confident match; per-slot cache skips unchanged frames."""
        crop = self._crop(frame, box)
        sig = self._signature(crop)
        cache = self._cache.get(slot_id)

        if cache is not None and cache.signature is not None:
            delta = float(np.abs(sig.astype(np.int16)
                                 - cache.signature.astype(np.int16)).mean())
            if delta < config.SIGNATURE_DELTA:
                return cache.name, cache.confidence       # cache hit

        # Skip empty / un-locked (grayed) slots so only confirmed picks show.
        if self._is_empty(crop) or self._is_grayed(crop):
            result_name, conf = None, 0.0
        else:
            mcrop = self._crop_expanded(frame, box, config.MATCH_EXPAND)
            results = [lib.match(mcrop, threshold) for lib in libraries]
            confident = [(n, c, m) for (n, c, m) in results
                         if m in ("template", "histogram") and n]
            if confident:
                raw_name, conf, _ = max(confident, key=lambda r: r[1])
            elif self.accept_low:
                lows = [(n, c, m) for (n, c, m) in results if n]
                raw_name, conf = max(lows, key=lambda r: r[1])[:2] if lows else (None, 0.0)
            else:
                raw_name, conf = None, 0.0
            if raw_name is None:
                result_name = None
            else:
                hero = self.db.get(raw_name)
                result_name = hero.name if hero else raw_name

        self._cache[slot_id] = _SlotCache(signature=sig, name=result_name,
                                          confidence=conf)
        return result_name, conf

    # ----- public API ------------------------------------------------------
    def detect(self, frame: "np.ndarray") -> DraftState:
        """Scan every slot box and return a fully-populated DraftState
        (including predicted lanes for both teams)."""
        _require_cv()
        state = DraftState()

        def scan(boxes, prefix, libraries, threshold=None, lock_gate=False):
            crops = [self._crop(frame, b) for b in boxes]
            # Relative-brightness lock gate (picks only, off unless configured).
            ref = 0.0
            if lock_gate and config.LOCKED_REL_BRIGHTNESS > 0:
                vals = [self._mean_value(c) for c, b in zip(crops, boxes)
                        if not self._is_empty(c)]
                ref = max(vals) if vals else 0.0
            out: List[Optional[str]] = []
            for i, (box, crop) in enumerate(zip(boxes, crops)):
                name, _ = self._classify_slot(frame, box, f"{prefix}{i}",
                                              libraries, threshold)
                if (name and ref > 0
                        and self._mean_value(crop) < config.LOCKED_REL_BRIGHTNESS * ref):
                    name = None                       # too dim => un-locked
                out.append(name)
            return out

        state.ally_picks = scan(self.layout.ally_picks, "ap", self.ally_libraries,
                                config.TEMPLATE_MATCH_THRESHOLD, lock_gate=True)
        state.enemy_picks = scan(self.layout.enemy_picks, "ep", self.enemy_libraries,
                                 config.ENEMY_MATCH_THRESHOLD, lock_gate=True)
        state.ally_bans = scan(self.layout.ally_bans, "ab",
                               self.ban_libraries, config.BAN_MATCH_THRESHOLD)
        state.enemy_bans = scan(self.layout.enemy_bans, "eb",
                                self.ban_libraries, config.BAN_MATCH_THRESHOLD)

        state.ally_lanes = assign_lanes(state.ally_picks, self.db)
        state.enemy_lanes = assign_lanes(state.enemy_picks, self.db)
        return state

    # ----- debug visualisation --------------------------------------------
    def draw_debug(self, frame: "np.ndarray", state: DraftState) -> "np.ndarray":
        """Return a copy of ``frame`` annotated with boxes + labels (handy for
        calibration without launching the PyQt overlay)."""
        _require_cv()
        out = frame.copy()

        def label_for(name, lanes):
            lane = next((ln for ln, hn in lanes.items() if hn == name), None)
            return f"{name.upper()} [{predicted_role_label(name, lane, self.db)}]"

        def draw(boxes, names, color, lanes, is_ban=False):
            for box, name in zip(boxes, names):
                if not name:
                    continue
                cv2.rectangle(out, (box.x, box.y), (box.x2, box.y2),
                              color, 2 if is_ban else config.THEME.box_thickness)
                if is_ban:
                    continue
                text = label_for(name, lanes)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                ty = box.y - 8 if box.y > 24 else box.y2 + th + 8
                cv2.rectangle(out, (box.x, ty - th - 6), (box.x + tw + 10, ty + 4),
                              (8, 16, 14), -1)
                cv2.putText(out, text, (box.x + 5, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 210), 1,
                            cv2.LINE_AA)

        c = config.THEME
        draw(self.layout.ally_picks, state.ally_picks, c.ally_box[:3], state.ally_lanes)
        draw(self.layout.enemy_picks, state.enemy_picks, c.enemy_box[:3], state.enemy_lanes)
        draw(self.layout.ally_bans, state.ally_bans, c.ban_box[:3], {}, is_ban=True)
        draw(self.layout.enemy_bans, state.enemy_bans, c.ban_box[:3], {}, is_ban=True)
        return out


# ===========================================================================
#  SELF-TEST  (synthetic frame -> verifies matching, caching, lane optimiser)
# ===========================================================================
if __name__ == "__main__":
    _require_cv()
    db = HeroDB.load("heroes.json")

    # --- pure-python optimiser check (no CV needed) -----------------------
    lanes = assign_lanes(["Guinevere", "Minsithar", "Zetian", "Aamon", "Estes"], db)
    print("assign_lanes (2 EXP fighters + mage + jungler + support):")
    for ln in config.LANES:
        print(f"   {ln:7} -> {lanes.get(ln, '-')}")

    # --- build a synthetic 2712x1220 frame with unique colour 'avatars' ----
    rng = np.random.default_rng(7)
    frame = np.full((config.RES_H, config.RES_W, 3), 18, np.uint8)
    lib = TemplateLibrary()
    planted = {
        "ap0": "Guinevere", "ap1": "Zetian", "ap2": "Minsithar",
        "ep0": "Alice", "ep1": "Khufra", "ep2": "Akai", "ep3": "Gord",
    }

    def paint(box, seed_name):
        # Deterministic SMOOTH, colourful patch per hero (low-frequency so it
        # survives the resize-correlation, like real hero art does).
        r = np.random.default_rng(abs(hash(seed_name)) % (2**32))
        hsv = np.zeros((box.h, box.w, 3), np.uint8)
        hsv[:, :, 0] = int(r.integers(0, 180)); hsv[:, :, 1] = 200; hsv[:, :, 2] = 190
        patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        for _ in range(3):
            col = tuple(int(x) for x in r.integers(0, 255, (3,)))
            x0, y0 = int(r.integers(0, box.w // 2)), int(r.integers(0, box.h // 2))
            cv2.rectangle(patch, (x0, y0),
                          (x0 + int(box.w * 0.4), y0 + int(box.h * 0.4)), col, -1)
        frame[box.y:box.y2, box.x:box.x2] = patch
        return patch

    L = config.LAYOUT
    slot_boxes = {f"ap{i}": L.ally_picks[i] for i in range(5)}
    slot_boxes.update({f"ep{i}": L.enemy_picks[i] for i in range(5)})
    for sid, hero_name in planted.items():
        patch = paint(slot_boxes[sid], hero_name)
        lib.add(hero_name, patch)            # register as the template

    det = DraftDetector(db, library=lib)

    t0 = time.perf_counter()
    state = det.detect(frame)
    t1 = time.perf_counter()
    state2 = det.detect(frame)               # identical frame -> all cache hits
    t2 = time.perf_counter()

    print(f"\nDetect #1 (cold): {(t1 - t0) * 1000:6.1f} ms")
    print(f"Detect #2 (cached): {(t2 - t1) * 1000:6.1f} ms  "
          f"(speed-up {((t1 - t0) / max(t2 - t1, 1e-6)):.1f}x)")
    print("\nAlly picks :", state.ally_picks)
    print("Enemy picks:", state.enemy_picks)
    print("Ally lanes :", state.ally_lanes)
    print("Enemy lanes:", state.enemy_lanes)

    ok = (state.ally_picks[:3] == ["Guinevere", "Zetian", "Minsithar"]
          and state.enemy_picks[:4] == ["Alice", "Khufra", "Akai", "Gord"])
    print("\nMATCH RESULT:", "PASS" if ok else "FAIL")

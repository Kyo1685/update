"""
icon_match.py
=============
Pixel-level ("structural") hero matching against the clean portrait art.
Pure numpy + OpenCV, so it works with or without the optional AI stack.

The draft screen draws every hero from the same circular portrait art the
template folders hold:

* **Ban icons** are that art scaled to a fixed diameter (0.86 x the ban box on
  the user's 1366x768 screen) under a red ban badge in one bottom corner
  (bottom-right on the ally row, bottom-left on the enemy row), slightly
  tinted.  Each hero's art is registered at that scale and correlated over the
  face only (badge corner and ring masked out).  On the real screenshots the
  true hero scores 0.91-0.99 and no other hero scores above 0.78.
* **Pick portraits** frame the art with a hero-specific zoom and offset (the
  enemy column mirrors it), so a pick is registered with a scale / offset /
  flip search.  That search is affordable for the few candidates DINOv2
  shortlists.  The fit also measures the portrait's contrast against the art:
  a hovered (pre-selected, not locked) hero is drawn at about half contrast.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import config

try:
    import numpy as np
    import cv2
except Exception:                       # pragma: no cover
    np = cv2 = None

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp")
Rank = List[Tuple[float, str]]          # (score, hero), best first
SIDES = ("ally", "enemy")


def hero_name_from_file(path: str) -> str:
    """Hero name for an image file, normalised like the template library.
    Memory files are named ``<hero>__<...>.png`` so a hero can keep several."""
    stem = os.path.splitext(os.path.basename(path))[0].split("__", 1)[0]
    return stem.replace("_", " ").strip().title()


def load_art(dirs: Sequence[str]) -> Dict[str, "np.ndarray"]:
    """One clean portrait per hero; the first folder that has the hero wins."""
    art: Dict[str, "np.ndarray"] = {}
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for pat in IMAGE_EXTS:
            for fp in sorted(glob.glob(os.path.join(d, pat))):
                name = hero_name_from_file(fp)
                if name in art:
                    continue
                img = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img is not None:
                    art[name] = img
    return art


def trim(img: "np.ndarray", frac: float) -> "np.ndarray":
    """Drop ``frac`` of the height/width from every edge."""
    if frac <= 0:
        return img
    h, w = img.shape[:2]
    dy, dx = int(round(h * frac)), int(round(w * frac))
    return img[dy:h - dy, dx:w - dx] if h - 2 * dy > 4 and w - 2 * dx > 4 else img


def circle_mask(d: int, inner: float = 1.0) -> "np.ndarray":
    yy, xx = np.mgrid[0:d, 0:d]
    r = d / 2.0
    return ((xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2) <= (inner * r) ** 2


def ban_mask(d: int, side: str, inner: float) -> "np.ndarray":
    """The part of a d x d ban icon that shows the hero: the disc minus the ban
    badge's bottom corner (ally row: bottom-right, enemy row: bottom-left)."""
    m = circle_mask(d, inner)
    yy, xx = np.mgrid[0:d, 0:d]
    low = yy > 0.55 * d
    m &= ~(low & ((xx > 0.55 * d) if side == "ally" else (xx < 0.45 * d)))
    return m


def _zero_mean_unit(px: "np.ndarray") -> "np.ndarray":
    """(n, 3) pixels -> per-channel zero-mean, jointly unit-norm vector."""
    px = px - px.mean(0, keepdims=True)
    v = px.ravel()
    return v / (float(np.linalg.norm(v)) + 1e-6)


@dataclass
class BanMatch:
    rank: Rank                          # (correlation, hero), best first
    geometry: Tuple[int, int, int]      # (diameter, x, y) of the icon in the crop

    @property
    def best(self) -> Tuple[float, Optional[str]]:
        return self.rank[0] if self.rank else (0.0, None)

    @property
    def margin(self) -> float:
        if not self.rank:
            return 0.0
        return self.rank[0][0] - (self.rank[1][0] if len(self.rank) > 1 else -1.0)


@dataclass
class PickFit:
    ncc: float                          # correlation of the registered art
    contrast: float                     # crop / art grey-level spread (face)
    geometry: Tuple[int, int, int, bool]    # (diameter, x, y, mirrored)


def decide_ban(match: BanMatch, min_score: Optional[float] = None,
               min_margin: Optional[float] = None
               ) -> Tuple[Optional[str], float, str]:
    """Structural verdict for a ban icon: (hero or None, score, method)."""
    min_score = config.BAN_NCC_MIN if min_score is None else min_score
    min_margin = config.BAN_NCC_MARGIN if min_margin is None else min_margin
    score, hero = match.best
    if hero is None:
        return None, 0.0, "none"
    if score < min_score:
        return None, score, "ncc-low"
    if match.margin < min_margin:
        return None, score, "ncc-tie"
    return hero, score, "ncc"


class IconMatcher:
    """Registered correlation of slot crops against the clean hero art."""

    def __init__(self, art: Dict[str, "np.ndarray"]):
        self.art = dict(art)
        self.names = sorted(self.art)
        self._lookup = {n.lower(): n for n in self.names}
        self._ban_mats: Dict[Tuple[int, str], Tuple] = {}
        self._resized: Dict[Tuple[str, int, bool], "np.ndarray"] = {}
        self._masks: Dict[Tuple[int, float], "np.ndarray"] = {}

    @classmethod
    def from_dirs(cls, dirs: Optional[Sequence[str]] = None) -> "IconMatcher":
        return cls(load_art(config.ICON_ART_DIRS if dirs is None else dirs))

    def __len__(self) -> int:
        return len(self.names)

    def resolve(self, hero: Optional[str]) -> Optional[str]:
        """The art's own spelling of ``hero`` (case-insensitive), or None."""
        return self._lookup.get(hero.lower()) if hero else None

    def has(self, hero: str) -> bool:
        return self.resolve(hero) is not None

    # ----- shared helpers ---------------------------------------------------
    def _art_at(self, name: str, d: int, flip: bool = False) -> "np.ndarray":
        key = (name, d, flip)
        img = self._resized.get(key)
        if img is None:
            src = cv2.flip(self.art[name], 1) if flip else self.art[name]
            img = cv2.resize(src, (d, d), interpolation=cv2.INTER_AREA)
            self._resized[key] = img
        return img

    def _circle(self, d: int, inner: float) -> "np.ndarray":
        key = (d, inner)
        m = self._masks.get(key)
        if m is None:
            m = circle_mask(d, inner).astype(np.uint8)
            self._masks[key] = m
        return m

    @staticmethod
    def _diameters(size: int, ratios: Sequence[float]) -> List[int]:
        return sorted({max(8, int(round(size * r))) for r in ratios})

    # ----- bans ---------------------------------------------------------------
    def _ban_templates(self, d: int, side: str):
        key = (d, side)
        mats = self._ban_mats.get(key)
        if mats is None:
            m = ban_mask(d, side, config.BAN_ICON_INNER)
            rows = [_zero_mean_unit(self._art_at(n, d).astype(np.float32)[m])
                    for n in self.names]
            mats = (m, np.stack(rows) if rows else np.zeros((0, 3 * int(m.sum())),
                                                             np.float32))
            self._ban_mats[key] = mats
        return mats

    def rank_ban(self, crop: "np.ndarray", side: str) -> BanMatch:
        """Every hero's best correlation with the icon in a ban crop.  The icon's
        diameter and position are searched (both depend on the calibration), so
        the result also carries the geometry of the best-matching hero."""
        if not self.names:
            return BanMatch([], (0, 0, 0))
        size = min(crop.shape[:2])
        pad = config.BAN_ICON_PAD
        big = cv2.copyMakeBorder(crop[:, :, :3], pad, pad, pad, pad,
                                 cv2.BORDER_REPLICATE).astype(np.float32)
        best = np.full(len(self.names), -np.inf, np.float32)
        where = [None] * len(self.names)
        for d in self._diameters(size, config.BAN_ICON_RATIOS):
            if d > min(big.shape[:2]):
                continue
            m, T = self._ban_templates(d, side)
            # Every d x d window of the padded crop at once: (ny, nx, 3, d, d).
            win = np.lib.stride_tricks.sliding_window_view(big, (d, d), axis=(0, 1))
            ny, nx = win.shape[:2]
            px = win[..., m].reshape(ny * nx, 3, -1).transpose(0, 2, 1)   # (n, P, 3)
            px = px - px.mean(1, keepdims=True)
            Q = px.reshape(ny * nx, -1)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-6
            S = T @ Q.T                                   # (heroes, windows)
            arg = S.argmax(1)
            top = S[np.arange(len(self.names)), arg]
            better = top > best
            for i in np.nonzero(better)[0]:
                y, x = divmod(int(arg[i]), nx)
                where[i] = (d, x - pad, y - pad)
            best = np.where(better, top, best)
        order = np.argsort(-best, kind="stable")
        rank = [(float(best[i]), self.names[i]) for i in order if np.isfinite(best[i])]
        geometry = where[order[0]] if rank else (size, 0, 0)
        return BanMatch(rank, geometry)

    @staticmethod
    def ban_view(crop: "np.ndarray", geometry: Tuple[int, int, int], side: str,
                 size: int = 160) -> "np.ndarray":
        """The icon cut out at ``geometry``, scaled to ``size`` and masked like
        ``ban_reference`` - the like-for-like input DINOv2 compares."""
        d, x, y = geometry
        if d <= 0:                              # nothing registered (no art)
            return cv2.resize(crop[:, :, :3], (size, size), interpolation=cv2.INTER_CUBIC)
        pad = max(0, -x, -y, x + d - crop.shape[1], y + d - crop.shape[0])
        big = cv2.copyMakeBorder(crop[:, :, :3], pad, pad, pad, pad,
                                 cv2.BORDER_REPLICATE) if pad else crop[:, :, :3]
        icon = big[y + pad:y + pad + d, x + pad:x + pad + d]
        icon = cv2.resize(icon, (size, size), interpolation=cv2.INTER_CUBIC)
        icon[~ban_mask(size, side, 0.96)] = 0
        return icon

    def ban_reference(self, name: str, side: str, diameter: int,
                      size: int = 160) -> "np.ndarray":
        """Clean art rendered like a ban icon on this screen: shrunk to the
        icon's on-screen diameter (same softness), with the badge corner and
        the ring masked exactly as ``ban_view`` masks the live icon."""
        small = self._art_at(name, max(8, int(diameter)))
        ref = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
        ref[~ban_mask(size, side, 0.96)] = 0
        return ref

    # ----- picks --------------------------------------------------------------
    def fit_pick(self, crop: "np.ndarray", name: str) -> Optional[PickFit]:
        """Register ``name``'s art inside a pick portrait (scale, offset and
        mirror searched) and measure how well it explains the crop."""
        name = self.resolve(name)
        if name is None:
            return None
        c = trim(crop[:, :, :3], config.PICK_TRIM)
        size = min(c.shape[:2])
        pad = int(round(size * 0.3))
        big = cv2.copyMakeBorder(c, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        limit = min(big.shape[:2])
        lo, hi = config.PICK_FIT_SCALES

        def match(img, d, flip, inner=config.PICK_FIT_INNER):
            r = cv2.matchTemplate(img, self._art_at(name, d, flip), cv2.TM_CCOEFF_NORMED,
                                  mask=self._circle(d, inner))
            r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, mx, _, loc = cv2.minMaxLoc(r)
            return float(mx), loc

        # Coarse: half resolution, 12 scales x both orientations.
        half = cv2.resize(big, (big.shape[1] // 2, big.shape[0] // 2),
                          interpolation=cv2.INTER_AREA)
        coarse = sorted({int(round(size * (lo + (hi - lo) * i / 11))) for i in range(12)})
        best = (-2.0, None)
        for flip in (False, True):
            for d in coarse:
                h = d // 2
                if 8 <= h <= min(half.shape[:2]):
                    mx, loc = match(half, h, flip)
                    if mx > best[0]:
                        best = (mx, (d, 2 * loc[0], 2 * loc[1], flip))
        if best[1] is None:
            return None
        # Refine: full resolution, nearby scales, a few px around the coarse hit.
        step = max(2, (coarse[1] - coarse[0]) if len(coarse) > 1 else 2)
        d0, x0, y0, flip = best[1]
        best = (-2.0, None)
        for d in range(max(8, d0 - step), min(limit, d0 + step) + 1, 2):
            s = 4 + abs(d - d0)
            ya, xa = max(0, y0 - s), max(0, x0 - s)
            win = big[ya:min(big.shape[0], y0 + d0 + s + 1),
                      xa:min(big.shape[1], x0 + d0 + s + 1)]
            if win.shape[0] < d or win.shape[1] < d:
                continue
            mx, loc = match(win, d, flip)
            if mx > best[0]:
                best = (mx, (d, xa + loc[0], ya + loc[1], flip))
        if best[1] is None:
            return None
        ncc, (d, x, y, flip) = best
        m = self._circle(d, config.PICK_FIT_INNER).astype(bool)
        art = cv2.cvtColor(self._art_at(name, d, flip), cv2.COLOR_BGR2GRAY)[m]
        seen = cv2.cvtColor(big[y:y + d, x:x + d], cv2.COLOR_BGR2GRAY)[m]
        contrast = float(seen.std()) / max(float(art.std()), 1e-3)
        return PickFit(ncc, contrast, (d, x - pad, y - pad, flip))

    def referee(self, crop: "np.ndarray",
                fits: Optional[Dict[str, Optional[PickFit]]] = None
                ) -> Callable[[Sequence[str]], Dict[str, float]]:
        """``decide_pick``'s tie-breaker: registered correlation per candidate
        (fits are memoised in ``fits`` so the lock check can reuse them)."""
        fits = {} if fits is None else fits

        def score(names: Sequence[str]) -> Dict[str, float]:
            out = {}
            for n in names:
                if n not in fits:
                    fits[n] = self.fit_pick(crop, n)
                if fits[n] is not None:
                    out[n] = fits[n].ncc
            return out
        return score


def ban_diameter(layout=None) -> int:
    """Typical on-screen ban icon diameter for the calibrated layout."""
    layout = config.LAYOUT if layout is None else layout
    sizes = [min(b.w, b.h) for b in list(getattr(layout, "ally_bans", []))
             + list(getattr(layout, "enemy_bans", []))]
    if not sizes:
        return 44
    return max(8, int(round(float(np.median(sizes)) * config.BAN_ICON_DIAMETER)))


def is_hover(fit: Optional[PickFit]) -> bool:
    """A pre-selected (hovered, not locked) portrait is drawn at roughly half
    contrast.  Judged only when the art registers well enough to compare."""
    return bool(fit is not None and fit.ncc >= config.LOCK_FIT_MIN
                and fit.contrast < config.LOCK_MIN_CONTRAST)

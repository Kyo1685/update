"""
recognizer.py
=============
Hero recognition with a double-check: two independent recognisers must agree
before a slot gets a name.

PICKS   DINOv2 (a small local vision model) turns every portrait into a feature
        "fingerprint" and names a slot by its nearest reference:
            best similarity  < DINO_MIN_SIM                   -> blank
            best leads the runner-up by >= DINO_CLEAR_MARGIN  -> that hero
            near-tie -> the clean art is registered into the portrait
                        (icon_match.py) as a referee; its winner must be one of
                        the tied candidates, otherwise blank
        then the LOCKED-IN check: a hovered (pre-selected) hero is drawn at
        about half contrast, so it reads "NOT PICKED" instead of a pick.
BANS    every ban icon is the portrait art at a fixed scale under the ban
        badge, so the registered correlation decides (icon_match.py) and
        DINOv2 - fed the same masked icon - must agree.  Heroes with no public
        art (Sora) fall back to crops remembered from the screen.

Everything that needs torch is optional.  ``decide`` and ``HeroIndex`` are plain
numpy, and ``DinoRecognizer`` takes any object with ``embed(list_of_bgr)``, so
the logic is testable without the model.  ``DinoRecognizer.load`` returns None,
and logs why, when torch / transformers / the model are unavailable; the
detector then keeps its template pipeline.
"""

from __future__ import annotations

import glob
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import config
from icon_match import (IMAGE_EXTS, SIDES, BanMatch, IconMatcher, ban_diameter,
                        decide_ban as decide_ban_ncc, hero_name_from_file, is_hover,
                        trim)

try:
    import numpy as np
    import cv2
except Exception:                       # pragma: no cover
    np = cv2 = None

try:                                    # optional AI stack (requirements-ai.txt)
    import torch
    import torchvision                  # noqa: F401  (the image processor needs it)
    import transformers
    from transformers import AutoImageProcessor, AutoModel
except Exception:                       # pragma: no cover
    torch = transformers = AutoImageProcessor = AutoModel = None

try:
    from PIL import Image
except Exception:                       # pragma: no cover
    Image = None

PREP_VERSION = "cls+meanpatch/v1"       # bump to invalidate cached embeddings

Rank = List[Tuple[float, str]]          # (similarity, hero), best first
Referee = Callable[[Sequence[str]], Dict[str, float]]

__all__ = ["DinoRecognizer", "DinoEmbedder", "HeroIndex", "Verdict", "decide",
           "sim_of", "format_rank", "hero_name_from_file", "ai_stack_available"]


def ai_stack_available() -> bool:
    return None not in (torch, transformers, AutoModel, Image, np, cv2)


# ===========================================================================
#  THE DOUBLE-CHECK  (pure python)
# ===========================================================================
def decide(rank: Rank, referee: Optional[Referee] = None,
           min_sim: Optional[float] = None, clear_margin: Optional[float] = None,
           tie_topk: Optional[int] = None) -> Tuple[Optional[str], float, str]:
    """Turn a pick's similarity ranking into (hero or None, similarity, method).

    method: ``dino`` (clear win), ``dino+ncc`` (near-tie, the referee agrees
    with DINOv2's best), ``ncc`` (near-tie, the referee is sure of another tied
    candidate), ``dino-tie`` / ``dino-low`` (blank), ``none`` (no refs).
    ``referee(candidates)`` returns {hero: registered correlation}; it is
    expensive, so it only runs on a near-tie."""
    min_sim = config.DINO_MIN_SIM if min_sim is None else min_sim
    clear_margin = config.DINO_CLEAR_MARGIN if clear_margin is None else clear_margin
    tie_topk = config.DINO_TIE_TOPK if tie_topk is None else tie_topk
    if not rank:
        return None, 0.0, "none"
    s1, h1 = rank[0]
    s2 = rank[1][0] if len(rank) > 1 else -1.0
    if s1 < min_sim:
        return None, s1, "dino-low"
    if s1 - s2 >= clear_margin:
        return h1, s1, "dino"
    tied = [h for s, h in rank[:tie_topk] if s >= s1 - clear_margin]
    scores = referee(tied) if referee is not None else {}
    if scores:
        order = sorted(scores.items(), key=lambda kv: -kv[1])
        best, n1 = order[0]
        n2 = order[1][1] if len(order) > 1 else -1.0
        if best.lower() == h1.lower() and n1 >= config.PICK_NCC_AGREE:
            return h1, s1, "dino+ncc"
        if n1 >= config.PICK_NCC_STRONG and n1 - n2 >= config.PICK_NCC_MARGIN:
            return best, next(s for s, h in rank if h == best), "ncc"
    return None, s1, "dino-tie"


def sim_of(rank: Rank, hero: str, topk: Optional[int] = None) -> Optional[float]:
    """Similarity of ``hero`` in ``rank``, or None if it isn't within ``topk``."""
    for s, h in (rank if topk is None else rank[:topk]):
        if h.lower() == hero.lower():
            return s
    return None


def format_rank(rank: Rank, n: int = 3) -> str:
    return ", ".join(f"{h}:{s:.2f}" for s, h in rank[:n])


@dataclass
class Verdict:
    hero: Optional[str]
    score: float
    method: str
    locked: bool = True                 # False: a hovered (pre-selected) pick
    rank: Rank = field(default_factory=list)
    detail: str = ""


# ===========================================================================
#  REFERENCE INDEX  (pure numpy)
# ===========================================================================
class HeroIndex:
    """Reference fingerprints, one row per image.  A hero usually has several
    (downloaded art in several framings, crops remembered from the screen); a
    hero's similarity to a query is its best row.

    Rows are CLEAN (downloaded portraits) or SCREEN (crops taken off the draft
    screen).  A screen crop carries the slot's overlay - ring, badges, the red
    ban slash - which every other slot of that type shares, so it resembles
    all of them (different heroes: up to 0.79).  It therefore only counts as
    evidence at DINO_SCREEN_MIN or above, where a genuine re-sighting of the
    same art lands (0.90+ even with a 4 px box shift)."""

    def __init__(self):
        self._vecs = None                       # (n_refs, dim) float32
        self._hero_ids = np.zeros((0,), np.int64)
        self._screen = np.zeros((0,), bool)
        self._heroes: List[str] = []            # id -> display name
        self._ids: Dict[str, int] = {}          # lower-case name -> id

    def _hero_id(self, hero: str) -> int:
        key = hero.lower()
        if key not in self._ids:
            self._ids[key] = len(self._heroes)
            self._heroes.append(hero)
        return self._ids[key]

    def add(self, hero: str, vecs, screen: bool = False) -> None:
        vecs = np.atleast_2d(np.asarray(vecs, np.float32))
        ids = np.full(len(vecs), self._hero_id(hero), np.int64)
        self._vecs = vecs.copy() if self._vecs is None else np.vstack([self._vecs, vecs])
        self._hero_ids = np.concatenate([self._hero_ids, ids])
        self._screen = np.concatenate([self._screen, np.full(len(vecs), screen)])

    @classmethod
    def from_rows(cls, heroes: Sequence[str], vecs,
                  screen: Optional[Sequence[bool]] = None) -> "HeroIndex":
        idx = cls()
        if len(heroes) == 0:
            return idx
        idx._vecs = np.asarray(vecs, np.float32)
        idx._hero_ids = np.array([idx._hero_id(h) for h in heroes], np.int64)
        idx._screen = (np.zeros(len(heroes), bool) if screen is None
                       else np.asarray(screen, bool))
        return idx

    def __len__(self) -> int:
        return len(self._heroes)

    @property
    def n_refs(self) -> int:
        return 0 if self._vecs is None else self._vecs.shape[0]

    def heroes(self) -> List[str]:
        return list(self._heroes)

    def refs_of(self, hero: str, clean_only: bool = False):
        hid = self._ids.get(hero.lower())
        if hid is None or self._vecs is None:
            return np.zeros((0, 0), np.float32)
        rows = self._hero_ids == hid
        if clean_only:
            rows &= ~self._screen
        return self._vecs[rows]

    @property
    def n_screen(self) -> int:
        return int(self._screen.sum())

    def rank(self, query) -> Rank:
        if self._vecs is None:
            return []
        sims = self._vecs @ np.asarray(query, np.float32)
        if self._screen.any():
            sims = np.where(self._screen & (sims < config.DINO_SCREEN_MIN),
                            -np.inf, sims)
        best = np.full(len(self._heroes), -np.inf, np.float32)
        np.maximum.at(best, self._hero_ids, sims)
        order = np.argsort(-best, kind="stable")
        return [(float(best[i]), self._heroes[i]) for i in order
                if np.isfinite(best[i])]


def _content_key(img, signature: str) -> str:
    """Cache key of the exact pixels being embedded (so identical images from
    different folders - or a folder of mirrored art - are embedded once)."""
    h = hashlib.md5(np.ascontiguousarray(img).tobytes())
    h.update(repr(img.shape).encode())
    h.update(signature.encode())
    return h.hexdigest()


def _load_cache(path: Optional[str]) -> Dict[str, "np.ndarray"]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            return dict(zip(z["keys"].tolist(), z["vecs"]))
    except Exception:
        return {}                               # corrupt/old cache: rebuild


def _save_cache(path: Optional[str], cache: Dict[str, "np.ndarray"]) -> None:
    if not path or not cache:
        return
    keys = list(cache)
    tmp = path + ".tmp.npz"
    np.savez(tmp, keys=np.array(keys), vecs=np.stack([cache[k] for k in keys]))
    os.replace(tmp, path)


# ===========================================================================
#  QUERY / REFERENCE VIEWS
# ===========================================================================
def pick_view(crop: "np.ndarray") -> "np.ndarray":
    """What DINOv2 sees of a pick slot: the portrait minus its frame edge (where
    the slot border and any overlay box line sit)."""
    return trim(crop[:, :, :3], config.PICK_TRIM)


def _centre(img: "np.ndarray", zoom: float) -> "np.ndarray":
    if zoom >= 0.999:
        return img
    h, w = img.shape[:2]
    m = max(8, int(round(min(h, w) * zoom)))
    y, x = (h - m) // 2, (w - m) // 2
    return img[y:y + m, x:x + m]


def _soft(img: "np.ndarray", px: int) -> "np.ndarray":
    h, w = img.shape[:2]
    if px <= 0 or min(h, w) <= px:
        return img
    small = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def pick_reference_views(img: "np.ndarray") -> List["np.ndarray"]:
    """The framings pick slots show one portrait in: both orientations (the
    enemy column mirrors the art), zoomed in, and at screen softness."""
    out = []
    for o in (img, cv2.flip(img, 1)):
        for z in config.PICK_REF_ZOOMS:
            v = _centre(o, z)
            out.append(v)
            if config.PICK_REF_SOFT:
                out.append(_soft(v, config.PICK_REF_SOFT))
    return out


def screen_kind(path: str, img: "np.ndarray", layout=None) -> str:
    """'pick' or 'ban' for a crop remembered from the screen.  Memory files
    say so in their name (``<hero>__pick__<id>.png``); older crops are sized
    like the slot they came from."""
    parts = os.path.splitext(os.path.basename(path))[0].split("__")
    if len(parts) >= 3 and parts[1] in ("pick", "ban"):
        return parts[1]
    layout = config.LAYOUT if layout is None else layout
    side = min(img.shape[:2])
    try:
        bans = [min(b.w, b.h) for b in list(layout.ally_bans) + list(layout.enemy_bans)]
        picks = [min(b.w, b.h) for b in list(layout.ally_picks) + list(layout.enemy_picks)]
        return ("ban" if abs(side - float(np.median(bans)))
                < abs(side - float(np.median(picks))) else "pick")
    except Exception:
        return "ban" if side < 64 else "pick"


# ===========================================================================
#  REFERENCES  (every reference image, embedded once and cached)
# ===========================================================================
@dataclass
class References:
    pick: HeroIndex                     # clean framings + remembered pick crops
    ban: Dict[str, HeroIndex]           # side -> clean art rendered as ban icons
    screen_ban: HeroIndex               # remembered ban crops (heroes w/o art)
    ignored: List[str] = field(default_factory=list)   # rejected memories


def _image_files(dirs: Sequence[str]):
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for pat in IMAGE_EXTS:
            for fp in sorted(glob.glob(os.path.join(d, pat))):
                img = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img is not None:
                    yield fp, img


def build_references(embedder, icons: IconMatcher, clean_dirs: Sequence[str],
                     screen_dirs: Sequence[str] = (), cache_path: Optional[str] = None,
                     log: Callable[[str], None] = print,
                     ban_px: Optional[int] = None) -> References:
    """Embed every reference view.  Embeddings are cached by pixel content, so
    after the first run only new or changed images are embedded.

    Crops remembered from the screen are double-checked here too: one that
    looks clearly more like another hero than the one it is filed under (an
    old mislabel) is ignored rather than allowed to repeat the mistake."""
    signature = embedder.signature
    cache = _load_cache(cache_path)
    pending: Dict[str, "np.ndarray"] = {}
    used = set()

    def key_of(img) -> str:
        k = _content_key(img, signature)
        used.add(k)
        if k not in cache:
            pending.setdefault(k, img)
        return k

    # Clean pick framings, deduplicated by content (the template folders
    # hold the same art, some of it mirrored).
    pick_rows: List[Tuple[str, str]] = []
    seen = set()
    for fp, img in _image_files(clean_dirs):
        hero = hero_name_from_file(fp)
        for v in pick_reference_views(img):
            k = key_of(v)
            if (hero.lower(), k) not in seen:
                seen.add((hero.lower(), k))
                pick_rows.append((hero, k))
    # Ban icons: each hero's art rendered at this screen's icon size.
    ban_px = ban_diameter() if ban_px is None else ban_px
    ban_rows = {side: [(h, key_of(icons.ban_reference(h, side, ban_px)))
                       for h in icons.names] for side in SIDES}
    # Remembered screen crops.
    screen_pick: List[Tuple[str, str, str]] = []
    screen_ban: List[Tuple[str, str, str, "np.ndarray"]] = []
    for fp, img in _image_files(screen_dirs):
        hero = hero_name_from_file(fp)
        if screen_kind(fp, img) == "pick":
            screen_pick.append((hero, key_of(pick_view(img)), fp))
        else:
            screen_ban.append((hero, key_of(img[:, :, :3]), fp, img))

    if pending:
        keys = list(pending)
        log(f"[dino] fingerprinting {len(keys)} reference images "
            f"(only when the art changes)...")
        t0 = time.perf_counter()
        step = 256
        for i in range(0, len(keys), step):
            chunk = keys[i:i + step]
            for k, v in zip(chunk, embedder.embed([pending[k] for k in chunk])):
                cache[k] = v
            if len(keys) > step:
                log(f"[dino]   {min(i + step, len(keys))}/{len(keys)}")
        log(f"[dino]   done in {time.perf_counter() - t0:.1f}s")
        _save_cache(cache_path, {k: cache[k] for k in used})

    def index(rows, screen=False) -> HeroIndex:
        if not rows:
            return HeroIndex()
        return HeroIndex.from_rows([h for h, k in rows],
                                   np.stack([cache[k] for h, k in rows]),
                                   screen=[screen] * len(rows))

    pick = index(pick_rows)
    ignored: List[str] = []
    for hero, k, fp in screen_pick:
        vec = cache[k]
        if len(pick.refs_of(hero, clean_only=True)):
            rank = pick.rank(vec)
            own = sim_of(rank, hero)
            if own is None or own < rank[0][0] - config.DINO_SCREEN_TOLERANCE:
                ignored.append(f"{fp} (filed as {hero}, looks like "
                               f"{rank[0][1]} {rank[0][0]:.2f} vs {own or 0:.2f})")
                continue
        pick.add(hero, vec, screen=True)
    sban = HeroIndex()
    for hero, k, fp, img in screen_ban:
        if icons.has(hero):
            continue            # the art itself recognises this hero's ban icon
        verdicts = [decide_ban_ncc(icons.rank_ban(img, side)) for side in SIDES]
        other = next((v for v in verdicts if v[0] is not None), None)
        if other is not None:
            ignored.append(f"{fp} (filed as {hero}, but it is {other[0]}'s "
                           f"ban icon {other[1]:.2f})")
            continue
        sban.add(hero, cache[k], screen=True)
    return References(pick, {s: index(ban_rows[s]) for s in SIDES}, sban, ignored)


# ===========================================================================
#  DINOv2 EMBEDDER  (needs torch)
# ===========================================================================
class DinoEmbedder:
    """DINOv2 feature extractor: CLS token + mean patch token of the last
    hidden state, L2-normalised - the exact pipeline measured on the real
    draft crops."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 threads: Optional[int] = None, log: Callable[[str], None] = print):
        if not ai_stack_available():
            raise RuntimeError("torch + torchvision + transformers + pillow are "
                               "required (pip install -r requirements-ai.txt)")
        self.model_name = model_name or config.DINO_MODEL
        threads = config.DINO_THREADS if threads is None else threads
        if threads and threads > 0:
            torch.set_num_threads(min(int(threads), os.cpu_count() or 1))
        dev = (device or config.DINO_DEVICE).lower()
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev
        self.processor, self.model = self._load(log)
        self.model.to(self.device).eval()
        self.dim = 2 * int(self.model.config.hidden_size)

    def _load(self, log):
        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        try:                                    # cached -> no network needed
            return (AutoImageProcessor.from_pretrained(self.model_name,
                                                       local_files_only=True),
                    AutoModel.from_pretrained(self.model_name, local_files_only=True))
        except OSError:                         # not in the local cache yet
            log(f"[dino] downloading {self.model_name} (one-time, ~90 MB)...")
            return (AutoImageProcessor.from_pretrained(self.model_name),
                    AutoModel.from_pretrained(self.model_name))

    @property
    def signature(self) -> str:
        """Everything that shapes an embedding (part of every cache key)."""
        return f"{self.model_name}|{PREP_VERSION}|transformers-{transformers.__version__}"

    @staticmethod
    def _pil(bgr):
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        return Image.fromarray(cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB))

    def embed(self, images_bgr: Sequence, batch: Optional[int] = None):
        batch = batch or config.DINO_BATCH
        out = []
        for i in range(0, len(images_bgr), batch):
            pil = [self._pil(im) for im in images_bgr[i:i + batch]]
            with torch.inference_mode():
                x = self.processor(images=pil, return_tensors="pt")
                x = {k: v.to(self.device) for k, v in x.items()}
                h = self.model(**x).last_hidden_state
                v = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
                v = torch.nn.functional.normalize(v, dim=1)
            out.append(v.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.dim), np.float32)
        return np.concatenate(out)


# ===========================================================================
#  RECOGNIZER  (embedder + references + double-check + memory)
# ===========================================================================
class DinoRecognizer:
    def __init__(self, embedder, refs: References, icons: IconMatcher,
                 learn_dir: Optional[str] = None):
        self.embedder = embedder
        self.refs = refs
        self.index = refs.pick                  # pick references (diagnostics)
        self.icons = icons
        self.learn_dir = learn_dir
        self.embed_calls = 0                    # images embedded (tests/diagnostics)

    @classmethod
    def load(cls, clean_dirs: Optional[Sequence[str]] = None,
             screen_dirs: Optional[Sequence[str]] = None,
             cache_path: Optional[str] = None, learn_dir: Optional[str] = None,
             icons: Optional[IconMatcher] = None,
             log: Callable[[str], None] = print) -> Optional["DinoRecognizer"]:
        """Build the recognizer, or return None - with the reason logged - when
        the AI stack or the model is unavailable.  ``learn_dir`` (where new
        memories are saved) is always read as a screen folder too."""
        if not ai_stack_available():
            log("[dino] torch/torchvision/transformers/pillow not installed - using "
                "template matching.  To enable: pip install -r requirements-ai.txt")
            return None
        clean_dirs = list(config.DINO_CLEAN_DIRS if clean_dirs is None else clean_dirs)
        screen_dirs = list(config.DINO_SCREEN_DIRS if screen_dirs is None else screen_dirs)
        learn_dir = config.DINO_LEARN_DIR if learn_dir is None else learn_dir
        if learn_dir and learn_dir not in screen_dirs:
            screen_dirs.append(learn_dir)
        cache_path = config.DINO_CACHE_PATH if cache_path is None else cache_path
        try:
            t0 = time.perf_counter()
            icons = icons if icons is not None else IconMatcher.from_dirs()
            emb = DinoEmbedder(log=log)
            refs = build_references(emb, icons, clean_dirs, screen_dirs,
                                    cache_path, log)
        except Exception as exc:
            log(f"[dino] unavailable ({type(exc).__name__}: {exc}) - "
                "using template matching")
            return None
        if len(refs.pick) == 0:
            log("[dino] no reference images found - using template matching")
            return None
        for msg in refs.ignored:
            log(f"[dino] ignoring a remembered crop that fails the double-check: {msg}")
        log(f"[dino] ready: {emb.model_name} on {emb.device}, {len(refs.pick)} heroes "
            f"/ {refs.pick.n_refs} pick references ({refs.pick.n_screen} from your "
            f"screen), ban icons for {len(icons)} heroes "
            f"({time.perf_counter() - t0:.1f}s)")
        return cls(emb, refs, icons, learn_dir)

    sim_of = staticmethod(sim_of)
    format_rank = staticmethod(format_rank)
    pick_view = staticmethod(pick_view)

    @property
    def label(self) -> str:
        return (f"{getattr(self.embedder, 'model_name', 'dino')} on "
                f"{getattr(self.embedder, 'device', '?')}")

    def embed(self, images: Sequence):
        self.embed_calls += len(images)
        return self.embedder.embed(list(images))

    def ban_view(self, crop, side: str, match: BanMatch):
        return self.icons.ban_view(crop, match.geometry, side)

    # ----- decisions ----------------------------------------------------------
    def decide_pick(self, crop, vec) -> Verdict:
        rank = self.refs.pick.rank(vec)
        fits: Dict = {}
        hero, score, method = decide(rank, self.icons.referee(crop, fits))
        locked = True
        if hero is not None:
            if hero not in fits:
                fits[hero] = self.icons.fit_pick(crop, hero)
            locked = not is_hover(fits[hero])
        detail = f"{method} {score:.2f} | top3 {format_rank(rank)}"
        judged = {n: f for n, f in fits.items() if f is not None}
        if judged:
            detail += " | fit " + ", ".join(f"{n}:{f.ncc:.2f}" for n, f in judged.items())
        if hero is not None and hero in judged:
            detail += f" | contrast {judged[hero].contrast:.2f}"
            if not locked:
                detail += " (hover - not locked)"
        return Verdict(hero, score, method, locked, rank, detail)

    def decide_ban(self, crop, side: str, match: BanMatch, vec=None) -> Verdict:
        hero, score, method = decide_ban_ncc(match)
        drank = self.refs.ban[side].rank(vec) if vec is not None else []
        if hero is not None and drank:
            own = sim_of(drank, hero, config.BAN_DINO_TOPK)
            if own is None or own < drank[0][0] - config.BAN_DINO_TOLERANCE:
                hero, method = None, "ncc/dino-conflict"
            else:
                method = "ncc+dino"
        elif hero is None and len(self.refs.screen_ban):
            # No public art matches: a crop remembered from the screen may
            # (heroes missing from every portrait DB, e.g. Sora).
            srank = self.refs.screen_ban.rank(self.embed([crop[:, :, :3]])[0])
            if srank and (len(srank) == 1 or srank[0][0] - srank[1][0] >= 0.02):
                score, hero = srank[0]
                method = "memory"
        detail = (f"{method} {score:.2f} | art {format_rank(match.rank)}"
                  + (f" | dino {format_rank(drank)}" if drank else ""))
        return Verdict(hero, score, method, True, match.rank, detail)

    # ----- memory -------------------------------------------------------------
    def maybe_learn(self, hero: str, crop, vec, rank: Rank) -> bool:
        """Remember a confident clear win (a locked pick) as an extra reference
        for ``hero`` - saved to learn_dir and live immediately.  Skips crops
        that add nothing (near-duplicates) and caps references per hero."""
        if not self.learn_dir or not rank or rank[0][1].lower() != hero.lower():
            return False
        s1 = rank[0][0]
        s2 = rank[1][0] if len(rank) > 1 else -1.0
        if s1 < config.DINO_LEARN_MIN or s1 - s2 < config.DINO_LEARN_MARGIN:
            return False
        refs = self.refs.pick.refs_of(hero)
        if len(refs) and float(np.max(refs @ vec)) >= config.DINO_LEARN_DUP_SIM:
            return False
        stem = hero.strip().lower()
        if len(glob.glob(os.path.join(self.learn_dir, f"{glob.escape(stem)}__*.png"))) \
                >= config.DINO_LEARN_MAX_PER_HERO:
            return False
        try:
            os.makedirs(self.learn_dir, exist_ok=True)
            uid = hashlib.md5(np.ascontiguousarray(crop).tobytes()).hexdigest()[:10]
            cv2.imwrite(os.path.join(self.learn_dir, f"{stem}__pick__{uid}.png"), crop)
        except Exception:
            return False
        self.refs.pick.add(hero, vec, screen=True)
        return True

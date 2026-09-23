"""
recognizer.py
=============
DINOv2 hero recognition.  Every hero portrait becomes a feature "fingerprint"
(DINOv2-small CLS token + mean patch token, L2-normalised), and a slot is named
by its nearest reference, then double-checked:

    best similarity  < DINO_MIN_SIM                 -> blank (empty / junk slot)
    best leads the runner-up by >= DINO_CLEAR_MARGIN -> that hero
    near-tie                                        -> the template matcher's
                                                       pick, if it is one of
                                                       DINOv2's top candidates;
                                                       otherwise blank

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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import config

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

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp")
PREP_VERSION = "cls+meanpatch/v1"       # bump to invalidate cached embeddings

Rank = List[Tuple[float, str]]          # (similarity, hero), best first


def ai_stack_available() -> bool:
    return None not in (torch, transformers, AutoModel, Image, np, cv2)


def hero_name_from_file(path: str) -> str:
    """Hero name for a reference file, normalised like the template library.
    Memory files are named ``<hero>__<id>.png`` so a hero can keep several."""
    stem = os.path.splitext(os.path.basename(path))[0].split("__", 1)[0]
    return stem.replace("_", " ").strip().title()


# ===========================================================================
#  THE DOUBLE-CHECK  (pure python)
# ===========================================================================
def decide(rank: Rank, tiebreak: Optional[Callable[[], Optional[str]]] = None,
           min_sim: Optional[float] = None, clear_margin: Optional[float] = None,
           tie_topk: Optional[int] = None) -> Tuple[Optional[str], float, str]:
    """Turn a similarity ranking into (hero or None, similarity, method).

    method: ``dino`` (clear win), ``dino+tmpl`` (near-tie settled by the
    template matcher), ``dino-tie`` / ``dino-low`` (blank), ``none`` (no refs).
    ``tiebreak`` returns the template matcher's top hero; it is expensive, so
    it is only called on a near-tie."""
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
    pick = tiebreak() if tiebreak is not None else None
    if pick:
        for s, h in rank[:tie_topk]:
            if h.lower() == pick.lower():
                return h, s, "dino+tmpl"
    return None, s1, "dino-tie"


def sim_of(rank: Rank, hero: str, topk: Optional[int] = None) -> Optional[float]:
    """Similarity of ``hero`` in ``rank``, or None if it isn't within ``topk``."""
    for s, h in (rank if topk is None else rank[:topk]):
        if h.lower() == hero.lower():
            return s
    return None


def format_rank(rank: Rank, n: int = 3) -> str:
    return ", ".join(f"{h}:{s:.2f}" for s, h in rank[:n])


# ===========================================================================
#  REFERENCE INDEX  (pure numpy)
# ===========================================================================
class HeroIndex:
    """Reference fingerprints, one row per image.  A hero usually has several
    (downloaded art, its mirror, crops remembered from the screen); a hero's
    similarity to a query is its best row.

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

    def refs_of(self, hero: str):
        hid = self._ids.get(hero.lower())
        if hid is None or self._vecs is None:
            return np.zeros((0, 0), np.float32)
        return self._vecs[self._hero_ids == hid]

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


def build_index(embedder, clean_dirs: Sequence[str], screen_dirs: Sequence[str] = (),
                cache_path: Optional[str] = None,
                log: Callable[[str], None] = print) -> HeroIndex:
    """Embed every reference image (and its horizontal mirror, so orientation
    never matters): downloaded art from ``clean_dirs``, crops taken off the
    screen from ``screen_dirs``.  Embeddings are cached by pixel content, so
    after the first run only new or changed images are embedded."""
    signature = embedder.signature
    rows: List[Tuple[str, str, bool]] = []      # (hero, key, is_screen)
    pending: Dict[str, "np.ndarray"] = {}       # key -> image still to embed
    cache = _load_cache(cache_path)
    seen = set()
    sources = [(d, False) for d in clean_dirs] + [(d, True) for d in screen_dirs]
    for d, is_screen in sources:
        if not d or not os.path.isdir(d):
            continue
        for pat in IMAGE_EXTS:
            for fp in sorted(glob.glob(os.path.join(d, pat))):
                img = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                hero = hero_name_from_file(fp)
                for variant in (img, cv2.flip(img, 1)):
                    key = _content_key(variant, signature)
                    if (hero.lower(), key) in seen:
                        continue
                    seen.add((hero.lower(), key))
                    rows.append((hero, key, is_screen))
                    if key not in cache:
                        pending.setdefault(key, variant)
    if pending:
        keys = list(pending)
        log(f"[dino] embedding {len(keys)} reference images "
            f"(only when the art changes)...")
        t0 = time.perf_counter()
        for k, v in zip(keys, embedder.embed([pending[k] for k in keys])):
            cache[k] = v
        log(f"[dino]   done in {time.perf_counter() - t0:.1f}s")
        _save_cache(cache_path, {k: cache[k] for _, k, _ in rows})
    if not rows:
        return HeroIndex()
    return HeroIndex.from_rows([h for h, _, _ in rows],
                               np.stack([cache[k] for _, k, _ in rows]),
                               screen=[s for _, _, s in rows])


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
#  RECOGNIZER  (embedder + index + double-check + memory)
# ===========================================================================
class DinoRecognizer:
    def __init__(self, embedder, index: HeroIndex, learn_dir: Optional[str] = None):
        self.embedder = embedder
        self.index = index
        self.learn_dir = learn_dir
        self.embed_calls = 0                    # images embedded (tests/diagnostics)

    @classmethod
    def load(cls, clean_dirs: Optional[Sequence[str]] = None,
             screen_dirs: Optional[Sequence[str]] = None,
             cache_path: Optional[str] = None, learn_dir: Optional[str] = None,
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
            emb = DinoEmbedder(log=log)
            index = build_index(emb, clean_dirs, screen_dirs, cache_path, log)
        except Exception as exc:
            log(f"[dino] unavailable ({type(exc).__name__}: {exc}) - "
                "using template matching")
            return None
        if len(index) == 0:
            log("[dino] no reference images found - using template matching")
            return None
        log(f"[dino] ready: {emb.model_name} on {emb.device}, {len(index)} heroes / "
            f"{index.n_refs} references ({index.n_screen} from your screen) "
            f"({time.perf_counter() - t0:.1f}s)")
        return cls(emb, index, learn_dir)

    sim_of = staticmethod(sim_of)
    format_rank = staticmethod(format_rank)

    @property
    def label(self) -> str:
        return (f"{getattr(self.embedder, 'model_name', 'dino')} on "
                f"{getattr(self.embedder, 'device', '?')}")

    def embed(self, crops: Sequence):
        self.embed_calls += len(crops)
        return self.embedder.embed(list(crops))

    def rank(self, vec) -> Rank:
        return self.index.rank(vec)

    def decide(self, rank: Rank, tiebreak=None):
        return decide(rank, tiebreak)

    def maybe_learn(self, hero: str, crop, vec, rank: Rank) -> bool:
        """Remember a confident clear win as an extra reference for ``hero``
        (saved to learn_dir and live immediately).  Skips crops that add
        nothing (near-duplicates) and caps references per hero."""
        if not self.learn_dir or not rank or rank[0][1].lower() != hero.lower():
            return False
        s1 = rank[0][0]
        s2 = rank[1][0] if len(rank) > 1 else -1.0
        if s1 < config.DINO_LEARN_MIN or s1 - s2 < config.DINO_LEARN_MARGIN:
            return False
        refs = self.index.refs_of(hero)
        if len(refs) and float(np.max(refs @ vec)) >= config.DINO_LEARN_DUP_SIM:
            return False
        stem = hero.strip().lower()
        if len(glob.glob(os.path.join(self.learn_dir, f"{glob.escape(stem)}__*.png"))) \
                >= config.DINO_LEARN_MAX_PER_HERO:
            return False
        try:
            os.makedirs(self.learn_dir, exist_ok=True)
            uid = hashlib.md5(np.ascontiguousarray(crop).tobytes()).hexdigest()[:10]
            cv2.imwrite(os.path.join(self.learn_dir, f"{stem}__{uid}.png"), crop)
        except Exception:
            return False
        self.index.add(hero, vec, screen=True)
        return True

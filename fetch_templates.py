"""
fetch_templates.py
==================
One-shot downloader that fills ``templates/`` with a portrait for EVERY hero,
so you never have to screenshot 130 heroes by hand.

It reads a JSON source that pairs each hero with an image URL, downloads each
picture, centre-crops it to a square and saves it as ``templates/<hero>.png``
(named so the detector resolves it against ``heroes.json``).

Default source
--------------
A public community database (p3hndrx/MLBB-API) whose ``portrait`` field is a
128x128 square hosted on Moonton's own CDN.  It is fully pluggable - point it
at any JSON via flags / ``config.TEMPLATE_SOURCE_*``:

    python fetch_templates.py                 # download the whole roster
    python fetch_templates.py --only-db       # only heroes in heroes.json
    python fetch_templates.py --overwrite     # re-download existing files
    python fetch_templates.py --source-url <url> --image-key icon --record-path data

Notes
-----
* These portraits are Moonton game assets fetched for personal, local use as
  template-matching references; they are git-ignored and never committed.
* The portraits frame the hero like the in-game locked slot, so they match well
  with the histogram fallback; for the last few % of accuracy you can later drop
  a hand-cropped draft portrait over any specific file.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import config

try:
    import numpy as np
    import cv2
except Exception:                                   # pragma: no cover
    np = None
    cv2 = None

_UA = {"User-Agent": "mlbb-draft-overlay/1.0 (+templates)"}


def _require_cv() -> None:
    if np is None or cv2 is None:
        raise RuntimeError("numpy + opencv-python required: "
                           "pip install numpy opencv-python")


def _norm(s: object) -> str:
    """Aggressive normaliser for fuzzy name matching ('X.Borg' -> 'xborg')."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# Some community sources carry a sentinel "None"/"null" hero at id 0.
_PLACEHOLDER = {"", "none", "null"}


def _is_placeholder(name: object) -> bool:
    return _norm(name) in _PLACEHOLDER


def _dig(payload: object, dotted: Optional[str]) -> object:
    node = payload
    for part in (dotted.split(".") if dotted else []):
        node = node[part]                          # type: ignore[index]
    return node


def _http_json(url: str, timeout: float) -> object:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_bytes(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _square(img: "np.ndarray", size: int) -> "np.ndarray":
    """Centre-crop to a square and resize to ``size`` x ``size``."""
    h, w = img.shape[:2]
    m = min(h, w)
    y, x = (h - m) // 2, (w - m) // 2
    crop = img[y:y + m, x:x + m]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _safe_filename(name: str) -> str:
    """Filesystem-safe template filename; the detector lower/title-cases and
    resolves it against heroes.json case-insensitively."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "", name).strip().lower()
    return f"{cleaned}.png"


def build_index(entries: Sequence[dict],
                name_keys: Sequence[str]) -> Dict[str, dict]:
    """Map normalised hero name/slug -> source record."""
    idx: Dict[str, dict] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        for k in name_keys:
            v = e.get(k)
            if v and not _is_placeholder(v):
                idx.setdefault(_norm(v), e)
    return idx


def resolve(name: str, idx: Dict[str, dict], fuzzy: float = 0.86
            ) -> Optional[dict]:
    key = _norm(name)
    if key in idx:
        return idx[key]
    match = difflib.get_close_matches(key, list(idx.keys()), n=1, cutoff=fuzzy)
    return idx[match[0]] if match else None


def fetch_templates(source_url: str = config.TEMPLATE_SOURCE_URL,
                    record_path: Optional[str] = config.TEMPLATE_SOURCE_RECORD_PATH,
                    name_keys: Sequence[str] = config.TEMPLATE_SOURCE_NAME_KEYS,
                    image_key: str = config.TEMPLATE_SOURCE_IMAGE_KEY,
                    out_dir: str = config.TEMPLATE_DIR,
                    heroes_path: str = "heroes.json",
                    size: int = config.TEMPLATE_FETCH_SIZE,
                    only_db: bool = False,
                    overwrite: bool = False,
                    limit: Optional[int] = None,
                    timeout: float = 20.0) -> Tuple[int, int, List[str]]:
    """Download portraits. Returns (saved, skipped, unmatched_names)."""
    _require_cv()
    os.makedirs(out_dir, exist_ok=True)

    payload = _http_json(source_url, timeout)
    records = _dig(payload, record_path)
    if isinstance(records, dict):
        records = list(records.values())
    if not isinstance(records, list):
        raise ValueError("source record_path did not yield a list of heroes")
    idx = build_index(records, name_keys)

    # Decide what to download.
    if only_db:
        with open(heroes_path, "r", encoding="utf-8") as fh:
            db = json.load(fh)
        wanted = [(h["name"], resolve(h["name"], idx)) for h in db["heroes"]]
    else:
        # whole roster straight from the source
        seen = set()
        wanted = []
        for e in records:
            nm = e.get(name_keys[0]) or e.get("name")
            if nm and not _is_placeholder(nm) and _norm(nm) not in seen:
                seen.add(_norm(nm))
                wanted.append((str(nm), e))

    if limit is not None:
        wanted = wanted[:limit]

    saved = skipped = 0
    unmatched: List[str] = []
    total = len(wanted)
    for i, (display_name, entry) in enumerate(wanted, 1):
        if entry is None or not entry.get(image_key):
            unmatched.append(display_name)
            print(f"[{i:3}/{total}] {display_name:18} -- no source image", file=sys.stderr)
            continue
        path = os.path.join(out_dir, _safe_filename(display_name))
        if os.path.exists(path) and not overwrite:
            skipped += 1
            print(f"[{i:3}/{total}] {display_name:18} -- exists, skip")
            continue
        try:
            raw = _http_bytes(entry[image_key], timeout)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("decode failed")
            cv2.imwrite(path, _square(img, size))
            saved += 1
            print(f"[{i:3}/{total}] {display_name:18} -> {os.path.basename(path)}")
        except Exception as exc:                    # keep going on any failure
            unmatched.append(display_name)
            print(f"[{i:3}/{total}] {display_name:18} -- FAILED: {exc}", file=sys.stderr)

    return saved, skipped, unmatched


def main() -> int:
    p = argparse.ArgumentParser(description="Download hero portrait templates.")
    p.add_argument("--source-url", default=config.TEMPLATE_SOURCE_URL)
    p.add_argument("--record-path", default=config.TEMPLATE_SOURCE_RECORD_PATH)
    p.add_argument("--name-keys", default=",".join(config.TEMPLATE_SOURCE_NAME_KEYS),
                   help="comma-separated keys to match heroes by")
    p.add_argument("--image-key", default=config.TEMPLATE_SOURCE_IMAGE_KEY)
    p.add_argument("--out", default=config.TEMPLATE_DIR)
    p.add_argument("--heroes", default="heroes.json")
    p.add_argument("--size", type=int, default=config.TEMPLATE_FETCH_SIZE)
    p.add_argument("--only-db", action="store_true",
                   help="only download heroes present in heroes.json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="cap downloads (testing)")
    p.add_argument("--timeout", type=float, default=20.0)
    args = p.parse_args()

    try:
        saved, skipped, unmatched = fetch_templates(
            source_url=args.source_url,
            record_path=args.record_path,
            name_keys=tuple(k.strip() for k in args.name_keys.split(",")),
            image_key=args.image_key,
            out_dir=args.out,
            heroes_path=args.heroes,
            size=args.size,
            only_db=args.only_db,
            overwrite=args.overwrite,
            limit=args.limit,
            timeout=args.timeout,
        )
    except Exception as exc:
        sys.stderr.write(f"[fatal] {exc}\n")
        return 2

    print(f"\nDone: {saved} downloaded, {skipped} skipped, "
          f"{len(unmatched)} unmatched.")
    if unmatched:
        print("Unmatched (add manually or adjust --name-keys): "
              + ", ".join(unmatched))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

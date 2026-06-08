"""
import_icons.py
===============
Turn a LOCAL folder of hero icons into detector templates - including the
common case of **circular icons with transparent corners** (128x128 RGBA).

Why a separate step?  The detector reads templates as opaque BGR.  A circular
PNG has transparent corners whose RGB is undefined; if you let OpenCV drop the
alpha you get garbage/black fringes that hurt matching.  This tool composites
each icon over a solid background first, so every template is clean and square.

Usage
-----
    # extract your archive somewhere, then:
    python import_icons.py --src /path/to/heroes        # import ALL icons
    python import_icons.py --src ./heroes --only-db      # only heroes.json heroes
    python import_icons.py --src ./heroes --shape inscribed   # face-only crop
    python import_icons.py --src ./heroes --bg 12,18,16 --overwrite

Filenames become hero names (resolved against heroes.json case-insensitively),
so "Luo Yi.png", "X.Borg.png", "Yi Sun-shin.png" all just work.

Shapes
------
    bbox      (default) full icon composited on --bg; framing matches a slot box
              calibrated to the whole avatar.
    inscribed central square that fits inside the circle - pure face, zero
              transparency; use if your slot boxes are tight on the face.
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import config
from fetch_templates import _norm, _safe_filename

try:
    import numpy as np
    import cv2
except Exception:                                   # pragma: no cover
    np = None
    cv2 = None


def _require_cv() -> None:
    if np is None or cv2 is None:
        raise RuntimeError("numpy + opencv-python required: "
                           "pip install numpy opencv-python")


def composite_on_bg(img: "np.ndarray", bg_bgr: Tuple[int, int, int]) -> "np.ndarray":
    """Flatten a BGRA (or BGR) image onto a solid BGR background."""
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bg = np.array(bg_bgr, np.float32).reshape(1, 1, 3)
        out = bgr * alpha + bg * (1.0 - alpha)
        return out.astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, :3]


def inscribed_square(img: "np.ndarray") -> "np.ndarray":
    """Largest axis-aligned square fully inside the inscribed circle."""
    h, w = img.shape[:2]
    d = min(h, w)
    side = int(d / (2 ** 0.5))                      # diameter / sqrt(2)
    cy, cx = h // 2, w // 2
    half = side // 2
    return img[cy - half:cy + half, cx - half:cx + half]


def square_resize(img: "np.ndarray", size: int) -> "np.ndarray":
    h, w = img.shape[:2]
    m = min(h, w)
    y, x = (h - m) // 2, (w - m) // 2
    return cv2.resize(img[y:y + m, x:x + m], (size, size),
                      interpolation=cv2.INTER_AREA)


def _db_index(heroes_path: str) -> Dict[str, str]:
    with open(heroes_path, "r", encoding="utf-8") as fh:
        db = json.load(fh)["heroes"]
    return {_norm(h["name"]): h["name"] for h in db}


def import_icons(src: str,
                 out_dir: str = config.TEMPLATE_DIR,
                 size: int = config.TEMPLATE_FETCH_SIZE,
                 bg_bgr: Tuple[int, int, int] = (0, 0, 0),
                 shape: str = "bbox",
                 heroes_path: str = "heroes.json",
                 only_db: bool = False,
                 overwrite: bool = False
                 ) -> Tuple[int, int, List[str]]:
    """Returns (saved, skipped, skipped_not_in_db)."""
    _require_cv()
    if not os.path.isdir(src):
        raise NotADirectoryError(f"--src not found: {src}")
    os.makedirs(out_dir, exist_ok=True)

    files: List[str] = []
    for ext in ("*.png", "*.webp", "*.jpg", "*.jpeg"):
        files.extend(glob.glob(os.path.join(src, ext)))
    files.sort()

    # Always build the DB index so we can canonicalise filenames to our exact
    # spelling (e.g. pack's "Minsitthar" -> our "Minsithar"), which keeps the
    # detector's labels in sync with the engine.
    db_idx = _db_index(heroes_path)
    db_keys = list(db_idx.keys())
    saved = skipped = 0
    skipped_db: List[str] = []
    total = len(files)

    for i, f in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(f))[0]
        key = _norm(stem)
        canon = db_idx.get(key)
        if canon is None:
            match = difflib.get_close_matches(key, db_keys, n=1, cutoff=0.86)
            canon = db_idx[match[0]] if match else None
        if only_db and canon is None:
            skipped_db.append(stem)
            continue
        dst = os.path.join(out_dir, _safe_filename(canon if canon else stem))
        if os.path.exists(dst) and not overwrite:
            skipped += 1
            continue
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[{i:3}/{total}] {stem:18} -- unreadable, skip", file=sys.stderr)
            continue
        comp = composite_on_bg(img, bg_bgr)
        if shape == "inscribed":
            comp = inscribed_square(comp)
        cv2.imwrite(dst, square_resize(comp, size))
        saved += 1
        print(f"[{i:3}/{total}] {stem:18} -> {os.path.basename(dst)}")

    return saved, skipped, skipped_db


def _coverage(out_dir: str, heroes_path: str) -> Tuple[int, int, List[str]]:
    have = {_norm(os.path.splitext(f)[0]) for f in os.listdir(out_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))}
    with open(heroes_path, "r", encoding="utf-8") as fh:
        db = json.load(fh)["heroes"]
    missing = [h["name"] for h in db if _norm(h["name"]) not in have]
    return len(db) - len(missing), len(db), missing


def main() -> int:
    p = argparse.ArgumentParser(description="Import local hero icons as templates.")
    p.add_argument("--src", required=True, help="folder of icon images")
    p.add_argument("--out", default=config.TEMPLATE_DIR)
    p.add_argument("--size", type=int, default=config.TEMPLATE_FETCH_SIZE)
    p.add_argument("--bg", default="0,0,0", help="background as R,G,B for transparent corners")
    p.add_argument("--shape", choices=("bbox", "inscribed"), default="bbox")
    p.add_argument("--heroes", default="heroes.json")
    p.add_argument("--only-db", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    try:
        r, g, b = (int(x) for x in args.bg.split(","))
        bg_bgr = (b, g, r)                          # cv2 is BGR
    except Exception:
        sys.stderr.write("--bg must be 'R,G,B'\n")
        return 2

    try:
        saved, skipped, skipped_db = import_icons(
            src=args.src, out_dir=args.out, size=args.size, bg_bgr=bg_bgr,
            shape=args.shape, heroes_path=args.heroes,
            only_db=args.only_db, overwrite=args.overwrite)
    except Exception as exc:
        sys.stderr.write(f"[fatal] {exc}\n")
        return 2

    have, total, missing = _coverage(args.out, args.heroes)
    print(f"\nDone: {saved} written, {skipped} skipped"
          + (f", {len(skipped_db)} not in DB" if args.only_db else ""))
    print(f"heroes.json coverage: {have}/{total} have a template.")
    if missing:
        print("DB heroes still missing a template: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

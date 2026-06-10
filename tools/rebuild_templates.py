"""
tools/rebuild_templates.py
==========================
Nuke-and-rebuild EVERY template set from the up-to-date source, with one
canonical orientation policy:

    templates/         square base  (enemy library seed)        - as-is
    templates_circle/  circular     (ally picks + bans)         - as-is (right)
    templates_ally/    circular     (ally override pack)        - as-is (right)
    templates_enemy/   square       (enemy override pack)       - MIRRORED (left)

All four sets are regenerated for every hero in heroes.json from the same
fresh download (alpha-composited onto a dark background, centre-squared,
160x160), so the sets can never drift apart, stale art is impossible, and
stray/duplicate files (e.g. a second Minsitthar spelling) are wiped.

Heroes the source lacks fall back to existing local art (searched across the
old sets) so the rebuild never loses a hero.

    python tools/rebuild_templates.py            # rebuild everything
    python tools/rebuild_templates.py --dry-run  # show plan, write nothing
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from fetch_templates import (_http_json, _http_bytes, _dig, build_index,
                             resolve, _safe_filename)

try:
    import numpy as np
    import cv2
except Exception:                       # pragma: no cover
    np = cv2 = None

BG = (18, 18, 18)          # composite background (matches the dark draft UI)
SIZE = 160                 # canonical template size


def _decode(raw: bytes) -> "np.ndarray":
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("decode failed")
    return img


def _composite(img: "np.ndarray") -> "np.ndarray":
    """Flatten alpha onto BG so transparent corners can't skew matching."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        bg = np.empty_like(rgb)
        bg[:] = BG
        return (rgb * a + bg * (1.0 - a)).astype(np.uint8)
    return img[:, :, :3]


def _square(img: "np.ndarray", size: int = SIZE) -> "np.ndarray":
    h, w = img.shape[:2]
    m = min(h, w)
    y, x = (h - m) // 2, (w - m) // 2
    return cv2.resize(img[y:y + m, x:x + m], (size, size),
                      interpolation=cv2.INTER_AREA)


def _existing_art(fname_stem: str) -> "np.ndarray | None":
    """Fallback: pull the hero's art from whatever old set still has it."""
    for d in (config.TEMPLATE_CIRCLE_DIR, config.TEMPLATE_DIR,
              config.TEMPLATE_ALLY_DIR, config.TEMPLATE_ENEMY_DIR):
        for ext in ("png", "jpg", "jpeg", "webp"):
            p = os.path.join(d, f"{fname_stem}.{ext}")
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    return img
    return None


def _wipe_images(folder: str) -> int:
    """Remove every image in ``folder`` (keeps README/other files)."""
    n = 0
    for ext in ("png", "jpg", "jpeg", "webp"):
        for p in glob.glob(os.path.join(folder, f"*.{ext}")):
            os.remove(p)
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description="Rebuild all template sets from source.")
    p.add_argument("--source-url", default=config.TEMPLATE_SOURCE_URL)
    p.add_argument("--record-path", default=config.TEMPLATE_SOURCE_RECORD_PATH)
    p.add_argument("--image-key", default="portrait")
    p.add_argument("--heroes", default="heroes.json")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if cv2 is None:
        sys.stderr.write("needs numpy + opencv-python\n")
        return 2

    with open(args.heroes, "r", encoding="utf-8") as fh:
        heroes = [h["name"] for h in json.load(fh)["heroes"]]

    payload = _http_json(args.source_url, args.timeout)
    records = _dig(payload, args.record_path)
    if isinstance(records, dict):
        records = list(records.values())
    idx = build_index(records, config.TEMPLATE_SOURCE_NAME_KEYS)
    print(f"source: {len(records)} records | db: {len(heroes)} heroes\n")

    # Stage everything first; only swap the real folders if the build is sane.
    stage = tempfile.mkdtemp(prefix="tmpl_rebuild_")
    sets = {"square": os.path.join(stage, "square"),
            "circle": os.path.join(stage, "circle"),
            "ally": os.path.join(stage, "ally"),
            "enemy": os.path.join(stage, "enemy")}
    for d in sets.values():
        os.makedirs(d, exist_ok=True)

    fetched, fellback, missing = [], [], []
    for i, name in enumerate(sorted(heroes), 1):
        stem = os.path.splitext(_safe_filename(name))[0]
        rec = resolve(name, idx)
        img = None
        src = ""
        url = rec.get(args.image_key) if rec else None
        if url:
            try:
                img = _decode(_http_bytes(url, args.timeout))
                src = "source"
            except Exception as exc:
                print(f"[{i:3}] {name:18} download failed ({exc}); trying local",
                      file=sys.stderr)
        if img is None:
            img = _existing_art(stem)
            src = "local-fallback"
        if img is None:
            missing.append(name)
            print(f"[{i:3}] {name:18} -- NO ART ANYWHERE", file=sys.stderr)
            continue

        base = _square(_composite(img))            # as-is = ally/right-facing
        mirrored = cv2.flip(base, 1)               # enemy = left-facing
        cv2.imwrite(os.path.join(sets["square"], f"{stem}.png"), base)
        cv2.imwrite(os.path.join(sets["circle"], f"{stem}.png"), base)
        cv2.imwrite(os.path.join(sets["ally"], f"{stem}.png"), base)
        cv2.imwrite(os.path.join(sets["enemy"], f"{stem}.png"), mirrored)
        (fetched if src == "source" else fellback).append(name)
        print(f"[{i:3}] {name:18} {src}")

    built = len(fetched) + len(fellback)
    print(f"\nstaged {built}/{len(heroes)}  "
          f"({len(fetched)} fresh, {len(fellback)} kept-local, {len(missing)} missing)")
    if missing:
        print("missing entirely:", ", ".join(missing))
    if built < len(heroes) - 2:
        sys.stderr.write("too many missing - NOT swapping folders.\n")
        return 1
    if args.dry_run:
        print("(dry-run: folders untouched)")
        shutil.rmtree(stage, ignore_errors=True)
        return 0

    targets = {"square": config.TEMPLATE_DIR, "circle": config.TEMPLATE_CIRCLE_DIR,
               "ally": config.TEMPLATE_ALLY_DIR, "enemy": config.TEMPLATE_ENEMY_DIR}
    for key, dst in targets.items():
        os.makedirs(dst, exist_ok=True)
        removed = _wipe_images(dst)
        n = 0
        for f in sorted(os.listdir(sets[key])):
            shutil.move(os.path.join(sets[key], f), os.path.join(dst, f))
            n += 1
        print(f"{dst}: wiped {removed} old, wrote {n} new")
    shutil.rmtree(stage, ignore_errors=True)
    print("\nRebuild complete. Orientation: ally/circle/square as-is (right), "
          "enemy mirrored (left).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
